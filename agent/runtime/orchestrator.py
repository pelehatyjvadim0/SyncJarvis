from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Awaitable, Callable

from openai import AsyncOpenAI

from agent.config.settings import AppSettings
from agent.logging.history_logger import HistoryLogger
from agent.models.action import ActionResult, AgentAction
from agent.models.plan import ExecutionPlan, Subtask
from agent.models.state import AgentState
from agent.models.task import TaskMode
from agent.planner.plan_builder import build_execution_plan
from agent.policies.base import BaseTaskPolicy
from agent.runtime.goal_verifier import verify_user_goal_satisfied_llm
from agent.runtime.memory import RuntimeMemory
from agent.runtime.policy_registry import default_policies
from agent.runtime.react_loop.config import LoopConfig, RunCostStats
from agent.runtime.react_loop.loop import SubtaskReActLoop
from agent.tools.browser_executor import BrowserToolExecutor


@dataclass
class OrchestratorConfig:
    model_cheap: str
    model_smart: str
    temperature: float
    max_total_steps: int
    max_steps_per_subtask: int
    smart_cooldown_steps: int
    history_dir: str

    @classmethod
    def from_app_settings(cls, s: AppSettings) -> OrchestratorConfig:
        # Маппинг AppSettings в параметры оркестратора без чтения окружения здесь.
        return cls(
            model_cheap=s.openrouter_model_cheap,
            model_smart=s.openrouter_model_smart,
            temperature=s.orchestrator_temperature,
            max_total_steps=s.max_total_steps,
            max_steps_per_subtask=s.max_subtask_steps,
            smart_cooldown_steps=s.smart_cooldown_steps,
            history_dir=s.history_dir,
        )


@dataclass
class _PauseContext:
    # Снимок для продолжения подзадачи после подтверждения опасного действия в CLI.
    subtask: Subtask
    policy: BaseTaskPolicy
    memory: RuntimeMemory
    step_offset_at_subtask_start: int
    executed_steps_at_awaiting: int
    guarded: AgentAction


def format_cost_summary_line(
    *,
    total_cost_usd: float,
    total_llm_steps: int,
    total_cheap_steps: int,
    total_smart_steps: int,
) -> str:
    # Одна строка сводки стоимости для стрима - убирает дублирование в _drive_plan.
    avg_step_cost = total_cost_usd / total_llm_steps if total_llm_steps else 0.0
    return (
        f"[COST] total_usd={total_cost_usd:.6f} | llm_steps={total_llm_steps} | "
        f"cheap={total_cheap_steps} | smart={total_smart_steps} | avg_step_usd={avg_step_cost:.6f}"
    )


class TaskOrchestrator:
    # Высокий уровень: план из цели, последовательность подзадач, агрегаты шагов и стоимости.

    def __init__(
        self,
        executor: BrowserToolExecutor,
        settings: AppSettings,
        policies: dict[TaskMode, BaseTaskPolicy] | None = None,
        config: OrchestratorConfig | None = None,
    ):
        # Собирает SubtaskReActLoop с LoopConfig из настроек, логгер истории и счётчики одного «run» пользовательской цели.
        self.executor = executor
        self.settings = settings
        self.config = config or OrchestratorConfig.from_app_settings(settings)
        self.history_logger = HistoryLogger(history_dir=self.config.history_dir)
        self.policies = policies if policies is not None else default_policies()
        self.loop = SubtaskReActLoop(
            executor=executor,
            settings=settings,
            config=LoopConfig(
                model_cheap=self.config.model_cheap,
                model_smart=self.config.model_smart,
                temperature=self.config.temperature,
                pricing=settings.pricing,
                history_dir=self.config.history_dir,
                smart_cooldown_steps=self.config.smart_cooldown_steps,
                captcha_max_consecutive_waits=settings.captcha_max_consecutive_waits,
            ),
            history_logger=self.history_logger,
        )
        self._pause: _PauseContext | None = None
        self._plan: ExecutionPlan | None = None
        self._user_goal: str = ""
        self._total_steps: int = 0
        self._total_cost_usd: float = 0.0
        self._total_cheap_steps: int = 0
        self._total_smart_steps: int = 0
        self._total_llm_steps: int = 0
        self._completion_notes: list[str] = []
        self._had_partial: bool = False
        self.final_report_markdown: str = ""

    def _policy_for_mode(self, mode: TaskMode) -> BaseTaskPolicy:
        # Возвращает политику по режиму подзадачи с запасным GENERIC.
        return self.policies.get(mode, self.policies[TaskMode.GENERIC])

    def _merge_subtask_costs(self, used_steps: int, cost_stats: RunCostStats) -> None:
        # Добавляет статистику одной подзадачи (или её продолжения после resume) к глобальным счётчикам run.
        self._total_steps += used_steps
        self._total_cost_usd += cost_stats.total_cost_usd
        self._total_cheap_steps += cost_stats.cheap_steps
        self._total_smart_steps += cost_stats.smart_steps
        self._total_llm_steps += cost_stats.llm_steps

    async def resume_after_dangerous_confirmation(
        self,
        *,
        stream_callback: Callable[[str], Awaitable[None]],
        confirmed_result: ActionResult,
    ) -> AgentState:
        # Дозапускает ReAct после подтверждённого опасного действия, затем продолжает оставшийся план через _drive_plan.
        ctx = self._pause
        plan = self._plan
        if not ctx or not plan:
            await stream_callback("[RESUME] Нет сохранённого контекста ожидания.")
            return AgentState.ERROR
        guarded = self.loop._last_guarded_at_pause or ctx.guarded
        state, used_steps, cost_stats = await self.loop.run_subtask(
            subtask=ctx.subtask,
            policy=ctx.policy,
            memory=ctx.memory,
            stream_callback=stream_callback,
            max_steps=self.config.max_steps_per_subtask,
            step_offset=ctx.step_offset_at_subtask_start,
            resume_last_action=guarded,
            resume_last_result=confirmed_result,
            resume_executed_steps=ctx.executed_steps_at_awaiting,
        )
        self._merge_subtask_costs(used_steps, cost_stats)
        self._pause = None
        await stream_callback(
            f"[RESUME] Подзадача после подтверждения: state={state.value}, steps_used={used_steps}"
        )
        if state == AgentState.FINISHED:
            self._completion_notes.append(f"{ctx.subtask.id}: {ctx.memory.done_reason or 'успешно'}")
            plan.mark_current_done()
        elif state == AgentState.AWAITING_USER_CONFIRMATION:
            g = self.loop._last_guarded_at_pause or guarded
            self._pause = _PauseContext(
                subtask=ctx.subtask,
                policy=ctx.policy,
                memory=ctx.memory,
                step_offset_at_subtask_start=ctx.step_offset_at_subtask_start,
                executed_steps_at_awaiting=self.loop.last_subtask_executed_step_total,
                guarded=g,
            )
            await stream_callback(format_cost_summary_line(
                total_cost_usd=self._total_cost_usd,
                total_llm_steps=self._total_llm_steps,
                total_cheap_steps=self._total_cheap_steps,
                total_smart_steps=self._total_smart_steps,
            ))
            return state
        elif state in (AgentState.ERROR, AgentState.SUBTASK_STEP_LIMIT, AgentState.BLOCKED_CAPTCHA):
            return state
        return await self._drive_plan(stream_callback)

    async def _drive_plan(self, stream_callback: Callable[[str], Awaitable[None]]) -> AgentState:
        # Цикл по подзадачам активного self._plan с лимитами, паузой на подтверждение и опциональным LLM-verify в конце.
        plan = self._plan
        if not plan:
            return AgentState.ERROR
        user_goal = self._user_goal

        while not plan.is_finished():
            current = plan.current_subtask()
            if not current:
                break
            policy = self._policy_for_mode(current.mode)
            await stream_callback(f"Старт {current.id} [{current.mode.value}]")
            memory = RuntimeMemory()
            step_base = self._total_steps
            state, used_steps, cost_stats = await self.loop.run_subtask(
                subtask=current,
                policy=policy,
                memory=memory,
                stream_callback=stream_callback,
                max_steps=self.config.max_steps_per_subtask,
                step_offset=step_base,
            )
            self._merge_subtask_costs(used_steps, cost_stats)
            if self._total_steps >= self.config.max_total_steps:
                await stream_callback(
                    format_cost_summary_line(
                        total_cost_usd=self._total_cost_usd,
                        total_llm_steps=self._total_llm_steps,
                        total_cheap_steps=self._total_cheap_steps,
                        total_smart_steps=self._total_smart_steps,
                    )
                )
                await stream_callback("Достигнут общий лимит шагов.")
                return AgentState.ERROR if not self._had_partial else AgentState.PARTIAL
            if state == AgentState.AWAITING_USER_CONFIRMATION:
                g = self.loop._last_guarded_at_pause
                if g is None:
                    g = AgentAction(thought="", action="click", params={})
                self._pause = _PauseContext(
                    subtask=current,
                    policy=policy,
                    memory=memory,
                    step_offset_at_subtask_start=step_base,
                    executed_steps_at_awaiting=self.loop.last_subtask_executed_step_total,
                    guarded=g,
                )
                await stream_callback(
                    format_cost_summary_line(
                        total_cost_usd=self._total_cost_usd,
                        total_llm_steps=self._total_llm_steps,
                        total_cheap_steps=self._total_cheap_steps,
                        total_smart_steps=self._total_smart_steps,
                    )
                )
                return state
            if state == AgentState.BLOCKED_CAPTCHA:
                await stream_callback(
                    format_cost_summary_line(
                        total_cost_usd=self._total_cost_usd,
                        total_llm_steps=self._total_llm_steps,
                        total_cheap_steps=self._total_cheap_steps,
                        total_smart_steps=self._total_smart_steps,
                    )
                )
                await stream_callback(f"Капча: {memory.done_reason}")
                return AgentState.BLOCKED_CAPTCHA
            if state == AgentState.SUBTASK_STEP_LIMIT:
                await stream_callback(
                    format_cost_summary_line(
                        total_cost_usd=self._total_cost_usd,
                        total_llm_steps=self._total_llm_steps,
                        total_cheap_steps=self._total_cheap_steps,
                        total_smart_steps=self._total_smart_steps,
                    )
                )
                await stream_callback(f"Лимит шагов подзадачи {current.id}: {memory.done_reason}")
                if self.settings.continue_after_subtask_step_limit:
                    self._had_partial = True
                    self._completion_notes.append(f"{current.id}: (пропуск по лимиту) {memory.done_reason}")
                    plan.mark_current_done()
                    continue
                return AgentState.SUBTASK_STEP_LIMIT
            if state == AgentState.ERROR:
                await stream_callback(
                    format_cost_summary_line(
                        total_cost_usd=self._total_cost_usd,
                        total_llm_steps=self._total_llm_steps,
                        total_cheap_steps=self._total_cheap_steps,
                        total_smart_steps=self._total_smart_steps,
                    )
                )
                await stream_callback(f"Подзадача {current.id} завершилась ошибкой.")
                return AgentState.ERROR
            self._completion_notes.append(f"{current.id}: {memory.done_reason or 'успешно'}")
            plan.mark_current_done()
            await stream_callback(f"Подзадача {current.id} завершена. Причина: {memory.done_reason or 'успешно'}")

        await stream_callback(
            format_cost_summary_line(
                total_cost_usd=self._total_cost_usd,
                total_llm_steps=self._total_llm_steps,
                total_cheap_steps=self._total_cheap_steps,
                total_smart_steps=self._total_smart_steps,
            )
        )
        if not plan.is_finished():
            return AgentState.ERROR
        summary = "\n".join(self._completion_notes) if self._completion_notes else user_goal
        if self.settings.goal_verify_llm:
            satisfied = await verify_user_goal_satisfied_llm(
                self.settings,
                executor=self.executor,
                user_goal=user_goal,
                done_reason_summary=summary,
                stream_callback=stream_callback,
            )
            if not satisfied:
                await stream_callback(
                    "[VERIFY] Итог: цель не подтверждена (см. выше [VERIFY-RESULT] / [VERIFY-FAIL] / [VERIFY-DEBUG]). "
                    + (
                        "Сессия: PARTIAL (AGENT_GOAL_VERIFY_FAIL_SOFT=1)."
                        if self.settings.goal_verify_fail_soft
                        else "Сессия завершается с ERROR."
                    )
                )
                if self.settings.goal_verify_fail_soft:
                    self._had_partial = True
                    self._completion_notes.append(
                        "Финальная LLM-проверка цели: satisfied=false (мягкий режим, см. лог verify)."
                    )
                else:
                    return AgentState.ERROR
        terminal_state = AgentState.PARTIAL if self._had_partial else AgentState.FINISHED
        if terminal_state == AgentState.FINISHED:
            try:
                self.final_report_markdown = await self.generate_final_report(stream_callback=stream_callback)
            except Exception as exc:  # noqa: BLE001
                await stream_callback(f"[FINAL-REPORT] Не удалось сгенерировать отчёт: {exc}")
                self.final_report_markdown = ""
        return terminal_state

    def _build_final_report_user_prompt(self) -> str:
        # Собирает компактный контекст для финального отчёта: история шагов + финальное наблюдение страницы.
        history_tail = self.loop.context_history[-20:]
        obs_tail = self.loop.last_observation[:25]
        compact_obs = [
            {
                "role": x.role,
                "name": (x.name or "")[:120],
                "value": str(x.value)[:120] if x.value is not None else "",
            }
            for x in obs_tail
        ]
        return (
            f"Цель пользователя:\n{self._user_goal}\n\n"
            f"Выполненные подзадачи:\n{json.dumps(self._completion_notes, ensure_ascii=False)}\n\n"
            f"context.history (последние шаги):\n{json.dumps(history_tail, ensure_ascii=False)}\n\n"
            f"last_observation (видимые элементы):\n{json.dumps(compact_obs, ensure_ascii=False)}\n"
        )

    async def generate_final_report(self, *, stream_callback: Callable[[str], Awaitable[None]]) -> str:
        # Генерирует итоговый markdown-отчёт по завершению run (отдельный system prompt + контекст истории и финального экрана).
        await stream_callback("[FINAL-REPORT] Генерирую финальный отчёт.")
        system_prompt = (
            "Ты — элитный аналитик-ассистент. Твоя задача: составить краткий, но информативный отчет о работе агента.\n"
            "Стиль: профессиональный, лаконичный, дружелюбный. Используй Markdown и эмодзи.\n\n"
            "Структура ответа:\n"
            "🤖 **SyncJarvis**: [Краткий итог: задача выполнена/не выполнена, общий статус]\n\n"
            "**Хронология действий:**\n"
            "• [Действие 1] → [Результат]\n"
            "• [Действие 2] → [Результат]\n"
            "• [И так далее: только важные этапы и найденные данные]\n\n"
            "🏁 **Итог**: [Конкретный финальный результат: цена, ссылка, подтверждение или статус]\n\n"
            "🖼 **На экране сейчас**: [Короткое описание финального состояния страницы и ключевых видимых элементов]"
        )
        user_prompt = self._build_final_report_user_prompt()
        headers: dict[str, str] = {}
        if self.settings.openrouter_http_referer:
            headers["HTTP-Referer"] = self.settings.openrouter_http_referer
        if self.settings.openrouter_x_title:
            headers["X-Title"] = self.settings.openrouter_x_title
        client = AsyncOpenAI(
            api_key=self.settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers=headers or None,
        )
        response = await client.chat.completions.create(
            model=self.settings.openrouter_model_cheap,
            temperature=0.2,
            max_tokens=450,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        report = (response.choices[0].message.content or "").strip()
        if report:
            await stream_callback("[FINAL-REPORT] Отчёт сформирован.")
            return report
        await stream_callback("[FINAL-REPORT] Пустой ответ модели, использую fallback.")
        return (
            "🤖 **SyncJarvis**: Задача завершена.\n\n"
            "**Результат выполнения:**\n"
            + "\n".join(f"✅ {sentence.strip()}" for x in (self._completion_notes or ["Подзадачи выполнены."]) for sentence in x.replace('\n', '. ').split('. ') if sentence.strip())
            + "\n\nФинальное состояние:\nОткрыта финальная страница по целевой задаче."
        )

    async def run(self, user_goal: str, stream_callback: Callable[[str], Awaitable[None]]) -> AgentState:
        # Точка входа: строит план, сбрасывает счётчики run, делегирует исполнение в _drive_plan.
        self.loop.reset_session_context()
        await stream_callback(
            f"[PLANNER] Запрос разбиения цели через LLM: model={self.settings.openrouter_model_smart}, "
            f"temperature={self.settings.planner_temperature}"
        )
        plan, plan_report = await build_execution_plan(self.settings, user_goal)
        if plan_report.planner_thought:
            await stream_callback(f"[PLANNER] thought={plan_report.planner_thought}")
        if plan_report.error:
            await stream_callback(f"[PLANNER] error={plan_report.error}")
        if plan_report.planner_subtasks_dropped > 0:
            await stream_callback(
                f"[PLANNER] Обрезано подзадач сверх лимита: {plan_report.planner_subtasks_dropped} "
                f"(AGENT_PLANNER_MAX_SUBTASKS={self.settings.planner_max_subtasks})."
            )
        if not plan.subtasks:
            await stream_callback("Планировщик не сформировал подзадачи.")
            return AgentState.ERROR

        self._plan = plan
        self._user_goal = user_goal
        self._pause = None
        self._total_steps = 0
        self._total_cost_usd = 0.0
        self._total_cheap_steps = 0
        self._total_smart_steps = 0
        self._total_llm_steps = 0
        self._completion_notes = []
        self._had_partial = False
        self.final_report_markdown = ""

        await stream_callback(f"План: {len(plan.subtasks)} подзадач.")
        for subtask in plan.subtasks:
            await stream_callback(f"- {subtask.id}: [{subtask.mode.value}] {subtask.goal}")

        return await self._drive_plan(stream_callback)
