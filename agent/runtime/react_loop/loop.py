from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Awaitable, Callable

from agent.config.settings import AppSettings
from agent.llm.clients.actor import ActorLLMClient
from agent.llm.services.router import ModelRoute, ModelRouter
from agent.logging.history_logger import HistoryLogger
from agent.models.action import ActionResult, AgentAction
from agent.models.observation import InteractiveElement
from agent.models.plan import Subtask
from agent.models.state import AgentState
from agent.llm.prompts.actor import (
    ordered_observation_for_actor_prompt,
    resolve_actor_element_index,
    serialize_observation_window_for_actor_prompt,
)
from agent.runtime.react_loop.grounding import (
    should_apply_grounding_search_wait,
    should_run_grounding,
)
from agent.perception.accessibility import collect_interactive_elements
from agent.policies.base import BaseTaskPolicy
from agent.runtime.anti_loop import apply_global_anti_loop
from agent.runtime.memory import RuntimeMemory
from agent.runtime.react_loop.captcha import is_captcha_present
from agent.runtime.react_loop.config import LoopConfig, RunCostStats
from agent.runtime.react_loop.page_fingerprint import snapshot_page_fingerprint
from agent.runtime.react_loop.persistence import persist_step
from agent.runtime.react_loop.step_utils import action_signature, format_runtime_context
from agent.runtime.react_loop.telemetry import build_step_telemetry
from agent.runtime.security import is_confirmation_required
from agent.runtime.self_correction import build_correction_hint
from agent.tools.browser_executor import BrowserToolExecutor


@dataclass(frozen=True)
class _LlmDecision:
    model_route: ModelRoute
    proposed: AgentAction
    model_name: str
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class _CaptchaIterationOutcome:
    last_action: AgentAction
    last_result: ActionResult


class SubtaskReActLoop:
    # Один ReAct-цикл по подзадаче: наблюдение, капча, LLM, гарды, исполнение, телеметрия.

    def __init__(
        self,
        executor: BrowserToolExecutor,
        settings: AppSettings,
        config: LoopConfig,
        history_logger: HistoryLogger | None = None,
    ):
        self.executor = executor
        self._settings = settings
        self.config = config
        self.actor = ActorLLMClient(
            api_key=settings.openrouter_api_key,
            model=config.model_smart,
            temperature=config.temperature,
            referer=settings.openrouter_http_referer,
            title=settings.openrouter_x_title,
            request_max_tokens=settings.actor_response_max_tokens,
            prompt_limits=settings.prompt_limits,
        )
        self.model_router = ModelRouter(
            cheap_model=config.model_cheap,
            smart_model=config.model_smart,
            smart_cooldown_steps=config.smart_cooldown_steps,
        )
        self.history_logger = history_logger or HistoryLogger(history_dir=config.history_dir)
        # Заполняется перед возвратом AWAITING_USER_CONFIRMATION - для resume_after_dangerous_confirmation в оркестраторе.
        self._last_guarded_at_pause: AgentAction | None = None
        # Последнее значение счётчика шагов внутри run_subtask - нужно для resume (абсолютный счётчик подзадачи).
        self.last_subtask_executed_step_total: int = 0
        # Короткая история действий текущего run для финального отчёта.
        self.context_history: list[dict[str, Any]] = []
        # Последнее наблюдение страницы (используется в финальном отчёте).
        self.last_observation: list[InteractiveElement] = []
        self._max_context_history: int = 200
        # Конец прошлой итерации цикла подзадачи: URL страницы (для grounding: смена без полного reload).
        self._loop_end_url: str | None = None
        # Последний fingerprint, для которого уже выполняли grounding (дебаунс).
        self._last_grounding_fingerprint: str | None = None

    def reset_session_context(self) -> None:
        # Сбрасывает историю/наблюдение перед новым run оркестратора.
        self.context_history = []
        self.last_observation = []
        self._loop_end_url = None
        self._last_grounding_fingerprint = None

    def _refresh_loop_url_snapshot(self) -> None:
        try:
            p = self.executor.page
            if p and not p.is_closed():
                self._loop_end_url = p.url
        except Exception:
            pass

    def _append_context_history(self, item: dict[str, Any]) -> None:
        # Держит ограниченный хвост истории, чтобы не раздувать память в долгой сессии.
        self.context_history.append(item)
        if len(self.context_history) > self._max_context_history:
            self.context_history = self.context_history[-self._max_context_history :]

    def _require_page(self):
        # Делегируем в executor: при self.page=None он перепривязывается к живой вкладке из context.pages (нужно для recovery).
        return self.executor._require_page()

    async def _handle_captcha(
        self,
        *,
        page,
        observation: list[InteractiveElement],
        subtask: Subtask,
        memory: RuntimeMemory,
        global_step: int,
        stream_callback: Callable[[str], Awaitable[None]],
    ) -> _CaptchaIterationOutcome | None:
        # Если на странице капча - один wait, телеметрия CAPTCHA_WAIT, persist; иначе None.
        if not await is_captcha_present(page, observation):
            return None
        await stream_callback(
            "[CAPTCHA] Обнаружена капча. Ожидаю, пока пользователь решит её вручную."
        )
        wait_action = AgentAction(
            thought="Обнаружена капча - ожидаю ручного решения пользователем.",
            action="wait",
            params={"seconds": 3.0},
        )
        wait_result = await self.executor.execute_action("wait", {"seconds": 3.0}, observation)
        telemetry = build_step_telemetry(
            subtask=subtask,
            memory=memory,
            observation=observation,
            phase="CAPTCHA_WAIT",
            action_signature=action_signature(wait_action),
            progress_score=memory.last_progress_score,
        )
        await persist_step(
            page=page,
            history_logger=self.history_logger,
            global_step=global_step,
            observation=observation,
            llm_response=wait_action,
            result=wait_result,
            telemetry=telemetry,
        )
        await stream_callback(f"[Шаг {global_step}] result={wait_result.message} | success={wait_result.success}")
        self._append_context_history(
            {
                "step": global_step,
                "mode": subtask.mode.value,
                "goal": subtask.goal,
                "action": wait_action.action,
                "params": wait_action.params,
                "result_success": wait_result.success,
                "result_message": wait_result.message,
            }
        )
        return _CaptchaIterationOutcome(last_action=wait_action, last_result=wait_result)

    async def _make_decision(
        self,
        *,
        subtask: Subtask,
        policy: BaseTaskPolicy,
        observation: list[InteractiveElement],
        obs_window: list[InteractiveElement],
        last_action: AgentAction | None,
        last_action_result: ActionResult | None,
        memory: RuntimeMemory,
        global_step: int,
        page,
    ) -> _LlmDecision:
        # Выбор модели (ModelRouter) и запрос к ActorLLMClient; сетевые ретраи из настроек.
        runtime_context = format_runtime_context(memory)
        model_route = self.model_router.select(
            task_mode=subtask.mode,
            last_action_result=last_action_result,
            memory_view=memory.guard_view(),
            current_step=global_step,
        )
        if model_route.tier == "smart":
            memory.last_smart_step = global_step
        if self._settings.observation_fusion_multimodal:
            if should_apply_grounding_search_wait(last_action, last_action_result, subtask.mode):
                await asyncio.sleep(self._settings.grounding_min_wait_seconds)
            shot = Path(self.config.history_dir) / f"fusion_{global_step:03d}.png"
            await page.screenshot(path=str(shot), full_page=False)
            compact = serialize_observation_window_for_actor_prompt(obs_window, self.actor.prompt_limits)
            compact_json = json.dumps(compact, ensure_ascii=False)
            prev_status = "Нет предыдущего результата"
            if last_action_result:
                prev_status = (
                    f"success={last_action_result.success}, changed={last_action_result.changed}, "
                    f"message={last_action_result.message}, error={last_action_result.error}"
                )
            decision = await self.actor.decide_fusion_step_action(
                subtask_goal=subtask.goal,
                task_mode=subtask.mode,
                mode_rules=policy.mode_rules(),
                runtime_context=runtime_context[:400],
                last_step_summary=prev_status,
                self_check_hint=(memory.self_check_hint or "")[:400],
                screenshot_path=str(shot),
                compact_observation_json=compact_json,
                model_override=model_route.model,
                max_transport_retries=self._settings.llm_transport_max_retries,
            )
            return _LlmDecision(
                model_route=model_route,
                proposed=decision.action,
                model_name=decision.model_used,
                prompt_tokens=decision.prompt_tokens,
                completion_tokens=decision.completion_tokens,
            )
        decision = await self.actor.decide_action(
            subtask_goal=subtask.goal,
            task_mode=subtask.mode,
            observation=observation,
            last_action_result=last_action_result,
            runtime_context=runtime_context,
            mode_rules=policy.mode_rules(),
            model_override=model_route.model,
            max_transport_retries=self._settings.llm_transport_max_retries,
            self_check_hint=(memory.self_check_hint or "")[:400],
        )
        return _LlmDecision(
            model_route=model_route,
            proposed=decision.action,
            model_name=decision.model_used,
            prompt_tokens=decision.prompt_tokens,
            completion_tokens=decision.completion_tokens,
        )

    async def _handle_llm_error(
        self,
        exc: ValueError,
        *,
        subtask: Subtask,
        policy: BaseTaskPolicy,
        observation: list[InteractiveElement],
        memory: RuntimeMemory,
        global_step: int,
        page,
        stream_callback: Callable[[str], Awaitable[None]],
    ) -> tuple[AgentAction, ActionResult]:
        # Невалидный JSON от актора - fallback политики, REPLAN-телеметрия, persist, стрим.
        result = ActionResult(
            success=False,
            message="Ответ модели невалиден - используется fallback.",
            changed=False,
            error=str(exc),
        )
        action = policy.fallback_action(subtask.goal, observation, reason="Невалидный JSON модели")
        telemetry = build_step_telemetry(
            subtask=subtask,
            memory=memory,
            observation=observation,
            phase="REPLAN",
            action_signature=action_signature(action),
            progress_score=memory.last_progress_score,
            model_tier="fallback",
            model_used="fallback_action",
            estimated_cost_usd=0.0,
        )
        await persist_step(
            page=page,
            history_logger=self.history_logger,
            global_step=global_step,
            observation=observation,
            llm_response=action,
            result=result,
            telemetry=telemetry,
        )
        await stream_callback(f"[Шаг {global_step}] result={result.message} | success={result.success}")
        return action, result

    def _apply_guards(
        self,
        policy: BaseTaskPolicy,
        subtask_goal: str,
        observation: list[InteractiveElement],
        proposed: AgentAction,
        memory: RuntimeMemory,
    ) -> AgentAction:
        # Цепочка anti_loop_guard (режим) - apply_global_anti_loop - refine_after_global_anti_loop.
        guarded = policy.anti_loop_guard(
            subtask_goal, observation, proposed, memory_view=memory.guard_view()
        )
        guarded = apply_global_anti_loop(guarded, memory)
        return policy.refine_after_global_anti_loop(guarded, memory)

    @staticmethod
    def _refine_self_check_repeat_click(
        proposed: AgentAction,
        last_action: AgentAction | None,
        memory: RuntimeMemory,
    ) -> AgentAction:
        # Если самопроверка оставила подсказку и модель снова кликает тот же ax_id — скролл, подсказку сбрасываем.
        if not (memory.self_check_hint or "").strip():
            return proposed
        if proposed.action != "click" or last_action is None or last_action.action != "click":
            return proposed
        pa = proposed.params.get("ax_id")
        la = last_action.params.get("ax_id")
        if not (isinstance(pa, str) and isinstance(la, str) and pa and pa == la):
            return proposed
        memory.self_check_hint = ""
        return AgentAction(
            thought="Самопроверка: цель не достигнута — не повторяю тот же ax_id, делаю scroll.",
            action="scroll",
            params={"direction": "down", "amount": 600},
        )

    @staticmethod
    def _should_visual_stagnation_recovery(
        subtask: Subtask,
        memory: RuntimeMemory,
        last_action: AgentAction | None,
        last_action_result: ActionResult | None,
    ) -> bool:
        # Vision при застревании на click в TRANSACTION/GENERIC (поиск miss обрабатывается отдельно).
        if subtask.mode.value not in ("TRANSACTION", "GENERIC"):
            return False
        if last_action is None or last_action.action != "click":
            return False
        if last_action_result is None or not last_action_result.success:
            return False
        if memory.stagnation_steps < 2 and memory.repeat_count < 2:
            return False
        return True

    async def _attempt_visual_recovery(
        self,
        *,
        subtask: Subtask,
        observation: list[InteractiveElement],
        memory: RuntimeMemory,
        last_action_result: ActionResult,
        global_step: int,
        page,
        stream_callback: Callable[[str], Awaitable[None]],
        vision_reason: str = "search_miss",
    ) -> tuple[AgentAction, ActionResult, _LlmDecision]:
        # Скриншот + multimodal LLM: click_xy/scroll/wait (поиск или застревание на click).
        vision_path = Path(self.config.history_dir) / f"vision_probe_{global_step:03d}.png"
        await page.screenshot(path=str(vision_path), full_page=False)
        vision = await self.actor.decide_visual_recovery(
            subtask_goal=subtask.goal,
            screenshot_path=str(vision_path),
            last_error=last_action_result.message,
            model_override=self.config.model_cheap,
            max_transport_retries=self._settings.llm_transport_max_retries,
        )
        action_name = vision.action if vision.action in {"click_xy", "scroll", "wait"} else "wait"
        params = vision.params if isinstance(vision.params, dict) else {}
        guarded = AgentAction(
            thought=f"Vision recovery: {vision.reason or 'попытка восстановить управление по скриншоту'}",
            action=action_name,
            params=params,
        )
        await stream_callback(
            f"[VISION] trigger={vision_reason}, action={guarded.action}, params={guarded.params}, "
            f"model_reason={vision.reason or '-'}"
        )
        result = await self.executor.execute_action(guarded.action, guarded.params, observation)
        llm = _LlmDecision(
            model_route=ModelRoute(tier="cheap", model=self.config.model_cheap, reason="Vision recovery"),
            proposed=guarded,
            model_name=vision.model_used,
            prompt_tokens=vision.prompt_tokens,
            completion_tokens=vision.completion_tokens,
        )
        return guarded, result, llm

    async def _attempt_grounding(
        self,
        *,
        subtask: Subtask,
        policy: BaseTaskPolicy,
        observation: list[InteractiveElement],
        obs_window: list[InteractiveElement],
        memory: RuntimeMemory,
        last_action: AgentAction | None,
        last_action_result: ActionResult | None,
        global_step: int,
        page,
        stream_callback: Callable[[str], Awaitable[None]],
        current_fingerprint: str,
    ) -> tuple[AgentAction, ActionResult, _LlmDecision]:
        # Скрин + multimodal: цель + список element_index; одно действие, затем тот же путь fingerprint/changed, что у основного шага.
        if should_apply_grounding_search_wait(last_action, last_action_result, subtask.mode):
            await asyncio.sleep(self._settings.grounding_min_wait_seconds)
        shot = Path(self.config.history_dir) / f"grounding_{global_step:03d}.png"
        await page.screenshot(path=str(shot), full_page=False)
        compact = serialize_observation_window_for_actor_prompt(obs_window, self.actor.prompt_limits)
        compact_json = json.dumps(compact, ensure_ascii=False)
        gd = await self.actor.decide_grounding_action(
            subtask_goal=subtask.goal,
            task_mode=subtask.mode,
            screenshot_path=str(shot),
            compact_observation_json=compact_json,
            model_override=self.config.model_cheap,
            max_transport_retries=self._settings.llm_transport_max_retries,
        )
        proposed = gd.action
        resolved, idx_warn = resolve_actor_element_index(proposed, obs_window)
        if idx_warn:
            await stream_callback(idx_warn)
        guarded = self._apply_guards(policy, subtask.goal, observation, resolved, memory)
        guarded = self._refine_self_check_repeat_click(guarded, last_action, memory)
        await stream_callback(
            f"[GROUNDING] action={guarded.action} | params={guarded.params} | "
            f"model={gd.model_used} | tokens in/out={gd.prompt_tokens}/{gd.completion_tokens}"
        )
        before_fp: str | None = None
        if guarded.action in ("click", "navigate", "type"):
            try:
                before_fp = await snapshot_page_fingerprint(page)
            except Exception:
                before_fp = None
        result = await self.executor.execute_action(guarded.action, guarded.params, observation)
        if (
            before_fp is not None
            and result.success
            and guarded.action in ("click", "navigate", "type")
        ):
            try:
                await asyncio.sleep(0.1)
                after_fp = await snapshot_page_fingerprint(page)
                if after_fp == before_fp:
                    result = result.model_copy(update={"changed": False})
            except Exception:
                pass
        llm = _LlmDecision(
            model_route=ModelRoute(tier="cheap", model=self.config.model_cheap, reason="Grounding"),
            proposed=guarded,
            model_name=gd.model_used,
            prompt_tokens=gd.prompt_tokens,
            completion_tokens=gd.completion_tokens,
        )
        self._last_grounding_fingerprint = current_fingerprint
        return guarded, result, llm

    async def _update_performance_metrics(
        self,
        *,
        subtask: Subtask,
        policy: BaseTaskPolicy,
        memory: RuntimeMemory,
        cost_stats: RunCostStats,
        observation: list[InteractiveElement],
        guarded: AgentAction,
        result: ActionResult,
        llm: _LlmDecision,
        signature: str,
        global_step: int,
        page,
        stream_callback: Callable[[str], Awaitable[None]],
    ) -> AgentState | None:
        # Прогресс и стагнация в памяти, стоимость, телеметрия EXECUTION, persist, регистрация cost_stats, терминальные статусы.
        progress = policy.compute_progress(observation, guarded, result)
        if guarded.action == "type" and not result.success and result.reason_code == "target_not_editable":
            memory.type_not_editable_streak += 1
            await stream_callback(f"[GUARD] reason=target_not_editable | streak={memory.type_not_editable_streak}")
        elif guarded.action == "type":
            memory.type_not_editable_streak = 0
        if (
            guarded.action == "type"
            and not result.success
            and result.reason_code == "search_input_not_found"
        ):
            memory.search_target_miss_streak += 1
            await stream_callback(f"[GUARD] reason=search_input_not_found | streak={memory.search_target_miss_streak}")
        elif guarded.action in {"type", "click", "navigate"} and result.success:
            memory.search_target_miss_streak = 0
        if progress <= memory.last_progress_score:
            memory.stagnation_steps += 1
        else:
            memory.stagnation_steps = 0
        memory.last_progress_score = progress

        est_cost = self.config.pricing.estimate_cost_usd(
            prompt_tokens=llm.prompt_tokens,
            completion_tokens=llm.completion_tokens,
            tier=llm.model_route.tier,
        )
        telemetry = build_step_telemetry(
            subtask=subtask,
            memory=memory,
            observation=observation,
            phase="EXECUTION",
            action_signature=signature,
            progress_score=progress,
            model_tier=llm.model_route.tier,
            model_used=llm.model_name or llm.model_route.model,
            prompt_tokens=llm.prompt_tokens,
            completion_tokens=llm.completion_tokens,
            estimated_cost_usd=est_cost,
        )
        await persist_step(
            page=page,
            history_logger=self.history_logger,
            global_step=global_step,
            observation=observation,
            llm_response=guarded,
            result=result,
            telemetry=telemetry,
        )
        cost_stats.register(tier=llm.model_route.tier, cost_usd=telemetry.estimated_cost_usd)

        await stream_callback(
            f"[Шаг {global_step}] result={result.message} | success={result.success} | "
            f"estimated_cost_usd={telemetry.estimated_cost_usd:.6f}"
        )
        self._append_context_history(
            {
                "step": global_step,
                "mode": subtask.mode.value,
                "goal": subtask.goal,
                "thought": guarded.thought,
                "action": guarded.action,
                "params": guarded.params,
                "result_success": result.success,
                "result_message": result.message,
                "reason_code": result.reason_code,
                "cost_usd": round(telemetry.estimated_cost_usd, 6),
            }
        )

        if result.state == AgentState.AWAITING_USER_CONFIRMATION:
            return AgentState.AWAITING_USER_CONFIRMATION

        force_done, force_reason = policy.check_force_finish_after_execution(guarded, result, memory)
        if force_done:
            memory.done_reason = force_reason
            return AgentState.FINISHED

        if guarded.action == "finish":
            memory.done_reason = "LLM явно завершил подзадачу."
            return AgentState.FINISHED

        return None

    # Главная корутина выполнения подзадачи. Здесь происходит основной цикл шагов агента.
    # Параметры resume_* нужны для возобновления после приостановки на опасном действии.
    async def run_subtask(
        self,
        subtask: Subtask,
        policy: BaseTaskPolicy,
        memory: RuntimeMemory,
        stream_callback: Callable[[str], Awaitable[None]],
        max_steps: int,
        step_offset: int = 0,
        *,
        resume_last_action: AgentAction | None = None,
        resume_last_result: ActionResult | None = None,
        resume_executed_steps: int = 0,
    ) -> tuple[AgentState, int, RunCostStats]:
        # Сбрасываем последнее опасное действие при входе в функцию.
        # Это нужно, чтобы знать, где останавливаться при подтверждении пользователя.
        self._last_guarded_at_pause = None
        self._loop_end_url = None
        self._last_grounding_fingerprint = None
        # Восстанавливаем последнее выполнение действия, если были на паузе (резюмирование).
        last_action_result: ActionResult | None = resume_last_result
        last_action: AgentAction | None = resume_last_action
        # Устанавливаем количество уже выполненных шагов для поддержки "resume".
        executed_steps = resume_executed_steps
        steps_at_entry = resume_executed_steps
        # Создаем объект для сбора статистики по стоимости выполнения шагов.
        cost_stats = RunCostStats()
        # Счетчик подряд идущих встреч с капчей для определения блокировки.
        captcha_streak = 0
        # Счетчик подряд неудачных попыток сбора observation — чтобы в случае сбоев пробовать другой способ.
        observation_collect_fail_streak = 0
        # Счетчик пустых observation, чтобы не попасть в бесконечный цикл скроллов.
        empty_observation_streak = 0

        # Внутренняя функция для вычисления количества шагов, сделанных за текущий вызов.
        # Используется для корректного учета общего количества выполненных шагов после паузы.
        def _steps_delta() -> int:
            return executed_steps - steps_at_entry

        # Унифицированная функция возврата результата выполнения подзадачи.
        # Сохраняет счетчик шагов, чтобы можно было правильно продолжить впоследствии.
        def _ret(state: AgentState) -> tuple[AgentState, int, RunCostStats]:
            self.last_subtask_executed_step_total = executed_steps
            return state, _steps_delta(), cost_stats

        # Вспомогательная функция: пытается собрать интерактивные элементы со страницы.
        # При ошибке перепривязывает страницу и выполняет повторный сбор (например, если страница устарела).
        async def _collect_observation_with_recovery(page) -> tuple[object, list[InteractiveElement]]:
            try:
                return page, await collect_interactive_elements(page)
            except Exception as exc: 
                # Сообщаем о падении сбора дерева доступности.
                await stream_callback(f"[OBSERVATION] Ошибка сбора accessibility дерева: {exc}")
                self.executor.page = None
                recovered_page = self._require_page()
                # Если страница была перепривязана, сигнализируем об этом в логе.
                if recovered_page is not page:
                    await stream_callback("[OBSERVATION] Перепривязка к активной вкладке и повторный сбор.")
                try:
                    return recovered_page, await collect_interactive_elements(recovered_page)
                except Exception as exc2:  
                    # Если и повторная попытка не удалась, возвращаем пустой список observation.
                    await stream_callback(f"[OBSERVATION] Повторный сбор после перепривязки не удался: {exc2}")
                    return recovered_page, []

        # Основной цикл — пока не достигнут лимит шагов max_steps.
        # На каждом шаге выполняется наблюдение, принятие решения, выполнение действия и реакция на результат.
        while executed_steps < max_steps:
            # Увеличиваем счетчик шагов субзадачи.
            executed_steps += 1
            # Рассчитываем глобальный номер шага с учетом смещения (step_offset).
            global_step = step_offset + executed_steps
            # Получаем текущую страницу для взаимодействия.
            page = self._require_page()
            url_changed_since_last = self._loop_end_url is not None and page.url != self._loop_end_url

            # Пытаемся получить интерактивные элементы. При необходимости перепривязываем страницу.
            page, observation = await _collect_observation_with_recovery(page)
            if observation:
                # Если удалось собрать элементы, сбрасываем счетчик неудач сбора.
                observation_collect_fail_streak = 0
            else:
                # Если не удалось собрать, увеличиваем счетчик ошибочного сбора.
                observation_collect_fail_streak += 1
            # Если подряд несколько неудач — выполняем fallback: ждем, перепривязываем, снова пробуем собрать элементы.
            if observation_collect_fail_streak >= 2:
                await stream_callback("[OBSERVATION] Controlled fallback: wait + page rebind.")
                await self.executor.execute_action("wait", {"seconds": 0.6}, observation)
                self.executor.page = None
                page = self._require_page()
                page, observation = await _collect_observation_with_recovery(page)
                observation_collect_fail_streak = 0 if observation else observation_collect_fail_streak
            # Сохраняем текущее observation для возможного последующего анализа.
            self.last_observation = list(observation)
            # Если интерактивных элементов совсем нет, считаем такие случаи.
            if len(observation) == 0:
                empty_observation_streak += 1
                # Если подряд несколько пустых observation — завершить с ошибкой.
                if empty_observation_streak >= 3:
                    memory.done_reason = "Observation пустой после нескольких попыток recovery."
                    return _ret(AgentState.ERROR)
                # Если это не критическая ситуация — пробуем прокрутить страницу вниз, вдруг элементы вне viewport.
                await stream_callback(
                    "[OBSERVATION] Нет видимых интерактивных элементов в viewport — выполняю scroll вниз."
                )
                await page.mouse.wheel(0, 720)  # Скроллим вниз
                await asyncio.sleep(0.12)  # Ждем, чтобы страница успела отрисоваться
                page, observation = await _collect_observation_with_recovery(page)
                self.last_observation = list(observation)
            # Если элементов мало — возможно, есть смысл подсказать пользователю или LLM-агенту про скролл.
            elif len(observation) < 3:
                empty_observation_streak = 0
                await stream_callback(
                    "[HINT] Видимых элементов мало — цель может быть ниже или выше; при необходимости сделай scroll."
                )
            else:
                # Сбрасываем счетчик пустых observation, если элементы найдены.
                empty_observation_streak = 0

            obs_window = ordered_observation_for_actor_prompt(observation, self.actor.prompt_limits)

            # Формируем подсказку по коррекции (например, если последнее действие было неудачным).
            correction_hint = build_correction_hint(last_action_result)
            # Логируем текущий шаг и подсказку по коррекции шагов.
            await stream_callback(f"[Шаг {global_step}] [{subtask.mode.value}] self_correction={correction_hint}")

            # Проверка на наличие капчи (captcha) на странице.
            if await is_captcha_present(page, observation):
                # Увеличиваем счетчик подряд встреченных капчей.
                captcha_streak += 1
                # Если превышен лимит ожиданий при капче — выполняем выход.
                if captcha_streak > self.config.captcha_max_consecutive_waits:
                    memory.done_reason = "Превышен лимит ожиданий при капче."
                    return _ret(AgentState.BLOCKED_CAPTCHA)
                # Если капча обнаружена — вызываем специальную обработку (_handle_captcha).
                captcha_out = await self._handle_captcha(
                    page=page,
                    observation=observation,
                    subtask=subtask,
                    memory=memory,
                    global_step=global_step,
                    stream_callback=stream_callback,
                )
                # После попытки работаем с результатом: если был возврат, откатываем шаг и повторяем цикл.
                if captcha_out is not None:
                    last_action = captcha_out.last_action
                    last_action_result = captcha_out.last_result
                    executed_steps -= 1
                    self._refresh_loop_url_snapshot()
                    continue
            else:
                # Если капчи нет, сбрасываем счетчик подряд встреч капч.
                captcha_streak = 0

            # Проверяем по политике subtask — не пора ли завершаться (например, цель достигнута).
            done, done_reason = policy.is_done(subtask.goal, last_action, last_action_result)
            if done:
                # Если задача завершена — сохраняем причину и возвращаем соответсвующий статус.
                memory.done_reason = done_reason
                return _ret(AgentState.FINISHED)

            # # Self-check: по умолчанию только после успешного шага; опционально — после неуспешного click в SELECTION/TRANSACTION,
            # # чтобы LLM увидел observation «товар уже в корзине», хотя Playwright вернул timeout/перехват.
            _self_check_modes = frozenset({"SELECTION", "TRANSACTION"})
            _run_goal_self_check = (
                self._settings.subtask_goal_self_check_llm
                and last_action is not None
                and last_action_result is not None
                and (
                    last_action_result.success
                    or (
                        self._settings.subtask_goal_self_check_after_failed_click
                        and last_action.action == "click"
                        and subtask.mode.value in _self_check_modes
                    )
                )
            )
            if _run_goal_self_check:
                try:
                    # Формируем контекст выполнения для передачи LLM.
                    runtime_context = format_runtime_context(memory)
                    # Запрашиваем у LLM анализ: достигнута ли цель подзадачи.
                    goal_check = await self.actor.assess_goal_reached(
                        subtask_goal=subtask.goal,
                        observation=observation,
                        last_action=last_action,
                        last_action_result=last_action_result,
                        runtime_context=runtime_context,
                        model_override=self.config.model_cheap,
                        max_transport_retries=self._settings.llm_transport_max_retries,
                    )
                    # Оцениваем стоимость такого запроса.
                    check_cost = self.config.pricing.estimate_cost_usd(
                        prompt_tokens=goal_check.prompt_tokens,
                        completion_tokens=goal_check.completion_tokens,
                        tier="cheap",
                    )
                    # Регистрируем стоимость в статистике.
                    cost_stats.register(tier="cheap", cost_usd=check_cost)
                    # Логируем результат самопроверки.
                    await stream_callback(
                        f"[SELF-CHECK] goal_reached={goal_check.goal_reached} | "
                        f"reason={goal_check.reason or '-'} | "
                        f"model={goal_check.model_used} | "
                        f"tokens(in={goal_check.prompt_tokens}, out={goal_check.completion_tokens}) | "
                        f"estimated_cost_usd={check_cost:.6f}"
                    )
                    # Увеличиваем счетчик самопроверок.
                    memory.self_check_count += 1
                    if goal_check.goal_reached:
                        # Если цель достигнута по мнению LLM, завершаем выполнение.
                        memory.self_check_hint = ""
                        memory.done_reason = goal_check.reason or "LLM self-check: цель подзадачи достигнута."
                        return _ret(AgentState.FINISHED)
                    # Иначе сохраняем причину неудачи (сокращаем до 300 символов).
                    memory.self_check_hint = (goal_check.reason or "")[:300]
                except Exception as exc:  # noqa: BLE001
                    # Логируем ошибку самопроверки, не прерываем цикл.
                    await stream_callback(f"[SELF-CHECK] Пропускаю из-за ошибки: {exc}")

            try:
                current_fp = await snapshot_page_fingerprint(page)
            except Exception:
                current_fp = ""
            run_grounding = False
            gr_reason = ""
            if not self._settings.observation_fusion_multimodal:
                run_grounding, gr_reason = should_run_grounding(
                    settings=self._settings,
                    subtask_mode=subtask.mode,
                    last_action=last_action,
                    last_action_result=last_action_result,
                    current_fingerprint=current_fp,
                    last_grounding_fingerprint=self._last_grounding_fingerprint,
                    url_changed_since_last_step=url_changed_since_last,
                )
            if run_grounding:
                await stream_callback(f"[GROUNDING] start | {gr_reason}")
                try:
                    guarded, result, llm = await self._attempt_grounding(
                        subtask=subtask,
                        policy=policy,
                        observation=observation,
                        obs_window=obs_window,
                        memory=memory,
                        last_action=last_action,
                        last_action_result=last_action_result,
                        global_step=global_step,
                        page=page,
                        stream_callback=stream_callback,
                        current_fingerprint=current_fp,
                    )
                except Exception as exc:  # noqa: BLE001
                    await stream_callback(f"[GROUNDING] Пропуск из-за ошибки: {exc}")
                else:
                    sig = action_signature(guarded)
                    memory.update_signature(sig)
                    memory.update_after_action(guarded)
                    terminal = await self._update_performance_metrics(
                        subtask=subtask,
                        policy=policy,
                        memory=memory,
                        cost_stats=cost_stats,
                        observation=observation,
                        guarded=guarded,
                        result=result,
                        llm=llm,
                        signature=sig,
                        global_step=global_step,
                        page=page,
                        stream_callback=stream_callback,
                    )
                    last_action_result = result
                    last_action = guarded
                    self._refresh_loop_url_snapshot()
                    if terminal is not None:
                        if terminal == AgentState.AWAITING_USER_CONFIRMATION:
                            self._last_guarded_at_pause = guarded
                        return _ret(terminal)
                    continue

            try:
                # Проверка для режима поиска: если цель поиска не найдена несколько раз,
                # инициируем специальное восстановление через визуальный анализ.
                if (
                    subtask.mode.value == "SEARCH"
                    and memory.search_target_miss_streak >= 3
                    and last_action_result is not None
                ):
                    await stream_callback(
                        f"[GUARD] reason=search_target_miss_streak | streak={memory.search_target_miss_streak}"
                    )
                    # Пытаемся восстановить прогресс — вызываем процедуру визуального восстановления.
                    guarded, result, llm = await self._attempt_visual_recovery(
                        subtask=subtask,
                        observation=observation,
                        memory=memory,
                        last_action_result=last_action_result,
                        global_step=global_step,
                        page=page,
                        stream_callback=stream_callback,
                        vision_reason="search_miss",
                    )
                    # Обновляем подпись действия и память агента.
                    sig = action_signature(guarded)
                    memory.update_signature(sig)
                    memory.update_after_action(guarded)
                    # Обновляем метрики и проверяем, пора ли завершать.
                    terminal = await self._update_performance_metrics(
                        subtask=subtask,
                        policy=policy,
                        memory=memory,
                        cost_stats=cost_stats,
                        observation=observation,
                        guarded=guarded,
                        result=result,
                        llm=llm,
                        signature=sig,
                        global_step=global_step,
                        page=page,
                        stream_callback=stream_callback,
                    )
                    # Сохраняем последнее действие и результат, чтобы можно было продолжать после восстановления.
                    last_action_result = result
                    last_action = guarded
                    # Увеличиваем счетчик восстановлений через компьютерное зрение.
                    memory.vision_recovery_count += 1
                    if terminal is not None:
                        # Если требуется подтверждение пользователя, запоминаем guarded для паузы.
                        if terminal == AgentState.AWAITING_USER_CONFIRMATION:
                            self._last_guarded_at_pause = guarded
                        # Завершаем выполнение с нужным статусом.
                        return _ret(terminal)
                    self._refresh_loop_url_snapshot()
                    continue
                # Если обнаружена стагнация (например, цикл одинаковых кликов), пробуем восстановление.
                if self._should_visual_stagnation_recovery(
                    subtask, memory, last_action, last_action_result
                ) and last_action_result is not None:
                    await stream_callback(
                        "[GUARD] reason=stagnation_click | vision recovery "
                        f"(stagnation_steps={memory.stagnation_steps}, repeat_count={memory.repeat_count})"
                    )
                    # Запускаем визуальное восстановление как при поисковом тупике.
                    guarded, result, llm = await self._attempt_visual_recovery(
                        subtask=subtask,
                        observation=observation,
                        memory=memory,
                        last_action_result=last_action_result,
                        global_step=global_step,
                        page=page,
                        stream_callback=stream_callback,
                        vision_reason="stagnation_click",
                    )
                    sig = action_signature(guarded)
                    memory.update_signature(sig)
                    memory.update_after_action(guarded)
                    terminal = await self._update_performance_metrics(
                        subtask=subtask,
                        policy=policy,
                        memory=memory,
                        cost_stats=cost_stats,
                        observation=observation,
                        guarded=guarded,
                        result=result,
                        llm=llm,
                        signature=sig,
                        global_step=global_step,
                        page=page,
                        stream_callback=stream_callback,
                    )
                    last_action_result = result
                    last_action = guarded
                    memory.vision_recovery_count += 1
                    if terminal is not None:
                        if terminal == AgentState.AWAITING_USER_CONFIRMATION:
                            self._last_guarded_at_pause = guarded
                        return _ret(terminal)
                    self._refresh_loop_url_snapshot()
                    continue
                # Если не требуется визуальное восстановление — LLM: либо fusion (скрин + a11y), либо текстовый актёр.
                if self._settings.observation_fusion_multimodal:
                    await stream_callback("[FUSION] Решение по скриншоту viewport и списку a11y (сверка с изображением).")
                llm = await self._make_decision(
                    subtask=subtask,
                    policy=policy,
                    observation=observation,
                    obs_window=obs_window,
                    last_action=last_action,
                    last_action_result=last_action_result,
                    memory=memory,
                    global_step=global_step,
                    page=page,
                )
                resolved_proposed, idx_warn = resolve_actor_element_index(llm.proposed, obs_window)
                if idx_warn:
                    await stream_callback(idx_warn)
                llm = replace(llm, proposed=resolved_proposed)
            except ValueError as exc:
                # Если возникла ошибка при генерации действия LLM, обрабатываем её через специальный обработчик.
                last_action, last_action_result = await self._handle_llm_error(
                    exc,
                    subtask=subtask,
                    policy=policy,
                    observation=observation,
                    memory=memory,
                    global_step=global_step,
                    page=page,
                    stream_callback=stream_callback,
                )
                self._refresh_loop_url_snapshot()
                continue

            # Применяем защитные политики и фильтры к предложенному LLM действию.
            guarded = self._apply_guards(policy, subtask.goal, observation, llm.proposed, memory)
            after_policy = guarded
            # Дополнительное уточнение: если был повторный клик (self_check), корректируем действие.
            guarded = self._refine_self_check_repeat_click(guarded, last_action, memory)
            # Логируем, если произошло изменение в действиях вследствие этой корректировки.
            if guarded.action != after_policy.action or guarded.params != after_policy.params:
                ax_dbg = after_policy.params.get("ax_id", "")
                await stream_callback(
                    f"[GUARD] reason=self_check_repeat_click | ax_id={ax_dbg}"
                )
            # Если политика/глобальные guard-и изменили действие по сравнению с оригинальным LLM-ответом, логируем это.
            if after_policy.action != llm.proposed.action or after_policy.params != llm.proposed.params:
                await stream_callback(
                    "[GUARD] reason=policy_or_global_guard_changed_action | "
                    f"type_not_editable_streak={memory.type_not_editable_streak} | "
                    f"search_target_miss_streak={memory.search_target_miss_streak}"
                )
            # Формируем компактную уникальную подпись действия для отслеживания повторов.
            sig = action_signature(guarded)
            # Обновляем хранилище подписей, чтобы видеть, какие действия уже были осуществлены.
            memory.update_signature(sig)
            # Фиксируем факт выполнения действия (для политики и анализа LLM).
            memory.update_after_action(guarded)

            # Обрезаем поле "thought" и подготавливаем его для лога — можно анализировать reasoning LLM.
            thought_text = (guarded.thought or "").strip()
            # Выводим подробный лог о рассуждении и предстоящем действии с моделью и токенами.
            await stream_callback(
                "\n[THOUGHT]\n"
                f"{thought_text}\n"
                "────────────────────────────────────────────────\n"
                f"[Шаг {global_step}] [{subtask.mode.value}] model={llm.model_route.tier}:"
                f"{llm.model_name or llm.model_route.model} | reason={llm.model_route.reason} | "
                f"tokens(in={llm.prompt_tokens}, out={llm.completion_tokens}) | "
                f"action={guarded.action} | params={guarded.params}"
            )
            # Если действие помечено как опасное — уведомляем пользователя и требуем подтверждение.
            if is_confirmation_required(guarded):
                await stream_callback(
                    "[SECURITY] Действие помечено как опасное, требуется подтверждение пользователя."
                )

            # Инициализируем fingerprint (отпечаток страницы) для последующего сравнения до/после действия, если действие изменяющее.
            before_fp: str | None = None
            if guarded.action in ("click", "navigate", "type"):
                try:
                    before_fp = await snapshot_page_fingerprint(page)
                except Exception:
                    before_fp = None

            # Выполняем конечное действие на странице (например, клик, ввод текста).
            result = await self.executor.execute_action(guarded.action, guarded.params, observation)

            # Сравниваем fingerprint страницы до и после действия — если ничего не изменилось, помечаем результат как не изменивший состояние.
            if (
                before_fp is not None
                and result.success
                and guarded.action in ("click", "navigate", "type")
            ):
                try:
                    await asyncio.sleep(0.1)  # Ждем, чтобы страница успела обновиться.
                    after_fp = await snapshot_page_fingerprint(page)
                    if after_fp == before_fp:
                        result = result.model_copy(update={"changed": False})
                except Exception:
                    pass

            # Обновляем метрики, проверяем условия завершения и выполняем прагматический возврат, если нужно.
            terminal = await self._update_performance_metrics(
                subtask=subtask,
                policy=policy,
                memory=memory,
                cost_stats=cost_stats,
                observation=observation,
                guarded=guarded,
                result=result,
                llm=llm,
                signature=sig,
                global_step=global_step,
                page=page,
                stream_callback=stream_callback,
            )
            # Сохраняем последнее выполненное действие и результат.
            last_action_result = result
            last_action = guarded
            self._refresh_loop_url_snapshot()
            # Если metrika говорит, что пора выйти (например, опасное действие) — делаем это.
            if terminal is not None:
                if terminal == AgentState.AWAITING_USER_CONFIRMATION:
                    self._last_guarded_at_pause = guarded
                return _ret(terminal)

        # Если цикл завершился по лимиту шагов — указываем причину.
        memory.done_reason = memory.done_reason or "Исчерпан лимит шагов подзадачи."
        return _ret(AgentState.SUBTASK_STEP_LIMIT)
