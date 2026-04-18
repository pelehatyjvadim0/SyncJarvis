from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable

from agent.config.settings import AppSettings
from agent.llm.clients.actor import ActorLLMClient
from agent.llm.services.router import ModelRouter
from agent.logging.history_logger import HistoryLogger
from agent.models.action import ActionResult, AgentAction
from agent.models.observation import InteractiveElement
from agent.models.plan import Subtask
from agent.models.state import AgentState
from agent.llm.prompts.actor import ordered_observation_for_actor_prompt
from agent.runtime.react_loop.grounding import should_run_grounding
from agent.policies.base import BaseTaskPolicy
from agent.runtime.memory import RuntimeMemory
from agent.runtime.react_loop.captcha import is_captcha_present
from agent.runtime.react_loop.config import LoopConfig, RunCostStats
from agent.runtime.react_loop.engine.action_executor import (
    apply_guards,
    refine_self_check_repeat_click,
    run_guarded_action_with_fingerprint_and_metrics,
    update_performance_metrics,
)
from agent.runtime.react_loop.engine.decision_maker import (
    handle_llm_error,
    make_decision,
    resolve_decision_element_indexes,
)
from agent.runtime.react_loop.step_utils import action_signature, format_runtime_context
from agent.runtime.self_correction import build_correction_hint
from agent.tools.browser_executor import BrowserToolExecutor
from agent.runtime.react_loop.engine.types import (
    _CaptchaIterationOutcome,
    _LlmDecision,
)
from agent.runtime.react_loop.components.captcha_solver import handle_captcha_iteration
from agent.runtime.react_loop.components.fingerprinting import (
    page_url_changed_since_stored,
    read_live_url_from_executor_or_none,
    safe_current_fingerprint,
)
from agent.runtime.react_loop.components.grounding import attempt_grounding_step
from agent.runtime.react_loop.components.vision_recovery import attempt_visual_recovery
from agent.runtime.react_loop.utils.observation_builder import (
    run_subtask_observation_collection_phase,
)


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
        url = read_live_url_from_executor_or_none(self.executor)
        if url is not None:
            self._loop_end_url = url

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
        return await handle_captcha_iteration(
            page=page,
            observation=observation,
            subtask=subtask,
            memory=memory,
            global_step=global_step,
            stream_callback=stream_callback,
            executor=self.executor,
            history_logger=self.history_logger,
            append_context_history=self._append_context_history,
        )

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
        return await make_decision(
            subtask=subtask,
            policy=policy,
            observation=observation,
            obs_window=obs_window,
            last_action=last_action,
            last_action_result=last_action_result,
            memory=memory,
            global_step=global_step,
            page=page,
            actor=self.actor,
            model_router=self.model_router,
            settings=self._settings,
            config=self.config,
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
        return await handle_llm_error(
            exc,
            subtask=subtask,
            policy=policy,
            observation=observation,
            memory=memory,
            global_step=global_step,
            page=page,
            stream_callback=stream_callback,
            history_logger=self.history_logger,
        )

    def _apply_guards(
        self,
        policy: BaseTaskPolicy,
        subtask_goal: str,
        observation: list[InteractiveElement],
        proposed: AgentAction,
        memory: RuntimeMemory,
    ) -> AgentAction:
        return apply_guards(policy, subtask_goal, observation, proposed, memory)

    @staticmethod
    def _refine_self_check_repeat_click(
        proposed: AgentAction,
        last_action: AgentAction | None,
        memory: RuntimeMemory,
    ) -> AgentAction:
        return refine_self_check_repeat_click(proposed, last_action, memory)

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
        return await attempt_visual_recovery(
            subtask=subtask,
            observation=observation,
            memory=memory,
            last_action_result=last_action_result,
            global_step=global_step,
            page=page,
            stream_callback=stream_callback,
            vision_reason=vision_reason,
            history_dir=Path(self.config.history_dir),
            model_cheap=self.config.model_cheap,
            settings=self._settings,
            actor=self.actor,
            executor=self.executor,
        )

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
        return await attempt_grounding_step(
            subtask=subtask,
            observation=observation,
            obs_window=obs_window,
            memory=memory,
            last_action=last_action,
            last_action_result=last_action_result,
            global_step=global_step,
            page=page,
            stream_callback=stream_callback,
            current_fingerprint=current_fingerprint,
            settings=self._settings,
            model_cheap=self.config.model_cheap,
            history_dir=Path(self.config.history_dir),
            actor=self.actor,
            executor=self.executor,
            apply_guards=lambda resolved: apply_guards(policy, subtask.goal, observation, resolved, memory),
            refine_self_check_repeat_click=lambda g: refine_self_check_repeat_click(g, last_action, memory),
            on_grounding_fingerprint_committed=lambda fp: setattr(self, "_last_grounding_fingerprint", fp),
        )

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
        return await update_performance_metrics(
            subtask=subtask,
            policy=policy,
            memory=memory,
            cost_stats=cost_stats,
            observation=observation,
            guarded=guarded,
            result=result,
            llm=llm,
            signature=signature,
            global_step=global_step,
            page=page,
            stream_callback=stream_callback,
            config=self.config,
            history_logger=self.history_logger,
            append_context_history=self._append_context_history,
        )

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

        # Основной цикл — пока не достигнут лимит шагов max_steps.
        # На каждом шаге выполняется наблюдение, принятие решения, выполнение действия и реакция на результат.
        while executed_steps < max_steps:
            # Увеличиваем счетчик шагов субзадачи.
            executed_steps += 1
            # Рассчитываем глобальный номер шага с учетом смещения (step_offset).
            global_step = step_offset + executed_steps
            # Получаем текущую страницу для взаимодействия.
            page = self._require_page()
            url_changed_since_last = page_url_changed_since_stored(self._loop_end_url, page.url)

            _obs_phase = await run_subtask_observation_collection_phase(
                page,
                memory=memory,
                stream_callback=stream_callback,
                executor=self.executor,
                require_page=self._require_page,
                set_last_observation=lambda obs: setattr(self, "last_observation", list(obs)),
                observation_collect_fail_streak=observation_collect_fail_streak,
                empty_observation_streak=empty_observation_streak,
            )
            page = _obs_phase.page
            observation = _obs_phase.observation
            observation_collect_fail_streak = _obs_phase.observation_collect_fail_streak
            empty_observation_streak = _obs_phase.empty_observation_streak
            if _obs_phase.agent_state_if_terminal is not None:
                return _ret(_obs_phase.agent_state_if_terminal)

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

            current_fp = await safe_current_fingerprint(page)
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
                llm = await resolve_decision_element_indexes(llm, obs_window, stream_callback)
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

            guarded, result, llm, terminal = await run_guarded_action_with_fingerprint_and_metrics(
                llm=llm,
                last_action=last_action,
                policy=policy,
                subtask=subtask,
                observation=observation,
                memory=memory,
                page=page,
                stream_callback=stream_callback,
                global_step=global_step,
                cost_stats=cost_stats,
                executor=self.executor,
                config=self.config,
                history_logger=self.history_logger,
                append_context_history=self._append_context_history,
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
