from __future__ import annotations

from typing import TYPE_CHECKING, Awaitable, Callable

from agent.llm.prompts.actor import ordered_observation_for_actor_prompt
from agent.models.action import ActionResult, AgentAction
from agent.models.observation import InteractiveElement
from agent.models.plan import Subtask
from agent.models.state import AgentState
from agent.policies.base import BaseTaskPolicy
from agent.runtime.memory import RuntimeMemory
from agent.runtime.react_loop.components.captcha_detect import is_captcha_present
from agent.runtime.react_loop.components.fingerprinting import (
    page_url_changed_since_stored,
    safe_current_fingerprint,
)
from agent.runtime.react_loop.config import RunCostStats
from agent.runtime.react_loop.engine.action_executor import run_guarded_action_with_fingerprint_and_metrics
from agent.runtime.react_loop.engine.decision_maker import resolve_decision_element_indexes
from agent.runtime.react_loop.components.grounding import should_run_grounding
from agent.runtime.self_correction import build_correction_hint
from agent.runtime.react_loop.step_utils import action_signature, format_runtime_context
from agent.runtime.react_loop.utils.observation_builder import run_subtask_observation_collection_phase

if TYPE_CHECKING:
    from agent.runtime.react_loop.loop import SubtaskReActLoop


async def run_subtask_pipeline(
    loop: SubtaskReActLoop,
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
    loop._last_guarded_at_pause = None
    loop._loop_end_url = None
    loop._last_grounding_fingerprint = None
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
        loop.last_subtask_executed_step_total = executed_steps
        return state, _steps_delta(), cost_stats

    # Основной цикл — пока не достигнут лимит шагов max_steps.
    # На каждом шаге выполняется наблюдение, принятие решения, выполнение действия и реакция на результат.
    while executed_steps < max_steps:
        # Увеличиваем счетчик шагов субзадачи.
        executed_steps += 1
        # Рассчитываем глобальный номер шага с учетом смещения (step_offset).
        global_step = step_offset + executed_steps
        # Получаем текущую страницу для взаимодействия.
        page = loop._require_page()
        url_changed_since_last = page_url_changed_since_stored(loop._loop_end_url, page.url)

        _obs_phase = await run_subtask_observation_collection_phase(
            page,
            memory=memory,
            stream_callback=stream_callback,
            executor=loop.executor,
            require_page=loop._require_page,
            set_last_observation=lambda obs: setattr(loop, "last_observation", list(obs)),
            observation_collect_fail_streak=observation_collect_fail_streak,
            empty_observation_streak=empty_observation_streak,
        )
        page = _obs_phase.page
        observation = _obs_phase.observation
        observation_collect_fail_streak = _obs_phase.observation_collect_fail_streak
        empty_observation_streak = _obs_phase.empty_observation_streak
        if _obs_phase.agent_state_if_terminal is not None:
            return _ret(_obs_phase.agent_state_if_terminal)

        obs_window = ordered_observation_for_actor_prompt(observation, loop.actor.prompt_limits)

        # Формируем подсказку по коррекции (например, если последнее действие было неудачным).
        correction_hint = build_correction_hint(last_action_result)
        # Логируем текущий шаг и подсказку по коррекции шагов.
        await stream_callback(f"[Шаг {global_step}] [{subtask.mode.value}] self_correction={correction_hint}")

        # Проверка на наличие капчи (captcha) на странице.
        if await is_captcha_present(page, observation):
            # Увеличиваем счетчик подряд встреченных капчей.
            captcha_streak += 1
            # Если превышен лимит ожиданий при капче — выполняем выход.
            if captcha_streak > loop.config.captcha_max_consecutive_waits:
                memory.done_reason = "Превышен лимит ожиданий при капче."
                return _ret(AgentState.BLOCKED_CAPTCHA)
            # Если капча обнаружена — вызываем специальную обработку (_handle_captcha).
            captcha_out = await loop._handle_captcha(
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
                loop._refresh_loop_url_snapshot()
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
        # # чтобы LLM увидел observation, хотя Playwright вернул timeout/перехват.
        _self_check_modes = frozenset({"SELECTION", "TRANSACTION"})
        _run_goal_self_check = (
            loop._settings.subtask_goal_self_check_llm
            and last_action is not None
            and last_action_result is not None
            and (
                last_action_result.success
                or (
                    loop._settings.subtask_goal_self_check_after_failed_click
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
                goal_check = await loop.actor.assess_goal_reached(
                    subtask_goal=subtask.goal,
                    observation=observation,
                    last_action=last_action,
                    last_action_result=last_action_result,
                    runtime_context=runtime_context,
                    model_override=loop.config.model_cheap,
                    max_transport_retries=loop._settings.llm_transport_max_retries,
                )
                # Оцениваем стоимость такого запроса.
                check_cost = loop.config.pricing.estimate_cost_usd(
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
        if not loop._settings.observation_fusion_multimodal:
            run_grounding, gr_reason = should_run_grounding(
                settings=loop._settings,
                subtask_mode=subtask.mode,
                last_action=last_action,
                last_action_result=last_action_result,
                current_fingerprint=current_fp,
                last_grounding_fingerprint=loop._last_grounding_fingerprint,
                url_changed_since_last_step=url_changed_since_last,
            )
        if run_grounding:
            await stream_callback(f"[GROUNDING] start | {gr_reason}")
            try:
                guarded, result, llm = await loop._attempt_grounding(
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
                terminal = await loop._update_performance_metrics(
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
                loop._refresh_loop_url_snapshot()
                if terminal is not None:
                    if terminal == AgentState.AWAITING_USER_CONFIRMATION:
                        loop._last_guarded_at_pause = guarded
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
                guarded, result, llm = await loop._attempt_visual_recovery(
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
                terminal = await loop._update_performance_metrics(
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
                        loop._last_guarded_at_pause = guarded
                    # Завершаем выполнение с нужным статусом.
                    return _ret(terminal)
                loop._refresh_loop_url_snapshot()
                continue
            # Если обнаружена стагнация (например, цикл одинаковых кликов), пробуем восстановление.
            if loop._should_visual_stagnation_recovery(
                subtask, memory, last_action, last_action_result
            ) and last_action_result is not None:
                await stream_callback(
                    "[GUARD] reason=stagnation_click | vision recovery "
                    f"(stagnation_steps={memory.stagnation_steps}, repeat_count={memory.repeat_count})"
                )
                # Запускаем визуальное восстановление как при поисковом тупике.
                guarded, result, llm = await loop._attempt_visual_recovery(
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
                terminal = await loop._update_performance_metrics(
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
                        loop._last_guarded_at_pause = guarded
                    return _ret(terminal)
                loop._refresh_loop_url_snapshot()
                continue
            # Если не требуется визуальное восстановление — LLM: либо fusion (скрин + a11y), либо текстовый актёр.
            if loop._settings.observation_fusion_multimodal:
                await stream_callback("[FUSION] Решение по скриншоту viewport и списку a11y (сверка с изображением).")
            llm = await loop._make_decision(
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
            last_action, last_action_result = await loop._handle_llm_error(
                exc,
                subtask=subtask,
                policy=policy,
                observation=observation,
                memory=memory,
                global_step=global_step,
                page=page,
                stream_callback=stream_callback,
            )
            loop._refresh_loop_url_snapshot()
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
            executor=loop.executor,
            config=loop.config,
            history_logger=loop.history_logger,
            append_context_history=loop._append_context_history,
        )
        # Сохраняем последнее выполненное действие и результат.
        last_action_result = result
        last_action = guarded
        loop._refresh_loop_url_snapshot()
        # Если metrika говорит, что пора выйти (например, опасное действие) — делаем это.
        if terminal is not None:
            if terminal == AgentState.AWAITING_USER_CONFIRMATION:
                loop._last_guarded_at_pause = guarded
            return _ret(terminal)

    # Если цикл завершился по лимиту шагов — указываем причину.
    memory.done_reason = memory.done_reason or "Исчерпан лимит шагов подзадачи."
    return _ret(AgentState.SUBTASK_STEP_LIMIT)
