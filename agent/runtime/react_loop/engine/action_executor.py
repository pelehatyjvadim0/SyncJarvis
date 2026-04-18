from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from agent.config.settings import AppSettings
from agent.logging.history_logger import HistoryLogger
from agent.models.action import ActionResult, AgentAction
from agent.models.observation import InteractiveElement
from agent.models.plan import Subtask
from agent.models.state import AgentState
from agent.policies.base import BaseTaskPolicy
from agent.runtime.anti_loop import apply_global_anti_loop
from agent.runtime.memory import RuntimeMemory
from agent.runtime.react_loop.components.fingerprinting import (
    before_fingerprint_if_mutating_action,
    maybe_adjust_result_changed_after_mutating_action,
)
from agent.runtime.react_loop.config import LoopConfig, RunCostStats
from agent.runtime.react_loop.engine.types import _LlmDecision
from agent.runtime.react_loop.step_utils import action_signature
from agent.runtime.react_loop.utils.persistence import persist_step
from agent.runtime.react_loop.utils.telemetry import build_step_telemetry
from agent.runtime.security import is_confirmation_required
from agent.tools.browser_executor import BrowserToolExecutor


def apply_guards(
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


def refine_self_check_repeat_click(
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


async def update_performance_metrics(
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
    config: LoopConfig,
    history_logger: HistoryLogger,
    append_context_history: Callable[[dict[str, Any]], None],
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

    est_cost = config.pricing.estimate_cost_usd(
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
        history_logger=history_logger,
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
    append_context_history(
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


async def run_guarded_action_with_fingerprint_and_metrics(
    *,
    llm: _LlmDecision,
    last_action: AgentAction | None,
    policy: BaseTaskPolicy,
    subtask: Subtask,
    observation: list[InteractiveElement],
    memory: RuntimeMemory,
    page,
    stream_callback: Callable[[str], Awaitable[None]],
    global_step: int,
    cost_stats: RunCostStats,
    executor: BrowserToolExecutor,
    config: LoopConfig,
    history_logger: HistoryLogger,
    append_context_history: Callable[[dict[str, Any]], None],
) -> tuple[AgentAction, ActionResult, _LlmDecision, AgentState | None]:
    # Применяем защитные политики и фильтры к предложенному LLM действию.
    guarded = apply_guards(policy, subtask.goal, observation, llm.proposed, memory)
    after_policy = guarded
    # Дополнительное уточнение: если был повторный клик (self_check), корректируем действие.
    guarded = refine_self_check_repeat_click(guarded, last_action, memory)
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
    before_fp = await before_fingerprint_if_mutating_action(page, guarded.action)

    # Выполняем конечное действие на странице (например, клик, ввод текста).
    result = await executor.execute_action(guarded.action, guarded.params, observation)

    # Сравниваем fingerprint страницы до и после действия — если ничего не изменилось, помечаем результат как не изменивший состояние.
    result = await maybe_adjust_result_changed_after_mutating_action(
        page,
        result,
        guarded_action=guarded.action,
        before_fp=before_fp,
    )

    # Обновляем метрики, проверяем условия завершения и выполняем прагматический возврат, если нужно.
    terminal = await update_performance_metrics(
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
        config=config,
        history_logger=history_logger,
        append_context_history=append_context_history,
    )
    return guarded, result, llm, terminal
