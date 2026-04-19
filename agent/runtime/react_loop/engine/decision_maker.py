from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import replace

from agent.config.settings import AppSettings
from agent.llm.clients.actor import ActorLLMClient
from agent.llm.prompts.actor import resolve_actor_element_index
from agent.llm.services.router import ModelRouter
from agent.logging.history_logger import HistoryLogger
from agent.models.action import ActionResult, AgentAction
from agent.models.observation import InteractiveElement
from agent.models.plan import Subtask
from agent.policies.base import BaseTaskPolicy
from agent.runtime.memory import RuntimeMemory
from agent.runtime.react_loop.components.fusion_step_snapshot import build_fusion_step_snapshot
from agent.runtime.react_loop.config import LoopConfig
from agent.runtime.react_loop.engine.types import _LlmDecision
from agent.runtime.react_loop.observability.replan_recording import persist_replan_fallback_step
from agent.runtime.react_loop.step_utils import format_runtime_context


async def make_decision(
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
    actor: ActorLLMClient,
    model_router: ModelRouter,
    settings: AppSettings,
    config: LoopConfig,
) -> _LlmDecision:
    # Выбор модели (ModelRouter) и запрос к ActorLLMClient; сетевые ретраи из настроек.
    runtime_context = format_runtime_context(memory)
    model_route = model_router.select(
        task_mode=subtask.mode,
        last_action_result=last_action_result,
        memory_view=memory.guard_view(),
        current_step=global_step,
    )
    if model_route.tier == "smart":
        memory.last_smart_step = global_step
    # Единый путь: viewport PNG + цель/контекст; a11y только как подпись к тому же кадру (fusion).
    snap = await build_fusion_step_snapshot(
        page=page,
        obs_window=obs_window,
        last_action=last_action,
        last_action_result=last_action_result,
        memory=memory,
        settings=settings,
        config=config,
        prompt_limits=actor.prompt_limits,
        global_step=global_step,
        subtask=subtask,
    )
    decision = await actor.decide_fusion_step_action(
        subtask_goal=subtask.goal,
        task_mode=subtask.mode,
        mode_rules=policy.mode_rules(),
        runtime_context=runtime_context[:400],
        last_step_summary=snap.last_step_summary,
        self_check_hint=(memory.self_check_hint or "")[:400],
        screenshot_path=snap.screenshot_path,
        compact_observation_json=snap.compact_observation_json,
        coordinate_priority_hint=snap.coordinate_priority_hint,
        model_override=model_route.model,
        max_transport_retries=settings.llm_transport_max_retries,
    )
    return _LlmDecision(
        model_route=model_route,
        proposed=decision.action,
        model_name=decision.model_used,
        prompt_tokens=decision.prompt_tokens,
        completion_tokens=decision.completion_tokens,
    )


async def handle_llm_error(
    exc: ValueError,
    *,
    subtask: Subtask,
    policy: BaseTaskPolicy,
    observation: list[InteractiveElement],
    observation_window: list[InteractiveElement] | None = None,  # при REPLAN после fusion шаг всё ещё можно сопоставить с окном промпта
    memory: RuntimeMemory,
    global_step: int,
    page,
    stream_callback: Callable[[str], Awaitable[None]],
    history_logger: HistoryLogger,
) -> tuple[AgentAction, ActionResult]:
    # Невалидный JSON от актора - fallback политики, REPLAN-телеметрия, persist, стрим.
    result = ActionResult(
        success=False,
        message="Ответ модели невалиден - используется fallback.",
        changed=False,
        error=str(exc),
    )
    action = policy.fallback_action(subtask.goal, observation, reason="Невалидный JSON модели")
    await persist_replan_fallback_step(
        subtask=subtask,
        memory=memory,
        observation=observation,
        observation_window=observation_window,
        action=action,
        result=result,
        global_step=global_step,
        page=page,
        history_logger=history_logger,
        stream_callback=stream_callback,
    )
    return action, result


async def resolve_decision_element_indexes(
    llm: _LlmDecision,
    obs_window: list[InteractiveElement],
    stream_callback: Callable[[str], Awaitable[None]],
) -> _LlmDecision:
    resolved_proposed, idx_warn = resolve_actor_element_index(llm.proposed, obs_window)
    if idx_warn:
        await stream_callback(idx_warn)
    return replace(llm, proposed=resolved_proposed)
