from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path

from agent.config.settings import AppSettings
from agent.llm.clients.actor import ActorLLMClient
from agent.llm.prompts.actor import (
    resolve_actor_element_index,
    serialize_observation_window_for_actor_prompt,
)
from agent.llm.services.router import ModelRouter
from agent.logging.history_logger import HistoryLogger
from agent.models.action import ActionResult, AgentAction
from agent.models.observation import InteractiveElement
from agent.models.plan import Subtask
from agent.policies.base import BaseTaskPolicy
from agent.runtime.memory import RuntimeMemory
from agent.runtime.react_loop.config import LoopConfig
from agent.runtime.react_loop.engine.types import _LlmDecision
from agent.runtime.react_loop.components.grounding import should_apply_grounding_search_wait
from agent.runtime.react_loop.step_utils import action_signature, format_runtime_context
from agent.runtime.react_loop.utils.persistence import persist_step
from agent.runtime.react_loop.utils.telemetry import build_step_telemetry


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
    if settings.observation_fusion_multimodal:
        if should_apply_grounding_search_wait(last_action, last_action_result, subtask.mode):
            await asyncio.sleep(settings.grounding_min_wait_seconds)
        shot = Path(config.history_dir) / f"fusion_{global_step:03d}.png"
        await page.screenshot(path=str(shot), full_page=False)
        compact = serialize_observation_window_for_actor_prompt(obs_window, actor.prompt_limits)
        compact_json = json.dumps(compact, ensure_ascii=False)
        prev_status = "Нет предыдущего результата"
        if last_action_result:
            prev_status = (
                f"success={last_action_result.success}, changed={last_action_result.changed}, "
                f"message={last_action_result.message}, error={last_action_result.error}"
            )
        decision = await actor.decide_fusion_step_action(
            subtask_goal=subtask.goal,
            task_mode=subtask.mode,
            mode_rules=policy.mode_rules(),
            runtime_context=runtime_context[:400],
            last_step_summary=prev_status,
            self_check_hint=(memory.self_check_hint or "")[:400],
            screenshot_path=str(shot),
            compact_observation_json=compact_json,
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
    decision = await actor.decide_action(
        subtask_goal=subtask.goal,
        task_mode=subtask.mode,
        observation=observation,
        last_action_result=last_action_result,
        runtime_context=runtime_context,
        mode_rules=policy.mode_rules(),
        model_override=model_route.model,
        max_transport_retries=settings.llm_transport_max_retries,
        self_check_hint=(memory.self_check_hint or "")[:400],
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
        history_logger=history_logger,
        global_step=global_step,
        observation=observation,
        llm_response=action,
        result=result,
        telemetry=telemetry,
    )
    await stream_callback(f"[Шаг {global_step}] result={result.message} | success={result.success}")
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
