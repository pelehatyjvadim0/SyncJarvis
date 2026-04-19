"""Исполнение guarded-действия; метрики/persist — в ``action_executor``. Вызывается из ``pipeline``."""

from __future__ import annotations

from agent.models.state import AgentState
from agent.runtime.react_loop.components.crop_action_verify import maybe_verify_click_type_crop
from agent.runtime.react_loop.engine.action_executor import run_guarded_action_with_fingerprint_and_metrics
from agent.runtime.react_loop.engine.phases.context import PipelineIterationContext
from agent.runtime.react_loop.engine.phases.types import PhaseStepResult


async def run_execute_phase(ctx: PipelineIterationContext) -> PhaseStepResult:
    assert ctx.llm is not None
    ctx.last_crop_verify_path = None
    crop_out = await maybe_verify_click_type_crop(ctx, ctx.llm.proposed)
    if crop_out == "retry":
        ctx.executed_steps -= 1
        return PhaseStepResult(mode="next_iteration")
    guarded, result, _llm, terminal = await run_guarded_action_with_fingerprint_and_metrics(
        llm=ctx.llm,
        last_action=ctx.last_action,
        policy=ctx.policy,
        subtask=ctx.subtask,
        observation=ctx.observation,
        observation_window=ctx.obs_window,
        memory=ctx.memory,
        page=ctx.page,
        stream_callback=ctx.stream_callback,
        global_step=ctx.global_step,
        cost_stats=ctx.cost_stats,
        executor=ctx.loop.executor,
        config=ctx.loop.config,
        history_logger=ctx.loop.history_logger,
        append_context_history=ctx.loop._append_context_history,
    )
    ctx.memory.crop_verify_no_streak = 0
    ctx.last_action_result = result
    ctx.last_action = guarded
    ctx.loop._refresh_loop_url_snapshot()
    if terminal is not None:
        if terminal == AgentState.AWAITING_USER_CONFIRMATION:
            ctx.loop._last_guarded_at_pause = guarded
        return PhaseStepResult(mode="halt", halt_state=terminal)
    return PhaseStepResult(mode="proceed")
