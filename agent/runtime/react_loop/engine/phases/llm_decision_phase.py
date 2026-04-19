"""Решение актёра и разбор индексов. Вызывается из ``pipeline``."""

from __future__ import annotations

from agent.runtime.react_loop.engine.decision_maker import (
    handle_llm_error,
    make_decision,
    resolve_decision_element_indexes,
)
from agent.runtime.react_loop.engine.phases.context import PipelineIterationContext
from agent.runtime.react_loop.engine.phases.types import PhaseStepResult


async def run_llm_decision_phase(ctx: PipelineIterationContext) -> PhaseStepResult:
    try:
        await ctx.stream_callback(
            "[FUSION] Решение по скриншоту viewport и списку a11y (сверка с изображением)."
        )
        ctx.llm = await make_decision(
            subtask=ctx.subtask,
            policy=ctx.policy,
            observation=ctx.observation,
            obs_window=ctx.obs_window,
            last_action=ctx.last_action,
            last_action_result=ctx.last_action_result,
            memory=ctx.memory,
            global_step=ctx.global_step,
            page=ctx.page,
            actor=ctx.loop.actor,
            model_router=ctx.loop.model_router,
            settings=ctx.loop._settings,
            config=ctx.loop.config,
        )
        ctx.llm = await resolve_decision_element_indexes(
            ctx.llm, ctx.obs_window, ctx.stream_callback
        )
    except ValueError as exc:
        await ctx.stream_callback(f"[ACTOR-PARSE] {exc}")
        ctx.last_action, ctx.last_action_result = await handle_llm_error(
            exc,
            subtask=ctx.subtask,
            policy=ctx.policy,
            observation=ctx.observation,
            observation_window=ctx.obs_window,
            memory=ctx.memory,
            global_step=ctx.global_step,
            page=ctx.page,
            stream_callback=ctx.stream_callback,
            history_logger=ctx.loop.history_logger,
        )
        ctx.loop._refresh_loop_url_snapshot()
        return PhaseStepResult(mode="next_iteration")

    return PhaseStepResult(mode="proceed")
