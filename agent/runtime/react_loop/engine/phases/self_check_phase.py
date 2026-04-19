"""LLM self-check цели подзадачи. Вызывается из ``pipeline``."""

from __future__ import annotations

from agent.models.state import AgentState
from agent.runtime.react_loop.components.goal_self_check_snapshot import build_goal_self_check_snapshot
from agent.runtime.react_loop.engine.phases.context import PipelineIterationContext
from agent.runtime.react_loop.engine.phases.types import PhaseStepResult
from agent.runtime.react_loop.step_utils import format_runtime_context


async def run_goal_self_check_phase(ctx: PipelineIterationContext) -> PhaseStepResult:
    # Self-check: по умолчанию только после успешного шага; опционально — после неуспешного click в SELECTION/TRANSACTION,
    # чтобы LLM увидел observation, хотя Playwright вернул timeout/перехват.
    _self_check_modes = frozenset({"SELECTION", "TRANSACTION"})
    _run_goal_self_check = (
        ctx.loop._settings.subtask_goal_self_check_llm
        and ctx.last_action is not None
        and ctx.last_action_result is not None
        and (
            ctx.last_action_result.success
            or (
                ctx.loop._settings.subtask_goal_self_check_after_failed_click
                and ctx.last_action.action == "click"
                and ctx.subtask.mode.value in _self_check_modes
            )
        )
    )
    if not _run_goal_self_check:
        return PhaseStepResult(mode="proceed")

    assert ctx.last_action is not None and ctx.last_action_result is not None
    try:
        runtime_context = format_runtime_context(ctx.memory)
        snap = await build_goal_self_check_snapshot(
            page=ctx.page,
            observation=ctx.observation,
            prompt_limits=ctx.loop.actor.prompt_limits,
            history_dir=ctx.loop.config.history_dir,
            global_step=ctx.global_step,
        )
        goal_check = await ctx.loop.actor.assess_goal_reached(
            subtask_goal=ctx.subtask.goal,
            screenshot_path=snap.screenshot_path,
            compact_observation_json=snap.compact_observation_json,
            last_action=ctx.last_action,
            last_action_result=ctx.last_action_result,
            runtime_context=runtime_context,
            model_override=ctx.loop.config.model_cheap,
            max_transport_retries=ctx.loop._settings.llm_transport_max_retries,
        )
        check_cost = ctx.loop.config.pricing.estimate_cost_usd(
            prompt_tokens=goal_check.prompt_tokens,
            completion_tokens=goal_check.completion_tokens,
            tier="cheap",
        )
        ctx.cost_stats.register(tier="cheap", cost_usd=check_cost)
        await ctx.stream_callback(
            f"[SELF-CHECK] goal_reached={goal_check.goal_reached} | "
            f"reason={goal_check.reason or '-'} | "
            f"model={goal_check.model_used} | "
            f"tokens(in={goal_check.prompt_tokens}, out={goal_check.completion_tokens}) | "
            f"estimated_cost_usd={check_cost:.6f}"
        )
        ctx.memory.self_check_count += 1
        if goal_check.goal_reached:
            ctx.memory.self_check_hint = ""
            ctx.memory.done_reason = goal_check.reason or "LLM self-check: цель подзадачи достигнута."
            return PhaseStepResult(mode="halt", halt_state=AgentState.FINISHED)
        ctx.memory.self_check_hint = (goal_check.reason or "")[:300]
    except Exception as exc:  # noqa: BLE001
        await ctx.stream_callback(f"[SELF-CHECK] Пропускаю из-за ошибки: {exc}")

    return PhaseStepResult(mode="proceed")
