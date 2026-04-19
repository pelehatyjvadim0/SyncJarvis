"""Ветки vision recovery; при срабатывании — ``persist_metrics``. Вызывается из ``pipeline``."""

from __future__ import annotations

from pathlib import Path

from agent.models.action import ActionResult, AgentAction
from agent.models.plan import Subtask
from agent.models.state import AgentState
from agent.runtime.memory import RuntimeMemory
from agent.runtime.react_loop.components.vision_recovery import attempt_visual_recovery
from agent.runtime.react_loop.engine.phases.context import PipelineIterationContext
from agent.runtime.react_loop.engine.phases.persist_metrics import persist_step_execution_metrics
from agent.runtime.react_loop.engine.phases.types import PhaseStepResult
from agent.runtime.react_loop.step_utils import action_signature


def should_visual_stagnation_recovery(
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


async def run_vision_recovery_phase(ctx: PipelineIterationContext) -> PhaseStepResult:
    if (
        ctx.subtask.mode.value == "SEARCH"
        and ctx.memory.search_target_miss_streak >= 3
        and ctx.last_action_result is not None
    ):
        await ctx.stream_callback(
            f"[GUARD] reason=search_target_miss_streak | streak={ctx.memory.search_target_miss_streak}"
        )
        guarded, result, llm = await attempt_visual_recovery(
            subtask=ctx.subtask,
            observation=ctx.observation,
            memory=ctx.memory,
            last_action_result=ctx.last_action_result,
            global_step=ctx.global_step,
            page=ctx.page,
            stream_callback=ctx.stream_callback,
            vision_reason="search_miss",
            history_dir=Path(ctx.loop.config.history_dir),
            model_cheap=ctx.loop.config.model_cheap,
            settings=ctx.loop._settings,
            actor=ctx.loop.actor,
            executor=ctx.loop.executor,
        )
        sig = action_signature(guarded)
        ctx.memory.update_signature(sig)
        ctx.memory.update_after_action(guarded)
        terminal = await persist_step_execution_metrics(
            subtask=ctx.subtask,
            policy=ctx.policy,
            memory=ctx.memory,
            cost_stats=ctx.cost_stats,
            observation=ctx.observation,
            observation_window=ctx.obs_window,
            guarded=guarded,
            result=result,
            llm=llm,
            signature=sig,
            global_step=ctx.global_step,
            page=ctx.page,
            stream_callback=ctx.stream_callback,
            config=ctx.loop.config,
            history_logger=ctx.loop.history_logger,
            append_context_history=ctx.loop._append_context_history,
        )
        ctx.last_action_result = result
        ctx.last_action = guarded
        ctx.memory.vision_recovery_count += 1
        if terminal is not None:
            if terminal == AgentState.AWAITING_USER_CONFIRMATION:
                ctx.loop._last_guarded_at_pause = guarded
            return PhaseStepResult(mode="halt", halt_state=terminal)
        ctx.loop._refresh_loop_url_snapshot()
        return PhaseStepResult(mode="next_iteration")

    if (
        should_visual_stagnation_recovery(
            ctx.subtask, ctx.memory, ctx.last_action, ctx.last_action_result
        )
        and ctx.last_action_result is not None
    ):
        await ctx.stream_callback(
            "[GUARD] reason=stagnation_click | vision recovery "
            f"(stagnation_steps={ctx.memory.stagnation_steps}, repeat_count={ctx.memory.repeat_count})"
        )
        guarded, result, llm = await attempt_visual_recovery(
            subtask=ctx.subtask,
            observation=ctx.observation,
            memory=ctx.memory,
            last_action_result=ctx.last_action_result,
            global_step=ctx.global_step,
            page=ctx.page,
            stream_callback=ctx.stream_callback,
            vision_reason="stagnation_click",
            history_dir=Path(ctx.loop.config.history_dir),
            model_cheap=ctx.loop.config.model_cheap,
            settings=ctx.loop._settings,
            actor=ctx.loop.actor,
            executor=ctx.loop.executor,
        )
        sig = action_signature(guarded)
        ctx.memory.update_signature(sig)
        ctx.memory.update_after_action(guarded)
        terminal = await persist_step_execution_metrics(
            subtask=ctx.subtask,
            policy=ctx.policy,
            memory=ctx.memory,
            cost_stats=ctx.cost_stats,
            observation=ctx.observation,
            observation_window=ctx.obs_window,
            guarded=guarded,
            result=result,
            llm=llm,
            signature=sig,
            global_step=ctx.global_step,
            page=ctx.page,
            stream_callback=ctx.stream_callback,
            config=ctx.loop.config,
            history_logger=ctx.loop.history_logger,
            append_context_history=ctx.loop._append_context_history,
        )
        ctx.last_action_result = result
        ctx.last_action = guarded
        ctx.memory.vision_recovery_count += 1
        if terminal is not None:
            if terminal == AgentState.AWAITING_USER_CONFIRMATION:
                ctx.loop._last_guarded_at_pause = guarded
            return PhaseStepResult(mode="halt", halt_state=terminal)
        ctx.loop._refresh_loop_url_snapshot()
        return PhaseStepResult(mode="next_iteration")

    return PhaseStepResult(mode="proceed")
