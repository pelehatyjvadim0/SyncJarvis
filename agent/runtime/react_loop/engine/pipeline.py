"""Линейный дирижёр шага подзадачи: порядок фаз = прежний монолитный цикл.

Итерация ``while`` (после инкремента счётчика шага):
  observation → self_check → grounding → vision_recovery → llm_decision → execute.

``persist_metrics.persist_step_execution_metrics`` вызывается из vision-ветки
(фаза grounding в цикле остаётся как no-op при viewport-first fusion). Метрики основного execute — внутри
``action_executor.run_guarded_action_with_fingerprint_and_metrics``.
При ``next_iteration`` из ``execute_phase`` (crop-verify smart) счётчик шага откатывается внутри фазы, цикл ``continue``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Awaitable, Callable

from agent.models.action import ActionResult, AgentAction
from agent.models.plan import Subtask
from agent.models.state import AgentState
from agent.policies.base import BaseTaskPolicy
from agent.runtime.memory import RuntimeMemory
from agent.runtime.react_loop.config import RunCostStats
from agent.runtime.react_loop.engine.phases.context import PipelineIterationContext
from agent.runtime.react_loop.engine.phases.execute_phase import run_execute_phase
from agent.runtime.react_loop.engine.phases.grounding_phase import run_grounding_phase
from agent.runtime.react_loop.engine.phases.llm_decision_phase import run_llm_decision_phase
from agent.runtime.react_loop.engine.phases.observation_phase import run_observation_phase
from agent.runtime.react_loop.engine.phases.self_check_phase import run_goal_self_check_phase
from agent.runtime.react_loop.engine.phases.types import PhaseStepResult
from agent.runtime.react_loop.engine.phases.vision_recovery_phase import run_vision_recovery_phase

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
    loop._last_guarded_at_pause = None
    loop._loop_end_url = None
    loop._last_grounding_fingerprint = None

    cost_stats = RunCostStats()
    steps_at_entry = resume_executed_steps

    ctx = PipelineIterationContext(
        loop=loop,
        subtask=subtask,
        policy=policy,
        memory=memory,
        stream_callback=stream_callback,
        cost_stats=cost_stats,
        step_offset=step_offset,
        executed_steps=resume_executed_steps,
        last_action=resume_last_action,
        last_action_result=resume_last_result,
        captcha_streak=0,
        observation_collect_fail_streak=0,
        empty_observation_streak=0,
    )

    def _steps_delta() -> int:
        return ctx.executed_steps - steps_at_entry

    def _ret(state: AgentState) -> tuple[AgentState, int, RunCostStats]:
        loop.last_subtask_executed_step_total = ctx.executed_steps
        return state, _steps_delta(), cost_stats

    def _consume(r: PhaseStepResult) -> tuple[AgentState, int, RunCostStats] | None:
        if r.mode == "halt":
            assert r.halt_state is not None
            return _ret(r.halt_state)
        return None

    while ctx.executed_steps < max_steps:
        ctx.executed_steps += 1
        ctx.global_step = step_offset + ctx.executed_steps

        r_obs = await run_observation_phase(ctx)
        out = _consume(r_obs)
        if out is not None:
            return out
        if r_obs.mode == "next_iteration":
            continue

        r_self = await run_goal_self_check_phase(ctx)
        out = _consume(r_self)
        if out is not None:
            return out

        r_gr = await run_grounding_phase(ctx)
        out = _consume(r_gr)
        if out is not None:
            return out
        if r_gr.mode == "next_iteration":
            continue

        r_vis = await run_vision_recovery_phase(ctx)
        out = _consume(r_vis)
        if out is not None:
            return out
        if r_vis.mode == "next_iteration":
            continue

        r_llm = await run_llm_decision_phase(ctx)
        if r_llm.mode == "next_iteration":
            continue

        r_ex = await run_execute_phase(ctx)
        out = _consume(r_ex)
        if out is not None:
            return out
        if r_ex.mode == "next_iteration":
            continue

    memory.done_reason = memory.done_reason or "Исчерпан лимит шагов подзадачи."
    return _ret(AgentState.SUBTASK_STEP_LIMIT)
