"""Сбор observation, капча, ранний ``policy.is_done``. Вызывается из ``pipeline``."""

from __future__ import annotations

from agent.llm.prompts.actor import ordered_observation_for_actor_prompt
from agent.models.state import AgentState
from agent.runtime.react_loop.components.captcha import handle_captcha_iteration, is_captcha_present
from agent.runtime.react_loop.components.fingerprinting import page_url_changed_since_stored
from agent.runtime.react_loop.engine.phases.context import PipelineIterationContext
from agent.runtime.react_loop.engine.phases.types import PhaseStepResult
from agent.runtime.self_correction import build_correction_hint
from agent.runtime.react_loop.utils.observation_builder import run_subtask_observation_collection_phase


async def run_observation_phase(ctx: PipelineIterationContext) -> PhaseStepResult:
    ctx.page = ctx.loop.executor.require_page()
    ctx.url_changed_since_last = page_url_changed_since_stored(
        ctx.loop._loop_end_url, ctx.page.url
    )

    _obs_phase = await run_subtask_observation_collection_phase(
        ctx.page,
        memory=ctx.memory,
        stream_callback=ctx.stream_callback,
        executor=ctx.loop.executor,
        require_page=ctx.loop.executor.require_page,
        set_last_observation=lambda obs: setattr(ctx.loop, "last_observation", list(obs)),
        observation_collect_fail_streak=ctx.observation_collect_fail_streak,
        empty_observation_streak=ctx.empty_observation_streak,
    )
    ctx.page = _obs_phase.page
    ctx.observation = _obs_phase.observation
    ctx.observation_collect_fail_streak = _obs_phase.observation_collect_fail_streak
    ctx.empty_observation_streak = _obs_phase.empty_observation_streak
    if _obs_phase.agent_state_if_terminal is not None:
        return PhaseStepResult(mode="halt", halt_state=_obs_phase.agent_state_if_terminal)

    ctx.obs_window = ordered_observation_for_actor_prompt(
        ctx.observation, ctx.loop.actor.prompt_limits
    )

    correction_hint = build_correction_hint(ctx.last_action_result)
    await ctx.stream_callback(
        f"[Шаг {ctx.global_step}] [{ctx.subtask.mode.value}] self_correction={correction_hint}"
    )

    if await is_captcha_present(ctx.page, ctx.observation):
        ctx.captcha_streak += 1
        if ctx.captcha_streak > ctx.loop.config.captcha_max_consecutive_waits:
            ctx.memory.done_reason = "Превышен лимит ожиданий при капче."
            return PhaseStepResult(mode="halt", halt_state=AgentState.BLOCKED_CAPTCHA)
        captcha_out = await handle_captcha_iteration(
            page=ctx.page,
            observation=ctx.observation,
            subtask=ctx.subtask,
            memory=ctx.memory,
            global_step=ctx.global_step,
            stream_callback=ctx.stream_callback,
            executor=ctx.loop.executor,
            history_logger=ctx.loop.history_logger,
            append_context_history=ctx.loop._append_context_history,
        )
        if captcha_out is not None:
            ctx.last_action = captcha_out.last_action
            ctx.last_action_result = captcha_out.last_result
            ctx.executed_steps -= 1
            ctx.loop._refresh_loop_url_snapshot()
            return PhaseStepResult(mode="next_iteration")
    else:
        ctx.captcha_streak = 0

    done, done_reason = ctx.policy.is_done(
        ctx.subtask.goal, ctx.last_action, ctx.last_action_result
    )
    if done:
        ctx.memory.done_reason = done_reason
        return PhaseStepResult(mode="halt", halt_state=AgentState.FINISHED)

    return PhaseStepResult(mode="proceed")
