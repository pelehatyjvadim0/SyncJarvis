from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from agent.logging.history_logger import HistoryLogger
from agent.models.action import ActionResult, AgentAction
from agent.models.observation import InteractiveElement
from agent.models.plan import Subtask
from agent.runtime.memory import RuntimeMemory
from agent.runtime.react_loop.step_utils import action_signature
from agent.runtime.react_loop.utils.persistence import persist_step
from agent.runtime.react_loop.utils.telemetry import build_step_telemetry


async def persist_replan_fallback_step(
    *,
    subtask: Subtask,
    memory: RuntimeMemory,
    observation: list[InteractiveElement],
    observation_window: list[InteractiveElement] | None,
    action: AgentAction,
    result: ActionResult,
    global_step: int,
    page: Any,
    history_logger: HistoryLogger,
    stream_callback: Callable[[str], Awaitable[None]],
) -> None:
    """REPLAN после невалидного JSON: телеметрия + persist + stream (инвариант как в ``handle_llm_error``)."""
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
        observation_window=observation_window,
        llm_response=action,
        result=result,
        telemetry=telemetry,
    )
    await stream_callback(f"[Шаг {global_step}] result={result.message} | success={result.success}")
