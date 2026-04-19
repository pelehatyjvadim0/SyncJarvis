"""Общий вызов `update_performance_metrics` после guarded-действия.

Не является отдельной «вершиной» графа `run_subtask_pipeline`: вызывается только из
`grounding_phase` и `vision_recovery_phase` (там же, где persist/metrics были в монолитном
`pipeline.py` до выноса фаз). Основной путь после LLM — `execute_phase` →
`run_guarded_action_with_fingerprint_and_metrics` в `action_executor` (телеметрия внутри).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from agent.logging.history_logger import HistoryLogger
from agent.models.action import ActionResult, AgentAction
from agent.models.observation import InteractiveElement
from agent.models.plan import Subtask
from agent.models.state import AgentState
from agent.policies.base import BaseTaskPolicy
from agent.runtime.memory import RuntimeMemory
from agent.runtime.react_loop.config import LoopConfig, RunCostStats
from agent.runtime.react_loop.engine.action_executor import update_performance_metrics
from agent.runtime.react_loop.engine.types import _LlmDecision


async def persist_step_execution_metrics(
    *,
    subtask: Subtask,
    policy: BaseTaskPolicy,
    memory: RuntimeMemory,
    cost_stats: RunCostStats,
    observation: list[InteractiveElement],
    observation_window: list[InteractiveElement] | None,
    guarded: AgentAction,
    result: ActionResult,
    llm: _LlmDecision,
    signature: str,
    global_step: int,
    page: Any,
    stream_callback: Callable[[str], Awaitable[None]],
    config: LoopConfig,
    history_logger: HistoryLogger,
    append_context_history: Callable[[dict[str, Any]], None],
) -> AgentState | None:
    return await update_performance_metrics(
        subtask=subtask,
        policy=policy,
        memory=memory,
        cost_stats=cost_stats,
        observation=observation,
        observation_window=observation_window,
        guarded=guarded,
        result=result,
        llm=llm,
        signature=signature,
        global_step=global_step,
        page=page,
        stream_callback=stream_callback,
        config=config,
        history_logger=history_logger,
        append_context_history=append_context_history,
    )
