"""Мутабельное состояние шага; передаётся во все ``run_*_phase``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from agent.models.action import ActionResult, AgentAction
from agent.models.observation import InteractiveElement
from agent.models.plan import Subtask
from agent.policies.base import BaseTaskPolicy
from agent.runtime.memory import RuntimeMemory
from agent.runtime.react_loop.config import RunCostStats
from agent.runtime.react_loop.engine.types import _LlmDecision

if TYPE_CHECKING:
    from agent.runtime.react_loop.loop import SubtaskReActLoop


@dataclass
class PipelineIterationContext:
    """Состояние одной итерации (и накопители между итерациями), передаваемое в фазы."""

    loop: SubtaskReActLoop
    subtask: Subtask
    policy: BaseTaskPolicy
    memory: RuntimeMemory
    stream_callback: Callable[[str], Awaitable[None]]
    cost_stats: RunCostStats
    step_offset: int
    executed_steps: int
    last_action: AgentAction | None
    last_action_result: ActionResult | None
    captcha_streak: int = 0
    observation_collect_fail_streak: int = 0
    empty_observation_streak: int = 0
    page: Any = None
    global_step: int = 0
    url_changed_since_last: bool = False
    observation: list[InteractiveElement] = field(default_factory=list)
    obs_window: list[InteractiveElement] = field(default_factory=list)
    llm: _LlmDecision | None = None
    # Путь к PNG последнего crop-verify (smart), если фаза сняла кроп; для отладки и логов.
    last_crop_verify_path: str | None = None
