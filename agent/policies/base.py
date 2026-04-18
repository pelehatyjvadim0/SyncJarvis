from __future__ import annotations

from abc import ABC, abstractmethod

from agent.models.action import ActionResult, AgentAction
from agent.models.observation import InteractiveElement
from agent.models.task import TaskMode
from agent.runtime.memory import MemoryGuardView, RuntimeMemory


class BaseTaskPolicy(ABC):
    mode: TaskMode = TaskMode.GENERIC

    @abstractmethod
    def mode_rules(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def is_done(self, subtask_goal: str, last_action: AgentAction | None, last_result: ActionResult | None) -> tuple[bool, str]:
        raise NotImplementedError

    @abstractmethod
    def compute_progress(
        self,
        observation: list[InteractiveElement],
        action: AgentAction,
        result: ActionResult,
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    def anti_loop_guard(
        self,
        subtask_goal: str,
        observation: list[InteractiveElement],
        proposed_action: AgentAction,
        memory_view: MemoryGuardView,
    ) -> AgentAction:
        raise NotImplementedError

    @abstractmethod
    def fallback_action(self, subtask_goal: str, observation: list[InteractiveElement], reason: str) -> AgentAction:
        raise NotImplementedError

    def refine_after_global_anti_loop(self, guarded: AgentAction, memory: RuntimeMemory) -> AgentAction:
        return guarded

    def check_force_finish_after_execution(
        self, guarded: AgentAction, result: ActionResult, memory: RuntimeMemory
    ) -> tuple[bool, str]:
        return False, ""

