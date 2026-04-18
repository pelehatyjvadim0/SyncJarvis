from __future__ import annotations

from agent.models.action import ActionResult, AgentAction
from agent.models.observation import InteractiveElement
from agent.models.task import TaskMode
from agent.policies.base import BaseTaskPolicy
from agent.runtime.memory import MemoryGuardView


class GenericPolicy(BaseTaskPolicy):
    mode = TaskMode.GENERIC

    def mode_rules(self) -> str:
        return "Работай аккуратно и выбирай действие с наибольшим шансом прогресса."

    def is_done(self, subtask_goal: str, last_action: AgentAction | None, last_result: ActionResult | None) -> tuple[bool, str]:
        if last_action and last_action.action == "finish" and last_result and last_result.success:
            return True, "LLM завершил подзадачу."
        return False, ""

    def compute_progress(self, observation: list[InteractiveElement], action: AgentAction, result: ActionResult) -> int:
        score = 1 if result.success else -1
        if action.action == "scroll":
            score -= 1
        if not result.changed:
            score -= 1
        return score

    def anti_loop_guard(
        self,
        subtask_goal: str,
        observation: list[InteractiveElement],
        proposed_action: AgentAction,
        memory_view: MemoryGuardView,
    ) -> AgentAction:
        if memory_view.repeat_count >= 2:
            return AgentAction(thought="Обнаружен цикл, делаю паузу.", action="wait", params={"seconds": 1.0})
        return proposed_action

    def fallback_action(self, subtask_goal: str, observation: list[InteractiveElement], reason: str) -> AgentAction:
        return AgentAction(thought=f"Fallback: {reason}", action="wait", params={"seconds": 1.0})

