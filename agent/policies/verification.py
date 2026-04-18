from __future__ import annotations

from agent.models.action import ActionResult, AgentAction
from agent.models.observation import InteractiveElement
from agent.models.task import TaskMode
from agent.policies.generic import GenericPolicy


class VerificationPolicy(GenericPolicy):
    mode = TaskMode.VERIFICATION

    def mode_rules(self) -> str:
        return "Проверяй, что целевое действие уже выполнено. Если подтверждение есть - заверши подзадачу."

    def compute_progress(self, observation: list[InteractiveElement], action: AgentAction, result: ActionResult) -> int:
        if action.action == "wait":
            return 0
        return super().compute_progress(observation, action, result)

