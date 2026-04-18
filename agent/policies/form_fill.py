from __future__ import annotations

from agent.models.action import ActionResult, AgentAction
from agent.models.observation import InteractiveElement
from agent.models.task import TaskMode
from agent.policies.generic import GenericPolicy


class FormFillPolicy(GenericPolicy):
    mode = TaskMode.FORM_FILL

    def mode_rules(self) -> str:
        return "Последовательно заполняй поля и отправляй форму. Избегай лишнего scroll."

    def compute_progress(self, observation: list[InteractiveElement], action: AgentAction, result: ActionResult) -> int:
        score = 0
        if action.action == "type" and result.success:
            score += 2
        if action.action == "click" and result.success:
            score += 1
        if not result.success:
            score -= 1
        return score

