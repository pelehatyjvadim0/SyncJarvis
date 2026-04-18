from __future__ import annotations

from agent.models.action import ActionResult, AgentAction
from agent.models.observation import InteractiveElement
from agent.models.task import TaskMode
from agent.policies.generic import GenericPolicy


class NavigationPolicy(GenericPolicy):
    mode = TaskMode.NAVIGATION

    def mode_rules(self) -> str:
        return "Приоритет - открыть нужную страницу через navigate, затем завершить подзадачу."

    def is_done(self, subtask_goal: str, last_action: AgentAction | None, last_result: ActionResult | None) -> tuple[bool, str]:
        if not last_action or not last_result:
            return False, ""
        if last_action.action == "navigate" and last_result.success:
            return True, "Навигация выполнена."
        if last_action.action == "finish" and last_result.success:
            return True, "Подзадача навигации завершена."
        return False, ""

    def compute_progress(self, observation: list[InteractiveElement], action: AgentAction, result: ActionResult) -> int:
        if action.action == "navigate" and result.success:
            return 3
        return super().compute_progress(observation, action, result)

