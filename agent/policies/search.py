from __future__ import annotations

from agent.models.action import ActionResult, AgentAction
from agent.models.observation import InteractiveElement
from agent.models.task import TaskMode
from agent.policies.generic import GenericPolicy
from agent.runtime.memory import MemoryGuardView


class SearchPolicy(GenericPolicy):
    mode = TaskMode.SEARCH

    def mode_rules(self) -> str:
        return (
            "Сначала используй поля searchbox/textbox для ввода запроса, "
            "потом Enter или клик по кнопке поиска. Избегай лишнего scroll."
        )

    def _find_search_input(self, observation: list[InteractiveElement]) -> InteractiveElement | None:
        for item in observation:
            if item.role not in {"searchbox", "textbox"}:
                continue
            name = (item.name or "").lower()
            if "поиск" in name or "search" in name or not name:
                return item
        return None

    def anti_loop_guard(
        self,
        subtask_goal: str,
        observation: list[InteractiveElement],
        proposed_action: AgentAction,
        memory_view: MemoryGuardView,
    ) -> AgentAction:
        if proposed_action.action == "type" and memory_view.type_not_editable_streak >= 2:
            input_el = self._find_search_input(observation)
            if input_el:
                return AgentAction(
                    thought="Обнаружены повторные ошибки ввода в не-редактируемый элемент, переключаюсь на явный поиск через валидное поле.",
                    action="type",
                    params={"ax_id": input_el.ax_id, "text": subtask_goal, "press_enter": True},
                )
        if proposed_action.action == "scroll" and memory_view.scroll_streak >= 2:
            input_el = self._find_search_input(observation)
            if input_el:
                return AgentAction(
                    thought="Слишком много scroll, переключаюсь на поиск через ввод.",
                    action="type",
                    params={"ax_id": input_el.ax_id, "text": subtask_goal, "press_enter": True},
                )
            return AgentAction(thought="Слишком много scroll, делаю паузу.", action="wait", params={"seconds": 0.8})
        return super().anti_loop_guard(subtask_goal, observation, proposed_action, memory_view)

    def compute_progress(self, observation: list[InteractiveElement], action: AgentAction, result: ActionResult) -> int:
        score = 0
        if self._find_search_input(observation):
            score += 1
        if action.action == "type" and result.success:
            score += 2
        if action.action == "click" and result.success:
            score += 1
        if action.action == "scroll":
            score -= 1
        if not result.success:
            score -= 1
        return score

