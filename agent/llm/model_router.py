from __future__ import annotations

from dataclasses import dataclass

from agent.models.action import ActionResult
from agent.models.task import TaskMode
from agent.runtime.memory import MemoryGuardView


@dataclass
class ModelRoute:
    tier: str
    model: str
    reason: str


class ModelRouter:
    def __init__(self, cheap_model: str, smart_model: str, smart_cooldown_steps: int = 2):
        self.cheap_model = cheap_model
        self.smart_model = smart_model
        self.smart_cooldown_steps = max(0, smart_cooldown_steps)

    def select(
        self,
        task_mode: TaskMode,
        last_action_result: ActionResult | None,
        memory_view: MemoryGuardView,
        current_step: int,
    ) -> ModelRoute:
        if task_mode in {TaskMode.COMMUNICATION, TaskMode.TRANSACTION, TaskMode.VERIFICATION}:
            return ModelRoute(tier="smart", model=self.smart_model, reason="Критичный режим задачи.")

        wants_smart = False
        smart_reason = ""

        if memory_view.stagnation_steps >= 2:
            wants_smart = True
            smart_reason = "Стагнация шага."

        if memory_view.repeat_count >= 2:
            wants_smart = True
            smart_reason = smart_reason or "Повторение действий."

        if last_action_result and (not last_action_result.success or not last_action_result.changed):
            wants_smart = True
            smart_reason = smart_reason or "Нет прогресса на прошлом шаге."

        if wants_smart:
            last_smart_step = memory_view.last_smart_step
            if current_step - last_smart_step <= self.smart_cooldown_steps:
                return ModelRoute(
                    tier="cheap",
                    model=self.cheap_model,
                    reason="Smart cooldown - экономим токены между эскалациями.",
                )
            return ModelRoute(tier="smart", model=self.smart_model, reason=smart_reason)

        return ModelRoute(tier="cheap", model=self.cheap_model, reason="Обычный шаг.")
