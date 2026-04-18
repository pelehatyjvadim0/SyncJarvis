from __future__ import annotations

from pydantic import BaseModel, Field

from agent.models.task import TaskMode


class Subtask(BaseModel):
    id: str
    title: str
    mode: TaskMode
    goal: str
    done: bool = False


class ExecutionPlan(BaseModel):
    user_goal: str
    subtasks: list[Subtask] = Field(default_factory=list)
    current_index: int = 0

    def current_subtask(self) -> Subtask | None:
        if self.current_index < 0 or self.current_index >= len(self.subtasks):
            return None
        return self.subtasks[self.current_index]

    def mark_current_done(self) -> None:
        current = self.current_subtask()
        if current:
            current.done = True
            self.current_index += 1

    def is_finished(self) -> bool:
        return self.current_index >= len(self.subtasks)

