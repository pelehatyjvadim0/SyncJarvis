from __future__ import annotations

from pydantic import BaseModel, Field

from agent.models.task import TaskMode


class PlannerSubtask(BaseModel):
    title: str
    mode: TaskMode
    goal: str


class PlannerResponse(BaseModel):
    thought: str = ""
    subtasks: list[PlannerSubtask] = Field(default_factory=list)

