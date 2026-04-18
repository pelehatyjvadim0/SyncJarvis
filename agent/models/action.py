from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent.models.state import AgentState


class AgentAction(BaseModel):
    thought: str = Field(..., description="Короткое рассуждение перед следующим действием")
    action: str = Field(
        ...,
        description="Название инструмента: navigate, click, type, scroll, wait, finish",
    )
    params: dict[str, Any] = Field(default_factory=dict)


class ActionResult(BaseModel):
    success: bool
    message: str
    # Машиночитаемая причина результата (для устойчивых guard'ов и аналитики без парсинга message).
    reason_code: str = ""
    changed: bool = True
    is_dangerous: bool = False
    state: AgentState = AgentState.RUNNING
    screenshot_path: str | None = None
    error: str | None = None

