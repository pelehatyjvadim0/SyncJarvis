from __future__ import annotations

from pydantic import BaseModel

from agent.models.action import ActionResult, AgentAction
from agent.models.observation import InteractiveElement
from agent.models.telemetry import StepTelemetry


class StepLog(BaseModel):
    step: int
    observation: list[InteractiveElement]
    llm_response: AgentAction
    result: ActionResult
    telemetry: StepTelemetry | None = None

