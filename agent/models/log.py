from __future__ import annotations

from pydantic import BaseModel

from agent.models.action import ActionResult, AgentAction
from agent.models.observation import InteractiveElement
from agent.models.telemetry import StepTelemetry


class StepLog(BaseModel):
    step: int
    observation: list[InteractiveElement]
    # Окно наблюдения в том же порядке, что ушло в промпт актёра (sorted + trim); element_index в llm_response относится к нему, а не к сырому observation.
    observation_window: list[InteractiveElement] | None = None
    llm_response: AgentAction
    result: ActionResult
    telemetry: StepTelemetry | None = None

