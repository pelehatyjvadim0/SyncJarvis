from __future__ import annotations

from dataclasses import dataclass

from agent.llm.services.router import ModelRoute
from agent.models.action import ActionResult, AgentAction


@dataclass(frozen=True)
class _LlmDecision:
    model_route: ModelRoute
    proposed: AgentAction
    model_name: str
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class _CaptchaIterationOutcome:
    last_action: AgentAction
    last_result: ActionResult
