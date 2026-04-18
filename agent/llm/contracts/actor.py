from __future__ import annotations

from dataclasses import dataclass

from agent.models.action import AgentAction


@dataclass
class ActorDecision:
    action: AgentAction
    model_used: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class GoalCheckDecision:
    goal_reached: bool
    reason: str
    model_used: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class VisualRecoveryDecision:
    action: str
    params: dict
    reason: str
    model_used: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class GroundingDecision:
    action: AgentAction
    model_used: str
    prompt_tokens: int
    completion_tokens: int
