from __future__ import annotations

from agent.models.action import AgentAction


def is_confirmation_required(action: AgentAction) -> bool:
    return bool(action.params.get("is_dangerous", False))

