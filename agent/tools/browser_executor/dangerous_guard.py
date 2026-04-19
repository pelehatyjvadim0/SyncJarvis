from __future__ import annotations

from agent.models.action import ActionResult
from agent.models.observation import InteractiveElement
from agent.models.state import AgentState


def guard_dangerous(dangerous: bool, action_name: str, element: InteractiveElement) -> ActionResult | None:
    # Если действие помечено опасным и пользователь ещё не подтвердил — возвращает AWAITING_USER_CONFIRMATION вместо DOM-операции.
    if not dangerous:
        return None
    return ActionResult(
        success=False,
        message=f"Требуется подтверждение пользователя для {action_name}: {element.name or element.role}",
        reason_code="awaiting_user_confirmation",
        changed=False,
        is_dangerous=True,
        state=AgentState.AWAITING_USER_CONFIRMATION,
    )
