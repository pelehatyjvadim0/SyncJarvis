from __future__ import annotations

from typing import Any

from agent.models.action import ActionResult
from agent.models.observation import InteractiveElement
from agent.models.state import AgentState
from . import element_actions


async def execute_action(
    ex: Any,
    tool_name: str,
    params: dict[str, Any],
    observation: list[InteractiveElement],
    *,
    user_confirmed_dangerous: bool = False,
) -> ActionResult:
    # Диспетчер инструментов агента: navigate/wait/scroll/click/type/finish; при AWAITING на опасном действии сохраняет параметры в _pending_*.
    try:
        if tool_name == "navigate":
            return await element_actions.navigate(ex, params["url"])
        if tool_name == "wait":
            return await element_actions.wait_tool(ex, float(params.get("seconds", 1.0)))
        if tool_name == "scroll":
            return await element_actions.scroll(
                ex, params.get("direction", "down"), element_actions.parse_scroll_amount(params.get("amount", 600))
            )
        if tool_name == "click_xy":
            return await element_actions.click_xy(ex, float(params.get("x", 0.0)), float(params.get("y", 0.0)))
        if tool_name in {"click", "type"}:
            ax_id = params.get("ax_id")
            element = next((x for x in observation if x.ax_id == ax_id), None)
            if not element:
                return ActionResult(
                    success=False,
                    message=f"Элемент с ax_id='{ax_id}' не найден в текущем наблюдении",
                    reason_code="target_not_found",
                    changed=False,
                )
            dangerous = bool(params.get("is_dangerous", False))
            if tool_name == "click":
                res = await element_actions.click(ex, element, dangerous=dangerous, user_confirmed=user_confirmed_dangerous)
            else:
                if not element_actions.is_text_input_element(element):
                    fallback = element_actions.pick_best_text_input(observation)
                    if fallback is None:
                        return ActionResult(
                            success=False,
                            message=(
                                f"ax_id='{ax_id}' указывает на role='{element.role}', а для type нужно поле ввода; "
                                "подходящий textbox/searchbox не найден"
                            ),
                            reason_code="search_input_not_found",
                            changed=False,
                        )
                    element = fallback
                raw_text = params.get("text")
                if raw_text is None:
                    raw_text = params.get("value", "")
                res = await element_actions.type_text(
                    ex,
                    element,
                    text=str(raw_text),
                    press_enter=bool(params.get("press_enter", False)),
                    dangerous=dangerous,
                    user_confirmed=user_confirmed_dangerous,
                )
            if res.state == AgentState.AWAITING_USER_CONFIRMATION:
                ex._pending_tool = tool_name
                ex._pending_params = dict(params)
            else:
                ex.clear_pending_dangerous_action()
            return res
        if tool_name == "finish":
            return ActionResult(success=True, message="Задача завершена", changed=False, state=AgentState.FINISHED)
        return ActionResult(success=False, message=f"Неизвестный инструмент: {tool_name}", changed=False)
    except Exception as exc:  # noqa: BLE001
        return ActionResult(success=False, message="Ошибка выполнения инструмента", reason_code="tool_dispatch_error", changed=False, error=str(exc))


async def execute_pending_dangerous_confirmation(
    ex: Any,
    observation: list[InteractiveElement],
) -> ActionResult:
    # Повторно выполняет отложенное опасное click/type после подтверждения пользователя (см. has_pending_dangerous_action).
    if not ex.has_pending_dangerous_action():
        return ActionResult(success=False, message="Нет отложенного опасного действия", reason_code="no_pending_dangerous_action", changed=False)
    tool = ex._pending_tool
    params = ex._pending_params or {}
    ex.clear_pending_dangerous_action()
    assert tool is not None
    return await execute_action(ex, tool, params, observation, user_confirmed_dangerous=True)
