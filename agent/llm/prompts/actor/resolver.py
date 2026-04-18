from __future__ import annotations

from agent.models.action import AgentAction
from agent.models.observation import InteractiveElement


def resolve_actor_element_index(
    proposed: AgentAction,
    observation_window: list[InteractiveElement],
) -> tuple[AgentAction, str | None]:
    # После ответа LLM: element_index → реальный ax_id. Обратная совместимость: только ax_id без индекса — без изменений.
    if proposed.action not in ("click", "type"):
        return proposed, None
    params = dict(proposed.params)
    if "element_index" not in params:
        return proposed, None
    raw_idx = params.get("element_index")
    try:
        idx = int(raw_idx)
    except (TypeError, ValueError):
        return proposed, f"[GUARD] reason=element_index_invalid | value={raw_idx!r}"
    n = len(observation_window)
    if idx < 0 or idx >= n:
        range_hint = "список элементов в промпте пуст" if n == 0 else f"допустимо 0..{n - 1}"
        wait = AgentAction(
            thought=f"{(proposed.thought or '').strip()}\n[Система: element_index={idx} вне диапазона ({range_hint})]".strip(),
            action="wait",
            params={"seconds": 0.45},
        )
        return wait, f"[GUARD] reason=element_index_out_of_range | index={idx} | window_size={n}"
    el = observation_window[idx]
    params["ax_id"] = el.ax_id
    return proposed.model_copy(update={"params": params}), None
