from __future__ import annotations

from agent.config.settings import ActorPromptLimits
from agent.models.observation import InteractiveElement

from agent.llm.prompts.actor.text_utils import _trim_text


def _observation_sort_key(x: InteractiveElement) -> tuple:
    return (
        0 if x.focused else 1,
        0 if x.disabled is False else 1,
        0 if (x.name or "").strip() else 1,
        0 if x.role in {"searchbox", "textbox", "button", "link"} else 1,
        x.ax_id,
    )


def ordered_observation_for_actor_prompt(
    observation: list[InteractiveElement],
    limits: ActorPromptLimits,
) -> list[InteractiveElement]:
    # Тот же порядок и лимит, что в промпте актора — для резолва element_index → ax_id в цикле.
    sorted_items = sorted(observation, key=_observation_sort_key)
    return sorted_items[: limits.max_observation_items]


def serialize_observation_window_for_actor_prompt(
    window: list[InteractiveElement],
    limits: ActorPromptLimits,
) -> list[dict]:
    # В промпт: короткий element_index; длинный ax_id не показываем — чтобы модель не копировала путь дерева.
    compact: list[dict] = []
    for element_index, item in enumerate(window):
        row: dict = {
            "element_index": element_index,
            "role": item.role,
            "name": _trim_text(item.name, limits.max_text_field_len),
            "disabled": item.disabled,
            "focused": item.focused,
            "value": _trim_text(str(item.value) if item.value is not None else "", limits.max_text_field_len),
            "index_within_role_name": item.index_within_role_name,
        }
        if item.bbox_doc_x is not None and item.bbox_doc_y is not None:
            row["bbox_doc"] = {
                "x": round(item.bbox_doc_x, 1),
                "y": round(item.bbox_doc_y, 1),
                "w": round(item.bbox_doc_w or 0.0, 1),
                "h": round(item.bbox_doc_h or 0.0, 1),
            }
        if item.parent_anchor:
            row["parent"] = item.parent_anchor
        compact.append(row)
    return compact


def _serialize_observation_for_prompt(
    observation: list[InteractiveElement],
    limits: ActorPromptLimits,
) -> list[dict]:
    window = ordered_observation_for_actor_prompt(observation, limits)
    return serialize_observation_window_for_actor_prompt(window, limits)
