from __future__ import annotations

import json

from agent.config.settings import ActorPromptLimits
from agent.models.action import ActionResult, AgentAction
from agent.models.observation import InteractiveElement
from agent.models.task import TaskMode


def _trim_text(value: str | None, max_len: int) -> str:
    if not value:
        return ""
    text = value.strip()
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}..."


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
        compact.append(row)
    return compact


def _serialize_observation_for_prompt(
    observation: list[InteractiveElement],
    limits: ActorPromptLimits,
) -> list[dict]:
    window = ordered_observation_for_actor_prompt(observation, limits)
    return serialize_observation_window_for_actor_prompt(window, limits)


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


def _build_light_context(runtime_context: str) -> str:
    # Возвращает полный runtime_context (без фильтрации полей), чтобы модель видела все streak/счётчики.
    return runtime_context[:400]


def build_actor_prompt(
    subtask_goal: str,
    task_mode: TaskMode,
    observation: list[InteractiveElement],
    last_action_result: ActionResult | None,
    runtime_context: str,
    mode_rules: str,
    limits: ActorPromptLimits,
    *,
    self_check_hint: str = "",
) -> str:
    prev_status = "Нет предыдущего результата"
    if last_action_result:
        prev_status = (
            f"success={last_action_result.success}, changed={last_action_result.changed}, "
            f"message={last_action_result.message}, error={last_action_result.error}"
        )
    compact_observation = _serialize_observation_for_prompt(observation, limits)
    light_runtime_context = _build_light_context(runtime_context)
    hint_block = ""
    sh = (self_check_hint or "").strip()
    if sh:
        hint_block = (
            "Самопроверка подзадачи (предыдущий шаг):\n"
            f"- {sh}\n"
            "- Если цель ещё не достигнута — не повторяй то же действие с тем же element_index (или тем же целевым элементом) "
            "без смены стратегии; выбери другой индекс, scroll или другое действие.\n\n"
        )

    return (
        "Ты агент веб-автоматизации.\n"
        "Верни СТРОГО JSON: thought, action, params.\n"
        "Не используй markdown.\n\n"
        f"Режим: {task_mode.value}\n"
        f"Цель подзадачи:\n{subtask_goal}\n\n"
        f"Результат прошлого шага:\n{prev_status}\n\n"
        f"Контекст рантайма:\n{light_runtime_context}\n\n"
        f"{hint_block}"
        f"Правила режима:\n{mode_rules}\n\n"
        "Интерактивные элементы:\n"
        f"{json.dumps(compact_observation, ensure_ascii=False)}\n\n"
        "Видимость:\n"
        "- В списке только элементы, попадающие в текущую видимую область (viewport) страницы.\n"
        "- Если нужной кнопки или поля нет в списке, вероятно оно ниже или выше — сначала сделай scroll, затем снова ориентируйся на новый список.\n\n"
        "Общие правила:\n"
        "- Используй только: navigate, click, type, scroll, wait, finish.\n"
        "- Для click и type обязательно передай целое params.element_index — значение из поля element_index "
        "в списке интерактивных элементов выше. Номера действуют только для этого списка в этом сообщении; "
        "на следующем шаге список будет другим — смотри только на новый список.\n"
        "- Обратная совместимость: если передаёшь только params.ax_id без element_index — допустимо, но предпочтительнее element_index.\n"
        "- Для action=type передавай текст только в params.text (не в params.value).\n"
        "- Если нужно отправить форму после ввода, используй params.press_enter=true или отдельный click по кнопке поиска.\n"
        "- Не используй CSS/XPath.\n"
        "- Если прошлый шаг не дал прогресса, выбери альтернативу.\n"
        "- Если элементов много, выбирай только самый релевантный к цели."
    )
