from __future__ import annotations

import json
import re

from agent.models.action import AgentAction

# Открывающий/закрывающий забор markdown; язык не только json (модели пишут javascript/js).
# \A / \Z — только начало/конец всего ответа, чтобы не съесть ``` внутри строк JSON.
_FENCE_OPEN = re.compile(
    r"\A\s*(?:```|~~~)\s*(?:json|javascript|js)?\s*",
    re.IGNORECASE,
)
_FENCE_CLOSE = re.compile(r"\s*(?:```|~~~)\s*\Z", re.IGNORECASE)
# Отдельная строка «json» сразу после забора — артефакт моделей.
_LEADING_JSON_LABEL = re.compile(r"\Ajson\s*(?:\n|\r\n|\r)", re.IGNORECASE)


def strip_markdown_json_fence(text: str) -> str:
    """Убирает ```json / ~~~json и закрывающий забор; опционально строку «json» перед телом."""
    t = (text or "").replace("\ufeff", "").strip()
    if not t:
        return t
    # Несколько вложенных заборов подряд — снимаем по одному, пока меняется (редко, но бывает).
    for _ in range(4):
        before = t
        t = _FENCE_OPEN.sub("", t, count=1).lstrip()
        if t == before:
            break
    t = _FENCE_CLOSE.sub("", t).rstrip()
    # Иногда после ```json идёт отдельная строка «json», затем JSON.
    for _ in range(3):
        m = _LEADING_JSON_LABEL.match(t)
        if not m:
            break
        t = t[m.end() :].lstrip()
    t = _FENCE_CLOSE.sub("", t).rstrip()
    return t.strip()


def parse_agent_action_json(raw_text: str) -> AgentAction:
    content = strip_markdown_json_fence(raw_text or "")
    if not content:
        raise ValueError("Модель вернула пустой ответ вместо JSON")
    try:
        data = json.loads(content)
        action = AgentAction.model_validate(data)
        return _strip_stale_ax_id_when_indexed(action)
    except Exception:  # noqa: BLE001
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = content[start : end + 1]
            try:
                data = json.loads(candidate)
                action = AgentAction.model_validate(data)
                return _strip_stale_ax_id_when_indexed(action)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Не удалось разобрать JSON-ответ модели: {exc}") from exc
        raise ValueError("Не удалось найти валидный JSON в ответе модели")


def _strip_stale_ax_id_when_indexed(action: AgentAction) -> AgentAction:
    # Текстовый режим актёра: модель может продублировать ax_id с прошлого шага — резолвер по индексу тогда перезапишет, но до резолва guard/логи видели бы конфликт.
    if action.action in {"click", "type"} and "element_index" in action.params:
        p = dict(action.params)
        p.pop("ax_id", None)
        return action.model_copy(update={"params": p})
    return action
