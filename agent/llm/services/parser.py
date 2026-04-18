from __future__ import annotations

import json

from agent.models.action import AgentAction


def parse_agent_action_json(raw_text: str) -> AgentAction:
    content = (raw_text or "").strip()
    if not content:
        raise ValueError("Модель вернула пустой ответ вместо JSON")
    try:
        data = json.loads(content)
        return AgentAction.model_validate(data)
    except Exception:  # noqa: BLE001
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = content[start : end + 1]
            try:
                data = json.loads(candidate)
                return AgentAction.model_validate(data)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Не удалось разобрать JSON-ответ модели: {exc}") from exc
        raise ValueError("Не удалось найти валидный JSON в ответе модели")
