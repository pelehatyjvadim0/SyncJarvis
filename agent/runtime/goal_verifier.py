from __future__ import annotations

import json
import re
from typing import Awaitable, Callable

from openai import AsyncOpenAI

from agent.config.settings import AppSettings


async def verify_user_goal_satisfied_llm(
    settings: AppSettings,
    *,
    user_goal: str,
    done_reason_summary: str,
    stream_callback: Callable[[str], Awaitable[None]],
) -> bool:
    # Один короткий запрос к LLM: вернёт ли JSON {"satisfied": true|false} по смыслу цели и краткому отчёту.
    if not settings.goal_verify_llm:
        return True
    headers: dict[str, str] = {}
    if settings.openrouter_http_referer:
        headers["HTTP-Referer"] = settings.openrouter_http_referer
    if settings.openrouter_x_title:
        headers["X-Title"] = settings.openrouter_x_title
    client = AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers=headers or None,
    )
    prompt = (
        "Оцени, достигнута ли цель пользователя по краткому отчёту агента.\n"
        f"Цель пользователя:\n{user_goal}\n\n"
        f"Отчёт / причины завершения подзадач:\n{done_reason_summary}\n\n"
        'Верни СТРОГО одну JSON-строку вида {"satisfied": true} или {"satisfied": false} без markdown.'
    )
    await stream_callback("[VERIFY] Запрос проверки достижения цели (LLM).")
    try:
        response = await client.chat.completions.create(
            model=settings.openrouter_model_cheap,
            max_tokens=80,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:  # noqa: BLE001
        await stream_callback(f"[VERIFY] Ошибка запроса, считаем цель достигнутой: {exc}")
        return True
    raw = (response.choices[0].message.content or "").strip()
    m = re.search(r"\{[^}]+\}", raw)
    if not m:
        await stream_callback("[VERIFY] Нет JSON в ответе, считаем цель достигнутой.")
        return True
    try:
        data = json.loads(m.group(0))
        return bool(data.get("satisfied"))
    except Exception:  # noqa: BLE001
        await stream_callback("[VERIFY] Невалидный JSON, считаем цель достигнутой.")
        return True
