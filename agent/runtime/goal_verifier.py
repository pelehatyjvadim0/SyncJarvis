"""Проверка достижения пользовательской цели после плана (viewport-first).

Инвариант: решение только по SCREEN + формулировка цели; сводка шагов — подпись.
При сетевой ошибке, отсутствии JSON или невалидном JSON — **не** считаем цель достигнутой (возврат False).
Снимок: ``{history_dir}/verify_user_goal.png`` — последний релевантный кадр сессии на момент вызова.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Awaitable, Callable

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

from agent.config.settings import AppSettings
from agent.llm.prompts.templates import llm_json_output_prohibitions_block, user_goal_verify_vision_instructions_block
from agent.llm.services.parser import strip_markdown_json_fence
from agent.tools.browser_executor import BrowserToolExecutor


async def verify_user_goal_satisfied_llm(
    settings: AppSettings,
    *,
    executor: BrowserToolExecutor,
    user_goal: str,
    done_reason_summary: str,
    stream_callback: Callable[[str], Awaitable[None]],
) -> bool:
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
    shot = Path(settings.history_dir) / "verify_user_goal.png"
    try:
        page = executor.require_page()
        shot.parent.mkdir(parents=True, exist_ok=True)
        await page.screenshot(path=str(shot), full_page=False)
    except Exception as exc:  # noqa: BLE001
        await stream_callback(f"[VERIFY] Не удалось снять viewport: {exc} — цель не подтверждена.")
        return False
    img_b64 = base64.b64encode(shot.read_bytes()).decode("ascii")
    prompt = (
        "Оцени по изображению viewport, достигнута ли цель пользователя.\n"
        f"{user_goal_verify_vision_instructions_block()}"
        f"{llm_json_output_prohibitions_block()}"
        "Схема: только {\"satisfied\": true} или {\"satisfied\": false}.\n\n"
        f"Цель пользователя:\n{user_goal}\n\n"
        f"Сводка шагов / причины завершения подзадач (контекст, не замена экрана):\n{done_reason_summary}\n"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ],
        }
    ]
    await stream_callback("[VERIFY] Запрос проверки цели (viewport + LLM).")
    await stream_callback(f"[VERIFY-DEBUG] Снимок для проверки: {shot.resolve()}")
    try:
        response = await client.chat.completions.create(
            model=settings.openrouter_model_cheap,
            max_tokens=120,
            temperature=0.0,
            messages=messages,
        )
    except Exception as exc:  # noqa: BLE001
        await stream_callback(f"[VERIFY] Ошибка запроса — цель не подтверждена: {exc}")
        logger.warning("goal verify HTTP failed: %s", exc, exc_info=True)
        return False

    choice = response.choices[0]
    raw_content = (choice.message.content or "").strip()
    finish_reason = getattr(choice, "finish_reason", None) or "unknown"
    model_used = getattr(response, "model", None) or settings.openrouter_model_cheap
    usage = response.usage
    pt = int(usage.prompt_tokens) if usage and usage.prompt_tokens is not None else None
    ct = int(usage.completion_tokens) if usage and usage.completion_tokens is not None else None
    await stream_callback(
        f"[VERIFY-DEBUG] api_model={model_used} finish_reason={finish_reason} "
        f"prompt_tokens={pt} completion_tokens={ct}"
    )
    if raw_content:
        one_line = raw_content.replace("\n", " ").replace("\r", "")
        preview = one_line[:500] + ("…" if len(one_line) > 500 else "")
        await stream_callback(f"[VERIFY-DEBUG] raw_response_preview={preview!r}")
        logger.info(
            "goal verify raw (len=%s finish=%s): %s",
            len(raw_content),
            finish_reason,
            one_line[:2000],
        )

    raw = strip_markdown_json_fence(raw_content)
    if not raw:
        await stream_callback(
            "[VERIFY-FAIL] Пустой ответ модели после strip — цель не подтверждена. "
            "Частые причины: обрезка по max_tokens, отказ модели вернуть JSON."
        )
        logger.warning("goal verify empty content finish_reason=%s", finish_reason)
        return False
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("not an object")
        satisfied = bool(data.get("satisfied"))
        payload_snip = json.dumps(data, ensure_ascii=False)[:400]
        await stream_callback(f"[VERIFY-RESULT] satisfied={satisfied} parsed={payload_snip}")
        if not satisfied:
            await stream_callback(
                "[VERIFY-FAIL] Модель вернула satisfied=false по текущему viewport. "
                "Сравните цель пользователя со снимком verify_user_goal.png (страница могла уйти с экрана корзины/товара; "
                "или модель консервативно не сопоставила UI с формулировкой цели)."
            )
            logger.info("goal verify not satisfied parsed=%s", payload_snip)
        return satisfied
    except Exception as exc:  # noqa: BLE001
        await stream_callback(
            f"[VERIFY-FAIL] Невалидный JSON после strip: {exc} | после strip (до 400 симв.): {raw[:400]!r}"
        )
        logger.warning("goal verify JSON parse error: %s raw=%r", exc, raw[:800], exc_info=True)
        return False
