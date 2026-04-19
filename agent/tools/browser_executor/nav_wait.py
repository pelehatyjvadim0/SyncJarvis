from __future__ import annotations

import asyncio
from typing import Any

from playwright.async_api import Page


def is_playwright_timeout(exc: BaseException) -> bool:
    return type(exc).__name__ == "TimeoutError" and "playwright" in (
        getattr(type(exc), "__module__", "") or ""
    )


async def post_settle_only(ex: Any) -> list[str]:
    # Без networkidle: ввод в поиск и т.п. дергает сеть постоянно — ждать «тишину» = до таймаута на каждый type.
    notes: list[str] = []
    if ex._navigate_post_settle_seconds > 0:
        await asyncio.sleep(ex._navigate_post_settle_seconds)
        notes.append(f"settle={ex._navigate_post_settle_seconds:.2f}s")
    return notes


async def networkidle_then_settle(ex: Any, page: Page) -> list[str]:
    # После load: «тишина сети» ~500ms без запросов; на реальных сайтах часто только таймаут — см. AGENT_BROWSER_NAVIGATE_NETWORKIDLE_TIMEOUT_MS и 0=выкл.
    notes: list[str] = []
    if ex._navigate_networkidle_timeout_ms > 0:
        try:
            await page.wait_for_load_state(
                "networkidle",
                timeout=float(ex._navigate_networkidle_timeout_ms),
            )
            notes.append("networkidle=ok")
        except Exception as exc:
            if not is_playwright_timeout(exc):
                raise
            notes.append(f"networkidle=timeout({ex._navigate_networkidle_timeout_ms}ms)")
    if ex._navigate_post_settle_seconds > 0:
        await asyncio.sleep(ex._navigate_post_settle_seconds)
        notes.append(f"settle={ex._navigate_post_settle_seconds:.2f}s")
    return notes


def load_state_after_mutation(ex: Any) -> str:
    # wait_for_load_state не поддерживает commit; для goto(commit) здесь достаточно domcontentloaded.
    if ex._navigate_wait_until == "commit":
        return "domcontentloaded"
    return ex._navigate_wait_until


async def await_after_page_mutation(ex: Any, page: Page) -> list[str]:
    # Клик по ссылке, submit, SPA-роут: ждём тот же класс готовности, что и после navigate (без вечного load на трекерах).
    notes: list[str] = []
    load_timeout = min(float(ex._navigate_timeout_ms), 120_000.0)
    state = load_state_after_mutation(ex)
    try:
        await page.wait_for_load_state(state, timeout=load_timeout)
        notes.append(f"{state}=ok")
    except Exception as exc:
        if not is_playwright_timeout(exc):
            raise
        notes.append(f"{state}=timeout({int(load_timeout)}ms)")
    notes.extend(await networkidle_then_settle(ex, page))
    return notes


def append_load_idle_suffix(message: str, notes: list[str]) -> str:
    if not notes:
        return message
    return f"{message} | load_idle: {', '.join(notes)}"
