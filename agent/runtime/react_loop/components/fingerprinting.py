from __future__ import annotations

import asyncio
from typing import Any

from agent.models.action import ActionResult
from agent.runtime.react_loop.page_fingerprint import snapshot_page_fingerprint
from agent.tools.browser_executor import BrowserToolExecutor


def read_live_url_from_executor_or_none(executor: BrowserToolExecutor) -> str | None:
    try:
        p = executor.page
        if p and not p.is_closed():
            return p.url
    except Exception:
        pass
    return None


def page_url_changed_since_stored(loop_end_url: str | None, page_url: str) -> bool:
    return loop_end_url is not None and page_url != loop_end_url


async def safe_current_fingerprint(page: Any) -> str:
    try:
        return await snapshot_page_fingerprint(page)
    except Exception:
        return ""


async def before_fingerprint_if_mutating_action(page: Any, guarded_action: str) -> str | None:
    before_fp: str | None = None
    if guarded_action in ("click", "navigate", "type"):
        try:
            before_fp = await snapshot_page_fingerprint(page)
        except Exception:
            before_fp = None
    return before_fp


async def maybe_adjust_result_changed_after_mutating_action(
    page: Any,
    result: ActionResult,
    *,
    guarded_action: str,
    before_fp: str | None,
) -> ActionResult:
    if (
        before_fp is not None
        and result.success
        and guarded_action in ("click", "navigate", "type")
    ):
        try:
            await asyncio.sleep(0.1)  # Ждем, чтобы страница успела обновиться.
            after_fp = await snapshot_page_fingerprint(page)
            if after_fp == before_fp:
                result = result.model_copy(update={"changed": False})
        except Exception:
            pass
    return result
