from __future__ import annotations

from typing import Any

from playwright.async_api import BrowserContext, Page

from agent.tools.browser_executor.url_priority import url_control_priority


def pick_best_page(context: BrowserContext) -> Page | None:
    # Не полагаемся на pages[0]: при CDP это часто фоновая about:blank, а нужная вкладка - другая / новее.
    alive: list[tuple[int, Page]] = [(idx, p) for idx, p in enumerate(context.pages) if not p.is_closed()]
    if not alive:
        return None
    alive.sort(key=lambda ip: (url_control_priority(ip[1].url or ""), -ip[0]))
    return alive[0][1]


def maybe_upgrade_from_blank_page(context: BrowserContext | None, current: Page) -> Page:
    # Если управляем пустой вкладкой, а в контексте уже есть страница с URL - переключаемся (target=_blank и т.п.).
    if not context or current.is_closed():
        return current
    cur_pri = url_control_priority(current.url or "")
    if cur_pri <= 1:
        return current
    better = pick_best_page(context)
    if better is None or better is current:
        return current
    if url_control_priority(better.url or "") < cur_pri:
        return better
    return current


def require_page(ex: Any) -> Page:
    """Возвращает живую страницу; если текущая вкладка закрыта, подхватывает другую из context.pages."""
    if ex.page and not ex.page.is_closed():
        ex.page = maybe_upgrade_from_blank_page(ex._context, ex.page)
        return ex.page
    if not ex._context:
        raise RuntimeError("Браузер не запущен")
    picked = pick_best_page(ex._context)
    if picked is not None:
        ex.page = picked
        return ex.page
    raise RuntimeError("Нет активной вкладки браузера")
