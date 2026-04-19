from __future__ import annotations

from typing import Any

from playwright.async_api import async_playwright

from agent.tools.browser_executor.page_selection import pick_best_page
from agent.tools.browser_executor.url_priority import url_control_priority


def _executor_viewport(ex: Any) -> dict[str, int]:
    w = int(getattr(ex, "_viewport_width", 1440))
    h = int(getattr(ex, "_viewport_height", 900))
    return {"width": w, "height": h}


async def browser_start(ex: Any) -> None:
    # Либо connect_over_cdp к уже запущенному Chromium (AGENT_BROWSER_CDP_URL), либо свой persistent-контекст.
    ex._playwright = await async_playwright().start()
    vp = _executor_viewport(ex)
    if ex._cdp_url:
        ex._cdp_browser = await ex._playwright.chromium.connect_over_cdp(ex._cdp_url)
        contexts = ex._cdp_browser.contexts
        if contexts:
            # Берём контекст, в котором есть «осмысленная» вкладка; иначе первый непустой по страницам.
            scored: list[tuple[int, int, Any]] = []
            for ctx in contexts:
                pages = [p for p in ctx.pages if not p.is_closed()]
                if not pages:
                    continue
                best_pri = min(url_control_priority(p.url or "") for p in pages)
                scored.append((best_pri, -len(pages), ctx))
            scored.sort(key=lambda t: (t[0], t[1]))
            ex._context = scored[0][2] if scored else contexts[0]
        else:
            ex._context = await ex._cdp_browser.new_context(viewport=vp)
        picked = pick_best_page(ex._context)
        ex.page = picked if picked is not None else await ex._context.new_page()
        try:
            await ex.page.bring_to_front()
        except Exception:
            pass
        return
    ex._context = await ex._playwright.chromium.launch_persistent_context(
        user_data_dir=ex._user_data_dir,
        headless=ex._headless,
        viewport=vp,
    )
    picked = pick_best_page(ex._context)
    ex.page = picked if picked is not None else await ex._context.new_page()
    try:
        await ex.page.bring_to_front()
    except Exception:
        pass


async def browser_stop(ex: Any) -> None:
    # CDP: только отключение Playwright от браузера (процесс Chromium не завершаем). Иначе - закрытие своего контекста.
    if ex._cdp_browser:
        await ex._cdp_browser.close()
        ex._cdp_browser = None
        ex._context = None
        ex.page = None
    elif ex._context:
        await ex._context.close()
        ex._context = None
        ex.page = None
    if ex._playwright:
        await ex._playwright.stop()
        ex._playwright = None
