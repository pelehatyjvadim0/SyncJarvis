from __future__ import annotations

from playwright.async_api import Page


async def snapshot_page_fingerprint(page: Page) -> str:
    # URL + длина видимого текста + лёгкий хэш начала body — для сравнения до/после click/navigate/type без доменных правил.
    url = page.url
    digest = await page.evaluate(
        """() => {
            const t = (document.body && document.body.innerText) ? document.body.innerText : "";
            let h = 2166136261 >>> 0;
            const n = Math.min(t.length, 5000);
            for (let i = 0; i < n; i++) {
                h ^= t.charCodeAt(i);
                h = Math.imul(h, 16777619) >>> 0;
            }
            return { len: t.length, h: h >>> 0 };
        }"""
    )
    if not digest or not isinstance(digest, dict):
        return f"{url}|0|0"
    return f"{url}|{int(digest.get('len', 0))}|{int(digest.get('h', 0))}"
