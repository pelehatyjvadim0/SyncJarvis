from __future__ import annotations

import re

from playwright.async_api import Page


async def is_captcha_present(page: Page | None, observation: list) -> bool:
    for item in observation:
        text = f"{item.role} {(item.name or '').lower()}"
        if any(
            token in text
            for token in ["captcha", "капча", "я не робот", "i am human", "recaptcha", "hcaptcha"]
        ):
            return True
    if not page:
        return False
    body_text = (await page.inner_text("body")).lower()
    if re.search(r"(captcha|капч|я не робот|i am not a robot|recaptcha|hcaptcha)", body_text):
        return True
    frame_urls = [frame.url.lower() for frame in page.frames]
    return any(("recaptcha" in url or "hcaptcha" in url) for url in frame_urls)
