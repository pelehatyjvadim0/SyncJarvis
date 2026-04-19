"""Единая съёмка viewport PNG (то же состояние, что видит пользователь)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


async def capture_viewport_png_to_file(page: Any, path: Path) -> None:
    """Скриншот видимой области (full_page=False), без изменения логики Playwright."""
    path.parent.mkdir(parents=True, exist_ok=True)
    await page.screenshot(path=str(path), full_page=False)
