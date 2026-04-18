from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from playwright.async_api import Page

from agent.models.log import StepLog


class HistoryLogger:
    def __init__(self, history_dir: str = "history"):
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    async def save_screenshot(self, page: Page, step: int) -> str:
        path = self.history_dir / f"step_{step:03d}.png"
        await page.screenshot(path=str(path), full_page=True)
        return str(path)

    def save_step_json(self, step_log: StepLog) -> str:
        path = self.history_dir / f"step_{step_log.step:03d}.json"
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **step_log.model_dump(),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

