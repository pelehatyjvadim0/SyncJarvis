from __future__ import annotations

from playwright.async_api import Page

from agent.logging.history_logger import HistoryLogger
from agent.models.action import ActionResult, AgentAction
from agent.models.log import StepLog
from agent.models.observation import InteractiveElement
from agent.models.telemetry import StepTelemetry


async def persist_step(
    *,
    page: Page,
    history_logger: HistoryLogger,
    global_step: int,
    observation: list[InteractiveElement],
    llm_response: AgentAction,
    result: ActionResult,
    telemetry: StepTelemetry,
) -> None:
    screenshot_path = await history_logger.save_screenshot(page, global_step)
    result.screenshot_path = screenshot_path
    history_logger.save_step_json(
        StepLog(
            step=global_step,
            observation=observation,
            llm_response=llm_response,
            result=result,
            telemetry=telemetry,
        )
    )
