from __future__ import annotations

from collections.abc import Awaitable, Callable

from agent.logging.history_logger import HistoryLogger
from agent.models.action import ActionResult, AgentAction
from agent.models.observation import InteractiveElement
from agent.models.plan import Subtask
from agent.runtime.memory import RuntimeMemory
from agent.runtime.react_loop.captcha import is_captcha_present
from agent.runtime.react_loop.engine.types import _CaptchaIterationOutcome
from agent.runtime.react_loop.step_utils import action_signature
from agent.runtime.react_loop.utils.persistence import persist_step
from agent.runtime.react_loop.utils.telemetry import build_step_telemetry
from agent.tools.browser_executor import BrowserToolExecutor


async def handle_captcha_iteration(
    *,
    page,
    observation: list[InteractiveElement],
    subtask: Subtask,
    memory: RuntimeMemory,
    global_step: int,
    stream_callback: Callable[[str], Awaitable[None]],
    executor: BrowserToolExecutor,
    history_logger: HistoryLogger,
    append_context_history: Callable[[dict], None],
) -> _CaptchaIterationOutcome | None:
    # Если на странице капча - один wait, телеметрия CAPTCHA_WAIT, persist; иначе None.
    if not await is_captcha_present(page, observation):
        return None
    await stream_callback(
        "[CAPTCHA] Обнаружена капча. Ожидаю, пока пользователь решит её вручную."
    )
    wait_action = AgentAction(
        thought="Обнаружена капча - ожидаю ручного решения пользователем.",
        action="wait",
        params={"seconds": 3.0},
    )
    wait_result = await executor.execute_action("wait", {"seconds": 3.0}, observation)
    telemetry = build_step_telemetry(
        subtask=subtask,
        memory=memory,
        observation=observation,
        phase="CAPTCHA_WAIT",
        action_signature=action_signature(wait_action),
        progress_score=memory.last_progress_score,
    )
    await persist_step(
        page=page,
        history_logger=history_logger,
        global_step=global_step,
        observation=observation,
        llm_response=wait_action,
        result=wait_result,
        telemetry=telemetry,
    )
    await stream_callback(f"[Шаг {global_step}] result={wait_result.message} | success={wait_result.success}")
    append_context_history(
        {
            "step": global_step,
            "mode": subtask.mode.value,
            "goal": subtask.goal,
            "action": wait_action.action,
            "params": wait_action.params,
            "result_success": wait_result.success,
            "result_message": wait_result.message,
        }
    )
    return _CaptchaIterationOutcome(last_action=wait_action, last_result=wait_result)
