from __future__ import annotations

from enum import Enum


class AgentState(str, Enum):
    RUNNING = "RUNNING"
    AWAITING_USER_CONFIRMATION = "AWAITING_USER_CONFIRMATION"
    FINISHED = "FINISHED"
    ERROR = "ERROR"
    # Подзадача исчерпала AGENT_MAX_SUBTASK_STEPS без успешного завершения.
    SUBTASK_STEP_LIMIT = "SUBTASK_STEP_LIMIT"
    # Капча не исчезла после AGENT_CAPTCHA_MAX_CONSECUTIVE_WAIT итераций ожидания.
    BLOCKED_CAPTCHA = "BLOCKED_CAPTCHA"
    # Часть подзадач выполнена, затем остановка по лимиту шага подзадачи при AGENT_CONTINUE_AFTER_SUBTASK_LIMIT=1.
    PARTIAL = "PARTIAL"

