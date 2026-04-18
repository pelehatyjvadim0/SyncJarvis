from __future__ import annotations

from agent.config.settings import AppSettings
from agent.models.action import ActionResult, AgentAction
from agent.models.task import TaskMode


def should_apply_grounding_search_wait(
    last_action: AgentAction | None,
    last_action_result: ActionResult | None,
    subtask_mode: TaskMode,
) -> bool:
    # Короткая пауза перед скрином: только после отправки поиска (type + Enter) в режиме SEARCH.
    if subtask_mode != TaskMode.SEARCH:
        return False
    if last_action is None or last_action.action != "type":
        return False
    if last_action_result is None or not last_action_result.success:
        return False
    return bool(last_action.params.get("press_enter"))


def should_run_grounding(
    *,
    settings: AppSettings,
    subtask_mode: TaskMode,
    last_action: AgentAction | None,
    last_action_result: ActionResult | None,
    current_fingerprint: str,
    last_grounding_fingerprint: str | None,
    url_changed_since_last_step: bool,
) -> tuple[bool, str]:
    """
    Решает, нужен ли шаг multimodal grounding (скрин + список element_index).
    Дебаунс: тот же fingerprint, что уже отработан grounding — пропуск.
    """
    if not settings.grounding_enabled:
        return False, "grounding_disabled"

    if last_grounding_fingerprint is not None and current_fingerprint == last_grounding_fingerprint:
        return False, "grounding_debounce_same_fingerprint"

    triggers: list[str] = []

    if settings.grounding_after_navigate and last_action is not None and last_action.action == "navigate":
        if last_action_result is not None and last_action_result.success:
            triggers.append("navigate_ok")

    if settings.grounding_after_search_submit:
        if (
            subtask_mode == TaskMode.SEARCH
            and last_action is not None
            and last_action.action == "type"
            and last_action_result is not None
            and last_action_result.success
            and bool(last_action.params.get("press_enter"))
        ):
            triggers.append("search_submit")

    if settings.grounding_after_fingerprint_change:
        if (
            last_action is not None
            and last_action.action in ("click", "type", "navigate")
            and last_action_result is not None
            and last_action_result.success
            and last_action_result.changed
            and subtask_mode.value in settings.grounding_modes
        ):
            triggers.append("fingerprint_changed")

    if settings.grounding_after_url_change and url_changed_since_last_step:
        triggers.append("url_changed")

    if not triggers:
        return False, "grounding_no_trigger"

    return True, "grounding:" + "+".join(triggers)
