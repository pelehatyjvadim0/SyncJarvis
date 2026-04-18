from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path

from agent.config.settings import AppSettings
from agent.llm.clients.actor import ActorLLMClient
from agent.llm.prompts.actor import resolve_actor_element_index, serialize_observation_window_for_actor_prompt
from agent.llm.services.router import ModelRoute
from agent.models.action import ActionResult, AgentAction
from agent.models.observation import InteractiveElement
from agent.models.plan import Subtask
from agent.models.task import TaskMode
from agent.runtime.memory import RuntimeMemory
from agent.runtime.react_loop.components.fingerprinting import (
    before_fingerprint_if_mutating_action,
    maybe_adjust_result_changed_after_mutating_action,
)
from agent.runtime.react_loop.engine.types import _LlmDecision
from agent.tools.browser_executor import BrowserToolExecutor


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


async def attempt_grounding_step(
    *,
    subtask: Subtask,
    observation: list[InteractiveElement],
    obs_window: list[InteractiveElement],
    memory: RuntimeMemory,
    last_action: AgentAction | None,
    last_action_result: ActionResult | None,
    global_step: int,
    page,
    stream_callback: Callable[[str], Awaitable[None]],
    current_fingerprint: str,
    settings: AppSettings,
    model_cheap: str,
    history_dir: Path,
    actor: ActorLLMClient,
    executor: BrowserToolExecutor,
    apply_guards: Callable[[AgentAction], AgentAction],
    refine_self_check_repeat_click: Callable[[AgentAction], AgentAction],
    on_grounding_fingerprint_committed: Callable[[str], None],
) -> tuple[AgentAction, ActionResult, _LlmDecision]:
    # Скрин + multimodal: цель + список element_index; одно действие, затем тот же путь fingerprint/changed, что у основного шага.
    if should_apply_grounding_search_wait(last_action, last_action_result, subtask.mode):
        await asyncio.sleep(settings.grounding_min_wait_seconds)
    shot = history_dir / f"grounding_{global_step:03d}.png"
    await page.screenshot(path=str(shot), full_page=False)
    compact = serialize_observation_window_for_actor_prompt(obs_window, actor.prompt_limits)
    compact_json = json.dumps(compact, ensure_ascii=False)
    gd = await actor.decide_grounding_action(
        subtask_goal=subtask.goal,
        task_mode=subtask.mode,
        screenshot_path=str(shot),
        compact_observation_json=compact_json,
        model_override=model_cheap,
        max_transport_retries=settings.llm_transport_max_retries,
    )
    proposed = gd.action
    resolved, idx_warn = resolve_actor_element_index(proposed, obs_window)
    if idx_warn:
        await stream_callback(idx_warn)
    guarded = apply_guards(resolved)
    guarded = refine_self_check_repeat_click(guarded)
    await stream_callback(
        f"[GROUNDING] action={guarded.action} | params={guarded.params} | "
        f"model={gd.model_used} | tokens in/out={gd.prompt_tokens}/{gd.completion_tokens}"
    )
    before_fp = await before_fingerprint_if_mutating_action(page, guarded.action)
    result = await executor.execute_action(guarded.action, guarded.params, observation)
    result = await maybe_adjust_result_changed_after_mutating_action(
        page,
        result,
        guarded_action=guarded.action,
        before_fp=before_fp,
    )
    llm = _LlmDecision(
        model_route=ModelRoute(tier="cheap", model=model_cheap, reason="Grounding"),
        proposed=guarded,
        model_name=gd.model_used,
        prompt_tokens=gd.prompt_tokens,
        completion_tokens=gd.completion_tokens,
    )
    on_grounding_fingerprint_committed(current_fingerprint)
    return guarded, result, llm
