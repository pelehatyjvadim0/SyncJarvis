from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

from agent.config.settings import AppSettings
from agent.llm.clients.actor import ActorLLMClient
from agent.llm.services.router import ModelRoute
from agent.models.action import ActionResult, AgentAction
from agent.models.observation import InteractiveElement
from agent.models.plan import Subtask
from agent.runtime.memory import RuntimeMemory
from agent.runtime.react_loop.engine.types import _LlmDecision
from agent.tools.browser_executor import BrowserToolExecutor


async def attempt_visual_recovery(
    *,
    subtask: Subtask,
    observation: list[InteractiveElement],
    memory: RuntimeMemory,
    last_action_result: ActionResult,
    global_step: int,
    page,
    stream_callback: Callable[[str], Awaitable[None]],
    vision_reason: str,
    history_dir: Path,
    model_cheap: str,
    settings: AppSettings,
    actor: ActorLLMClient,
    executor: BrowserToolExecutor,
) -> tuple[AgentAction, ActionResult, _LlmDecision]:
    # Скриншот + multimodal LLM: click_xy/scroll/wait (поиск или застревание на click).
    vision_path = history_dir / f"vision_probe_{global_step:03d}.png"
    await page.screenshot(path=str(vision_path), full_page=False)
    vision = await actor.decide_visual_recovery(
        subtask_goal=subtask.goal,
        screenshot_path=str(vision_path),
        last_error=last_action_result.message,
        model_override=model_cheap,
        max_transport_retries=settings.llm_transport_max_retries,
    )
    action_name = vision.action if vision.action in {"click_xy", "scroll", "wait"} else "wait"
    params = vision.params if isinstance(vision.params, dict) else {}
    guarded = AgentAction(
        thought=f"Vision recovery: {vision.reason or 'попытка восстановить управление по скриншоту'}",
        action=action_name,
        params=params,
    )
    await stream_callback(
        f"[VISION] trigger={vision_reason}, action={guarded.action}, params={guarded.params}, "
        f"model_reason={vision.reason or '-'}"
    )
    result = await executor.execute_action(guarded.action, guarded.params, observation)
    llm = _LlmDecision(
        model_route=ModelRoute(tier="cheap", model=model_cheap, reason="Vision recovery"),
        proposed=guarded,
        model_name=vision.model_used,
        prompt_tokens=vision.prompt_tokens,
        completion_tokens=vision.completion_tokens,
    )
    return guarded, result, llm
