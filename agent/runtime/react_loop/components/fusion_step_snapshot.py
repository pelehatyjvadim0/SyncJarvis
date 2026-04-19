"""Снимок viewport и JSON окна наблюдения для fusion multimodal шага (без вызова LLM)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.config.settings import AppSettings
from agent.llm.prompts.actor import serialize_observation_window_for_actor_prompt
from agent.models.action import ActionResult, AgentAction
from agent.models.observation import InteractiveElement
from agent.models.plan import Subtask
from agent.runtime.memory import RuntimeMemory
from agent.runtime.react_loop.components.grounding import should_apply_grounding_search_wait
from agent.runtime.react_loop.components.viewport_capture import capture_viewport_png_to_file
from agent.runtime.react_loop.config import LoopConfig
from agent.runtime.react_loop.step_utils import fusion_click_xy_recovery_hint, fusion_coordinate_priority_hint


@dataclass(frozen=True)
class FusionStepSnapshot:
    screenshot_path: str
    compact_observation_json: str
    last_step_summary: str
    coordinate_priority_hint: str


async def build_fusion_step_snapshot(
    *,
    page: Any,
    obs_window: list[InteractiveElement],
    last_action: AgentAction | None,
    last_action_result: ActionResult | None,
    memory: RuntimeMemory,
    settings: AppSettings,
    config: LoopConfig,
    prompt_limits: Any,
    global_step: int,
    subtask: Subtask,
) -> FusionStepSnapshot:
    if should_apply_grounding_search_wait(last_action, last_action_result, subtask.mode):
        await asyncio.sleep(settings.grounding_min_wait_seconds)
    shot = Path(config.history_dir) / f"fusion_{global_step:03d}.png"
    await capture_viewport_png_to_file(page, shot)
    compact = serialize_observation_window_for_actor_prompt(obs_window, prompt_limits)
    compact_json = json.dumps(compact, ensure_ascii=False)
    prev_status = "Нет предыдущего результата"
    if last_action_result:
        prev_status = (
            f"success={last_action_result.success}, changed={last_action_result.changed}, "
            f"message={last_action_result.message}, error={last_action_result.error}"
        )
    if last_action and (last_action.thought or "").strip():
        t = last_action.thought.strip()
        if len(t) > 520:
            t = t[:520] + "…"
        prev_status = f"{prev_status}\nПрошлый thought (сверь SCREEN с ожидаемым изменением UI): {t}"
    goh = (memory.last_guard_override_hint or "").strip()
    if goh:
        prev_status = f"{prev_status}\n{goh}"[:2000]
    coord_hint = fusion_coordinate_priority_hint(last_action, last_action_result, memory).strip()
    xy_rec = fusion_click_xy_recovery_hint(last_action, last_action_result, memory).strip()
    if coord_hint and xy_rec:
        coord_hint = f"{coord_hint}\n{xy_rec}"
    elif xy_rec:
        coord_hint = xy_rec
    return FusionStepSnapshot(
        screenshot_path=str(shot),
        compact_observation_json=compact_json,
        last_step_summary=prev_status,
        coordinate_priority_hint=coord_hint[:900],
    )
