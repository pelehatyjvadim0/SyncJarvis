"""Снимок viewport для LLM self-check цели подзадачи (без вызова LLM)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.llm.prompts.actor import (
    ordered_observation_for_actor_prompt,
    serialize_observation_window_for_actor_prompt,
)
from agent.models.observation import InteractiveElement
from agent.runtime.react_loop.components.viewport_capture import capture_viewport_png_to_file


@dataclass(frozen=True)
class GoalSelfCheckSnapshot:
    screenshot_path: str
    compact_observation_json: str


async def build_goal_self_check_snapshot(
    *,
    page: Any,
    observation: list[InteractiveElement],
    prompt_limits: Any,
    history_dir: str,
    global_step: int,
) -> GoalSelfCheckSnapshot:
    """Тот же viewport, что после observation; a11y — только компактная подпись к кадру."""
    shot = Path(history_dir) / f"selfcheck_{global_step:03d}.png"
    await capture_viewport_png_to_file(page, shot)
    window = ordered_observation_for_actor_prompt(observation, prompt_limits)
    compact = serialize_observation_window_for_actor_prompt(window, prompt_limits)
    return GoalSelfCheckSnapshot(
        screenshot_path=str(shot),
        compact_observation_json=json.dumps(compact, ensure_ascii=False),
    )
