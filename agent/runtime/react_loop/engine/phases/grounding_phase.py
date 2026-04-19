"""Ветка grounding (исторически multimodal refresh). При viewport-first fusion шаг актёра уже даёт SCREEN+a11y."""

from __future__ import annotations

from agent.runtime.react_loop.engine.phases.context import PipelineIterationContext
from agent.runtime.react_loop.engine.phases.types import PhaseStepResult


async def run_grounding_phase(ctx: PipelineIterationContext) -> PhaseStepResult:
    # Каждый шаг LLM — fusion (viewport PNG + подпись a11y); отдельная фаза grounding не добавляет второй источник истины.
    return PhaseStepResult(mode="proceed")
