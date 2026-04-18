from __future__ import annotations

from agent.models.observation import InteractiveElement
from agent.models.plan import Subtask
from agent.models.telemetry import StepTelemetry
from agent.runtime.memory import RuntimeMemory

from agent.runtime.react_loop.step_utils import search_candidates_count


def build_step_telemetry(
    *,
    subtask: Subtask,
    memory: RuntimeMemory,
    observation: list[InteractiveElement],
    phase: str,
    action_signature: str,
    progress_score: int,
    model_tier: str = "",
    model_used: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    estimated_cost_usd: float = 0.0,
    plan_source: str = "normalizer",
) -> StepTelemetry:
    return StepTelemetry(
        task_mode=subtask.mode,
        subtask_id=subtask.id,
        phase=phase,
        action_signature=action_signature,
        repeat_count=memory.repeat_count,
        scroll_streak=memory.scroll_streak,
        stagnation_steps=memory.stagnation_steps,
        progress_score=progress_score,
        search_candidates_count=search_candidates_count(observation),
        duplicate_blocked=memory.duplicate_blocked,
        model_tier=model_tier,
        model_used=model_used,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        estimated_cost_usd=estimated_cost_usd,
        plan_source=plan_source,
        done_reason=memory.done_reason,
    )
