from __future__ import annotations

from pydantic import BaseModel

from agent.models.task import TaskMode


class StepTelemetry(BaseModel):
    task_mode: TaskMode = TaskMode.GENERIC
    subtask_id: str = ""
    phase: str = "GENERAL"
    action_signature: str = ""
    repeat_count: int = 0
    scroll_streak: int = 0
    stagnation_steps: int = 0
    progress_score: int = 0
    search_candidates_count: int = 0
    duplicate_blocked: bool = False
    model_tier: str = ""
    model_used: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    plan_source: str = ""
    done_reason: str = ""

