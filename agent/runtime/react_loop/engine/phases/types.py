"""Результат фазы шага: управление ``while`` в ``pipeline.run_subtask_pipeline``.

Отдельно от ``engine.types`` (там структуры LLM/капчи для рантайма), здесь только
контроль потока итерации.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent.models.state import AgentState


@dataclass(frozen=True, slots=True)
class PhaseStepResult:
    """Управление потоком после фазы одного шага цикла подзадачи."""

    mode: Literal["proceed", "next_iteration", "halt"]
    halt_state: AgentState | None = None
