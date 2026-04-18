from __future__ import annotations

import json

from agent.config.settings import ActorPromptLimits
from agent.llm.prompts.actor.observation import _serialize_observation_for_prompt
from agent.llm.prompts.templates import (
    actor_decision_json_role_header,
    actor_general_rules_section,
    actor_visibility_section,
)
from agent.models.action import ActionResult
from agent.models.observation import InteractiveElement
from agent.models.task import TaskMode


def _build_light_context(runtime_context: str) -> str:
    # Возвращает полный runtime_context (без фильтрации полей), чтобы модель видела все streak/счётчики.
    return runtime_context[:400]


def build_actor_prompt(
    subtask_goal: str,
    task_mode: TaskMode,
    observation: list[InteractiveElement],
    last_action_result: ActionResult | None,
    runtime_context: str,
    mode_rules: str,
    limits: ActorPromptLimits,
    *,
    self_check_hint: str = "",
) -> str:
    prev_status = "Нет предыдущего результата"
    if last_action_result:
        prev_status = (
            f"success={last_action_result.success}, changed={last_action_result.changed}, "
            f"message={last_action_result.message}, error={last_action_result.error}"
        )
    compact_observation = _serialize_observation_for_prompt(observation, limits)
    light_runtime_context = _build_light_context(runtime_context)
    hint_block = ""
    sh = (self_check_hint or "").strip()
    if sh:
        hint_block = (
            "Самопроверка подзадачи (предыдущий шаг):\n"
            f"- {sh}\n"
            "- Если цель ещё не достигнута — не повторяй то же действие с тем же element_index (или тем же целевым элементом) "
            "без смены стратегии; выбери другой индекс, scroll или другое действие.\n\n"
        )

    return (
        f"{actor_decision_json_role_header()}"
        f"Режим: {task_mode.value}\n"
        f"Цель подзадачи:\n{subtask_goal}\n\n"
        f"Результат прошлого шага:\n{prev_status}\n\n"
        f"Контекст рантайма:\n{light_runtime_context}\n\n"
        f"{hint_block}"
        f"Правила режима:\n{mode_rules}\n\n"
        "Интерактивные элементы:\n"
        f"{json.dumps(compact_observation, ensure_ascii=False)}\n\n"
        f"{actor_visibility_section()}"
        f"{actor_general_rules_section()}"
    )
