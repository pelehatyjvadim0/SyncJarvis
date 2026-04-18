from __future__ import annotations

from agent.models.action import AgentAction
from agent.models.observation import InteractiveElement
from agent.runtime.memory import RuntimeMemory


def action_signature(action: AgentAction) -> str:
    ax_id = action.params.get("ax_id", "")
    text = str(action.params.get("text", ""))[:40]
    direction = action.params.get("direction", "")
    return f"{action.action}|{ax_id}|{direction}|{text}"


def search_candidates_count(observation: list[InteractiveElement]) -> int:
    return sum(1 for item in observation if item.role in {"searchbox", "textbox"})


def is_deep_context_mode(memory: RuntimeMemory) -> bool:
    return memory.stagnation_steps >= 2


def format_runtime_context(memory: RuntimeMemory) -> str:
    base = (
        f"repeat_count={memory.repeat_count}, scroll_streak={memory.scroll_streak}, "
        f"stagnation_steps={memory.stagnation_steps}, tried_ax_ids_count={len(memory.tried_ax_ids)}, "
        f"type_not_editable_streak={memory.type_not_editable_streak}, "
        f"search_target_miss_streak={memory.search_target_miss_streak}, "
        f"self_check_count={memory.self_check_count}, vision_recovery_count={memory.vision_recovery_count}"
    )
    if is_deep_context_mode(memory):
        return f"{base}, mode=DEEP_CONTEXT"
    return base
