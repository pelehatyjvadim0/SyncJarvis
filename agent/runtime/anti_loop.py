from __future__ import annotations

from agent.models.action import AgentAction
from agent.runtime.memory import RuntimeMemory


def apply_global_anti_loop(proposed: AgentAction, memory: RuntimeMemory) -> AgentAction:
    if memory.repeat_count >= 2:
        return AgentAction(thought="Глобальная антицикл-защита: пауза для перепланирования.", action="wait", params={"seconds": 1.0})
    if proposed.action == "scroll" and memory.scroll_streak >= 2:
        return AgentAction(thought="Слишком много scroll подряд, переключаюсь на паузу.", action="wait", params={"seconds": 0.8})
    if proposed.action == "scroll" and memory.stagnation_steps >= 3:
        return AgentAction(thought="Стагнация, блокирую очередной scroll.", action="wait", params={"seconds": 1.0})
    return proposed

