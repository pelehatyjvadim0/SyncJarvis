from __future__ import annotations

import unittest

from agent.models.action import AgentAction
from agent.runtime.anti_loop import apply_global_anti_loop
from agent.runtime.memory import RuntimeMemory


class AntiLoopTests(unittest.TestCase):
    def test_block_repeated_actions(self) -> None:
        memory = RuntimeMemory(repeat_count=2)
        proposed = AgentAction(thought="повтор", action="click", params={"ax_id": "1"})
        guarded = apply_global_anti_loop(proposed, memory)
        self.assertEqual(guarded.action, "wait")

    def test_block_scroll_streak(self) -> None:
        memory = RuntimeMemory(scroll_streak=2)
        proposed = AgentAction(thought="scroll", action="scroll", params={"direction": "down"})
        guarded = apply_global_anti_loop(proposed, memory)
        self.assertEqual(guarded.action, "wait")


if __name__ == "__main__":
    unittest.main()

