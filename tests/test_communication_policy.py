from __future__ import annotations

import unittest

from agent.models.action import ActionResult, AgentAction
from agent.policies.communication import CommunicationPolicy
from agent.runtime.memory import MemoryGuardView


class CommunicationPolicyTests(unittest.TestCase):
    def test_duplicate_message_blocked(self) -> None:
        policy = CommunicationPolicy()
        proposed = AgentAction(thought="send", action="type", params={"text": "Пока", "ax_id": "x"})
        guarded = policy.anti_loop_guard(
            subtask_goal="Попрощайся и заверши диалог",
            observation=[],
            proposed_action=proposed,
            memory_view=MemoryGuardView(last_sent_text="пока"),
        )
        self.assertEqual(guarded.action, "finish")

    def test_closing_done_after_click(self) -> None:
        policy = CommunicationPolicy()
        done, _ = policy.is_done(
            subtask_goal="Попрощайся и заверши диалог",
            last_action=AgentAction(thought="send", action="click", params={}),
            last_result=ActionResult(success=True, message="ok", changed=True),
        )
        self.assertTrue(done)


if __name__ == "__main__":
    unittest.main()

