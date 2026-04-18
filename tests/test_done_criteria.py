from __future__ import annotations

import unittest

from agent.models.action import ActionResult, AgentAction
from agent.policies.navigation import NavigationPolicy
from agent.policies.verification import VerificationPolicy


class DoneCriteriaTests(unittest.TestCase):
    def test_navigation_done_after_navigate_success(self) -> None:
        policy = NavigationPolicy()
        done, _ = policy.is_done(
            subtask_goal="Открой страницу",
            last_action=AgentAction(thought="go", action="navigate", params={"url": "https://example.com"}),
            last_result=ActionResult(success=True, message="ok", changed=True),
        )
        self.assertTrue(done)

    def test_verification_not_done_without_finish(self) -> None:
        policy = VerificationPolicy()
        done, _ = policy.is_done(
            subtask_goal="Проверь что действие выполнено",
            last_action=AgentAction(thought="wait", action="wait", params={"seconds": 1}),
            last_result=ActionResult(success=True, message="ok", changed=False),
        )
        self.assertFalse(done)


if __name__ == "__main__":
    unittest.main()

