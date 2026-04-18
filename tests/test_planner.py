from __future__ import annotations

import unittest

from agent.models.task import TaskMode
from agent.planner.plan_normalizer import normalize_execution_plan


class PlannerTests(unittest.TestCase):
    def test_decompose_multi_step_goal(self) -> None:
        goal = "Открой сайт и найди песню, потом отправь сообщение и попрощайся"
        plan = normalize_execution_plan(goal)
        self.assertGreaterEqual(len(plan.subtasks), 2)
        modes = [task.mode for task in plan.subtasks]
        self.assertIn(TaskMode.SEARCH, modes)
        self.assertIn(TaskMode.COMMUNICATION, modes)


if __name__ == "__main__":
    unittest.main()

