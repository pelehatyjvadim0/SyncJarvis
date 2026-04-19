"""Восстановление action/params из частично обрезанного fusion JSON."""

from __future__ import annotations

import unittest

from agent.llm.services.fusion_partial_recovery import try_recover_fusion_agent_action


class FusionRecoveryTests(unittest.TestCase):
    def test_recover_click_truncated(self) -> None:
        raw = '```json\n{"thought":"x","action":"click","params":{"element_index":7'
        r = try_recover_fusion_agent_action(raw)
        self.assertIsNotNone(r)
        assert r is not None
        self.assertEqual(r.action, "click")
        self.assertEqual(r.params.get("element_index"), 7)

    def test_recover_none_without_action(self) -> None:
        self.assertIsNone(try_recover_fusion_agent_action("not json at all"))

    def test_recover_wait(self) -> None:
        raw = '{"action":"wait","params":{"seconds":1.2}'
        r = try_recover_fusion_agent_action(raw)
        self.assertIsNotNone(r)
        assert r is not None
        self.assertEqual(r.action, "wait")
        self.assertAlmostEqual(r.params["seconds"], 1.2)
