from __future__ import annotations

import unittest

from agent.config.settings import ActorPromptLimits
from agent.models.action import AgentAction
from agent.models.observation import InteractiveElement
from agent.llm.prompts.actor_prompts import (
    ordered_observation_for_actor_prompt,
    resolve_actor_element_index,
    serialize_observation_window_for_actor_prompt,
)


class ElementIndexResolveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.limits = ActorPromptLimits(max_observation_items=20, max_text_field_len=80)

    def test_ordered_window_matches_serialize_length(self) -> None:
        obs = [
            InteractiveElement(ax_id="x.9", role="button", name="Z", disabled=False),
            InteractiveElement(ax_id="x.1", role="searchbox", name="Поиск", disabled=False),
        ]
        window = ordered_observation_for_actor_prompt(obs, self.limits)
        compact = serialize_observation_window_for_actor_prompt(window, self.limits)
        self.assertEqual(len(window), len(compact))
        for i, row in enumerate(compact):
            self.assertEqual(row["element_index"], i)
            self.assertNotIn("ax_id", row)

    def test_resolve_element_index_sets_ax_id(self) -> None:
        obs = [
            InteractiveElement(ax_id="path.b", role="button", name="Адрес", disabled=False),
            InteractiveElement(ax_id="path.a", role="searchbox", name="Поиск", disabled=False),
        ]
        window = ordered_observation_for_actor_prompt(obs, self.limits)
        proposed = AgentAction(thought="клик", action="click", params={"element_index": 0})
        resolved, warn = resolve_actor_element_index(proposed, window)
        self.assertIsNone(warn)
        self.assertEqual(resolved.params.get("ax_id"), "path.a")
        self.assertEqual(resolved.params.get("element_index"), 0)

    def test_resolve_backward_compat_ax_id_only(self) -> None:
        window = [
            InteractiveElement(ax_id="only", role="link", name="L", disabled=False),
        ]
        proposed = AgentAction(thought="t", action="click", params={"ax_id": "legacy-id"})
        resolved, warn = resolve_actor_element_index(proposed, window)
        self.assertIsNone(warn)
        self.assertEqual(resolved.params.get("ax_id"), "legacy-id")

    def test_resolve_out_of_range_returns_wait(self) -> None:
        window = [
            InteractiveElement(ax_id="a", role="button", name="A", disabled=False),
        ]
        proposed = AgentAction(thought="bad", action="type", params={"element_index": 9, "text": "x"})
        resolved, warn = resolve_actor_element_index(proposed, window)
        self.assertIn("element_index_out_of_range", warn or "")
        self.assertEqual(resolved.action, "wait")


if __name__ == "__main__":
    unittest.main()
