from __future__ import annotations

import unittest
from dataclasses import replace

from agent.config.settings import ActorPromptLimits, AgentPricing, AppSettings
from agent.models.action import ActionResult, AgentAction
from agent.models.task import TaskMode
from agent.runtime.react_loop.grounding import should_run_grounding


def _base_settings(**kwargs: object) -> AppSettings:
    base = AppSettings(
        openrouter_api_key="k",
        openrouter_model_fallback="m",
        openrouter_model_cheap="m",
        openrouter_model_smart="m",
        openrouter_http_referer=None,
        openrouter_x_title=None,
        max_total_steps=10,
        max_subtask_steps=5,
        smart_cooldown_steps=1,
        actor_response_max_tokens=200,
        prompt_limits=ActorPromptLimits(70, 80),
        orchestrator_temperature=0.2,
        planner_temperature=0.0,
        planner_max_subtasks=6,
        history_dir="h",
        pricing=AgentPricing(0, 0, 0, 0, 0, 0),
        use_llm_planner=False,
        captcha_max_consecutive_waits=5,
        llm_transport_max_retries=1,
        goal_verify_llm=False,
        continue_after_subtask_step_limit=False,
        browser_headless=True,
        browser_cdp_url=None,
        subtask_goal_self_check_llm=False,
        subtask_goal_self_check_after_failed_click=True,
        observation_fusion_multimodal=False,
        grounding_enabled=True,
        grounding_after_navigate=True,
        grounding_after_search_submit=True,
        grounding_after_fingerprint_change=True,
        grounding_after_url_change=True,
        grounding_modes=frozenset({"SEARCH", "SELECTION"}),
        grounding_min_wait_seconds=0.4,
    )
    return replace(base, **kwargs) if kwargs else base


class ShouldRunGroundingTests(unittest.TestCase):
    def test_disabled(self) -> None:
        s = _base_settings(grounding_enabled=False)
        ok, reason = should_run_grounding(
            settings=s,
            subtask_mode=TaskMode.SEARCH,
            last_action=AgentAction(thought="n", action="navigate", params={"url": "https://a"}),
            last_action_result=ActionResult(success=True, message="ok"),
            current_fingerprint="fp1",
            last_grounding_fingerprint=None,
            url_changed_since_last_step=False,
        )
        self.assertFalse(ok)
        self.assertIn("disabled", reason)

    def test_debounce_same_fingerprint(self) -> None:
        s = _base_settings()
        ok, _ = should_run_grounding(
            settings=s,
            subtask_mode=TaskMode.SEARCH,
            last_action=AgentAction(thought="n", action="navigate", params={"url": "https://a"}),
            last_action_result=ActionResult(success=True, message="ok"),
            current_fingerprint="same",
            last_grounding_fingerprint="same",
            url_changed_since_last_step=False,
        )
        self.assertFalse(ok)

    def test_navigate_trigger(self) -> None:
        s = _base_settings()
        ok, reason = should_run_grounding(
            settings=s,
            subtask_mode=TaskMode.NAVIGATION,
            last_action=AgentAction(thought="n", action="navigate", params={"url": "https://a"}),
            last_action_result=ActionResult(success=True, message="ok"),
            current_fingerprint="fp_new",
            last_grounding_fingerprint="fp_old",
            url_changed_since_last_step=False,
        )
        self.assertTrue(ok)
        self.assertIn("navigate_ok", reason)

    def test_search_submit_trigger(self) -> None:
        s = _base_settings()
        ok, reason = should_run_grounding(
            settings=s,
            subtask_mode=TaskMode.SEARCH,
            last_action=AgentAction(
                thought="t",
                action="type",
                params={"element_index": 0, "text": "x", "press_enter": True},
            ),
            last_action_result=ActionResult(success=True, message="ok"),
            current_fingerprint="b",
            last_grounding_fingerprint="a",
            url_changed_since_last_step=False,
        )
        self.assertTrue(ok)
        self.assertIn("search_submit", reason)

    def test_fingerprint_changed_respects_modes(self) -> None:
        s = _base_settings()
        ok, _ = should_run_grounding(
            settings=s,
            subtask_mode=TaskMode.GENERIC,
            last_action=AgentAction(thought="c", action="click", params={"ax_id": "1"}),
            last_action_result=ActionResult(success=True, message="ok", changed=True),
            current_fingerprint="x",
            last_grounding_fingerprint=None,
            url_changed_since_last_step=False,
        )
        self.assertFalse(ok)

        ok2, reason2 = should_run_grounding(
            settings=s,
            subtask_mode=TaskMode.SELECTION,
            last_action=AgentAction(thought="c", action="click", params={"ax_id": "1"}),
            last_action_result=ActionResult(success=True, message="ok", changed=True),
            current_fingerprint="x",
            last_grounding_fingerprint=None,
            url_changed_since_last_step=False,
        )
        self.assertTrue(ok2)
        self.assertIn("fingerprint_changed", reason2)

    def test_url_changed_trigger(self) -> None:
        s = _base_settings()
        ok, reason = should_run_grounding(
            settings=s,
            subtask_mode=TaskMode.GENERIC,
            last_action=None,
            last_action_result=None,
            current_fingerprint="u1",
            last_grounding_fingerprint=None,
            url_changed_since_last_step=True,
        )
        self.assertTrue(ok)
        self.assertIn("url_changed", reason)


if __name__ == "__main__":
    unittest.main()
