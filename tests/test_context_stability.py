from __future__ import annotations

import unittest

from agent.config.settings import ActorPromptLimits, AgentPricing, AppSettings
from agent.models.action import ActionResult, AgentAction
from agent.models.observation import InteractiveElement
from agent.models.task import TaskMode
from agent.policies.search import SearchPolicy
from agent.runtime.memory import MemoryGuardView
from agent.runtime.react_loop.config import LoopConfig
from agent.runtime.react_loop.loop import SubtaskReActLoop
from agent.runtime.memory import RuntimeMemory
from agent.runtime.react_loop.step_utils import format_runtime_context, fusion_click_xy_recovery_hint
from agent.tools.browser_executor import BrowserToolExecutor


def _settings() -> AppSettings:
    return AppSettings(
        openrouter_api_key="test-key",
        openrouter_model_fallback="openai/gpt-4o-mini",
        openrouter_model_cheap="openai/gpt-4o-mini",
        openrouter_model_smart="openai/gpt-4.1-mini",
        openrouter_http_referer=None,
        openrouter_x_title=None,
        max_total_steps=20,
        max_subtask_steps=10,
        smart_cooldown_steps=3,
        actor_response_max_tokens=200,
        prompt_limits=ActorPromptLimits(max_observation_items=70, max_text_field_len=80),
        orchestrator_temperature=0.2,
        planner_temperature=0.0,
        planner_max_subtasks=6,
        history_dir="history",
        pricing=AgentPricing(0, 0, 0, 0, 0, 0),
        use_llm_planner=True,
        captcha_max_consecutive_waits=20,
        llm_transport_max_retries=1,
        goal_verify_llm=False,
        goal_verify_fail_soft=False,
        continue_after_subtask_step_limit=False,
        browser_headless=True,
        browser_viewport_width=1440,
        browser_viewport_height=900,
        browser_cdp_url=None,
        subtask_goal_self_check_llm=False,
        subtask_goal_self_check_after_failed_click=True,
        observation_fusion_multimodal=False,
        grounding_enabled=False,
        grounding_after_navigate=True,
        grounding_after_search_submit=True,
        grounding_after_fingerprint_change=True,
        grounding_after_url_change=True,
        grounding_modes=frozenset({"SEARCH", "SELECTION"}),
        grounding_min_wait_seconds=0.4,
        browser_navigate_wait_until="domcontentloaded",
        browser_navigate_timeout_ms=30_000,
        browser_navigate_networkidle_timeout_ms=10_000,
        browser_navigate_post_settle_seconds=0.05,
    )


class ContextStabilityTests(unittest.TestCase):
    def test_runtime_context_contains_extended_streaks(self) -> None:
        mem = RuntimeMemory(
            type_not_editable_streak=2,
            search_target_miss_streak=3,
            self_check_count=4,
            vision_recovery_count=1,
        )
        text = format_runtime_context(mem)
        self.assertIn("type_not_editable_streak=2", text)
        self.assertIn("search_target_miss_streak=3", text)
        self.assertIn("self_check_count=4", text)
        self.assertIn("vision_recovery_count=1", text)

    def test_reason_code_for_type_without_input_candidates(self) -> None:
        import asyncio

        async def _run() -> str:
            ex = BrowserToolExecutor()
            obs = [InteractiveElement(ax_id="a1", role="link", name="Some link")]
            result = await ex.execute_action(
                "type",
                {"ax_id": "a1", "text": "hello"},
                obs,
            )
            self.assertFalse(result.success)
            return result.reason_code

        self.assertEqual(asyncio.run(_run()), "search_input_not_found")

    def test_context_history_is_bounded(self) -> None:
        ex = BrowserToolExecutor()
        s = _settings()
        loop = SubtaskReActLoop(
            executor=ex,
            settings=s,
            config=LoopConfig(
                model_cheap=s.openrouter_model_cheap,
                model_smart=s.openrouter_model_smart,
                temperature=0.2,
                pricing=s.pricing,
            ),
        )
        for i in range(260):
            loop._append_context_history({"step": i})  # noqa: SLF001
        self.assertEqual(len(loop.context_history), 200)
        self.assertEqual(loop.context_history[0]["step"], 60)

    def test_search_guard_triggers_by_streak(self) -> None:
        policy = SearchPolicy()
        proposed = AgentAction(
            thought="try type",
            action="type",
            params={"ax_id": "non-input", "text": "Бургер Кинг"},
        )
        observation = [
            InteractiveElement(ax_id="i1", role="searchbox", name="Поиск ресторана"),
            InteractiveElement(ax_id="l1", role="link", name="Ссылка"),
        ]
        guarded = policy.anti_loop_guard(
            subtask_goal="Найти Бургер Кинг",
            observation=observation,
            proposed_action=proposed,
            memory_view=MemoryGuardView(type_not_editable_streak=2),
        )
        self.assertEqual(guarded.action, "type")
        self.assertEqual(guarded.params.get("ax_id"), "i1")
        self.assertEqual(guarded.params.get("press_enter"), True)


class FusionClickXyHintTests(unittest.TestCase):
    def test_hint_on_tool_failure(self) -> None:
        act = AgentAction(thought="t", action="click_xy", params={"x": 1.0, "y": 2.0})
        res = ActionResult(success=False, message="err", changed=False, error="e")
        h = fusion_click_xy_recovery_hint(act, res, RuntimeMemory())
        self.assertIn("success=false", h)
        self.assertIn("x,y", h)

    def test_hint_on_repeat_signature(self) -> None:
        act = AgentAction(thought="t", action="click_xy", params={"x": 10.0, "y": 20.0})
        res = ActionResult(success=True, message="ok", changed=True)
        mem = RuntimeMemory(repeat_count=1)
        h = fusion_click_xy_recovery_hint(act, res, mem)
        self.assertIn("ПРИОРИТЕТ", h)
        self.assertIn("Прошлый thought", h)


if __name__ == "__main__":
    unittest.main()
