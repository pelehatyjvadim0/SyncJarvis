from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from playwright.async_api import async_playwright

from agent.config.settings import ActorPromptLimits, AgentPricing, AppSettings
from agent.models.action import AgentAction
from agent.models.plan import Subtask
from agent.models.state import AgentState
from agent.models.task import TaskMode
from agent.perception.accessibility import collect_all_interactive_elements, collect_interactive_elements
from agent.policies.generic import GenericPolicy
from agent.runtime.memory import RuntimeMemory
from agent.runtime.react_loop.config import LoopConfig
from agent.runtime.react_loop.loop import SubtaskReActLoop, _LlmDecision
from agent.tools.browser_executor import BrowserToolExecutor
from agent.llm.model_router import ModelRoute

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "viewport_scroll.html"


def _fixture_uri() -> str:
    return _FIXTURE.resolve().as_uri()


def _make_settings(history_dir: str) -> AppSettings:
    return AppSettings(
        openrouter_api_key="test-key",
        openrouter_model_fallback="openai/gpt-4o-mini",
        openrouter_model_cheap="openai/gpt-4o-mini",
        openrouter_model_smart="anthropic/claude-sonnet-4",
        openrouter_http_referer=None,
        openrouter_x_title=None,
        max_total_steps=20,
        max_subtask_steps=10,
        smart_cooldown_steps=5,
        actor_response_max_tokens=200,
        prompt_limits=ActorPromptLimits(max_observation_items=70, max_text_field_len=80),
        orchestrator_temperature=0.2,
        planner_temperature=0.0,
        planner_max_subtasks=6,
        history_dir=history_dir,
        pricing=AgentPricing(
            default_input_per_1m=0.0,
            default_output_per_1m=0.0,
            cheap_input_per_1m=0.0,
            cheap_output_per_1m=0.0,
            smart_input_per_1m=0.0,
            smart_output_per_1m=0.0,
        ),
        use_llm_planner=False,
        captcha_max_consecutive_waits=20,
        llm_transport_max_retries=1,
        goal_verify_llm=False,
        continue_after_subtask_step_limit=False,
        browser_headless=False,
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
    )


class E2EViewportFlowTests(IsolatedAsyncioTestCase):
    async def test_viewport_hides_below_fold_until_scroll(self) -> None:
        """До scroll в viewport-списке нет нижней кнопки; после wheel она появляется (bbox + фильтр)."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                page = await browser.new_page(viewport={"width": 900, "height": 600})
                await page.goto(_fixture_uri())
                visible_before = await collect_interactive_elements(page)
                names_before = {(e.name or "") for e in visible_before}
                self.assertTrue(any("видимости" in n or "viewport" in n.lower() for n in names_before))
                self.assertFalse(any("ниже" in n or "fold" in n.lower() for n in names_before))

                await page.mouse.wheel(0, 1400)
                await asyncio.sleep(0.15)
                visible_after = await collect_interactive_elements(page)
                names_after = {(e.name or "") for e in visible_after}
                self.assertTrue(any("ниже" in n or "fold" in n.lower() for n in names_after))

                for el in visible_after:
                    if el.bbox_doc_x is not None:
                        self.assertIsNotNone(el.bbox_doc_w)
            finally:
                await browser.close()

    async def test_collect_all_includes_offscreen_without_filter(self) -> None:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                page = await browser.new_page(viewport={"width": 900, "height": 600})
                await page.goto(_fixture_uri())
                all_el = await collect_all_interactive_elements(page)
                names = {(e.name or "") for e in all_el}
                self.assertTrue(any("видимости" in n or "viewport" in n.lower() for n in names))
                self.assertTrue(any("ниже" in n or "fold" in n.lower() for n in names))
            finally:
                await browser.close()

    async def test_react_loop_mock_llm_scroll_then_finish(self) -> None:
        """E2E: мок _make_decision (без сети) — scroll вниз, затем finish."""
        profile = tempfile.mkdtemp(prefix="sj-prof-")
        hist = tempfile.mkdtemp(prefix="sj-hist-")
        settings = _make_settings(hist)
        policy = GenericPolicy()
        memory = RuntimeMemory()
        subtask = Subtask(
            id="e2e-loop",
            title="e2e",
            mode=TaskMode.GENERIC,
            goal="Прокрутить и завершить",
        )
        cfg = LoopConfig(
            model_cheap=settings.openrouter_model_cheap,
            model_smart=settings.openrouter_model_smart,
            temperature=0.2,
            pricing=settings.pricing,
            history_dir=hist,
            smart_cooldown_steps=5,
            captcha_max_consecutive_waits=20,
        )
        executor = BrowserToolExecutor(user_data_dir=profile, headless=True)
        await executor.start()
        try:
            page = executor.page
            assert page is not None
            await page.goto(_fixture_uri())
            await page.set_viewport_size({"width": 900, "height": 600})
            loop = SubtaskReActLoop(executor=executor, settings=settings, config=cfg)

            route = ModelRoute(tier="cheap", model=settings.openrouter_model_cheap, reason="e2e-mock")
            dec1 = _LlmDecision(
                model_route=route,
                proposed=AgentAction(
                    thought="Сначала прокручиваю страницу вниз, чтобы увидеть цель.",
                    action="scroll",
                    params={"direction": "down", "amount": 1200},
                ),
                model_name=route.model,
                prompt_tokens=1,
                completion_tokens=1,
            )
            dec2 = _LlmDecision(
                model_route=route,
                proposed=AgentAction(
                    thought="Цель достигнута после прокрутки, завершаю.",
                    action="finish",
                    params={},
                ),
                model_name=route.model,
                prompt_tokens=1,
                completion_tokens=1,
            )

            async def _stream(_m: str) -> None:
                return None

            with patch.object(SubtaskReActLoop, "_make_decision", new_callable=AsyncMock) as md:
                md.side_effect = [dec1, dec2]
                state, _delta, _cost = await loop.run_subtask(
                    subtask=subtask,
                    policy=policy,
                    memory=memory,
                    stream_callback=_stream,
                    max_steps=8,
                    step_offset=0,
                )
                self.assertEqual(state, AgentState.FINISHED)
                self.assertEqual(md.call_count, 2)
        finally:
            await executor.stop()


if __name__ == "__main__":
    import unittest

    unittest.main()
