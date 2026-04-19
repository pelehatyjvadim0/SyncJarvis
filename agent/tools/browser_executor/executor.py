from __future__ import annotations

from typing import Any

from playwright.async_api import Browser, BrowserContext, Page

from agent.models.action import ActionResult
from agent.models.observation import InteractiveElement
from agent.tools.browser_executor import dom_actions, element_actions, lifecycle, page_selection, tool_dispatch


class BrowserToolExecutor:
    def __init__(
        self,
        user_data_dir: str = ".browser-profile",
        *,
        headless: bool = False,
        cdp_url: str | None = None,
        viewport_width: int = 1440,
        viewport_height: int = 900,
        navigate_wait_until: str = "domcontentloaded",
        navigate_timeout_ms: int = 180_000,
        navigate_networkidle_timeout_ms: int = 12_000,
        navigate_post_settle_seconds: float = 0.5,
    ):
        # Инициализирует профиль или CDP-endpoint и буфер отложенного опасного click/type для AWAITING_USER_CONFIRMATION.
        self._user_data_dir = user_data_dir
        self._headless = headless
        self._cdp_url = (cdp_url or "").strip() or None
        self._viewport_width = max(320, min(3840, int(viewport_width)))
        self._viewport_height = max(240, min(2160, int(viewport_height)))
        wu = (navigate_wait_until or "domcontentloaded").strip().lower()
        if wu not in {"domcontentloaded", "load", "commit", "networkidle"}:
            wu = "domcontentloaded"
        self._navigate_wait_until = wu
        self._navigate_timeout_ms = max(5_000, int(navigate_timeout_ms))
        self._navigate_networkidle_timeout_ms = max(0, int(navigate_networkidle_timeout_ms))
        self._navigate_post_settle_seconds = max(0.0, float(navigate_post_settle_seconds))
        self._playwright = None
        self._cdp_browser: Browser | None = None
        self._context: BrowserContext | None = None
        self.page: Page | None = None
        self._pending_tool: str | None = None
        self._pending_params: dict[str, Any] | None = None

    def require_page(self) -> Page:
        """Возвращает живую страницу; если текущая вкладка закрыта, подхватывает другую из context.pages."""
        return page_selection.require_page(self)

    def _require_page(self) -> Page:
        """Делегирует в :meth:`require_page` (внутренний вызов в подмодулях пакета)."""
        return self.require_page()

    async def start(self) -> None:
        await lifecycle.browser_start(self)

    async def stop(self) -> None:
        await lifecycle.browser_stop(self)

    async def screenshot_viewport_crop_around_element(
        self,
        element: InteractiveElement,
        *,
        size: int = 300,
        out_path: str,
    ) -> bool:
        return await dom_actions.screenshot_viewport_crop_around_element(self, element, size=size, out_path=out_path)

    def clear_pending_dangerous_action(self) -> None:
        # Сбрасывает отложенное опасное действие (отказ пользователя или завершение сценария).
        self._pending_tool = None
        self._pending_params = None

    def has_pending_dangerous_action(self) -> bool:
        # True, если ожидается подтверждённое выполнение click/type с is_dangerous.
        return self._pending_tool is not None and self._pending_params is not None

    async def navigate(self, url: str) -> ActionResult:
        return await element_actions.navigate(self, url)

    async def wait(self, seconds: float = 1.0) -> ActionResult:
        return await element_actions.wait_tool(self, seconds)

    async def scroll(self, direction: str = "down", amount: int = 600) -> ActionResult:
        return await element_actions.scroll(self, direction, amount)

    async def click_xy(self, x: float, y: float) -> ActionResult:
        return await element_actions.click_xy(self, x, y)

    async def click(self, element: InteractiveElement, dangerous: bool = False, *, user_confirmed: bool = False) -> ActionResult:
        return await element_actions.click(self, element, dangerous=dangerous, user_confirmed=user_confirmed)

    async def type(
        self,
        element: InteractiveElement,
        text: str,
        press_enter: bool = False,
        dangerous: bool = False,
        *,
        user_confirmed: bool = False,
    ) -> ActionResult:
        return await element_actions.type_text(
            self, element, text, press_enter=press_enter, dangerous=dangerous, user_confirmed=user_confirmed
        )

    async def execute_action(
        self,
        tool_name: str,
        params: dict[str, Any],
        observation: list[InteractiveElement],
        *,
        user_confirmed_dangerous: bool = False,
    ) -> ActionResult:
        return await tool_dispatch.execute_action(
            self, tool_name, params, observation, user_confirmed_dangerous=user_confirmed_dangerous
        )

    async def execute_pending_dangerous_confirmation(
        self,
        observation: list[InteractiveElement],
    ) -> ActionResult:
        return await tool_dispatch.execute_pending_dangerous_confirmation(self, observation)
