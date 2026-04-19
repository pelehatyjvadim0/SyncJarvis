from __future__ import annotations

import asyncio
from typing import Any

from agent.models.action import ActionResult
from agent.models.observation import InteractiveElement
from agent.tools.browser_executor.constants import TEXT_INPUT_ROLES
from agent.tools.browser_executor.dangerous_guard import guard_dangerous
from agent.tools.browser_executor.dom_actions import (
    click_with_overlap_recovery,
    is_interaction_obstructed,
    is_locator_editable,
    resolve_locator,
    scroll_locator_into_view_and_stabilize,
    try_click_tight_control_by_doc_geometry,
    try_unblock_interaction,
)
from agent.tools.browser_executor.nav_wait import append_load_idle_suffix, await_after_page_mutation, networkidle_then_settle, post_settle_only
from agent.tools.browser_executor.page_selection import require_page


async def navigate(ex: Any, url: str) -> ActionResult:
    # goto(wait_until) + networkidle + settle — см. AGENT_BROWSER_NAVIGATE_WAIT_UNTIL (по умолчанию domcontentloaded).
    try:
        page = require_page(ex)
        await page.goto(
            url,
            wait_until=ex._navigate_wait_until,  # type: ignore[arg-type]
            timeout=float(ex._navigate_timeout_ms),
        )
        notes = await networkidle_then_settle(ex, page)
        try:
            await page.bring_to_front()
        except Exception:
            pass
        extra = f" ({', '.join(notes)})" if notes else ""
        return ActionResult(success=True, message=f"Открыл страницу: {url}{extra}", changed=True)
    except Exception as exc:  # noqa: BLE001
        return ActionResult(success=False, message="Браузер не запущен", reason_code="browser_unavailable", changed=False, error=str(exc))


async def wait_tool(_ex: Any, seconds: float = 1.0) -> ActionResult:
    # Пауза исполнения (инструмент wait агента), DOM не трогает.
    await asyncio.sleep(seconds)
    return ActionResult(success=True, message=f"Подождал {seconds:.1f} сек", changed=False)


async def scroll(ex: Any, direction: str = "down", amount: int = 600) -> ActionResult:
    # Прокрутка колесом мыши вверх/вниз на заданное число пикселей.
    try:
        page = require_page(ex)
        delta = amount if direction == "down" else -amount
        await page.mouse.wheel(0, delta)
        return ActionResult(success=True, message=f"Прокрутил {direction} на {amount}px", changed=True)
    except Exception as exc:  # noqa: BLE001
        return ActionResult(success=False, message="Браузер не запущен", reason_code="browser_unavailable", changed=False, error=str(exc))


async def click_xy(ex: Any, x: float, y: float) -> ActionResult:
    # Клик по абсолютным координатам viewport (vision fallback, когда a11y-элемент не найден).
    try:
        page = require_page(ex)
        await page.mouse.click(float(x), float(y))
        idle = await await_after_page_mutation(ex, page)
        msg = append_load_idle_suffix(
            f"Кликнул по координатам ({float(x):.0f}, {float(y):.0f})",
            idle,
        )
        return ActionResult(success=True, message=msg, changed=True)
    except Exception as exc:  # noqa: BLE001
        return ActionResult(success=False, message="Ошибка выполнения инструмента", reason_code="tool_execution_error", changed=False, error=str(exc))


async def click(ex: Any, element: InteractiveElement, dangerous: bool = False, *, user_confirmed: bool = False) -> ActionResult:
    # Клик по элементу из наблюдения; при dangerous без user_confirmed — только запрос подтверждения, иначе клик с recovery перекрытий.
    blocked = None if user_confirmed else guard_dangerous(dangerous, "клика", element)
    if blocked:
        return blocked
    try:
        locator = resolve_locator(ex, element)
        await scroll_locator_into_view_and_stabilize(locator)
        geo = await try_click_tight_control_by_doc_geometry(ex, element)
        if geo is not None:
            return geo
        _, overlap_msg = await click_with_overlap_recovery(ex, locator)
        page = require_page(ex)
        idle = await await_after_page_mutation(ex, page)
        label = element.name or element.role
        if overlap_msg:
            return ActionResult(
                success=True,
                message=append_load_idle_suffix(f"Нажал на элемент '{label}'. {overlap_msg}", idle),
                changed=True,
            )
        return ActionResult(
            success=True,
            message=append_load_idle_suffix(f"Нажал на элемент '{label}'", idle),
            changed=True,
        )
    except Exception as exc:  # noqa: BLE001
        return ActionResult(success=False, message="Ошибка выполнения инструмента", changed=False, error=str(exc))


async def type_text(
    ex: Any,
    element: InteractiveElement,
    text: str,
    press_enter: bool = False,
    dangerous: bool = False,
    *,
    user_confirmed: bool = False,
) -> ActionResult:
    # Ввод текста в поле (fill + опционально Enter); та же схема опасности/подтверждения, что и у click.
    blocked = None if user_confirmed else guard_dangerous(dangerous, "ввода в поле", element)
    if blocked:
        return blocked
    text_norm = (text or "").strip()
    if not text_norm:
        return ActionResult(
            success=False,
            message="Пустой текст для type: ожидается params.text (или legacy params.value)",
            reason_code="empty_type_text",
            changed=False,
        )
    try:
        locator = resolve_locator(ex, element)
        await scroll_locator_into_view_and_stabilize(locator)
        editable, editable_reason = await is_locator_editable(locator)
        if not editable:
            return ActionResult(
                success=False,
                message=f"Выбранный элемент не поддерживает ввод: {editable_reason}",
                reason_code="target_not_editable",
                changed=False,
            )
        obstructed, details = await is_interaction_obstructed(ex, locator)
        if obstructed:
            await try_unblock_interaction(ex)
        await locator.fill(text_norm, timeout=5000)
        if press_enter:
            await locator.press("Enter")
        page = require_page(ex)
        # Enter/отправка формы может дать полноценный переход — ждём load+idle; обычный fill (поиск) не блокируем на networkidle.
        idle = (
            await await_after_page_mutation(ex, page)
            if press_enter
            else await post_settle_only(ex)
        )
        label = element.name or element.role
        if obstructed:
            return ActionResult(
                success=True,
                message=append_load_idle_suffix(
                    f"Ввел текст в '{label}'. Перед вводом снято перекрытие: {details or 'перекрывающий слой'}",
                    idle,
                ),
                changed=True,
            )
        return ActionResult(
            success=True,
            message=append_load_idle_suffix(f"Ввел текст в '{label}'", idle),
            changed=True,
        )
    except Exception as exc:  # noqa: BLE001
        return ActionResult(success=False, message="Ошибка выполнения инструмента", reason_code="tool_execution_error", changed=False, error=str(exc))


def is_text_input_element(element: InteractiveElement) -> bool:
    return element.role in TEXT_INPUT_ROLES


def pick_best_text_input(observation: list[InteractiveElement]) -> InteractiveElement | None:
    # Подбирает наиболее вероятное поле ввода для type, если модель ошиблась ax_id (динамический DOM) <3
    candidates = [x for x in observation if is_text_input_element(x) and x.disabled is not True]
    if not candidates:
        return None

    def _score(item: InteractiveElement) -> tuple[int, int, int, int, str]:
        name = (item.name or "").lower()
        return (
            0 if item.focused else 1,
            0 if item.role in {"searchbox", "textbox"} else 1,
            0 if ("поиск" in name or "search" in name) else 1,
            0 if name else 1,
            item.ax_id,
        )

    candidates.sort(key=_score)
    return candidates[0]


def parse_scroll_amount(raw: Any) -> int:
    # Нормализует amount для scroll: поддерживает int/str и слова medium/large/small.
    if isinstance(raw, (int, float)):
        return int(raw)
    text = str(raw or "").strip().lower()
    if not text:
        return 600
    aliases = {
        "small": 300,
        "medium": 600,
        "large": 1000,
    }
    if text in aliases:
        return aliases[text]
    try:
        return int(float(text))
    except ValueError:
        return 600
