from __future__ import annotations

import asyncio
from typing import Any

from playwright.async_api import Locator

from agent.models.action import ActionResult
from agent.models.observation import InteractiveElement
from agent.tools.browser_executor.constants import _SMALL_CONTROL_BBOX_MAX_PX
from agent.tools.browser_executor.nav_wait import append_load_idle_suffix, await_after_page_mutation
from agent.tools.browser_executor.page_selection import require_page


def resolve_locator(ex: Any, element: InteractiveElement) -> Locator:
    # Строит Playwright-locator по role/name и индексу среди элементов с тем же (role, name).
    page = require_page(ex)
    return page.get_by_role(element.role, name=element.name or None).nth(element.index_within_role_name)


async def scroll_locator_into_view_and_stabilize(locator: Locator) -> None:
    # Перед click/type: цель в центре viewport — меньше промахов и несовпадение bbox с визуалом.
    try:
        await locator.scroll_into_view_if_needed(timeout=10_000)
    except Exception:
        pass
    await asyncio.sleep(0.15)


async def screenshot_viewport_crop_around_element(
    ex: Any,
    element: InteractiveElement,
    *,
    size: int = 300,
    out_path: str,
) -> bool:
    # Кроп ~size×size вокруг центра bbox в координатах viewport (для smart crop-verify).
    if (
        element.bbox_doc_x is None
        or element.bbox_doc_y is None
        or element.bbox_doc_w is None
        or element.bbox_doc_h is None
    ):
        return False
    page = require_page(ex)
    cx = float(element.bbox_doc_x) + float(element.bbox_doc_w) / 2.0
    cy = float(element.bbox_doc_y) + float(element.bbox_doc_h) / 2.0
    half = float(size) / 2.0
    m = await page.evaluate(
        "() => ({sx: window.scrollX, sy: window.scrollY, iw: window.innerWidth, ih: window.innerHeight})"
    )
    sx = float(m["sx"])
    sy = float(m["sy"])
    iw = float(m["iw"])
    ih = float(m["ih"])
    vx = cx - sx
    vy = cy - sy
    clip_w = min(float(size), iw)
    clip_h = min(float(size), ih)
    clip_x = max(0.0, min(vx - half, iw - clip_w))
    clip_y = max(0.0, min(vy - half, ih - clip_h))
    if clip_w < 48 or clip_h < 48:
        return False
    await page.screenshot(
        path=out_path,
        clip={"x": clip_x, "y": clip_y, "width": clip_w, "height": clip_h},
    )
    return True


async def is_interaction_obstructed(ex: Any, locator: Locator) -> tuple[bool, str]:
    # Проверяет elementFromPoint в центре цели: если верхний элемент не цель и не предок — считаем перекрытие и собираем подсказку для лога.
    box = await locator.bounding_box()
    if not box:
        return False, ""

    page = require_page(ex)
    center_x = box["x"] + box["width"] / 2
    center_y = box["y"] + box["height"] / 2

    info = await page.evaluate(
        """
        ({ x, y, target }) => {
            const top = document.elementFromPoint(x, y);
            if (!top || !target) {
                return { obstructed: false, topTag: "", topRole: "", topText: "" };
            }

            const obstructed = top !== target && !target.contains(top);
            return {
                obstructed,
                topTag: top.tagName || "",
                topRole: top.getAttribute("role") || "",
                topText: (top.innerText || top.textContent || "").trim().slice(0, 60),
            };
        }
        """,
        {"x": center_x, "y": center_y, "target": await locator.element_handle()},
    )
    if not info or not info.get("obstructed"):
        return False, ""

    overlay_hint = info.get("topRole") or info.get("topTag") or "неизвестный элемент"
    overlay_text = info.get("topText") or ""
    return True, f"{overlay_hint} '{overlay_text}'".strip()


async def try_unblock_interaction(ex: Any) -> None:
    # Эвристика снятия перекрытий: Escape, клик в угол, небольшой скролл (модалки, баннеры, sticky-слои).
    page = require_page(ex)
    await page.keyboard.press("Escape")
    await asyncio.sleep(0.2)
    # Клик в левый верхний угол снимает фокус и иногда закрывает попапы.
    await page.mouse.click(8, 8)
    await asyncio.sleep(0.2)
    # Небольшой скролл помогает убрать sticky-оверлеи с центром экрана.
    await page.mouse.wheel(0, 220)
    await asyncio.sleep(0.2)


async def click_with_overlap_recovery(ex: Any, locator: Locator) -> tuple[bool, str]:
    # Клик с одной попыткой «разблокировки» при перехвате pointer events или заранее обнаруженном перекрытии.
    obstructed, details = await is_interaction_obstructed(ex, locator)
    if obstructed:
        await try_unblock_interaction(ex)
    try:
        await locator.click(timeout=5000)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        # # Типичный кейс маркетплейсов: ссылка карточки видна, но input количества перехватывает pointer — обычный click уходит в timeout.
        text = str(exc).lower()
        recoverable = (
            "intercepts pointer events" in text
            or "not receiving pointer events" in text
            or "timeout" in text
            or obstructed
        )
        if recoverable:
            await try_unblock_interaction(ex)
            try:
                await locator.click(timeout=5000)
                return True, f"Перед кликом снято перекрытие: {details or 'перехват/таймаут'}"
            except Exception:
                pass
            # # force=True: последняя линия обороны — Playwright не ждёт «полной» actionability; помогает когда UI уже обновился, но слой перекрытия остаётся.
            try:
                await locator.click(timeout=4500, force=True)
                return True, "Клик выполнен с force=True (fallback после перехвата/таймаута)."
            except Exception:
                raise exc
        raise


async def is_locator_editable(locator: Locator) -> tuple[bool, str]:
    # Проверяет, что резолвнутый DOM-элемент реально поддерживает fill.
    info = await locator.evaluate(
        """
        (el) => {
            const tag = (el.tagName || "").toLowerCase();
            const role = (el.getAttribute("role") || "").toLowerCase();
            const contentEditable = !!el.isContentEditable;
            const readOnly = !!el.readOnly || (el.getAttribute("aria-readonly") === "true");
            const isEditableTag = tag === "input" || tag === "textarea" || tag === "select";
            const isEditableRole = role === "textbox" || role === "searchbox" || role === "combobox";
            return { tag, role, contentEditable, readOnly, editable: (isEditableTag || contentEditable || isEditableRole) && !readOnly };
        }
        """
    )
    editable = bool(info and info.get("editable"))
    if editable:
        return True, ""
    tag = (info or {}).get("tag", "")
    role = (info or {}).get("role", "")
    return False, f"target_not_editable(tag={tag}, role={role})"


def tight_control_doc_bbox(element: InteractiveElement) -> tuple[float, float, float, float] | None:
    # Все четыре поля нужны: без полного bbox не кликаем «в воздух» — вернёмся к get_by_role().nth().
    if (
        element.bbox_doc_x is None
        or element.bbox_doc_y is None
        or element.bbox_doc_w is None
        or element.bbox_doc_h is None
    ):
        return None
    w = float(element.bbox_doc_w)
    h = float(element.bbox_doc_h)
    if w <= 0 or h <= 0 or w > _SMALL_CONTROL_BBOX_MAX_PX or h > _SMALL_CONTROL_BBOX_MAX_PX:
        return None
    return (float(element.bbox_doc_x), float(element.bbox_doc_y), w, h)


async def try_click_tight_control_by_doc_geometry(ex: Any, element: InteractiveElement) -> ActionResult | None:
    # Прокрутка к центру bbox в layout viewport + клик по координатам: центр привязан к этому ax_id, а не к n-му «Увеличить» в DOM.
    spec = tight_control_doc_bbox(element)
    if spec is None:
        return None
    x, y, w, h = spec
    try:
        page = require_page(ex)
        info = await page.evaluate(
            """
            ([rect]) => {
                const [bx, by, bw, bh] = rect;
                const cx = bx + bw / 2;
                const cy = by + bh / 2;
                window.scrollTo({
                    left: Math.max(0, cx - innerWidth / 2),
                    top: Math.max(0, cy - innerHeight / 2),
                    behavior: "auto",
                });
                return {
                    vx: cx - window.scrollX,
                    vy: cy - window.scrollY,
                    iw: innerWidth,
                    ih: innerHeight,
                };
            }
            """,
            [[x, y, w, h]],
        )
        if not isinstance(info, dict):
            return None
        vx = float(info["vx"])
        vy = float(info["vy"])
        iw = float(info["iw"])
        ih = float(info["ih"])
        margin = 6.0
        if not (margin <= vx <= iw - margin and margin <= vy <= ih - margin):
            return None
        await asyncio.sleep(0.12)
        await page.mouse.click(vx, vy)
        label = element.name or element.role
        idle = await await_after_page_mutation(ex, page)
        return ActionResult(
            success=True,
            message=append_load_idle_suffix(
                f"Нажал на элемент '{label}' (клик по bbox мелкого контрола в документе)",
                idle,
            ),
            changed=True,
        )
    except Exception:
        return None
