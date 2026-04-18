from __future__ import annotations

import asyncio
from typing import Any

from playwright.async_api import Browser, BrowserContext, Locator, Page, async_playwright

from agent.models.action import ActionResult
from agent.models.observation import InteractiveElement
from agent.models.state import AgentState

TEXT_INPUT_ROLES = {"textbox", "searchbox", "combobox"}


def _url_control_priority(url: str) -> int:
    # Меньше — лучше для выбора вкладки: http(s) предпочтительнее about:blank / новая вкладка / devtools.
    u = (url or "").strip().lower()
    if u.startswith("http://") or u.startswith("https://"):
        return 0
    if u.startswith("file:") or u.startswith("data:"):
        return 1
    if u in {"about:blank", "about:srcdoc"} or "new-tab-page" in u or u in {"edge://newtab/", "chrome://new-tab-page/"}:
        return 2
    if u.startswith("devtools://") or u.startswith("chrome-extension://") or u.startswith("chrome://") or u.startswith("edge://"):
        return 3
    return 2 if u.startswith("about:") else 1


class BrowserToolExecutor:
    def __init__(
        self,
        user_data_dir: str = ".browser-profile",
        *,
        headless: bool = False,
        cdp_url: str | None = None,
    ):
        # Инициализирует профиль или CDP-endpoint и буфер отложенного опасного click/type для AWAITING_USER_CONFIRMATION.
        self._user_data_dir = user_data_dir
        self._headless = headless
        self._cdp_url = (cdp_url or "").strip() or None
        self._playwright = None
        self._cdp_browser: Browser | None = None
        self._context: BrowserContext | None = None
        self.page: Page | None = None
        self._pending_tool: str | None = None
        self._pending_params: dict[str, Any] | None = None

    def _pick_best_page(self, context: BrowserContext) -> Page | None:
        # Не полагаемся на pages[0]: при CDP это часто фоновая about:blank, а нужная вкладка — другая / новее.
        alive: list[tuple[int, Page]] = [(idx, p) for idx, p in enumerate(context.pages) if not p.is_closed()]
        if not alive:
            return None
        alive.sort(key=lambda ip: (_url_control_priority(ip[1].url or ""), -ip[0]))
        return alive[0][1]

    def _maybe_upgrade_from_blank_page(self, current: Page) -> Page:
        # Если управляем пустой вкладкой, а в контексте уже есть страница с URL — переключаемся (target=_blank и т.п.).
        if not self._context or current.is_closed():
            return current
        cur_pri = _url_control_priority(current.url or "")
        if cur_pri <= 1:
            return current
        better = self._pick_best_page(self._context)
        if better is None or better is current:
            return current
        if _url_control_priority(better.url or "") < cur_pri:
            return better
        return current

    async def start(self) -> None:
        # Либо connect_over_cdp к уже запущенному Chromium (AGENT_BROWSER_CDP_URL), либо свой persistent-контекст.
        self._playwright = await async_playwright().start()
        if self._cdp_url:
            self._cdp_browser = await self._playwright.chromium.connect_over_cdp(self._cdp_url)
            contexts = self._cdp_browser.contexts
            if contexts:
                # Берём контекст, в котором есть «осмысленная» вкладка; иначе первый непустой по страницам.
                scored: list[tuple[int, int, BrowserContext]] = []
                for ctx in contexts:
                    pages = [p for p in ctx.pages if not p.is_closed()]
                    if not pages:
                        continue
                    best_pri = min(_url_control_priority(p.url or "") for p in pages)
                    scored.append((best_pri, -len(pages), ctx))
                scored.sort(key=lambda t: (t[0], t[1]))
                self._context = scored[0][2] if scored else contexts[0]
            else:
                self._context = await self._cdp_browser.new_context(viewport={"width": 1440, "height": 900})
            picked = self._pick_best_page(self._context)
            self.page = picked if picked is not None else await self._context.new_page()
            try:
                await self.page.bring_to_front()
            except Exception:
                pass
            return
        self._context = await self._playwright.chromium.launch_persistent_context(
            user_data_dir=self._user_data_dir,
            headless=self._headless,
            viewport={"width": 1440, "height": 900},
        )
        picked = self._pick_best_page(self._context)
        self.page = picked if picked is not None else await self._context.new_page()
        try:
            await self.page.bring_to_front()
        except Exception:
            pass

    async def stop(self) -> None:
        # CDP: только отключение Playwright от браузера (ваш процесс Chromium не завершаем). Иначе — закрытие своего контекста.
        if self._cdp_browser:
            await self._cdp_browser.close()
            self._cdp_browser = None
            self._context = None
            self.page = None
        elif self._context:
            await self._context.close()
            self._context = None
            self.page = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    def _require_page(self) -> Page:
        # Возвращает живую страницу; если текущая вкладка закрыта, подхватывает другую из context.pages.
        if self.page and not self.page.is_closed():
            self.page = self._maybe_upgrade_from_blank_page(self.page)
            return self.page
        if not self._context:
            raise RuntimeError("Браузер не запущен")
        picked = self._pick_best_page(self._context)
        if picked is not None:
            self.page = picked
            return self.page
        raise RuntimeError("Нет активной вкладки браузера")

    def _resolve_locator(self, element: InteractiveElement) -> Locator:
        # Строит Playwright-locator по role/name и индексу среди элементов с тем же (role, name).
        page = self._require_page()
        return page.get_by_role(element.role, name=element.name or None).nth(element.index_within_role_name)

    @staticmethod
    def _is_text_input_element(element: InteractiveElement) -> bool:
        return element.role in TEXT_INPUT_ROLES

    def _pick_best_text_input(self, observation: list[InteractiveElement]) -> InteractiveElement | None:
        # Подбирает наиболее вероятное поле ввода для type, если модель ошиблась ax_id (динамический DOM) <3
        candidates = [x for x in observation if self._is_text_input_element(x) and x.disabled is not True]
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

    def clear_pending_dangerous_action(self) -> None:
        # Сбрасывает отложенное опасное действие (отказ пользователя или завершение сценария).
        self._pending_tool = None
        self._pending_params = None

    def has_pending_dangerous_action(self) -> bool:
        # True, если ожидается подтверждённое выполнение click/type с is_dangerous.
        return self._pending_tool is not None and self._pending_params is not None

    def _guard_dangerous(self, dangerous: bool, action_name: str, element: InteractiveElement) -> ActionResult | None:
        # Если действие помечено опасным и пользователь ещё не подтвердил — возвращает AWAITING_USER_CONFIRMATION вместо DOM-операции.
        if not dangerous:
            return None
        return ActionResult(
            success=False,
            message=f"Требуется подтверждение пользователя для {action_name}: {element.name or element.role}",
            reason_code="awaiting_user_confirmation",
            changed=False,
            is_dangerous=True,
            state=AgentState.AWAITING_USER_CONFIRMATION,
        )

    async def _is_interaction_obstructed(self, locator: Locator) -> tuple[bool, str]:
        # Проверяет elementFromPoint в центре цели: если верхний элемент не цель и не предок — считаем перекрытие и собираем подсказку для лога.
        box = await locator.bounding_box()
        if not box:
            return False, ""

        page = self._require_page()
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

    async def _try_unblock_interaction(self) -> None:
        # Эвристика снятия перекрытий: Escape, клик в угол, небольшой скролл (модалки, баннеры, sticky-слои).
        page = self._require_page()
        await page.keyboard.press("Escape")
        await asyncio.sleep(0.2)
        # Клик в левый верхний угол снимает фокус и иногда закрывает попапы.
        await page.mouse.click(8, 8)
        await asyncio.sleep(0.2)
        # Небольшой скролл помогает убрать sticky-оверлеи с центром экрана.
        await page.mouse.wheel(0, 220)
        await asyncio.sleep(0.2)

    async def _click_with_overlap_recovery(self, locator: Locator) -> tuple[bool, str]:
        # Клик с одной попыткой «разблокировки» при перехвате pointer events или заранее обнаруженном перекрытии.
        obstructed, details = await self._is_interaction_obstructed(locator)
        if obstructed:
            await self._try_unblock_interaction()
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
                await self._try_unblock_interaction()
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

    async def _is_locator_editable(self, locator: Locator) -> tuple[bool, str]:
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

    async def navigate(self, url: str) -> ActionResult:
        # Переход по URL (domcontentloaded); при ошибке возвращает ActionResult с error, не бросает наружу.
        try:
            page = self._require_page()
            await page.goto(url, wait_until="domcontentloaded")
            try:
                await page.bring_to_front()
            except Exception:
                pass
            return ActionResult(success=True, message=f"Открыл страницу: {url}", changed=True)
        except Exception as exc:  # noqa: BLE001
            return ActionResult(success=False, message="Браузер не запущен", reason_code="browser_unavailable", changed=False, error=str(exc))

    async def wait(self, seconds: float = 1.0) -> ActionResult:
        # Пауза исполнения (инструмент wait агента), DOM не трогает.
        await asyncio.sleep(seconds)
        return ActionResult(success=True, message=f"Подождал {seconds:.1f} сек", changed=False)

    async def scroll(self, direction: str = "down", amount: int = 600) -> ActionResult:
        # Прокрутка колесом мыши вверх/вниз на заданное число пикселей.
        try:
            page = self._require_page()
            delta = amount if direction == "down" else -amount
            await page.mouse.wheel(0, delta)
            return ActionResult(success=True, message=f"Прокрутил {direction} на {amount}px", changed=True)
        except Exception as exc:  # noqa: BLE001
            return ActionResult(success=False, message="Браузер не запущен", reason_code="browser_unavailable", changed=False, error=str(exc))

    async def click_xy(self, x: float, y: float) -> ActionResult:
        # Клик по абсолютным координатам viewport (vision fallback, когда a11y-элемент не найден).
        try:
            page = self._require_page()
            await page.mouse.click(float(x), float(y))
            return ActionResult(success=True, message=f"Кликнул по координатам ({float(x):.0f}, {float(y):.0f})", changed=True)
        except Exception as exc:  # noqa: BLE001
            return ActionResult(success=False, message="Ошибка выполнения инструмента", reason_code="tool_execution_error", changed=False, error=str(exc))

    @staticmethod
    def _parse_scroll_amount(raw: Any) -> int:
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

    async def click(self, element: InteractiveElement, dangerous: bool = False, *, user_confirmed: bool = False) -> ActionResult:
        # Клик по элементу из наблюдения; при dangerous без user_confirmed — только запрос подтверждения, иначе клик с recovery перекрытий.
        # # Ретраи и force см. _click_with_overlap_recovery — без хардкода под конкретный сайт, только типовые сбои Playwright.
        blocked = None if user_confirmed else self._guard_dangerous(dangerous, "клика", element)
        if blocked:
            return blocked
        try:
            locator = self._resolve_locator(element)
            _, overlap_msg = await self._click_with_overlap_recovery(locator)
            label = element.name or element.role
            if overlap_msg:
                return ActionResult(success=True, message=f"Нажал на элемент '{label}'. {overlap_msg}", changed=True)
            return ActionResult(success=True, message=f"Нажал на элемент '{label}'", changed=True)
        except Exception as exc:  # noqa: BLE001
            return ActionResult(success=False, message="Ошибка выполнения инструмента", changed=False, error=str(exc))

    async def type(
        self,
        element: InteractiveElement,
        text: str,
        press_enter: bool = False,
        dangerous: bool = False,
        *,
        user_confirmed: bool = False,
    ) -> ActionResult:
        # Ввод текста в поле (fill + опционально Enter); та же схема опасности/подтверждения, что и у click.
        blocked = None if user_confirmed else self._guard_dangerous(dangerous, "ввода в поле", element)
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
            locator = self._resolve_locator(element)
            editable, editable_reason = await self._is_locator_editable(locator)
            if not editable:
                return ActionResult(
                    success=False,
                    message=f"Выбранный элемент не поддерживает ввод: {editable_reason}",
                    reason_code="target_not_editable",
                    changed=False,
                )
            obstructed, details = await self._is_interaction_obstructed(locator)
            if obstructed:
                await self._try_unblock_interaction()
            await locator.fill(text_norm, timeout=5000)
            if press_enter:
                await locator.press("Enter")
            label = element.name or element.role
            if obstructed:
                return ActionResult(
                    success=True,
                    message=f"Ввел текст в '{label}'. Перед вводом снято перекрытие: {details or 'перекрывающий слой'}",
                    changed=True,
                )
            return ActionResult(success=True, message=f"Ввел текст в '{label}'", changed=True)
        except Exception as exc:  # noqa: BLE001
            return ActionResult(success=False, message="Ошибка выполнения инструмента", reason_code="tool_execution_error", changed=False, error=str(exc))

    async def execute_action(
        self,
        tool_name: str,
        params: dict[str, Any],
        observation: list[InteractiveElement],
        *,
        user_confirmed_dangerous: bool = False,
    ) -> ActionResult:
        # Диспетчер инструментов агента: navigate/wait/scroll/click/type/finish; при AWAITING на опасном действии сохраняет параметры в _pending_*.
        try:
            if tool_name == "navigate":
                return await self.navigate(params["url"])
            if tool_name == "wait":
                return await self.wait(float(params.get("seconds", 1.0)))
            if tool_name == "scroll":
                return await self.scroll(params.get("direction", "down"), self._parse_scroll_amount(params.get("amount", 600)))
            if tool_name == "click_xy":
                return await self.click_xy(float(params.get("x", 0.0)), float(params.get("y", 0.0)))
            if tool_name in {"click", "type"}:
                ax_id = params.get("ax_id")
                element = next((x for x in observation if x.ax_id == ax_id), None)
                if not element:
                    return ActionResult(
                        success=False,
                        message=f"Элемент с ax_id='{ax_id}' не найден в текущем наблюдении",
                        reason_code="target_not_found",
                        changed=False,
                    )
                dangerous = bool(params.get("is_dangerous", False))
                if tool_name == "click":
                    res = await self.click(element, dangerous=dangerous, user_confirmed=user_confirmed_dangerous)
                else:
                    if not self._is_text_input_element(element):
                        fallback = self._pick_best_text_input(observation)
                        if fallback is None:
                            return ActionResult(
                                success=False,
                                message=(
                                    f"ax_id='{ax_id}' указывает на role='{element.role}', а для type нужно поле ввода; "
                                    "подходящий textbox/searchbox не найден"
                                ),
                                reason_code="search_input_not_found",
                                changed=False,
                            )
                        element = fallback
                    raw_text = params.get("text")
                    if raw_text is None:
                        raw_text = params.get("value", "")
                    res = await self.type(
                        element,
                        text=str(raw_text),
                        press_enter=bool(params.get("press_enter", False)),
                        dangerous=dangerous,
                        user_confirmed=user_confirmed_dangerous,
                    )
                if res.state == AgentState.AWAITING_USER_CONFIRMATION:
                    self._pending_tool = tool_name
                    self._pending_params = dict(params)
                else:
                    self.clear_pending_dangerous_action()
                return res
            if tool_name == "finish":
                return ActionResult(success=True, message="Задача завершена", changed=False, state=AgentState.FINISHED)
            return ActionResult(success=False, message=f"Неизвестный инструмент: {tool_name}", changed=False)
        except Exception as exc:  # noqa: BLE001
            return ActionResult(success=False, message="Ошибка выполнения инструмента", reason_code="tool_dispatch_error", changed=False, error=str(exc))

    async def execute_pending_dangerous_confirmation(
        self,
        observation: list[InteractiveElement],
    ) -> ActionResult:
        # Повторно выполняет отложенное опасное click/type после подтверждения пользователя (см. has_pending_dangerous_action).
        if not self.has_pending_dangerous_action():
            return ActionResult(success=False, message="Нет отложенного опасного действия", reason_code="no_pending_dangerous_action", changed=False)
        tool = self._pending_tool
        params = self._pending_params or {}
        self.clear_pending_dangerous_action()
        assert tool is not None
        return await self.execute_action(tool, params, observation, user_confirmed_dangerous=True)

