from __future__ import annotations

from typing import Any

from playwright.async_api import CDPSession, Page

from agent.models.observation import InteractiveElement


def _trim_anchor_label(name: Any, max_len: int = 72) -> str:
    s = (str(name) if name is not None else "").strip()
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


def _container_anchor_from_ancestors(ancestor_stack: list[tuple[str, Any]]) -> str | None:
    # Ближайший к цели предок-контейнер: section/article/listitem или явный текстовый заголовок (heading / именованный region).
    for role, name in reversed(ancestor_stack):
        r = str(role or "").lower()
        nm = _trim_anchor_label(name)
        if r in ("section", "article", "listitem"):
            label = "List item" if r == "listitem" else r.title()
            return f'{label} "{nm}"' if nm else label
        if r == "heading" and nm:
            return f'Heading "{nm}"'
        if r in ("banner", "header", "main", "navigation", "region", "complementary", "form") and nm:
            return f'{r.title()} "{nm}"'
    return None


INTERACTIVE_ROLES = {
    "button",
    "link",
    "textbox",
    "searchbox",
    "combobox",
    "checkbox",
    "radio",
    "switch",
    "menuitem",
    "menuitemcheckbox",
    "menuitemradio",
    "option",
    "tab",
    "slider",
    "spinbutton",
}


async def _layout_viewport_doc_rect(page: Page, padding: float) -> tuple[float, float, float, float]:
    # Прямоугольник видимой области в координатах документа (scroll + innerWidth/Height), расширенный на padding со всех сторон.
    m = await page.evaluate(
        "() => ({ sx: window.scrollX, sy: window.scrollY, "
        "vw: window.innerWidth, vh: window.innerHeight })"
    )
    sl = float(m["sx"])
    st = float(m["sy"])
    vw = float(m["vw"])
    vh = float(m["vh"])
    p = float(padding)
    return sl - p, st - p, sl + vw + p, st + vh + p


def _aabb_intersects(
    ax: float,
    ay: float,
    aw: float,
    ah: float,
    vx1: float,
    vy1: float,
    vx2: float,
    vy2: float,
) -> bool:
    # Пересечение двух оси-выровненных прямоугольников в одной системе координат (документ).
    return not (ax + aw < vx1 or ax > vx2 or ay + ah < vy1 or ay > vy2)


async def _fetch_document_bboxes(page: Page, backend_ids: list[int]) -> dict[int, tuple[float, float, float, float]]:
    # CDP DOM.getBoxModel: content quad в CSS px документа; сворачиваем в AABB (x, y, w, h).
    if not backend_ids:
        return {}
    session: CDPSession = await page.context.new_cdp_session(page)
    out: dict[int, tuple[float, float, float, float]] = {}
    try:
        for bid in backend_ids:
            try:
                model = await session.send("DOM.getBoxModel", {"backendNodeId": bid})
            except Exception:
                continue
            content = model.get("model", {}).get("content") or []
            if len(content) < 8:
                continue
            xs = [float(content[i]) for i in range(0, 8, 2)]
            ys = [float(content[i]) for i in range(1, 8, 2)]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            out[bid] = (x0, y0, x1 - x0, y1 - y0)
    finally:
        await session.detach()
    return out


async def collect_interactive_elements(
    page: Page,
    *,
    viewport_only: bool = True,
    viewport_padding: float = 12.0,
) -> list[InteractiveElement]:
    # AX-дерево + bbox в координатах документа (layout); при viewport_only остаются только элементы, пересекающие видимую область.
    session: CDPSession = await page.context.new_cdp_session(page)
    await session.send("Accessibility.enable")
    response = await session.send("Accessibility.getFullAXTree")
    await session.detach()

    ax_nodes = response.get("nodes", [])
    if not ax_nodes:
        return []

    raw_nodes: list[dict[str, Any]] = []
    nodes_by_id: dict[str, dict[str, Any]] = {}
    child_ids: set[str] = set()

    for node in ax_nodes:
        node_id = str(node.get("nodeId", ""))
        if not node_id:
            continue
        nodes_by_id[node_id] = node
        for child_id in node.get("childIds", []):
            child_ids.add(str(child_id))

    root_ids = [node_id for node_id in nodes_by_id if node_id not in child_ids]
    if not root_ids:
        root_ids = list(nodes_by_id.keys())

    def read_ax_value(node: dict[str, Any], field_name: str) -> Any:
        value = node.get(field_name)
        if isinstance(value, dict):
            return value.get("value")
        return value

    def read_property(node: dict[str, Any], prop_name: str) -> Any:
        for prop in node.get("properties", []):
            if prop.get("name") == prop_name:
                value = prop.get("value")
                if isinstance(value, dict):
                    return value.get("value")
                return value
        return None

    def walk(node_id: str, path: str = "0", ancestor_stack: list[tuple[str, Any]] | None = None) -> None:
        ancestor_stack = ancestor_stack or []
        node = nodes_by_id.get(node_id)
        if not node:
            return

        role = str(read_ax_value(node, "role") or "").lower()
        name = read_ax_value(node, "name")
        value = read_ax_value(node, "value")
        if role in INTERACTIVE_ROLES:
            anchor = _container_anchor_from_ancestors(ancestor_stack)
            raw_nodes.append(
                {
                    "ax_id": path,
                    "role": role,
                    "name": name,
                    "parent_anchor": anchor,
                    "dom_id": str(node.get("backendDOMNodeId")) if node.get("backendDOMNodeId") else None,
                    "disabled": read_property(node, "disabled"),
                    "focused": read_property(node, "focused"),
                    "checked": read_property(node, "checked"),
                    "pressed": read_property(node, "pressed"),
                    "expanded": read_property(node, "expanded"),
                    "selected": read_property(node, "selected"),
                    "value": value,
                }
            )

        next_stack = ancestor_stack + [(role, name)]
        for idx, child_id in enumerate(node.get("childIds", [])):
            walk(str(child_id), f"{path}.{idx}", next_stack)

    for root_idx, root_id in enumerate(root_ids):
        walk(root_id, str(root_idx), [])

    backend_ids: list[int] = []
    for node in raw_nodes:
        did = node.get("dom_id")
        if did and str(did).isdigit():
            backend_ids.append(int(str(did)))
    boxes = await _fetch_document_bboxes(page, backend_ids)

    for node in raw_nodes:
        did = node.get("dom_id")
        if did and str(did).isdigit():
            bid = int(str(did))
            b = boxes.get(bid)
            if b:
                node["bbox_doc_x"] = b[0]
                node["bbox_doc_y"] = b[1]
                node["bbox_doc_w"] = b[2]
                node["bbox_doc_h"] = b[3]

    counters: dict[tuple[str, str], int] = {}
    elements: list[InteractiveElement] = []
    for node in raw_nodes:
        name = (node.get("name") or "").strip()
        key = (node["role"], name)
        counters[key] = counters.get(key, 0) + 1
        node["index_within_role_name"] = counters[key] - 1
        elements.append(InteractiveElement.model_validate(node))

    if not viewport_only:
        return elements

    vrect = await _layout_viewport_doc_rect(page, viewport_padding)
    vx1, vy1, vx2, vy2 = vrect
    visible: list[InteractiveElement] = []
    for el in elements:
        if (
            el.bbox_doc_x is None
            or el.bbox_doc_y is None
            or el.bbox_doc_w is None
            or el.bbox_doc_h is None
        ):
            continue
        if _aabb_intersects(el.bbox_doc_x, el.bbox_doc_y, el.bbox_doc_w, el.bbox_doc_h, vx1, vy1, vx2, vy2):
            visible.append(el)
    return visible


async def collect_visible_interactive_elements(
    page: Page,
    *,
    viewport_padding: float = 12.0,
) -> list[InteractiveElement]:
    # Алиас: только пересечение с viewport (основной путь для LLM).
    return await collect_interactive_elements(page, viewport_only=True, viewport_padding=viewport_padding)


async def collect_all_interactive_elements(page: Page) -> list[InteractiveElement]:
    # Полный список интерактивных узлов с bbox без фильтра по viewport (отладка и тесты).
    return await collect_interactive_elements(page, viewport_only=False)
