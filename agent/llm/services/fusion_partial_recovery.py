"""Узкое восстановление action/params из частично битого fusion JSON (без ослабления контрактов модели)."""

from __future__ import annotations

import json
import re
from typing import Any

from agent.llm.services.parser import strip_markdown_json_fence
from agent.models.action import AgentAction


def try_recover_fusion_agent_action(raw: str) -> AgentAction | None:
    t = strip_markdown_json_fence(raw or "")
    ma = re.search(r'"action"\s*:\s*"([^"]+)"', t, re.IGNORECASE)
    if not ma:
        return None
    action = ma.group(1).strip().lower()
    allowed = {"click", "type", "scroll", "wait", "click_xy", "navigate", "finish"}
    if action not in allowed:
        return None
    thought = "Восстановлено из частичного JSON ответа модели."
    mt = re.search(r'"thought"\s*:\s*"((?:[^"\\]|\\.)*)"', t, re.DOTALL)
    if mt:
        try:
            thought = json.loads(f'"{mt.group(1)}"')
        except Exception:  # noqa: BLE001
            thought = mt.group(1).replace("\\n", " ")[:240]
    params: dict[str, Any] = {}
    ei = re.search(r'"element_index"\s*:\s*(\d+)', t)
    if ei:
        params["element_index"] = int(ei.group(1))
    if action in {"click", "type"} and "element_index" not in params:
        return None
    if action == "type":
        tx = re.search(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', t)
        if tx:
            try:
                params["text"] = json.loads(f'"{tx.group(1)}"')
            except Exception:  # noqa: BLE001
                params["text"] = tx.group(1)[:2000]
        if "text" not in params:
            return None
    if action == "click_xy":
        xm = re.search(r'"x"\s*:\s*([-+]?\d*\.?\d+)', t)
        ym = re.search(r'"y"\s*:\s*([-+]?\d*\.?\d+)', t)
        if not (xm and ym):
            return None
        try:
            params = {"x": float(xm.group(1)), "y": float(ym.group(1))}
        except ValueError:
            return None
    if action == "scroll":
        dm = re.search(r'"direction"\s*:\s*"([^"]+)"', t, re.I)
        am = re.search(r'"amount"\s*:\s*(\d+)', t)
        if dm and am:
            params = {"direction": dm.group(1).lower(), "amount": int(am.group(1))}
        else:
            return None
    if action == "wait":
        sm = re.search(r'"seconds"\s*:\s*([-+]?\d*\.?\d+)', t)
        if sm:
            params = {"seconds": float(sm.group(1))}
        else:
            params = {"seconds": 0.4}
    if action == "navigate":
        um = re.search(r'"url"\s*:\s*"((?:[^"\\]|\\.)*)"', t)
        if um:
            try:
                params = {"url": json.loads(f'"{um.group(1)}"')}
            except Exception:  # noqa: BLE001
                params = {"url": um.group(1)}
        else:
            return None
    if action == "finish":
        params = {}
    out = AgentAction(thought=thought[:500], action=action, params=params)
    if action in {"click", "type"} and "element_index" in out.params:
        p = dict(out.params)
        p.pop("ax_id", None)
        out = out.model_copy(update={"params": p})
    return out
