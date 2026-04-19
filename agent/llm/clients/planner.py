from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

from agent.llm.prompts.planner import build_planner_prompt
from agent.llm.services.parser import strip_markdown_json_fence
from agent.planner.plan_schema import PlannerResponse


def _json_parse_candidates(normalized: str) -> list[str]:
    """Полный текст и вырезка по первому «{» … последнему «}» (на случай префикса до JSON)."""
    t = normalized.strip()
    if not t:
        return []
    out: list[str] = [t]
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = t[start : end + 1]
        if snippet != t:
            out.append(snippet)
    return out


class PlannerLLMClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float,
        referer: str | None,
        title: str | None,
    ):
        self.model = model
        self.temperature = temperature
        headers: dict[str, str] = {}
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers=headers or None,
        )

    async def plan(self, user_goal: str, max_subtasks: int) -> PlannerResponse:
        prompt = build_planner_prompt(user_goal, max_subtasks)
        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=2_048,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = (response.choices[0].message.content or "").strip()
        normalized = strip_markdown_json_fence(raw)
        last_exc: BaseException | None = None
        for cand in _json_parse_candidates(normalized):
            try:
                data: Any = json.loads(cand)
                return PlannerResponse.model_validate(data)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue
        preview = raw.replace("\n", " ")[:280]
        msg = f"Planner JSON parse error: {last_exc}; raw_preview={preview!r}"
        if last_exc is not None:
            raise ValueError(msg) from last_exc
        raise ValueError(msg)
