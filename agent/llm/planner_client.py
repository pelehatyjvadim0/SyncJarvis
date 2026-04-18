from __future__ import annotations

import json

from openai import AsyncOpenAI

from agent.llm.prompts.planner import build_planner_prompt
from agent.planner.plan_schema import PlannerResponse


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
            max_tokens=600,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = (response.choices[0].message.content or "").strip()
        try:
            data = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            # Частый кейс: модель оборачивает JSON в ```json ... ``` несмотря на инструкцию.
            candidate = raw
            if "```" in candidate:
                candidate = candidate.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(candidate[start : end + 1])
                except Exception:  # noqa: BLE001
                    preview = raw.replace("\n", " ")[:240]
                    raise ValueError(f"Planner JSON parse error: {exc}; raw_preview='{preview}'") from exc
            else:
                preview = raw.replace("\n", " ")[:240]
                raise ValueError(f"Planner JSON parse error: {exc}; raw_preview='{preview}'") from exc
        try:
            return PlannerResponse.model_validate(data)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Planner schema validation error: {exc}") from exc
