from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError

from agent.config.settings import ActorPromptLimits
from agent.llm.parsing import parse_agent_action_json
from agent.llm.prompts.actor_prompts import (
    build_actor_prompt,
    ordered_observation_for_actor_prompt,
    serialize_observation_window_for_actor_prompt,
)
from agent.models.action import ActionResult, AgentAction
from agent.models.observation import InteractiveElement
from agent.models.task import TaskMode


@dataclass
class ActorDecision:
    action: AgentAction
    model_used: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class GoalCheckDecision:
    goal_reached: bool
    reason: str
    model_used: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class VisualRecoveryDecision:
    action: str
    params: dict
    reason: str
    model_used: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class GroundingDecision:
    action: AgentAction
    model_used: str
    prompt_tokens: int
    completion_tokens: int


class ActorLLMClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float,
        referer: str | None,
        title: str | None,
        request_max_tokens: int,
        prompt_limits: ActorPromptLimits,
    ):
        self.model = model
        self.temperature = temperature
        self.prompt_limits = prompt_limits
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
        self.request_max_tokens = request_max_tokens

    @staticmethod
    def _is_retryable_transport_error(exc: BaseException) -> bool:
        # Решает, стоит ли повторить запрос к API (сеть, таймаут, 5xx, rate limit).
        return isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)) or (
            isinstance(exc, APIStatusError) and getattr(exc, "status_code", 0) >= 500
        )

    async def _chat_with_retry(
        self,
        *,
        model_override: str | None,
        max_transport_retries: int,
        max_tokens: int,
        temperature: float,
        messages: Any,
        retry_backoff_base: float,
        failure_message: str,
    ):
        last_exc: BaseException | None = None
        for attempt in range(max(1, max_transport_retries)):
            try:
                return await self.client.chat.completions.create(
                    model=model_override or self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=messages,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if not self._is_retryable_transport_error(exc) or attempt >= max_transport_retries - 1:
                    raise
                await asyncio.sleep(retry_backoff_base * (attempt + 1))
        raise last_exc or RuntimeError(failure_message)

    @staticmethod
    def _parse_goal_check_json(raw_text: str) -> tuple[bool, str]:
        content = (raw_text or "").strip()
        if not content:
            raise ValueError("Пустой ответ self-check вместо JSON")
        try:
            data = json.loads(content)
        except Exception:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("Не удалось найти JSON в self-check ответе")
            data = json.loads(content[start : end + 1])
        goal_reached = bool(data.get("goal_reached", False))
        reason = str(data.get("reason", "") or "")
        return goal_reached, reason

    @staticmethod
    def _parse_visual_recovery_json(raw_text: str) -> tuple[str, dict, str]:
        content = (raw_text or "").strip()
        if not content:
            raise ValueError("Пустой ответ visual recovery вместо JSON")
        try:
            data = json.loads(content)
        except Exception:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("Не удалось найти JSON в visual recovery ответе")
            data = json.loads(content[start : end + 1])
        action = str(data.get("action", "wait") or "wait")
        params = data.get("params") if isinstance(data.get("params"), dict) else {}
        reason = str(data.get("reason", "") or "")
        return action, params, reason

    async def decide_action(
        self,
        subtask_goal: str,
        task_mode: TaskMode,
        observation: list[InteractiveElement],
        last_action_result: ActionResult | None,
        runtime_context: str,
        mode_rules: str,
        model_override: str | None = None,
        max_transport_retries: int = 3,
        self_check_hint: str = "",
    ) -> ActorDecision:
        # Собирает промпт и вызывает chat.completions с ограниченными повторами при сетевых сбоях; ValueError парсинга JSON действия не ретраится.
        prompt = build_actor_prompt(
            subtask_goal=subtask_goal,
            task_mode=task_mode,
            observation=observation,
            last_action_result=last_action_result,
            runtime_context=runtime_context,
            mode_rules=mode_rules,
            limits=self.prompt_limits,
            self_check_hint=self_check_hint,
        )
        response = await self._chat_with_retry(
            model_override=model_override,
            max_transport_retries=max_transport_retries,
            max_tokens=self.request_max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
            retry_backoff_base=0.4,
            failure_message="LLM request failed",
        )
        raw = response.choices[0].message.content or "{}"
        usage = response.usage
        return ActorDecision(
            action=parse_agent_action_json(raw),
            model_used=(response.model or model_override or self.model),
            prompt_tokens=int(usage.prompt_tokens) if usage and usage.prompt_tokens else 0,
            completion_tokens=int(usage.completion_tokens) if usage and usage.completion_tokens else 0,
        )

    async def assess_goal_reached(
        self,
        *,
        subtask_goal: str,
        observation: list[InteractiveElement],
        last_action: AgentAction,
        last_action_result: ActionResult,
        runtime_context: str,
        model_override: str | None = None,
        max_transport_retries: int = 3,
    ) -> GoalCheckDecision:
        # Микро self-check после шага: сравнивает цель подзадачи и текущее состояние страницы, возвращает JSON goal_reached/reason.
        window = ordered_observation_for_actor_prompt(observation, self.prompt_limits)
        compact_observation = serialize_observation_window_for_actor_prompt(window, self.prompt_limits)
        # # Подсказка про рассинхрон «UI vs инструмент»: на SPA last_action_result.success может быть false при уже выполненной цели (перехват клика, таймаут).
        prompt = (
            "Ты проверяешь, достигнута ли цель подзадачи после выполненного шага.\n"
            "Ответь СТРОГО JSON формата: {\"goal_reached\": true/false, \"reason\": \"кратко\"}.\n"
            "Не используй markdown.\n\n"
            "Если success=false у результата действия, но по списку элементов и цели подзадачи видно, что нужный "
            "итог уже отражён в UI (счётчики, выбранные состояния, нужный блок формы или списка по смыслу цели), "
            "ставь goal_reached=true и кратко укажи признаки в reason. Не требуй повторного клика ради формального success.\n\n"
            f"Цель подзадачи:\n{subtask_goal}\n\n"
            f"Последнее действие:\naction={last_action.action}, params={last_action.params}\n\n"
            f"Результат действия:\nsuccess={last_action_result.success}, changed={last_action_result.changed}, "
            f"message={last_action_result.message}, error={last_action_result.error}\n\n"
            f"Контекст рантайма:\n{runtime_context}\n\n"
            "Текущие видимые интерактивные элементы:\n"
            f"{json.dumps(compact_observation, ensure_ascii=False)}\n"
        )
        response = await self._chat_with_retry(
            model_override=model_override,
            max_transport_retries=max_transport_retries,
            max_tokens=min(120, self.request_max_tokens),
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
            retry_backoff_base=0.3,
            failure_message="Goal self-check request failed",
        )
        raw = response.choices[0].message.content or "{}"
        goal_reached, reason = self._parse_goal_check_json(raw)
        usage = response.usage
        return GoalCheckDecision(
            goal_reached=goal_reached,
            reason=reason,
            model_used=(response.model or model_override or self.model),
            prompt_tokens=int(usage.prompt_tokens) if usage and usage.prompt_tokens else 0,
            completion_tokens=int(usage.completion_tokens) if usage and usage.completion_tokens else 0,
        )

    async def decide_visual_recovery(
        self,
        *,
        subtask_goal: str,
        screenshot_path: str,
        last_error: str,
        model_override: str | None = None,
        max_transport_retries: int = 3,
    ) -> VisualRecoveryDecision:
        # Vision fallback: по скриншоту выбирает безопасное действие (click_xy/scroll/wait), когда a11y-элементы не дают продвинуться.
        img_bytes = Path(screenshot_path).read_bytes()
        img_b64 = base64.b64encode(img_bytes).decode("ascii")
        prompt = (
            "Ты помогаешь восстановить управление веб-агентом по скриншоту.\n"
            "Верни СТРОГО JSON: {\"action\": \"click_xy|scroll|wait\", \"params\": {...}, \"reason\": \"...\"}.\n"
            "Для click_xy верни params: {\"x\": number, \"y\": number} в координатах viewport.\n"
            "Для scroll верни params: {\"direction\": \"up|down\", \"amount\": 300|600|1000}.\n"
            "Если не уверен — action=wait.\n\n"
            f"Цель подзадачи: {subtask_goal}\n"
            f"Последняя ошибка: {last_error}\n"
        )
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ]}
        ]
        response = await self._chat_with_retry(
            model_override=model_override,
            max_transport_retries=max_transport_retries,
            max_tokens=min(160, self.request_max_tokens),
            temperature=0.1,
            messages=messages,
            retry_backoff_base=0.3,
            failure_message="Visual recovery request failed",
        )
        raw = response.choices[0].message.content or "{}"
        action, params, reason = self._parse_visual_recovery_json(raw)
        usage = response.usage
        return VisualRecoveryDecision(
            action=action,
            params=params,
            reason=reason,
            model_used=(response.model or model_override or self.model),
            prompt_tokens=int(usage.prompt_tokens) if usage and usage.prompt_tokens else 0,
            completion_tokens=int(usage.completion_tokens) if usage and usage.completion_tokens else 0,
        )

    @staticmethod
    def _parse_grounding_action_json(raw_text: str) -> AgentAction:
        content = (raw_text or "").strip()
        if not content:
            raise ValueError("Пустой ответ grounding вместо JSON")
        try:
            data = json.loads(content)
        except Exception:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("Не удалось найти JSON в grounding-ответе")
            data = json.loads(content[start : end + 1])
        thought = str(data.get("thought", "") or data.get("reason", "") or "")
        action = str(data.get("action", "wait") or "wait").strip().lower()
        params = data.get("params") if isinstance(data.get("params"), dict) else {}
        allowed = {"click", "type", "scroll", "wait", "click_xy", "navigate"}
        if action not in allowed:
            return AgentAction(
                thought=thought or "Неразрешённое действие в grounding, заменяю на wait.",
                action="wait",
                params={"seconds": 0.4},
            )
        return AgentAction(thought=thought, action=action, params=params)

    @staticmethod
    def _parse_fusion_step_action_json(raw_text: str) -> AgentAction:
        content = (raw_text or "").strip()
        if not content:
            raise ValueError("Пустой ответ fusion step вместо JSON")
        try:
            data = json.loads(content)
        except Exception:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("Не удалось найти JSON в fusion step ответе")
            data = json.loads(content[start : end + 1])
        thought = str(data.get("thought", "") or data.get("reason", "") or "")
        action = str(data.get("action", "wait") or "wait").strip().lower()
        params = data.get("params") if isinstance(data.get("params"), dict) else {}
        allowed = {"click", "type", "scroll", "wait", "click_xy", "navigate", "finish"}
        if action not in allowed:
            return AgentAction(
                thought=thought or "Неразрешённое действие в fusion step, заменяю на wait.",
                action="wait",
                params={"seconds": 0.4},
            )
        return AgentAction(thought=thought, action=action, params=params)

    async def decide_fusion_step_action(
        self,
        *,
        subtask_goal: str,
        task_mode: TaskMode,
        mode_rules: str,
        runtime_context: str,
        last_step_summary: str,
        self_check_hint: str,
        screenshot_path: str,
        compact_observation_json: str,
        model_override: str | None = None,
        max_transport_retries: int = 3,
    ) -> GroundingDecision:
        # Основной шаг агента: скрин + JSON a11y; модель явно сверяет видимое с деревом и возвращает одно действие.
        img_bytes = Path(screenshot_path).read_bytes()
        img_b64 = base64.b64encode(img_bytes).decode("ascii")
        hint_block = ""
        sh = (self_check_hint or "").strip()
        if sh:
            hint_block = (
                "Подсказка после самопроверки подзадачи (если была):\n"
                f"{sh}\n"
                "Учитывай её при выборе действия; не повторяй безуспешный клик по тому же смыслу.\n\n"
            )
        prompt = (
            "Ты веб-агент. Даны SCREEN (viewport screenshot) и JSON a11y elements.\n"
            "Приоритет: SCREEN > JSON.\n\n"
            "Правила:\n"
            "1) Если цель уже достигнута по SCREEN -> action=finish.\n"
            "2) Для click/type выбирай только params.element_index из JSON, который подтверждается SCREEN "
            "(совпадение по смыслу и положению).\n"
            "3) Если нужный контрол виден на SCREEN, но в JSON нет подходящего элемента или есть перекрытие -> "
            "action=scroll, click_xy или wait (не выдумывай element_index).\n"
            "4) Если элемент есть в JSON, но не виден/неактивен на SCREEN -> не кликай вслепую; scroll или wait.\n\n"
            "Верни СТРОГО JSON: {\"thought\":\"...\",\"action\":\"...\",\"params\":{...}}.\n"
            "Action: navigate | click | type | scroll | wait | click_xy | finish.\n"
            "click/type: params.element_index (int из списка ниже).\n"
            "type: params.text, optional params.press_enter.\n"
            "scroll: params.direction=up|down, params.amount=300|600|1000.\n"
            "wait: params.seconds.\n"
            "click_xy: params.x, params.y (viewport coords).\n"
            "navigate: params.url.\n"
            "finish: params={}. \n"
            "Для необратимых/денежных click/type укажи params.is_dangerous=true.\n"
            "Если не уверен -> action=wait.\n\n"
            f"Режим подзадачи: {task_mode.value}\n"
            f"Цель подзадачи:\n{subtask_goal}\n\n"
            f"Правила режима:\n{mode_rules}\n\n"
            f"Результат прошлого шага:\n{last_step_summary}\n\n"
            f"Контекст рантайма:\n{runtime_context}\n\n"
            f"{hint_block}"
            "Интерактивные элементы (element_index только из этого списка):\n"
            f"{compact_observation_json}\n"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            }
        ]
        response = await self._chat_with_retry(
            model_override=model_override,
            max_transport_retries=max_transport_retries,
            max_tokens=min(500, self.request_max_tokens),
            temperature=0.1,
            messages=messages,
            retry_backoff_base=0.3,
            failure_message="Fusion step request failed",
        )
        raw = response.choices[0].message.content or "{}"
        action = self._parse_fusion_step_action_json(raw)
        usage = response.usage
        return GroundingDecision(
            action=action,
            model_used=(response.model or model_override or self.model),
            prompt_tokens=int(usage.prompt_tokens) if usage and usage.prompt_tokens else 0,
            completion_tokens=int(usage.completion_tokens) if usage and usage.completion_tokens else 0,
        )

    async def decide_grounding_action(
        self,
        *,
        subtask_goal: str,
        task_mode: TaskMode,
        screenshot_path: str,
        compact_observation_json: str,
        model_override: str | None = None,
        max_transport_retries: int = 3,
    ) -> GroundingDecision:
        # Multimodal: скрин + нумерованный a11y-список; ответ — одно действие (приоритет element_index для click/type).
        img_bytes = Path(screenshot_path).read_bytes()
        img_b64 = base64.b64encode(img_bytes).decode("ascii")
        prompt = (
            "Ты веб-агент. По скриншоту viewport и JSON списка интерактивных элементов выбери ОДНО следующее действие.\n"
            "Верни СТРОГО JSON: {\"thought\": \"...\", \"action\": \"...\", \"params\": {...}}.\n"
            "Допустимые action: click, type, scroll, wait, click_xy, navigate.\n"
            "Для click и type укажи params.element_index — целое число из поля element_index списка (не копируй длинные id).\n"
            "Для type передай params.text; при необходимости params.press_enter.\n"
            "Для click_xy: params {\"x\": number, \"y\": number} в координатах viewport, если в списке нет нужного контрола.\n"
            "Для scroll: params {\"direction\": \"up|down\", \"amount\": 300|600|1000}.\n"
            "Для wait: params {\"seconds\": number}.\n"
            "Для navigate: params {\"url\": \"https://...\"}.\n"
            "Если не уверен — action=wait.\n\n"
            f"Режим подзадачи: {task_mode.value}\n"
            f"Цель подзадачи:\n{subtask_goal}\n\n"
            "Интерактивные элементы (element_index — только для этого списка):\n"
            f"{compact_observation_json}\n"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            }
        ]
        response = await self._chat_with_retry(
            model_override=model_override,
            max_transport_retries=max_transport_retries,
            max_tokens=min(400, self.request_max_tokens),
            temperature=0.1,
            messages=messages,
            retry_backoff_base=0.3,
            failure_message="Grounding request failed",
        )
        raw = response.choices[0].message.content or "{}"
        action = self._parse_grounding_action_json(raw)
        usage = response.usage
        return GroundingDecision(
            action=action,
            model_used=(response.model or model_override or self.model),
            prompt_tokens=int(usage.prompt_tokens) if usage and usage.prompt_tokens else 0,
            completion_tokens=int(usage.completion_tokens) if usage and usage.completion_tokens else 0,
        )
