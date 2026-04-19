from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

from openai import AsyncOpenAI

from agent.config.settings import ActorPromptLimits
from agent.llm.base import chat_with_retry
from agent.llm.contracts.actor import (
    ActorDecision,
    GoalCheckDecision,
    GroundingDecision,
    VisualRecoveryDecision,
)
from agent.llm.services.fusion_partial_recovery import try_recover_fusion_agent_action
from agent.llm.services.parser import parse_agent_action_json, strip_markdown_json_fence
from agent.llm.prompts.actor import (
    build_actor_prompt,
    ordered_observation_for_actor_prompt,
    serialize_observation_window_for_actor_prompt,
)
from agent.llm.prompts.templates import (
    fusion_step_thought_contract_block,
    goal_self_check_vision_instructions_block,
    llm_json_output_prohibitions_block,
)
from agent.models.action import ActionResult, AgentAction
from agent.models.observation import InteractiveElement
from agent.models.task import TaskMode

logger = logging.getLogger(__name__)


def _value_error_with_raw_appendix(
    exc: ValueError,
    raw: str | None,
    *,
    label: str,
    max_chars: int = 8000,
) -> ValueError:
    """Сохраняет исходное сообщение парсера и добавляет обрезанный сырой ответ модели (для стрима и логов)."""
    base = exc.args[0] if exc.args else str(exc)
    r = raw or ""
    if len(r) <= max_chars:
        preview = r
        suffix = ""
    else:
        preview = r[:max_chars]
        suffix = "\n... [обрезано для лога]"
    return ValueError(f"{base}\n--- {label} ({len(r)} симв.) ---\n{preview}{suffix}")


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
    def _load_json_object_from_raw_text(
        raw_text: str,
        *,
        empty_error: str,
        not_found_error: str,
    ) -> dict:
        content = strip_markdown_json_fence(raw_text or "")
        if not content:
            raise ValueError(empty_error)
        try:
            data = json.loads(content)
        except Exception as exc:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError(not_found_error) from exc
            try:
                data = json.loads(content[start : end + 1])
            except Exception as exc2:
                raise ValueError(f"{not_found_error}: {exc2}") from exc2
        if not isinstance(data, dict):
            raise ValueError(not_found_error)
        return data

    @staticmethod
    def _parse_goal_check_json(raw_text: str) -> tuple[bool, str]:
        data = ActorLLMClient._load_json_object_from_raw_text(
            raw_text,
            empty_error="Пустой ответ self-check вместо JSON",
            not_found_error="Не удалось найти JSON в self-check ответе",
        )
        goal_reached = bool(data.get("goal_reached", False))
        reason = str(data.get("reason", "") or "")
        return goal_reached, reason

    @staticmethod
    def _parse_visual_recovery_json(raw_text: str) -> tuple[str, dict, str]:
        data = ActorLLMClient._load_json_object_from_raw_text(
            raw_text,
            empty_error="Пустой ответ visual recovery вместо JSON",
            not_found_error="Не удалось найти JSON в visual recovery ответе",
        )
        action = str(data.get("action", "wait") or "wait")
        raw_params = data.get("params")
        params = raw_params if isinstance(raw_params, dict) else {}
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
        response = await chat_with_retry(
            self.client,
            default_model=self.model,
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
        try:
            action = parse_agent_action_json(raw)
        except ValueError as exc:
            logger.warning(
                "Actor text-mode JSON parse failed: %s | model=%s | raw_len=%s",
                exc,
                response.model or model_override or self.model,
                len(raw or ""),
                exc_info=True,
            )
            raise _value_error_with_raw_appendix(
                exc, raw, label="текстовый шаг актора: сырой ответ модели"
            ) from exc
        return ActorDecision(
            action=action,
            model_used=(response.model or model_override or self.model),
            prompt_tokens=int(usage.prompt_tokens) if usage and usage.prompt_tokens else 0,
            completion_tokens=int(usage.completion_tokens) if usage and usage.completion_tokens else 0,
        )

    async def assess_goal_reached(
        self,
        *,
        subtask_goal: str,
        screenshot_path: str,
        compact_observation_json: str,
        last_action: AgentAction,
        last_action_result: ActionResult,
        runtime_context: str,
        model_override: str | None = None,
        max_transport_retries: int = 3,
    ) -> GoalCheckDecision:
        # Self-check после шага: viewport-first (тот же PNG, что после observation), a11y только подпись.
        img_bytes = Path(screenshot_path).read_bytes()
        img_b64 = base64.b64encode(img_bytes).decode("ascii")
        prompt = (
            "Ты проверяешь по скриншоту viewport, достигнута ли цель подзадачи после выполненного шага.\n"
            f"{goal_self_check_vision_instructions_block()}"
            f"{llm_json_output_prohibitions_block()}"
            "Схема: {\"goal_reached\": true/false, \"reason\": \"кратко\"}.\n\n"
            f"Цель подзадачи:\n{subtask_goal}\n\n"
            f"Последнее действие:\naction={last_action.action}, params={last_action.params}\n\n"
            f"Thought перед действием (ожидание изменения UI — проверь по SCREEN):\n{(last_action.thought or '')[:480]}\n\n"
            "Результат действия (лог инструмента — не основание для goal_reached без подтверждения на SCREEN):\n"
            f"success={last_action_result.success}, changed={last_action_result.changed}, "
            f"message={last_action_result.message}, error={last_action_result.error}\n\n"
            f"Контекст рантайма:\n{runtime_context[:500]}\n\n"
            "Вспомогательный список a11y (не достаточен для goal_reached=true без согласования с SCREEN):\n"
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
        response = await chat_with_retry(
            self.client,
            default_model=self.model,
            model_override=model_override,
            max_transport_retries=max_transport_retries,
            max_tokens=min(220, self.request_max_tokens),
            temperature=0.0,
            messages=messages,
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
            f"{llm_json_output_prohibitions_block()}"
            "Схема: {\"action\": \"click_xy|scroll|wait\", \"params\": {...}, \"reason\": \"...\"}.\n"
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
        response = await chat_with_retry(
            self.client,
            default_model=self.model,
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
        data = ActorLLMClient._load_json_object_from_raw_text(
            raw_text,
            empty_error="Пустой ответ grounding вместо JSON",
            not_found_error="Не удалось найти JSON в grounding-ответе",
        )
        thought = str(data.get("thought", "") or data.get("reason", "") or "")
        action = str(data.get("action", "wait") or "wait").strip().lower()
        raw_params = data.get("params")
        # Копия params — иначе pop меняет объект из JSON и смешивает с другими разборами; та же логика, что fusion: индекс главнее ax_id.
        params = dict(raw_params) if isinstance(raw_params, dict) else {}
        allowed = {"click", "type", "scroll", "wait", "click_xy", "navigate"}
        if action not in allowed:
            return AgentAction(
                thought=thought or "Неразрешённое действие в grounding, заменяю на wait.",
                action="wait",
                params={"seconds": 0.4},
            )
        if action in {"click", "type"} and "element_index" in params:
            params.pop("ax_id", None)
        return AgentAction(thought=thought, action=action, params=params)

    @staticmethod
    def _parse_fusion_step_action_json(raw_text: str) -> AgentAction:
        data = ActorLLMClient._load_json_object_from_raw_text(
            raw_text,
            empty_error="Пустой ответ fusion step вместо JSON",
            not_found_error="Не удалось найти JSON в fusion step ответе",
        )
        thought = str(data.get("thought", "") or data.get("reason", "") or "")
        action = str(data.get("action", "wait") or "wait").strip().lower()
        raw_params = data.get("params")
        params = dict(raw_params) if isinstance(raw_params, dict) else {}
        allowed = {"click", "type", "scroll", "wait", "click_xy", "navigate", "finish"}
        if action not in allowed:
            return AgentAction(
                thought=thought or "Неразрешённое действие в fusion step, заменяю на wait.",
                action="wait",
                params={"seconds": 0.4},
            )
        # ax_id из JSON модели часто ошибочен или устарел; индекс — единственный выбор строки, резолвер сам проставит ax_id.
        if action in {"click", "type"} and "element_index" in params:
            params.pop("ax_id", None)
        if action == "click_xy":
            try:
                x = float(params.get("x"))
                y = float(params.get("y"))
            except (TypeError, ValueError):
                return AgentAction(
                    thought=thought or "Нужны числовые x,y для click_xy.",
                    action="wait",
                    params={"seconds": 0.45},
                )
            params = {"x": x, "y": y}
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
        coordinate_priority_hint: str = "",
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
        coord_block = ""
        cph = (coordinate_priority_hint or "").strip()
        if cph:
            coord_block = f"Сигнал рантайма (обязательно учти приоритет):\n{cph}\n\n"
        prompt = (
            "Ты веб-агент. Даны SCREEN (viewport screenshot) и JSON a11y elements.\n"
            "Приоритет: SCREEN > JSON.\n\n"
            f"{llm_json_output_prohibitions_block()}"
            f"{fusion_step_thought_contract_block()}"
            "Правила:\n"
            "1) Если цель уже достигнута по SCREEN -> action=finish.\n"
            "2) Иерархия click: предпочитай click/type с params.element_index, если на SCREEN однозначно одна строка JSON "
            "соответствует нужному контролу (нет соседних дублей с тем же визуалом/ярлыком).\n"
            "3) Используй click_xy, если: несколько похожих строк JSON на однозначный визуальный контрол; мелкие иконки (+/−, корзина); "
            "в блоке «Результат прошлого шага» есть сигнал рантайма про crop-verify или прошлый неуспешный click по индексу/координатам; "
            "нужный контрол на SCREEN не совпадает ни с одной строкой JSON.\n"
            "4) Обычно для click/type используй params.element_index из JSON, подтверждённый SCREEN. "
            "Если выше указан ПРИОРИТЕТ после неудачного индексного шага — в этом шаге верни click_xy с params.x, params.y "
            "(центр цели в viewport), а не снова только element_index.\n"
            "5) Если нужный контрол виден на SCREEN, но в JSON нет подходящего элемента или есть перекрытие -> "
            "action=scroll, click_xy или wait (не выдумывай element_index).\n"
            "6) Если элемент есть в JSON, но не виден/неактивен на SCREEN -> не кликай вслепую; scroll или wait.\n"
            "Если в «Результат прошлого шага» есть «Прошлый thought» — сначала проверь по SCREEN, сбылось ли ожидаемое изменение; "
            "если нет и цель подзадачи не достигнута — выбери исправляющее действие (другой индекс, click_xy, scroll, wait), не finish.\n"
            # Ниже три строки: фиксируют, что индекс — по отсортированному списку в промпте; снижают ложные клики вроде «Увеличить» не той карточки.
            "ВАЖНО: element_index относится ТОЛЬКО к строкам ниже в порядке перечисления (0 — первая строка JSON).\n"
            # Заставляет связать рассуждение с конкретной строкой; ловит рассинхрон thought vs выбранный индекс до исполнения.
            "Перед click/type в thought процитируй role, name и при наличии bbox_doc выбранной строки; если это не совпадает с тем, что видишь на SCREEN, не бери этот индекс — scroll или другой индекс.\n"
            # ax_id модель часто копирует неверно; канонический ax подставляет код по element_index — не засоряем ответ.
            "Для click/type не передавай ax_id в params — только element_index (и text/press_enter для type).\n\n"
            "Схема: {\"thought\": str, \"action\": str, \"params\": object}.\n"
            "Action: navigate | click | type | scroll | wait | click_xy | finish.\n"
            "click/type: params.element_index (int из списка ниже).\n"
            "type: params.text, optional params.press_enter.\n"
            "scroll: params.direction=up|down, params.amount=300|600|1000.\n"
            "wait: params.seconds.\n"
            "click_xy: params.x, params.y — числа в пикселях viewport (центр клика на SCREEN; не путай с bbox_doc документа).\n"
            "navigate: params.url.\n"
            "finish: params={}. \n"
            "Для необратимых/денежных click/type укажи params.is_dangerous=true.\n"
            "Если не уверен -> action=wait.\n\n"
            f"Режим подзадачи: {task_mode.value}\n"
            f"Цель подзадачи:\n{subtask_goal}\n\n"
            f"Правила режима:\n{mode_rules}\n\n"
            "Результат прошлого шага (если есть строка «СИСТЕМА:» — исполненное действие отличалось от предложения модели; учти это, не повторяй заблокированный паттерн):\n"
            f"{last_step_summary}\n\n"
            f"Контекст рантайма:\n{runtime_context}\n\n"
            f"{coord_block}"
            f"{hint_block}"
            "Интерактивные элементы (element_index только из этого списка; поле parent — ближайший контейнер в AX, "
            "если есть — используй для выбора строки на карточках/списках):\n"
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
        response = await chat_with_retry(
            self.client,
            default_model=self.model,
            model_override=model_override,
            max_transport_retries=max_transport_retries,
            # Не clamp к actor_response_max_tokens: при 300–400 легко режется JSON посередине → «невалидный ответ».
            max_tokens=min(1200, max(512, self.request_max_tokens)),
            temperature=0.1,
            messages=messages,
            retry_backoff_base=0.3,
            failure_message="Fusion step request failed",
        )
        raw = response.choices[0].message.content or "{}"
        model_used = response.model or model_override or self.model
        try:
            action = self._parse_fusion_step_action_json(raw)
        except ValueError as exc:
            recovered = try_recover_fusion_agent_action(raw)
            if recovered is not None:
                logger.warning(
                    "Fusion step: частичный JSON, восстановлено action=%s",
                    recovered.action,
                )
                action = recovered
            else:
                logger.warning(
                    "Fusion step JSON parse failed: %s | model=%s | raw_len=%s",
                    exc,
                    model_used,
                    len(raw or ""),
                    exc_info=True,
                )
                raise _value_error_with_raw_appendix(
                    exc, raw, label="fusion: сырой ответ модели"
                ) from exc
        usage = response.usage
        return GroundingDecision(
            action=action,
            model_used=model_used,
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
            f"{llm_json_output_prohibitions_block()}"
            # Grounding тоже идёт по отсортированному окну — те же правила, что fusion, чтобы индекс не «плавал» относительно DOM.
            "element_index — это порядок строк в списке ниже (0,1,2,...). В thought процитируй role+name выбранной строки.\n"
            "Для click/type не передавай ax_id — только element_index.\n"
            "Схема: {\"thought\": str, \"action\": str, \"params\": object}.\n"
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
        response = await chat_with_retry(
            self.client,
            default_model=self.model,
            model_override=model_override,
            max_transport_retries=max_transport_retries,
            max_tokens=min(800, max(384, self.request_max_tokens)),
            temperature=0.1,
            messages=messages,
            retry_backoff_base=0.3,
            failure_message="Grounding request failed",
        )
        raw = response.choices[0].message.content or "{}"
        model_used = response.model or model_override or self.model
        try:
            action = self._parse_grounding_action_json(raw)
        except ValueError as exc:
            logger.warning(
                "Grounding JSON parse failed: %s | model=%s | raw_len=%s",
                exc,
                model_used,
                len(raw or ""),
                exc_info=True,
            )
            raise _value_error_with_raw_appendix(
                exc, raw, label="grounding: сырой ответ модели"
            ) from exc
        usage = response.usage
        return GroundingDecision(
            action=action,
            model_used=model_used,
            prompt_tokens=int(usage.prompt_tokens) if usage and usage.prompt_tokens else 0,
            completion_tokens=int(usage.completion_tokens) if usage and usage.completion_tokens else 0,
        )

    async def verify_crop_element_target_visible_yes_no(
        self,
        *,
        crop_png_path: str,
        element_label: str,
        model_override: str,
        max_transport_retries: int = 3,
    ) -> tuple[bool, int, int]:
        # Кроп вокруг bbox + smart: один ответ YES/NO перед исполнением click/type.
        img_bytes = Path(crop_png_path).read_bytes()
        img_b64 = base64.b64encode(img_bytes).decode("ascii")
        prompt = (
            f"На фрагменте изображения виден ли элемент, соответствующий описанию: «{element_label}» "
            "(кнопка, поле ввода, иконка — достаточно узнаваемого фрагмента)? "
            "Ответь строго одним словом: YES или NO."
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
        response = await chat_with_retry(
            self.client,
            default_model=self.model,
            model_override=model_override,
            max_transport_retries=max_transport_retries,
            max_tokens=8,
            temperature=0.0,
            messages=messages,
            retry_backoff_base=0.3,
            failure_message="Crop element verify request failed",
        )
        raw = (response.choices[0].message.content or "").strip().upper()
        yes = raw.startswith("YES")
        usage = response.usage
        return (
            yes,
            int(usage.prompt_tokens) if usage and usage.prompt_tokens else 0,
            int(usage.completion_tokens) if usage and usage.completion_tokens else 0,
        )
