from __future__ import annotations

import re

from agent.models.action import ActionResult, AgentAction
from agent.models.observation import InteractiveElement
from agent.models.task import TaskMode
from agent.policies.generic import GenericPolicy
from agent.runtime.memory import MemoryGuardView, RuntimeMemory


class CommunicationPolicy(GenericPolicy):
    mode = TaskMode.COMMUNICATION

    def mode_rules(self) -> str:
        return (
            "Поддерживай контекст общения. "
            "Не отправляй одно и то же сообщение повторно. "
            "Если цель - попрощаться/завершить диалог, после успешной отправки выбирай finish."
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        normalized = CommunicationPolicy._normalize_text(text)
        return {token for token in re.split(r"[^a-zа-я0-9]+", normalized) if token}

    @staticmethod
    def _semantic_similarity(a: str, b: str) -> float:
        a_tokens = CommunicationPolicy._tokenize(a)
        b_tokens = CommunicationPolicy._tokenize(b)
        if not a_tokens or not b_tokens:
            return 0.0
        inter = len(a_tokens & b_tokens)
        union = len(a_tokens | b_tokens)
        if union == 0:
            return 0.0
        return inter / union

    @staticmethod
    def _is_closing_goal(goal: str) -> bool:
        value = goal.lower()
        keys = ["попрощ", "заверш", "законч", "пока", "до свид", "finish dialog", "end dialog"]
        return any(k in value for k in keys)

    def anti_loop_guard(
        self,
        subtask_goal: str,
        observation: list[InteractiveElement],
        proposed_action: AgentAction,
        memory_view: MemoryGuardView,
    ) -> AgentAction:
        if proposed_action.action == "type":
            new_text = self._normalize_text(str(proposed_action.params.get("text", "")))
            last_text = self._normalize_text(memory_view.last_sent_text)
            similarity = self._semantic_similarity(new_text, last_text)
            if new_text and (new_text == last_text or similarity >= 0.8):
                return AgentAction(thought="Блокирую дубль сообщения, завершаю диалог.", action="finish", params={})
        return super().anti_loop_guard(subtask_goal, observation, proposed_action, memory_view)

    def is_done(self, subtask_goal: str, last_action: AgentAction | None, last_result: ActionResult | None) -> tuple[bool, str]:
        if not last_action or not last_result:
            return False, ""
        if last_action.action == "finish" and last_result.success:
            return True, "Подзадача общения завершена."
        if self._is_closing_goal(subtask_goal) and last_action.action == "click" and last_result.success:
            return True, "Прощальное сообщение отправлено."
        return False, ""

    def compute_progress(self, observation: list[InteractiveElement], action: AgentAction, result: ActionResult) -> int:
        score = 0
        if action.action == "type" and result.success:
            score += 2
        if action.action == "click" and result.success:
            score += 2
        if action.action == "wait":
            score -= 1
        if not result.success:
            score -= 1
        return score

    def refine_after_global_anti_loop(self, guarded: AgentAction, memory: RuntimeMemory) -> AgentAction:
        if (
            guarded.action == "type"
            and memory.sent_message_count >= 1
            and memory.last_type_ax_id
            and guarded.params.get("ax_id") == memory.last_type_ax_id
        ):
            memory.duplicate_blocked = True
            return AgentAction(
                thought="Сообщение уже отправлено, блокирую повтор и завершаю подзадачу.",
                action="finish",
                params={},
            )
        return guarded

    def check_force_finish_after_execution(
        self, guarded: AgentAction, result: ActionResult, memory: RuntimeMemory
    ) -> tuple[bool, str]:
        if guarded.action == "click" and result.success and memory.last_type_ax_id:
            memory.sent_message_count += 1
            click_ax = guarded.params.get("ax_id")
            memory.last_send_click_ax_id = str(click_ax) if click_ax else ""
            return True, "Сообщение отправлено, диалоговая подзадача завершена."
        return False, ""

