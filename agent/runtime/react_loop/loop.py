from __future__ import annotations

from typing import Any, Awaitable, Callable

from agent.config.settings import AppSettings
from agent.llm.clients.actor import ActorLLMClient
from agent.llm.services.router import ModelRouter
from agent.logging.history_logger import HistoryLogger
from agent.models.action import ActionResult, AgentAction
from agent.models.observation import InteractiveElement
from agent.models.plan import Subtask
from agent.models.state import AgentState
from agent.policies.base import BaseTaskPolicy
from agent.runtime.memory import RuntimeMemory
from agent.runtime.react_loop.components.fingerprinting import read_live_url_from_executor_or_none
from agent.runtime.react_loop.config import LoopConfig, RunCostStats
from agent.runtime.react_loop.engine.pipeline import run_subtask_pipeline
from agent.runtime.react_loop.engine.types import _LlmDecision
from agent.tools.browser_executor import BrowserToolExecutor

__all__ = ["SubtaskReActLoop", "_LlmDecision"]


class SubtaskReActLoop:
    # Один ReAct-цикл по подзадаче: наблюдение, капча, LLM, гарды, исполнение, телеметрия.

    def __init__(
        self,
        executor: BrowserToolExecutor,
        settings: AppSettings,
        config: LoopConfig,
        history_logger: HistoryLogger | None = None,
    ):
        self.executor = executor
        self._settings = settings
        self.config = config
        self.actor = ActorLLMClient(
            api_key=settings.openrouter_api_key,
            model=config.model_smart,
            temperature=config.temperature,
            referer=settings.openrouter_http_referer,
            title=settings.openrouter_x_title,
            request_max_tokens=settings.actor_response_max_tokens,
            prompt_limits=settings.prompt_limits,
        )
        self.model_router = ModelRouter(
            cheap_model=config.model_cheap,
            smart_model=config.model_smart,
            smart_cooldown_steps=config.smart_cooldown_steps,
        )
        self.history_logger = history_logger or HistoryLogger(history_dir=config.history_dir)
        # Заполняется перед возвратом AWAITING_USER_CONFIRMATION - для resume_after_dangerous_confirmation в оркестраторе.
        self._last_guarded_at_pause: AgentAction | None = None
        # Последнее значение счётчика шагов внутри run_subtask - нужно для resume (абсолютный счётчик подзадачи).
        self.last_subtask_executed_step_total: int = 0
        # Короткая история действий текущего run для финального отчёта.
        self.context_history: list[dict[str, Any]] = []
        # Последнее наблюдение страницы (используется в финальном отчёте).
        self.last_observation: list[InteractiveElement] = []
        self._max_context_history: int = 200
        # Конец прошлой итерации цикла подзадачи: URL страницы (для grounding: смена без полного reload).
        self._loop_end_url: str | None = None
        # Последний fingerprint, для которого уже выполняли grounding (дебаунс).
        self._last_grounding_fingerprint: str | None = None

    def reset_session_context(self) -> None:
        # Сбрасывает историю/наблюдение перед новым run оркестратора.
        self.context_history = []
        self.last_observation = []
        self._loop_end_url = None
        self._last_grounding_fingerprint = None

    def _refresh_loop_url_snapshot(self) -> None:
        url = read_live_url_from_executor_or_none(self.executor)
        if url is not None:
            self._loop_end_url = url

    def _append_context_history(self, item: dict[str, Any]) -> None:
        # Держит ограниченный хвост истории, чтобы не раздувать память в долгой сессии.
        self.context_history.append(item)
        if len(self.context_history) > self._max_context_history:
            self.context_history = self.context_history[-self._max_context_history :]

    async def run_subtask(
        self,
        subtask: Subtask,
        policy: BaseTaskPolicy,
        memory: RuntimeMemory,
        stream_callback: Callable[[str], Awaitable[None]],
        max_steps: int,
        step_offset: int = 0,
        *,
        resume_last_action: AgentAction | None = None,
        resume_last_result: ActionResult | None = None,
        resume_executed_steps: int = 0,
    ) -> tuple[AgentState, int, RunCostStats]:
        return await run_subtask_pipeline(
            self,
            subtask,
            policy,
            memory,
            stream_callback,
            max_steps,
            step_offset,
            resume_last_action=resume_last_action,
            resume_last_result=resume_last_result,
            resume_executed_steps=resume_executed_steps,
        )
