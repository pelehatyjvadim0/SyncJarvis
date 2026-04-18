from __future__ import annotations

from pathlib import Path
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
from agent.runtime.react_loop.config import LoopConfig, RunCostStats
from agent.runtime.react_loop.engine.action_executor import (
    apply_guards,
    refine_self_check_repeat_click,
    update_performance_metrics,
)
from agent.runtime.react_loop.engine.decision_maker import (
    handle_llm_error,
    make_decision,
)
from agent.runtime.react_loop.engine.pipeline import run_subtask_pipeline
from agent.tools.browser_executor import BrowserToolExecutor
from agent.runtime.react_loop.engine.types import (
    _CaptchaIterationOutcome,
    _LlmDecision,
)
from agent.runtime.react_loop.components.captcha import handle_captcha_iteration
from agent.runtime.react_loop.components.fingerprinting import read_live_url_from_executor_or_none
from agent.runtime.react_loop.components.grounding import attempt_grounding_step
from agent.runtime.react_loop.components.vision_recovery import attempt_visual_recovery
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

    def _require_page(self):
        # Делегируем в executor: при self.page=None он перепривязывается к живой вкладке из context.pages (нужно для recovery).
        return self.executor._require_page()

    async def _handle_captcha(
        self,
        *,
        page,
        observation: list[InteractiveElement],
        subtask: Subtask,
        memory: RuntimeMemory,
        global_step: int,
        stream_callback: Callable[[str], Awaitable[None]],
    ) -> _CaptchaIterationOutcome | None:
        return await handle_captcha_iteration(
            page=page,
            observation=observation,
            subtask=subtask,
            memory=memory,
            global_step=global_step,
            stream_callback=stream_callback,
            executor=self.executor,
            history_logger=self.history_logger,
            append_context_history=self._append_context_history,
        )

    async def _make_decision(
        self,
        *,
        subtask: Subtask,
        policy: BaseTaskPolicy,
        observation: list[InteractiveElement],
        obs_window: list[InteractiveElement],
        last_action: AgentAction | None,
        last_action_result: ActionResult | None,
        memory: RuntimeMemory,
        global_step: int,
        page,
    ) -> _LlmDecision:
        return await make_decision(
            subtask=subtask,
            policy=policy,
            observation=observation,
            obs_window=obs_window,
            last_action=last_action,
            last_action_result=last_action_result,
            memory=memory,
            global_step=global_step,
            page=page,
            actor=self.actor,
            model_router=self.model_router,
            settings=self._settings,
            config=self.config,
        )

    async def _handle_llm_error(
        self,
        exc: ValueError,
        *,
        subtask: Subtask,
        policy: BaseTaskPolicy,
        observation: list[InteractiveElement],
        memory: RuntimeMemory,
        global_step: int,
        page,
        stream_callback: Callable[[str], Awaitable[None]],
    ) -> tuple[AgentAction, ActionResult]:
        return await handle_llm_error(
            exc,
            subtask=subtask,
            policy=policy,
            observation=observation,
            memory=memory,
            global_step=global_step,
            page=page,
            stream_callback=stream_callback,
            history_logger=self.history_logger,
        )

    def _apply_guards(
        self,
        policy: BaseTaskPolicy,
        subtask_goal: str,
        observation: list[InteractiveElement],
        proposed: AgentAction,
        memory: RuntimeMemory,
    ) -> AgentAction:
        return apply_guards(policy, subtask_goal, observation, proposed, memory)

    @staticmethod
    def _refine_self_check_repeat_click(
        proposed: AgentAction,
        last_action: AgentAction | None,
        memory: RuntimeMemory,
    ) -> AgentAction:
        return refine_self_check_repeat_click(proposed, last_action, memory)

    @staticmethod
    def _should_visual_stagnation_recovery(
        subtask: Subtask,
        memory: RuntimeMemory,
        last_action: AgentAction | None,
        last_action_result: ActionResult | None,
    ) -> bool:
        # Vision при застревании на click в TRANSACTION/GENERIC (поиск miss обрабатывается отдельно).
        if subtask.mode.value not in ("TRANSACTION", "GENERIC"):
            return False
        if last_action is None or last_action.action != "click":
            return False
        if last_action_result is None or not last_action_result.success:
            return False
        if memory.stagnation_steps < 2 and memory.repeat_count < 2:
            return False
        return True

    async def _attempt_visual_recovery(
        self,
        *,
        subtask: Subtask,
        observation: list[InteractiveElement],
        memory: RuntimeMemory,
        last_action_result: ActionResult,
        global_step: int,
        page,
        stream_callback: Callable[[str], Awaitable[None]],
        vision_reason: str = "search_miss",
    ) -> tuple[AgentAction, ActionResult, _LlmDecision]:
        return await attempt_visual_recovery(
            subtask=subtask,
            observation=observation,
            memory=memory,
            last_action_result=last_action_result,
            global_step=global_step,
            page=page,
            stream_callback=stream_callback,
            vision_reason=vision_reason,
            history_dir=Path(self.config.history_dir),
            model_cheap=self.config.model_cheap,
            settings=self._settings,
            actor=self.actor,
            executor=self.executor,
        )

    async def _attempt_grounding(
        self,
        *,
        subtask: Subtask,
        policy: BaseTaskPolicy,
        observation: list[InteractiveElement],
        obs_window: list[InteractiveElement],
        memory: RuntimeMemory,
        last_action: AgentAction | None,
        last_action_result: ActionResult | None,
        global_step: int,
        page,
        stream_callback: Callable[[str], Awaitable[None]],
        current_fingerprint: str,
    ) -> tuple[AgentAction, ActionResult, _LlmDecision]:
        return await attempt_grounding_step(
            subtask=subtask,
            observation=observation,
            obs_window=obs_window,
            memory=memory,
            last_action=last_action,
            last_action_result=last_action_result,
            global_step=global_step,
            page=page,
            stream_callback=stream_callback,
            current_fingerprint=current_fingerprint,
            settings=self._settings,
            model_cheap=self.config.model_cheap,
            history_dir=Path(self.config.history_dir),
            actor=self.actor,
            executor=self.executor,
            apply_guards=lambda resolved: apply_guards(policy, subtask.goal, observation, resolved, memory),
            refine_self_check_repeat_click=lambda g: refine_self_check_repeat_click(g, last_action, memory),
            on_grounding_fingerprint_committed=lambda fp: setattr(self, "_last_grounding_fingerprint", fp),
        )

    async def _update_performance_metrics(
        self,
        *,
        subtask: Subtask,
        policy: BaseTaskPolicy,
        memory: RuntimeMemory,
        cost_stats: RunCostStats,
        observation: list[InteractiveElement],
        guarded: AgentAction,
        result: ActionResult,
        llm: _LlmDecision,
        signature: str,
        global_step: int,
        page,
        stream_callback: Callable[[str], Awaitable[None]],
    ) -> AgentState | None:
        return await update_performance_metrics(
            subtask=subtask,
            policy=policy,
            memory=memory,
            cost_stats=cost_stats,
            observation=observation,
            guarded=guarded,
            result=result,
            llm=llm,
            signature=signature,
            global_step=global_step,
            page=page,
            stream_callback=stream_callback,
            config=self.config,
            history_logger=self.history_logger,
            append_context_history=self._append_context_history,
        )

    # Главная корутина выполнения подзадачи. Здесь происходит основной цикл шагов агента.
    # Параметры resume_* нужны для возобновления после приостановки на опасном действии.
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
