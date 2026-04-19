from __future__ import annotations

import asyncio

# Сбор observation на странице; parent_anchor и пр. — в agent.perception.accessibility при обходе AX.
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

from agent.models.observation import InteractiveElement
from agent.models.state import AgentState
from agent.perception.accessibility import collect_interactive_elements
from agent.runtime.memory import RuntimeMemory
from agent.tools.browser_executor import BrowserToolExecutor


async def collect_observation_with_recovery(
    page: Any,
    *,
    stream_callback: Callable[[str], Awaitable[None]],
    executor: BrowserToolExecutor,
    require_page: Callable[[], Any],
) -> tuple[Any, list[InteractiveElement]]:
    try:
        return page, await collect_interactive_elements(page)
    except Exception as exc:
        # Сообщаем о падении сбора дерева доступности.
        await stream_callback(f"[OBSERVATION] Ошибка сбора accessibility дерева: {exc}")
        executor.page = None
        recovered_page = require_page()
        # Если страница была перепривязана, сигнализируем об этом в логе.
        if recovered_page is not page:
            await stream_callback("[OBSERVATION] Перепривязка к активной вкладке и повторный сбор.")
        try:
            return recovered_page, await collect_interactive_elements(recovered_page)
        except Exception as exc2:
            # Если и повторная попытка не удалась, возвращаем пустой список observation.
            await stream_callback(f"[OBSERVATION] Повторный сбор после перепривязки не удался: {exc2}")
            return recovered_page, []


@dataclass
class SubtaskObservationCollectionPhaseResult:
    page: Any
    observation: list[InteractiveElement]
    observation_collect_fail_streak: int
    empty_observation_streak: int
    agent_state_if_terminal: AgentState | None = None


async def run_subtask_observation_collection_phase(
    page: Any,
    *,
    memory: RuntimeMemory,
    stream_callback: Callable[[str], Awaitable[None]],
    executor: BrowserToolExecutor,
    require_page: Callable[[], Any],
    set_last_observation: Callable[[Iterable[InteractiveElement]], None],
    observation_collect_fail_streak: int,
    empty_observation_streak: int,
) -> SubtaskObservationCollectionPhaseResult:
    # Пытаемся получить интерактивные элементы. При необходимости перепривязываем страницу.
    page, observation = await collect_observation_with_recovery(
        page,
        stream_callback=stream_callback,
        executor=executor,
        require_page=require_page,
    )
    if observation:
        # Если удалось собрать элементы, сбрасываем счетчик неудач сбора.
        observation_collect_fail_streak = 0
    else:
        # Если не удалось собрать, увеличиваем счетчик ошибочного сбора.
        observation_collect_fail_streak += 1
    # Если подряд несколько неудач — выполняем fallback: ждем, перепривязываем, снова пробуем собрать элементы.
    if observation_collect_fail_streak >= 2:
        await stream_callback("[OBSERVATION] Controlled fallback: wait + page rebind.")
        await executor.execute_action("wait", {"seconds": 0.6}, observation)
        executor.page = None
        page = require_page()
        page, observation = await collect_observation_with_recovery(
            page,
            stream_callback=stream_callback,
            executor=executor,
            require_page=require_page,
        )
        observation_collect_fail_streak = 0 if observation else observation_collect_fail_streak
    # Сохраняем текущее observation для возможного последующего анализа.
    set_last_observation(observation)
    # Если интерактивных элементов совсем нет, считаем такие случаи.
    if len(observation) == 0:
        empty_observation_streak += 1
        # Если подряд несколько пустых observation — завершить с ошибкой.
        if empty_observation_streak >= 3:
            memory.done_reason = "Observation пустой после нескольких попыток recovery."
            return SubtaskObservationCollectionPhaseResult(
                page=page,
                observation=observation,
                observation_collect_fail_streak=observation_collect_fail_streak,
                empty_observation_streak=empty_observation_streak,
                agent_state_if_terminal=AgentState.ERROR,
            )
        # Если это не критическая ситуация — пробуем прокрутить страницу вниз, вдруг элементы вне viewport.
        await stream_callback(
            "[OBSERVATION] Нет видимых интерактивных элементов в viewport — выполняю scroll вниз."
        )
        await page.mouse.wheel(0, 720)  # Скроллим вниз
        await asyncio.sleep(0.12)  # Ждем, чтобы страница успела отрисоваться
        page, observation = await collect_observation_with_recovery(
            page,
            stream_callback=stream_callback,
            executor=executor,
            require_page=require_page,
        )
        set_last_observation(observation)
    # Если элементов мало — возможно, есть смысл подсказать пользователю или LLM-агенту про скролл.
    elif len(observation) < 3:
        empty_observation_streak = 0
        await stream_callback(
            "[HINT] Видимых элементов мало — цель может быть ниже или выше; при необходимости сделай scroll."
        )
    else:
        # Сбрасываем счетчик пустых observation, если элементы найдены.
        empty_observation_streak = 0

    return SubtaskObservationCollectionPhaseResult(
        page=page,
        observation=observation,
        observation_collect_fail_streak=observation_collect_fail_streak,
        empty_observation_streak=empty_observation_streak,
        agent_state_if_terminal=None,
    )
