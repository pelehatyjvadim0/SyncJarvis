# Runtime Architecture

```text
agent/runtime/
├── ARCHITECTURE.md
├── memory.py              # RuntimeMemory, MemoryGuardView
├── orchestrator.py        # TaskOrchestrator — сессия задачи, вызов планировщика и цикла подзадач
├── anti_loop.py
├── goal_verifier.py       # Финальная проверка цели пользователя (viewport PNG + LLM; при сбое — не подтверждаем успех)
├── policy_registry.py
├── security.py
├── self_correction.py
└── react_loop/
    ├── loop.py            # Публичный вход: SubtaskReActLoop (инициализация LLM/роутера, делегирование в pipeline)
    ├── config.py
    ├── step_utils.py
    ├── engine/
    │   ├── pipeline.py    # Порядок фаз итерации (run_subtask_pipeline); halt / next_iteration
    │   ├── decision_maker.py
    │   ├── action_executor.py
    │   ├── types.py
    │   └── phases/        # Фазы: склейка + PhaseStepResult; не вызывают LLM напрямую без decision_maker
    ├── components/        # Узкие подсистемы: fusion snapshot, goal_self_check snapshot, viewport_capture, captcha, vision, fingerprint (grounding-runner остаётся в коде для тестов; фаза grounding в pipeline — no-op)
    ├── utils/             # observation_builder, telemetry, persistence
    └── observability/     # Тонкие фасады записи шага (например REPLAN persist + телеметрия)
```

Слой LLM (промпты, клиенты, контракты) — см. [`agent/llm/ARCHITECTURE.md`](../llm/ARCHITECTURE.md).

## Назначение

- **orchestrator** — верхний уровень: план, политика, вызов `SubtaskReActLoop`, паузы на опасные действия.
- **react_loop** — один ReAct-цикл по подзадаче: наблюдение → проверки → LLM → гварды → исполнение → метрики/история.

## Viewport-first (единый принцип)

- **Смысловые решения и «готово / не готово»** опираются на **то, что видно во viewport** (PNG `full_page=False`), а не на «успех инструмента» и не на a11y-список в отрыве от кадра.
- **Следующее действие (actor):** всегда `ActorLLMClient.decide_fusion_step_action` — viewport PNG + цель/контекст; сжатый a11y только как подпись к тому же кадру (`components/fusion_step_snapshot.build_fusion_step_snapshot` снимает PNG через `viewport_capture`).
- **Self-check цели подзадачи:** `components/goal_self_check_snapshot.build_goal_self_check_snapshot` → `assess_goal_reached` (multimodal); a11y не достаточен для `goal_reached=true` без согласования с SCREEN (см. `prompts/templates.goal_self_check_vision_instructions_block`).
- **Финальная проверка пользовательской цели:** `goal_verifier.verify_user_goal_satisfied_llm` — `{history_dir}/verify_user_goal.png` + цель + сводка шагов как подпись; при ошибке сети / парсинга — **не** считаем цель достигнутой (`False`).
- Отдельных режимов «текст vs vision» в цикле нет; `observation_fusion_multimodal` в настройках устарело (игнорируется для ветвления).

## Поток данных

1. `TaskOrchestrator` планирует подзадачи и для каждой создаёт/использует `SubtaskReActLoop`.
2. `SubtaskReActLoop.run_subtask` вызывает `engine.pipeline.run_subtask_pipeline(loop, ...)`.
3. **Pipeline** последовательно вызывает **phases** (`engine/phases/*`). Каждая фаза возвращает `PhaseStepResult` (`halt` | `next_iteration` | `proceed`).
4. Решение LLM: `engine.decision_maker.make_decision` → всегда fusion-снимок и `decide_fusion_step_action` → `action_executor` и `BrowserToolExecutor.execute_action`.

## Правила зависимостей

- `react_loop/phases` не импортирует `loop.py` (избегаем циклов); контекст передаётся через `PipelineIterationContext`.
- **Фаза** (`engine/phases/*`) = ветвление по состоянию шага и вызов компонентов/утилит; не дублировать бизнес-логику LLM внутри фазы без необходимости.
- **Компонент** (`react_loop/components/*`) = переиспользуемый блок с узким контрактом (captcha, fusion snapshot, grounding runner).
- `react_loop` не импортирует приватные методы исполнителя (`_require_page`); для страницы используется публичный `BrowserToolExecutor.require_page()`.
- Слой `agent/llm` — см. `agent/llm/ARCHITECTURE.md`; runtime импортирует **clients** / **services** / **prompts**, не наоборот.

## Метрики и история

- Телеметрия шага: `react_loop/utils/telemetry.build_step_telemetry`.
- Сохранение шага: `react_loop/utils/persistence.persist_step`.
- Сводная запись после исполнения в фазах: `engine/phases/persist_metrics.persist_step_execution_metrics` (моменты вызова зафиксированы в `pipeline.py` комментариями — не менять порядок без причины).
- REPLAN после невалидного JSON: `react_loop/observability/replan_recording.persist_replan_fallback_step`.

## Публичные точки входа

- `from agent.runtime.orchestrator import TaskOrchestrator, OrchestratorConfig`
- `from agent.runtime.react_loop.loop import SubtaskReActLoop`
- `from agent.tools.browser_executor import BrowserToolExecutor`

## Чеклист перед merge

- `python3 -m compileall agent`
- `pytest tests/test_grounding.py tests/test_context_stability.py tests/test_e2e_viewport_flow.py` (+ затронутые области)
