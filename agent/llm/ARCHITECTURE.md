# LLM Architecture

```text
agent/llm/
├── ARCHITECTURE.md
├── __init__.py
├── base.py
├── clients/
│   ├── __init__.py
│   ├── actor.py
│   └── planner.py
├── services/
│   ├── __init__.py
│   ├── parser.py
│   └── router.py
├── prompts/
│   ├── __init__.py
│   ├── planner.py
│   ├── templates.py
│   └── actor/
│       ├── __init__.py
│       ├── builder.py
│       ├── observation.py
│       ├── resolver.py
│       └── text_utils.py
└── contracts/
    ├── __init__.py
    ├── actor.py
    ├── planner.py
    └── router.py
```

Слой runtime и цикл подзадач — см. [`agent/runtime/ARCHITECTURE.md`](../runtime/ARCHITECTURE.md).

## 1) Назначение модуля

- `agent/llm` — слой работы с LLM для двух задач: планирование (`planner`) и выбор следующего действия (`actor`).
- Модуль собирает промпты, отправляет запросы в модель, парсит/валидирует ответы и возвращает типизированные контракты.
- Дополнительно содержит маршрутизацию моделей (`cheap/smart`) и transport/retry-примитивы.
- **Viewport-first:** основной цикл актёра — только multimodal fusion (`clients/actor.py` → `decide_fusion_step_action`); длинные инструкции про SCREEN / JSON / self-check / verify — в `prompts/templates.py`. Для частично битого JSON fusion допускается узкое восстановление `action`/`params` в `actor._try_recover_fusion_agent_action` без ослабления остальных контрактов.

## 2) Карта пакетов

- `base.py`
  - Общий transport-layer: retryable-ошибки и `chat_with_retry`.
- `clients/`
  - Клиенты LLM-вызовов: `PlannerLLMClient`, `ActorLLMClient`.
  - Оркестрируют вызов модели, но не содержат runtime-цикл.
- `services/`
  - Сервисные функции: парсинг action JSON и роутинг модели.
- `prompts/`
  - Сборка текстов промптов и подготовка входных данных для них.
  - `prompts/actor/*` — разложенная логика actor prompt (builder/observation/resolver/text_utils).
- `contracts/`
  - Dataclass-контракты выходов LLM-слоя (`ActorDecision`, `ModelRoute` и др.).

## 3) Потоки данных

- Planner flow
  - Цель пользователя -> `prompts.planner.build_planner_prompt` -> `clients.planner.PlannerLLMClient.plan` -> JSON parse/validate -> `PlannerResponse`.
- Actor flow (viewport-first)
  - `runtime.react_loop.components.fusion_step_snapshot` (viewport PNG + компактный a11y) → `clients.actor.ActorLLMClient.decide_fusion_step_action` (multimodal) → при частично битом JSON узкое `services.fusion_partial_recovery.try_recover_fusion_agent_action` → `prompts.actor.resolve_actor_element_index` по окну наблюдения.
  - Текстовый `build_actor_prompt` / `decide_action` остаётся в коде для совместимости и отладки, **основной цикл** оркестратора использует только fusion.

## 4) Зависимости между слоями (правила)

- Направление зависимостей:
  - `clients` -> `prompts`, `services`, `contracts`, `base`
  - `services` -> `contracts` (для `ModelRoute`)
  - `prompts` -> `models`, `config`, `prompts.templates`
- Что нельзя:
  - `prompts` не импортирует `clients`, `runtime`, сетевые SDK.
  - `services` не импортирует `clients` и runtime-циклы.
  - `contracts` не содержит бизнес-логику и внешние вызовы.

## 5) Публичные точки входа

- Канонические импорты:
  - `agent.llm.clients.actor.ActorLLMClient`
  - `agent.llm.clients.planner.PlannerLLMClient`
  - `agent.llm.services.router.ModelRoute`, `agent.llm.services.router.ModelRouter`
  - `agent.llm.services.parser.parse_agent_action_json`, `strip_markdown_json_fence`
  - `agent.llm.services.fusion_partial_recovery.try_recover_fusion_agent_action`
  - `agent.llm.prompts.actor.build_actor_prompt`
  - `agent.llm.prompts.actor.ordered_observation_for_actor_prompt`
  - `agent.llm.prompts.actor.serialize_observation_window_for_actor_prompt`
  - `agent.llm.prompts.actor.resolve_actor_element_index`
  - `agent.llm.prompts.planner.build_planner_prompt`
  - `agent.llm.base.chat_with_retry`, `agent.llm.base.is_retryable_transport_error`

## 6) Инварианты (для рефакторинга)

- Структурные изменения делаются в режиме **zero behavior change**.
- Retry/backoff/error semantics не менять без отдельного решения.
- Контракты prompt/response (строгий JSON, обязательные поля, порядок/смысл инструкций) считать стабильными.
- Изменение путей импорта допускается только с сохранением публичного API (через фасады `__init__.py` при необходимости).

## 7) Мини-чеклист перед merge

- `python3 -m compileall agent/llm`
- lint по измененным файлам (`ruff check ...`)
- ключевые тесты:
  - `tests/test_planner.py`
  - `tests/test_element_index_resolve.py`
  - `tests/test_grounding.py`
  - при необходимости `tests/test_anti_loop.py` / e2e-smoke
