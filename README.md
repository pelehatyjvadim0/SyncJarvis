# ReAct Browser Agent (Новая архитектура)

Проект использует двухуровневую схему:
- High-level оркестратор строит план из подзадач и переключает режимы.
- Low-level ReAct-цикл исполняет шаги внутри активной подзадачи.

## Ключевые директории

- `agent/planner/` - классификация интента и декомпозиция цели в подзадачи.
- `agent/policies/` - policy-правила по режимам задач:
  - NAVIGATION, SEARCH, COMMUNICATION, FORM_FILL, SELECTION, TRANSACTION, VERIFICATION, GENERIC.
- `agent/runtime/`
  - `orchestrator.py` - high-level выполнение плана,
  - `react_loop/loop.py` - low-level ReAct-цикл для одной подзадачи,
  - `memory.py`, `anti_loop.py` - память рантайма и антициклы.
- `agent/llm/`
  - `planner_client.py` - LLM-планировщик,
  - `actor_client.py` - LLM-исполнитель шагов,
  - `prompts/` - шаблоны промптов.
- `agent/models/` - доменные модели (`task`, `plan`, `telemetry`, `action`, `observation`, `state`, `log`).
- `agent/perception/` - сбор интерактивных элементов из accessibility tree (без CSS/XPath).
- `agent/tools/` - браузерные инструменты Playwright.
- `agent/logging/` - запись скриншотов и JSON-истории шагов.

## Поток данных

1. `TaskOrchestrator` получает цель пользователя.
2. `planner.plan_normalizer` строит `ExecutionPlan` из нескольких `Subtask`.
3. Для каждой подзадачи выбирается policy по `TaskMode`.
4. `agent.runtime.react_loop.loop.SubtaskReActLoop`:
   - собирает observation из accessibility tree (по умолчанию только элементы в **viewport** + bbox в координатах документа; полный список — `collect_all_interactive_elements`),
   - запрашивает `ActorLLMClient`,
   - применяет policy-guard и global anti-loop,
   - выполняет действие через `BrowserToolExecutor`,
   - пишет telemetry в `history/step_XXX.json`.
5. После завершения подзадачи оркестратор переключается к следующей.

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env
```

Минимум в `.env`:
- `OPENROUTER_API_KEY=...`
- `OPENROUTER_MODEL_CHEAP=openai/gpt-4o-mini` — массовые шаги актора (быстро и дёшево).
- `OPENROUTER_MODEL_SMART=anthropic/claude-sonnet-4` — эскалация через `ModelRouter` при стагнации/ошибках/критичных режимах.

Важно: если задать только `OPENROUTER_MODEL`, а `OPENROUTER_MODEL_CHEAP` и `OPENROUTER_MODEL_SMART` **не** указать, то оба уровня получат одну и ту же модель (см. `load_app_settings` в `agent/config/settings.py`) — это почти всегда дороже, чем развести cheap/smart.

Опционально:
- `OPENROUTER_MODEL=...` # fallback-имя модели; также подставляется в cheap/smart, если они не заданы
- `AGENT_MAX_TOTAL_STEPS=80`
- `AGENT_MAX_SUBTASK_STEPS=25`
- `AGENT_SMART_COOLDOWN_STEPS=2`
- `AGENT_PROMPT_MAX_OBSERVATION_ITEMS=70`
- `AGENT_PROMPT_MAX_TEXT_FIELD_LEN=80`
- `AGENT_ACTOR_RESPONSE_MAX_TOKENS=300`
- `AGENT_SUBTASK_GOAL_SELF_CHECK_LLM=false` # после успешного шага делает короткий LLM self-check "цель подзадачи уже достигнута?"
- `AGENT_PRICE_DEFAULT_INPUT_PER_1M=0`
- `AGENT_PRICE_DEFAULT_OUTPUT_PER_1M=0`
- `AGENT_PRICE_CHEAP_INPUT_PER_1M=0.8`
- `AGENT_PRICE_CHEAP_OUTPUT_PER_1M=4`
- `AGENT_PRICE_SMART_INPUT_PER_1M=3`
- `AGENT_PRICE_SMART_OUTPUT_PER_1M=15`
- `OPENROUTER_HTTP_REFERER=...`
- `OPENROUTER_X_TITLE=...`

Запуск:

```bash
python app.py
```

Браузер по умолчанию **с окном** (`AGENT_BROWSER_HEADLESS=false`). На сервере без дисплея поставь `AGENT_BROWSER_HEADLESS=true`.

Подключение к **уже открытому** Chromium (один ваш браузер, без второго окна от агента):

1. Закройте лишние окна или оставьте нужный профиль — агент возьмёт первый контекст и первую вкладку (или создаст вкладку, если вкладок нет).
2. Запустите Chrome/Chromium с отладкой, например:  
   `google-chrome --remote-debugging-port=9222`  
   (порт любой свободный; для Chromium — тот же флаг.)
3. В `.env`: `AGENT_BROWSER_CDP_URL=http://127.0.0.1:9222`  
   При заданном URL **`AGENT_BROWSER_HEADLESS` и профиль `.browser-profile` для нового запуска не используются** — управление идёт через CDP к вашему процессу.
4. Запустите `python app.py` — Playwright подключится к этому endpoint, ваш браузер не завершится при выходе из приложения (только отключение CDP).

## Как добавить новый TaskMode

1. Добавь режим в `agent/models/task.py`.
2. Создай policy в `agent/policies/` с методами:
   - `is_done()`
   - `compute_progress()`
   - `anti_loop_guard()`
   - `fallback_action()`
3. Подключи policy в `TaskOrchestrator.policies`.
4. При необходимости обнови правила классификации в `planner/intent_classifier.py`.

## Тестирование

```bash
python -m unittest discover -s tests -p "test_*.py"
python -m compileall agent app.py
```

E2E без реального LLM: `tests/test_e2e_viewport_flow.py` (Playwright + локальный HTML, мок `ActorLLMClient.decide_action` для сценария scroll→finish).

## Ограничения и принципы

- Только accessibility-based взаимодействие.
- Никаких хардкод CSS/XPath селекторов.
- Опасные действия требуют перехода в `AWAITING_USER_CONFIRMATION`.
- Антициклы и дедупликация сообщений реализуются через runtime memory + policy.
- Каскад моделей: cheap для обычных шагов, smart при стагнации/ошибках/критичных режимах.
- Ограничение эскалаций: `AGENT_SMART_COOLDOWN_STEPS` снижает частоту дорогих smart-вызовов.
- Стоимость шага: сохраняется в telemetry (`prompt_tokens`, `completion_tokens`, `estimated_cost_usd`).
- Итог сессии: в конце выводится `[COST] total_usd, llm_steps, cheap, smart, avg_step_usd`.

