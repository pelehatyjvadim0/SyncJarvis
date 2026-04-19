from __future__ import annotations

import os
from dataclasses import dataclass



def _env_str(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _env_opt_str(key: str) -> str | None:
    v = os.getenv(key)
    if v is None or not str(v).strip():
        return None
    return str(v).strip()


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _parse_grounding_modes(raw: str) -> frozenset[str]:
    items = [x.strip().upper() for x in (raw or "").split(",") if x.strip()]
    return frozenset(items) if items else frozenset({"SEARCH", "SELECTION"})


def _env_navigate_wait_until(key: str, default: str) -> str:
    raw = (os.getenv(key) or default).strip().lower()
    allowed = frozenset({"domcontentloaded", "load", "commit", "networkidle"})
    return raw if raw in allowed else default


def _env_viewport_dim(key: str, default: int, *, min_v: int, max_v: int) -> int:
    v = _env_int(key, default)
    return max(min_v, min(max_v, v))


@dataclass(frozen=True)
class AgentPricing:
    default_input_per_1m: float
    default_output_per_1m: float
    cheap_input_per_1m: float
    cheap_output_per_1m: float
    smart_input_per_1m: float
    smart_output_per_1m: float

    def price_per_1m(self, tier: str, kind: str) -> float:
        tier_l = tier.lower()
        kind_l = kind.lower()
        if tier_l == "fallback":
            return 0.0
        if tier_l == "cheap":
            return self.cheap_input_per_1m if kind_l == "input" else self.cheap_output_per_1m
        if tier_l == "smart":
            return self.smart_input_per_1m if kind_l == "input" else self.smart_output_per_1m
        if kind_l == "input":
            return self.default_input_per_1m
        return self.default_output_per_1m

    def estimate_cost_usd(self, prompt_tokens: int, completion_tokens: int, tier: str) -> float:
        inp = max(0.0, self.price_per_1m(tier, "input"))
        out = max(0.0, self.price_per_1m(tier, "output"))
        return (prompt_tokens / 1_000_000.0) * inp + (completion_tokens / 1_000_000.0) * out


@dataclass(frozen=True)
class ActorPromptLimits:
    max_observation_items: int
    max_text_field_len: int


@dataclass(frozen=True)
class AppSettings:
    openrouter_api_key: str
    openrouter_model_fallback: str
    openrouter_model_cheap: str
    openrouter_model_smart: str
    openrouter_http_referer: str | None
    openrouter_x_title: str | None
    max_total_steps: int
    max_subtask_steps: int
    smart_cooldown_steps: int
    actor_response_max_tokens: int
    prompt_limits: ActorPromptLimits
    orchestrator_temperature: float
    # Температура только для LLM-планировщика (JSON план). Низкая (0–0.2) — меньше лишних шагов; выше — больше вариативности.
    planner_temperature: float
    # Верхняя граница числа подзадач в плане (промпт планировщика + обрезка ответа модели).
    planner_max_subtasks: int
    history_dir: str
    pricing: AgentPricing
    # Deprecated: план теперь всегда строится через LLM-планировщик; флаг оставлен для обратной совместимости окружения.
    use_llm_planner: bool
    # Сколько подряд итераций «капча - wait» допускаем до BLOCKED_CAPTCHA.
    captcha_max_consecutive_waits: int
    # Повторы вызова chat.completions при сетевых/5xx ошибках (не при ValueError парсинга).
    llm_transport_max_retries: int
    # После FINISHED - короткий LLM-опрос «достигнута ли цель пользователя» (доп. запрос).
    goal_verify_llm: bool
    # Если verify вернул satisfied=false — не завершать ERROR, а PARTIAL (осторожно: отчёт может быть неточным).
    goal_verify_fail_soft: bool
    # При SUBTASK_STEP_LIMIT помечать подзадачу выполненной и идти дальше (осторожно: цель могла быть не достигнута).
    continue_after_subtask_step_limit: bool
    # True — Chromium без окна (CI/сервер); False — видимое окно (локальная отладка и капча руками).
    browser_headless: bool
    # Размер layout viewport для нового контекста Playwright (адаптивная вёрстка сайта зависит от этих пикселей).
    browser_viewport_width: int
    browser_viewport_height: int
    # Если задан (например http://127.0.0.1:9222) — Playwright подключается к уже запущенному Chromium по CDP, новый браузер не поднимается.
    browser_cdp_url: str | None
    # Включает микро LLM self-check после успешного шага: достигнута ли цель подзадачи на текущей странице.
    subtask_goal_self_check_llm: bool
    # При включённом self-check: для SELECTION/TRANSACTION вызывать проверку и после неуспешного click (товар уже в корзине, а инструмент вернул timeout).
    subtask_goal_self_check_after_failed_click: bool
    # Устарело: раньше переключатель «текст vs fusion»; сейчас актёр всегда viewport-first (поле читается из .env для совместимости дампов настроек).
    observation_fusion_multimodal: bool
    # Multimodal grounding: скрин + список element_index после ключевых событий (без хардкода под домены).
    grounding_enabled: bool
    grounding_after_navigate: bool
    grounding_after_search_submit: bool
    grounding_after_fingerprint_change: bool
    grounding_after_url_change: bool
    grounding_modes: frozenset[str]
    grounding_min_wait_seconds: float
    # navigate: goto(wait_until=...) + опционально networkidle + settle. ``load`` на крупных сайтах (hh.ru и т.д.) часто
    # зависает на минуты — по умолчанию ``domcontentloaded``. См. AGENT_BROWSER_NAVIGATE_WAIT_UNTIL.
    browser_navigate_wait_until: str
    browser_navigate_timeout_ms: int
    browser_navigate_networkidle_timeout_ms: int
    browser_navigate_post_settle_seconds: float


def load_app_settings() -> AppSettings:
    fb = _env_str("OPENROUTER_MODEL", "anthropic/claude-sonnet-4")
    cheap = _env_str("OPENROUTER_MODEL_CHEAP", fb)
    smart = _env_str("OPENROUTER_MODEL_SMART", fb)
    pricing = AgentPricing(
        default_input_per_1m=_env_float("AGENT_PRICE_DEFAULT_INPUT_PER_1M", 0.0),
        default_output_per_1m=_env_float("AGENT_PRICE_DEFAULT_OUTPUT_PER_1M", 0.0),
        cheap_input_per_1m=_env_float("AGENT_PRICE_CHEAP_INPUT_PER_1M", 0.0),
        cheap_output_per_1m=_env_float("AGENT_PRICE_CHEAP_OUTPUT_PER_1M", 0.0),
        smart_input_per_1m=_env_float("AGENT_PRICE_SMART_INPUT_PER_1M", 0.0),
        smart_output_per_1m=_env_float("AGENT_PRICE_SMART_OUTPUT_PER_1M", 0.0),
    )
    limits = ActorPromptLimits(
        max_observation_items=_env_int("AGENT_PROMPT_MAX_OBSERVATION_ITEMS", 70),
        max_text_field_len=_env_int("AGENT_PROMPT_MAX_TEXT_FIELD_LEN", 80),
    )
    _planner_t = _env_float("AGENT_PLANNER_TEMPERATURE", 0.0)
    planner_temperature = max(0.0, min(2.0, _planner_t))
    return AppSettings(
        openrouter_api_key=_env_str("OPENROUTER_API_KEY"),
        openrouter_model_fallback=fb,
        openrouter_model_cheap=cheap,
        openrouter_model_smart=smart,
        openrouter_http_referer=_env_opt_str("OPENROUTER_HTTP_REFERER"),
        openrouter_x_title=_env_opt_str("OPENROUTER_X_TITLE"),
        max_total_steps=_env_int("AGENT_MAX_TOTAL_STEPS", 80),
        max_subtask_steps=_env_int("AGENT_MAX_SUBTASK_STEPS", 25),
        smart_cooldown_steps=_env_int("AGENT_SMART_COOLDOWN_STEPS", 2),
        actor_response_max_tokens=_env_int("AGENT_ACTOR_RESPONSE_MAX_TOKENS", 300),
        prompt_limits=limits,
        orchestrator_temperature=0.2,
        planner_temperature=planner_temperature,
        planner_max_subtasks=max(1, min(50, _env_int("AGENT_PLANNER_MAX_SUBTASKS", 6))),
        history_dir="history",
        pricing=pricing,
        use_llm_planner=_env_bool("AGENT_USE_LLM_PLANNER", False),
        captcha_max_consecutive_waits=_env_int("AGENT_CAPTCHA_MAX_CONSECUTIVE_WAIT", 20),
        llm_transport_max_retries=_env_int("AGENT_LLM_TRANSPORT_MAX_RETRIES", 3),
        goal_verify_llm=_env_bool("AGENT_GOAL_VERIFY_LLM", False),
        goal_verify_fail_soft=_env_bool("AGENT_GOAL_VERIFY_FAIL_SOFT", False),
        continue_after_subtask_step_limit=_env_bool("AGENT_CONTINUE_AFTER_SUBTASK_LIMIT", False),
        browser_headless=_env_bool("AGENT_BROWSER_HEADLESS", False),
        browser_viewport_width=_env_viewport_dim(
            "AGENT_BROWSER_VIEWPORT_WIDTH", 1440, min_v=320, max_v=3840
        ),
        browser_viewport_height=_env_viewport_dim(
            "AGENT_BROWSER_VIEWPORT_HEIGHT", 900, min_v=240, max_v=2160
        ),
        browser_cdp_url=_env_opt_str("AGENT_BROWSER_CDP_URL"),
        subtask_goal_self_check_llm=_env_bool("AGENT_SUBTASK_GOAL_SELF_CHECK_LLM", False),
        subtask_goal_self_check_after_failed_click=_env_bool(
            "AGENT_SUBTASK_SELF_CHECK_AFTER_FAILED_CLICK", True
        ),
        observation_fusion_multimodal=_env_bool("AGENT_OBSERVATION_FUSION_MULTIMODAL", True),
        grounding_enabled=_env_bool("AGENT_GROUNDING_ENABLED", False),
        grounding_after_navigate=_env_bool("AGENT_GROUNDING_AFTER_NAVIGATE", True),
        grounding_after_search_submit=_env_bool("AGENT_GROUNDING_AFTER_SEARCH_SUBMIT", True),
        grounding_after_fingerprint_change=_env_bool("AGENT_GROUNDING_AFTER_FINGERPRINT_CHANGE", True),
        grounding_after_url_change=_env_bool("AGENT_GROUNDING_AFTER_URL_CHANGE", True),
        grounding_modes=_parse_grounding_modes(_env_str("AGENT_GROUNDING_MODES", "SEARCH,SELECTION")),
        grounding_min_wait_seconds=_env_int("AGENT_GROUNDING_MIN_WAIT_MS", 400) / 1000.0,
        browser_navigate_wait_until=_env_navigate_wait_until(
            "AGENT_BROWSER_NAVIGATE_WAIT_UNTIL", "domcontentloaded"
        ),
        browser_navigate_timeout_ms=max(5_000, _env_int("AGENT_BROWSER_NAVIGATE_TIMEOUT_MS", 180_000)),
        browser_navigate_networkidle_timeout_ms=max(
            0, _env_int("AGENT_BROWSER_NAVIGATE_NETWORKIDLE_TIMEOUT_MS", 12_000)
        ),
        browser_navigate_post_settle_seconds=max(
            0.0, _env_int("AGENT_BROWSER_NAVIGATE_POST_SETTLE_MS", 500) / 1000.0
        ),
    )
