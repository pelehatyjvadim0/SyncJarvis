# Configuration

Copy `.env.example` to `.env` and keep `.env` local. The only required value is an OpenRouter API key.

## Required

| Variable | Purpose |
| --- | --- |
| `OPENROUTER_API_KEY` | API key used for planner and actor requests. |
| `OPENROUTER_MODEL` | Fallback model when tier-specific values are not set. |

## Model routing and prompts

| Variables | Purpose |
| --- | --- |
| `OPENROUTER_MODEL_CHEAP`, `OPENROUTER_MODEL_SMART` | Models used by the two routing tiers. |
| `AGENT_SMART_COOLDOWN_STEPS` | Minimum spacing between smart-tier decisions. |
| `AGENT_PROMPT_MAX_OBSERVATION_ITEMS`, `AGENT_PROMPT_MAX_TEXT_FIELD_LEN` | Bounds for accessibility context sent to the actor. |
| `AGENT_ACTOR_RESPONSE_MAX_TOKENS`, `AGENT_LLM_TRANSPORT_MAX_RETRIES` | Response-size and retry limits. |

## Execution and safety

| Variables | Purpose |
| --- | --- |
| `AGENT_MAX_TOTAL_STEPS`, `AGENT_MAX_SUBTASK_STEPS` | Hard limits for a task and its subtasks. |
| `AGENT_PLANNER_MAX_SUBTASKS`, `AGENT_PLANNER_TEMPERATURE` | Planner output bounds and sampling control. |
| `AGENT_CAPTCHA_MAX_CONSECUTIVE_WAIT` | Number of consecutive CAPTCHA waits before the task is blocked. |
| `AGENT_GOAL_VERIFY_LLM`, `AGENT_GOAL_VERIFY_FAIL_SOFT` | Optional final-goal verification behavior. |
| `AGENT_CONTINUE_AFTER_SUBTASK_LIMIT` | Whether to continue after a subtask limit; leave disabled unless partial completion is acceptable. |

## Browser and local operation

| Variables | Purpose |
| --- | --- |
| `AGENT_BROWSER_HEADLESS` | Runs Chromium without a visible window when `true`. |
| `AGENT_BROWSER_VIEWPORT_WIDTH`, `AGENT_BROWSER_VIEWPORT_HEIGHT` | Layout viewport dimensions. |
| `AGENT_BROWSER_CDP_URL` | Optional connection to a browser already started with remote debugging. |
| `AGENT_OBSERVATION_FUSION_MULTIMODAL` | Keeps screenshot and accessibility context together for actor decisions. |

## Cost estimation and optional headers

`AGENT_PRICE_*_INPUT_PER_1M` and `AGENT_PRICE_*_OUTPUT_PER_1M` set local token-cost estimates. `OPENROUTER_HTTP_REFERER` and `OPENROUTER_X_TITLE` are optional request metadata. Review provider documentation before setting them.

Use conservative limits first. Model responses, site behavior, and network conditions vary; no configuration value makes browser automation risk-free.
