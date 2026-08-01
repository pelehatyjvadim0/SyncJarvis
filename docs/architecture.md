# Architecture

SyncJarvis is organized around a supervised browser-task lifecycle rather than a single unbounded model call.

## Components

| Component | Responsibility |
| --- | --- |
| Planner | Converts a user goal into bounded subtasks and task modes. |
| Orchestrator | Owns the session, task limits, policies, confirmation pauses, and final reporting. |
| ReAct loop | Repeats observation, decision, guarded execution, and persistence for one subtask. |
| Perception | Collects a compact accessibility-tree view of interactive page elements. |
| LLM layer | Builds prompts, calls the configured model tier, parses strict action responses, and handles retryable transport failures. |
| Browser executor | Runs Playwright actions and returns structured action results. |
| Safety controls | Apply confirmation, anti-loop, CAPTCHA, goal-check, and recovery rules. |

## One execution step

1. The loop captures the visible viewport and gathers the relevant accessibility context.
2. It applies observation checks, including blocked or CAPTCHA-like states.
3. The actor receives the task goal, screenshot, compact element data, and runtime context.
4. The proposed action is validated against task policy, loop guards, and confirmation requirements.
5. Playwright executes an allowed action and returns a structured result.
6. The loop records telemetry and persists a history entry before the next iteration.

## Contracts

The planner produces a bounded list of subtasks. The actor produces one structured action at a time. The executor returns whether the action succeeded, changed the page, or failed. The loop uses these contracts to decide whether to continue, retry, pause, finish, or stop.

## State transitions

The normal path is `planning → running → finished`. A task can instead enter `awaiting confirmation`, `blocked`, `partial`, or `error` when an action requires a user, a hard limit is reached, or the browser/LLM cannot continue safely. Success is not inferred solely from a successful browser call; the loop can use goal checks and page evidence.

The implementation-level module map is maintained with the code in `agent/runtime/ARCHITECTURE.md` and `agent/llm/ARCHITECTURE.md`.
