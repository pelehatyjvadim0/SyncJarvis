from __future__ import annotations

from dataclasses import dataclass

from agent.config.settings import AppSettings
from agent.llm.planner_client import PlannerLLMClient
from agent.models.plan import ExecutionPlan, Subtask


@dataclass(frozen=True)
class PlanBuildReport:
    # Диагностика построения плана, чтобы оркестратор мог прозрачно логировать причину пустого/ошибочного результата.
    source: str
    model: str
    planner_thought: str
    error: str
    subtask_count: int
    # Сколько подзадач от модели отброшено из-за лимита planner_max_subtasks.
    planner_subtasks_dropped: int


async def build_execution_plan(settings: AppSettings, user_goal: str) -> tuple[ExecutionPlan, PlanBuildReport]:
    # Собирает ExecutionPlan только через LLM-планировщик оркестратора (без normalizer-хардкода в runtime-пути).
    client = PlannerLLMClient(
        api_key=settings.openrouter_api_key,
        model=settings.openrouter_model_smart,
        temperature=settings.planner_temperature,
        referer=settings.openrouter_http_referer,
        title=settings.openrouter_x_title,
    )
    try:
        response = await client.plan(user_goal, settings.planner_max_subtasks)
    except Exception as exc:  # noqa: BLE001
        plan = ExecutionPlan(user_goal=user_goal, subtasks=[], current_index=0)
        return plan, PlanBuildReport(
            source="llm_planner",
            model=settings.openrouter_model_smart,
            planner_thought="",
            error=str(exc),
            subtask_count=0,
            planner_subtasks_dropped=0,
        )
    if not response.subtasks:
        plan = ExecutionPlan(user_goal=user_goal, subtasks=[], current_index=0)
        return plan, PlanBuildReport(
            source="llm_planner",
            model=settings.openrouter_model_smart,
            planner_thought=response.thought or "",
            error="Planner returned empty subtasks list",
            subtask_count=0,
            planner_subtasks_dropped=0,
        )
    cap = settings.planner_max_subtasks
    raw_list = response.subtasks[:cap]
    dropped = max(0, len(response.subtasks) - len(raw_list))
    subtasks: list[Subtask] = []
    for i, pst in enumerate(raw_list, start=1):
        subtasks.append(
            Subtask(
                id=f"task_{i}",
                title=pst.title or f"Подзадача {i}",
                mode=pst.mode,
                goal=pst.goal.strip() or user_goal,
            )
        )
    plan = ExecutionPlan(user_goal=user_goal, subtasks=subtasks, current_index=0)
    return plan, PlanBuildReport(
        source="llm_planner",
        model=settings.openrouter_model_smart,
        planner_thought=response.thought or "",
        error="",
        subtask_count=len(subtasks),
        planner_subtasks_dropped=dropped,
    )
