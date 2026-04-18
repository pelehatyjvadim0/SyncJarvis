from __future__ import annotations

from agent.models.plan import ExecutionPlan, Subtask
from agent.models.task import TaskMode
from agent.planner.intent_classifier import classify_primary_mode


def _split_goal_parts(user_goal: str) -> list[str]:
    # Режем только по явным связкам шагов - без одиночного " и ", чтобы не дробить фразы вроде "X и Y" в одной цели.
    separators = [" затем ", " потом ", " и после этого ", " после этого "]
    normalized = f" {user_goal.strip()} "
    for sep in separators:
        normalized = normalized.replace(sep, "|")
    parts = [x.strip(" .,!?:;\n\t") for x in normalized.split("|") if x.strip(" .,!?:;\n\t")]
    return parts if parts else [user_goal.strip()]


def normalize_execution_plan(user_goal: str) -> ExecutionPlan:
    parts = _split_goal_parts(user_goal)
    subtasks: list[Subtask] = []
    communication_terms = ["сообщ", "ответ", "чат", "напиши", "прощ", "диалог"]
    has_communication_intent = any(term in user_goal.lower() for term in communication_terms)

    for idx, part in enumerate(parts, start=1):
        mode = classify_primary_mode(part)
        # Если общая цель явно про диалог, а кусок не распознался как communication,
        # но содержит признаки общения, принудительно фиксируем режим.
        if has_communication_intent and mode == TaskMode.SEARCH:
            lowered = part.lower()
            if any(term in lowered for term in communication_terms):
                mode = TaskMode.COMMUNICATION
        subtasks.append(
            Subtask(
                id=f"task_{idx}",
                title=f"Подзадача {idx}",
                mode=mode,
                goal=part,
            )
        )

    if not subtasks:
        subtasks.append(
            Subtask(
                id="task_1",
                title="Подзадача 1",
                mode=TaskMode.GENERIC,
                goal=user_goal,
            )
        )

    return ExecutionPlan(user_goal=user_goal, subtasks=subtasks, current_index=0)

