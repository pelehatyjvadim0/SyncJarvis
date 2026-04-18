from __future__ import annotations

from agent.llm.prompts.templates import planner_plan_rules_block, planner_role_and_schema_block


def build_planner_prompt(user_goal: str, max_subtasks: int) -> str:
    return (
        f"{planner_role_and_schema_block()}"
        f"{planner_plan_rules_block(max_subtasks)}"
        f"Цель пользователя:\n{user_goal}"
    )
