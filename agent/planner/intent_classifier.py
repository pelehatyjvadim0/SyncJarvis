from __future__ import annotations

from agent.models.task import TaskMode


def classify_primary_mode(user_goal: str) -> TaskMode:
    goal = user_goal.lower()
    if any(x in goal for x in ["найти", "найди", "поиск", "search", "find", "ищи"]):
        return TaskMode.SEARCH
    if any(x in goal for x in ["сообщ", "ответ", "чат", "напиши", "прощ", "диалог"]):
        return TaskMode.COMMUNICATION
    if any(x in goal for x in ["заполни", "форма", "submit", "анкета"]):
        return TaskMode.FORM_FILL
    if any(x in goal for x in ["выбери", "select", "вариант", "фильтр"]):
        return TaskMode.SELECTION
    if any(x in goal for x in ["оплат", "удал", "подтверди", "купить"]):
        return TaskMode.TRANSACTION
    if any(x in goal for x in ["открой", "перейди", "navigate", "go to"]):
        return TaskMode.NAVIGATION
    if any(x in goal for x in ["проверь", "убедись", "verify"]):
        return TaskMode.VERIFICATION
    return TaskMode.GENERIC

