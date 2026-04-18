from __future__ import annotations

from agent.models.action import ActionResult


def build_correction_hint(last_action_result: ActionResult | None) -> str:
    if not last_action_result:
        return "Нет прошлого результата - выполняем первый шаг."
    if not last_action_result.success:
        return "Предыдущее действие завершилось ошибкой - нужен альтернативный путь."
    if not last_action_result.changed:
        return "Страница не изменилась - попробуй другое действие."
    return "Предыдущее действие успешно - можно продолжать план."

