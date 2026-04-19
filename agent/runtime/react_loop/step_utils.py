from __future__ import annotations

from agent.models.action import ActionResult, AgentAction
from agent.models.observation import InteractiveElement
from agent.runtime.memory import RuntimeMemory


def action_signature(action: AgentAction) -> str:
    ax_id = action.params.get("ax_id", "")
    text = str(action.params.get("text", ""))[:40]
    direction = action.params.get("direction", "")
    return f"{action.action}|{ax_id}|{direction}|{text}"


def search_candidates_count(observation: list[InteractiveElement]) -> int:
    return sum(1 for item in observation if item.role in {"searchbox", "textbox"})


def is_deep_context_mode(memory: RuntimeMemory) -> bool:
    return memory.stagnation_steps >= 2


def describe_guard_override_for_fusion(llm_proposed: AgentAction, executed: AgentAction) -> str:
    # Fusion иначе видит только wait/success и снова шлёт scroll; явно сообщаем, что scroll был подменён антициклом.
    if llm_proposed.action == executed.action and llm_proposed.params == executed.params:
        return ""
    base = (
        f"СИСТЕМА: модель предложила action={llm_proposed.action} params={llm_proposed.params}, "
        f"исполнено action={executed.action} params={executed.params} (антицикл или политика режима)."
    )
    if llm_proposed.action == "scroll" and executed.action == "wait":
        base += (
            " Scroll при стагнации/повторах часто заменяется на wait — не повторяй тот же scroll; "
            "нужны click/click_xy по элементу из списка, type, navigate или иной прогресс, иначе UI не сдвинется."
        )
    return base[:900]


def fusion_coordinate_priority_hint(
    last_action: AgentAction | None,
    last_action_result: ActionResult | None,
    memory: RuntimeMemory | None = None,
) -> str:
    # Crop-verify NO: действие не исполнялось — last_action_result часто пустой/старое; ориентируемся на self_check_hint.
    if memory is not None:
        sh = (memory.self_check_hint or "").lower()
        if "smart crop-verify" in sh or "crop-verify" in sh:
            return (
                "ПРИОРИТЕТ ЭТОГО ШАГА: smart crop-verify отклонил выбранный click/type по a11y (на кропе цель не "
                "совпала с ожиданием). НЕ повторяй тот же element_index и тот же ax_id. Выбери: action=click_xy с "
                "целочисленными params.x, params.y (центр цели в координатах viewport SCREEN), или другой "
                "element_index только если на SCREEN явно другой контрол, или scroll/wait."
            )
    # После неудачного click/type по element_index следующий fusion-шаг просим клик по пикселям — Playwright nth() часто промахивается на дубликатах.
    if not last_action or not last_action_result or last_action_result.success:
        return ""
    if last_action.action not in ("click", "type"):
        return ""
    if "element_index" not in last_action.params:
        return ""
    return (
        "ПРИОРИТЕТ ЭТОГО ШАГА: прошлый click/type через element_index (a11y) завершился с success=false. "
        "Не повторяй только другой element_index ради того же визуального контрола — верни action=click_xy с "
        "целочисленными params.x и params.y: центр нужной кнопки/иконки/поля в координатах viewport SCREEN "
        "(0,0 — левый верх видимой области). В thought кратко укажи, для какого элемента UI взял центр."
    )


def fusion_click_xy_recovery_hint(
    last_action: AgentAction | None,
    last_action_result: ActionResult | None,
    memory: RuntimeMemory | None = None,
) -> str:
    if not last_action or last_action.action != "click_xy" or not last_action_result:
        return ""
    if not last_action_result.success:
        return (
            "ПРИОРИТЕТ ЭТОГО ШАГА: прошлый click_xy завершился с success=false — возможен промах координат или перекрытие. "
            "Сверь SCREEN; возьми другие целочисленные x,y по центру цели, или однозначный click/type по element_index, "
            "или scroll/wait. Не повторяй те же x,y."
        )
    # Инструмент часто success=true даже при промахе; повтор той же сигнатуры click_xy — сигнал сменить координаты/стратегию.
    if memory is not None and memory.repeat_count >= 1:
        return (
            "ПРИОРИТЕТ ЭТОГО ШАГА: подряд повторяется тот же click_xy (та же сигнатура) без прогресса — смени x,y или выбери "
            "click/type по element_index, scroll или wait; сверь SCREEN с «Прошлый thought»."
        )
    return ""


def format_runtime_context(memory: RuntimeMemory) -> str:
    base = (
        f"repeat_count={memory.repeat_count}, scroll_streak={memory.scroll_streak}, "
        f"stagnation_steps={memory.stagnation_steps}, tried_ax_ids_count={len(memory.tried_ax_ids)}, "
        f"type_not_editable_streak={memory.type_not_editable_streak}, "
        f"search_target_miss_streak={memory.search_target_miss_streak}, "
        f"self_check_count={memory.self_check_count}, vision_recovery_count={memory.vision_recovery_count}"
    )
    if is_deep_context_mode(memory):
        return f"{base}, mode=DEEP_CONTEXT"
    return base
