"""Crop-verify перед исполнением click/type при tier=smart (см. ``execute_phase``)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from agent.models.action import AgentAction
from agent.tools.browser_executor.constants import _SMALL_CONTROL_BBOX_MAX_PX

if TYPE_CHECKING:
    from agent.runtime.react_loop.engine.phases.context import PipelineIterationContext


async def maybe_verify_click_type_crop(
    ctx: PipelineIterationContext,
    proposed: AgentAction,
) -> Literal["ok", "retry"]:
    # Только smart-маршрут LLM: кроп вокруг bbox + YES/NO; при NO — retry итерации без клика, подсказка в self_check_hint.
    if ctx.llm is None or ctx.llm.model_route.tier != "smart":
        return "ok"
    if proposed.action not in ("click", "type"):
        return "ok"
    ax_id = proposed.params.get("ax_id")
    if not isinstance(ax_id, str) or not ax_id:
        return "ok"
    element = next((e for e in ctx.observation if e.ax_id == ax_id), None)
    if element is None:
        return "ok"
    if (
        element.bbox_doc_x is None
        or element.bbox_doc_y is None
        or element.bbox_doc_w is None
        or element.bbox_doc_h is None
    ):
        return "ok"
    hist = Path(ctx.loop.config.history_dir)
    hist.mkdir(parents=True, exist_ok=True)
    crop_path = str(hist / f"crop_verify_{ctx.global_step:03d}.png")
    bw = float(element.bbox_doc_w)
    bh = float(element.bbox_doc_h)
    # Мелкие +/- и иконки — чуть больше контекста вокруг bbox для YES/NO verify.
    crop_size = 420 if max(bw, bh) <= float(_SMALL_CONTROL_BBOX_MAX_PX) else 300
    ok = await ctx.loop.executor.screenshot_viewport_crop_around_element(
        element, size=crop_size, out_path=crop_path
    )
    if not ok:
        return "ok"
    ctx.last_crop_verify_path = crop_path
    nm = (element.name or "").strip()
    label = f"{element.role} \"{nm}\"" if nm else element.role
    try:
        yes, pt, ct = await ctx.loop.actor.verify_crop_element_target_visible_yes_no(
            crop_png_path=crop_path,
            element_label=label[:220],
            model_override=ctx.loop.config.model_smart,
            max_transport_retries=ctx.loop._settings.llm_transport_max_retries,
        )
    except Exception as exc:
        await ctx.stream_callback(f"[CROP-VERIFY] skip | error={exc!s}")
        return "ok"
    est = ctx.loop.config.pricing.estimate_cost_usd(
        prompt_tokens=pt, completion_tokens=ct, tier="smart"
    )
    ctx.cost_stats.register(tier="smart", cost_usd=est)
    if yes:
        ctx.memory.crop_verify_no_streak = 0
        await ctx.stream_callback(f"[CROP-VERIFY] YES | crop={crop_path}")
        return "ok"
    ctx.memory.crop_verify_no_streak += 1
    n = ctx.memory.crop_verify_no_streak
    if n >= 2:
        ctx.memory.self_check_hint = (
            "Crop-verify NO (повторно): критично — не повторяй тот же click/type по element_index для той же цели. "
            "Ответь action=click_xy с целыми params.x, params.y в координатах viewport SCREEN (центр цели) "
            f"или заметный scroll; контекст отказа: «{label[:80]}»."
        )[:300]
        await ctx.stream_callback(
            f"[CROP-VERIFY] NO подряд {n} раз — усилена подсказка fusion, нужен click_xy или другой ход (crop={crop_path})."
        )
        ctx.memory.crop_verify_no_streak = 0
    else:
        ctx.memory.self_check_hint = (
            f"Crop-verify NO: на фрагменте вокруг «{label[:100]}» цель не подтверждена (перекрыта или вне кадра). "
            "Сделай scroll, другой element_index или click_xy по SCREEN."
        )[:300]
    await ctx.stream_callback(f"[CROP-VERIFY] NO | crop={crop_path} — повтор шага без исполнения.")
    return "retry"
