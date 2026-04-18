from __future__ import annotations

from dataclasses import dataclass, field

from agent.models.action import AgentAction


@dataclass(frozen=True)
class MemoryGuardView:
    repeat_count: int = 0
    scroll_streak: int = 0
    stagnation_steps: int = 0
    last_smart_step: int = -999999
    last_sent_text: str = ""
    type_not_editable_streak: int = 0
    search_target_miss_streak: int = 0
    self_check_count: int = 0
    vision_recovery_count: int = 0


@dataclass
class RuntimeMemory:
    recent_action_signatures: list[str] = field(default_factory=list)
    repeat_count: int = 0
    scroll_streak: int = 0
    stagnation_steps: int = 0
    last_progress_score: int = 0
    tried_ax_ids: set[str] = field(default_factory=set)
    last_sent_text: str = ""
    duplicate_blocked: bool = False
    done_reason: str = ""
    sent_message_count: int = 0
    last_type_ax_id: str = ""
    last_send_click_ax_id: str = ""
    last_smart_step: int = -999999
    type_not_editable_streak: int = 0
    search_target_miss_streak: int = 0
    self_check_count: int = 0
    vision_recovery_count: int = 0
    # Подсказка от assess_goal_reached при goal_reached=False — пробрасывается в Actor до сброса.
    self_check_hint: str = ""
    # Последний успешно запланированный click по ax_id (для гардов и отладки).
    last_click_ax_id: str = ""

    def update_signature(self, signature: str) -> None:
        if self.recent_action_signatures and self.recent_action_signatures[-1] == signature:
            self.repeat_count += 1
        else:
            self.repeat_count = 0
        self.recent_action_signatures.append(signature)
        self.recent_action_signatures = self.recent_action_signatures[-10:]

    def guard_view(self) -> MemoryGuardView:
        return MemoryGuardView(
            repeat_count=self.repeat_count,
            scroll_streak=self.scroll_streak,
            stagnation_steps=self.stagnation_steps,
            last_smart_step=self.last_smart_step,
            last_sent_text=self.last_sent_text,
            type_not_editable_streak=self.type_not_editable_streak,
            search_target_miss_streak=self.search_target_miss_streak,
            self_check_count=self.self_check_count,
            vision_recovery_count=self.vision_recovery_count,
        )

    def update_after_action(self, action: AgentAction) -> None:
        if action.action == "scroll":
            self.scroll_streak += 1
        else:
            self.scroll_streak = 0

        ax_id = action.params.get("ax_id")
        if isinstance(ax_id, str) and ax_id:
            self.tried_ax_ids.add(ax_id)

        if action.action == "type":
            self.last_sent_text = str(action.params.get("text", ""))
            ax = action.params.get("ax_id")
            if isinstance(ax, str):
                self.last_type_ax_id = ax

        if action.action == "click":
            ax = action.params.get("ax_id")
            self.last_click_ax_id = str(ax) if isinstance(ax, str) else ""

