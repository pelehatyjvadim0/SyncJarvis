from __future__ import annotations

from dataclasses import dataclass

from agent.config.settings import AgentPricing


@dataclass
class LoopConfig:
    model_cheap: str
    model_smart: str
    temperature: float
    pricing: AgentPricing
    history_dir: str = "history"
    smart_cooldown_steps: int = 2
    # Лимит подряд шагов CAPTCHA_WAIT до состояния BLOCKED_CAPTCHA.
    captcha_max_consecutive_waits: int = 20


@dataclass
class RunCostStats:
    total_cost_usd: float = 0.0
    cheap_steps: int = 0
    smart_steps: int = 0
    llm_steps: int = 0

    def register(self, tier: str, cost_usd: float) -> None:
        self.total_cost_usd += max(0.0, cost_usd)
        self.llm_steps += 1
        if tier == "cheap":
            self.cheap_steps += 1
        elif tier == "smart":
            self.smart_steps += 1
