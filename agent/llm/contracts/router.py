from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelRoute:
    tier: str
    model: str
    reason: str
