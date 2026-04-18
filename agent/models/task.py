from __future__ import annotations

from enum import Enum


class TaskMode(str, Enum):
    NAVIGATION = "NAVIGATION"
    SEARCH = "SEARCH"
    COMMUNICATION = "COMMUNICATION"
    FORM_FILL = "FORM_FILL"
    SELECTION = "SELECTION"
    TRANSACTION = "TRANSACTION"
    VERIFICATION = "VERIFICATION"
    GENERIC = "GENERIC"

