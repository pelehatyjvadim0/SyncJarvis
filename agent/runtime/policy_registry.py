from __future__ import annotations

from agent.models.task import TaskMode
from agent.policies.base import BaseTaskPolicy
from agent.policies.communication import CommunicationPolicy
from agent.policies.form_fill import FormFillPolicy
from agent.policies.generic import GenericPolicy
from agent.policies.navigation import NavigationPolicy
from agent.policies.search import SearchPolicy
from agent.policies.selection import SelectionPolicy
from agent.policies.transaction import TransactionPolicy
from agent.policies.verification import VerificationPolicy


def default_policies() -> dict[TaskMode, BaseTaskPolicy]:
    return {
        TaskMode.NAVIGATION: NavigationPolicy(),
        TaskMode.SEARCH: SearchPolicy(),
        TaskMode.COMMUNICATION: CommunicationPolicy(),
        TaskMode.FORM_FILL: FormFillPolicy(),
        TaskMode.SELECTION: SelectionPolicy(),
        TaskMode.TRANSACTION: TransactionPolicy(),
        TaskMode.VERIFICATION: VerificationPolicy(),
        TaskMode.GENERIC: GenericPolicy(),
    }
