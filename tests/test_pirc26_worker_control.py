"""Synthetic owner-frame refusal tests; actual launches live in runtime tests."""

import os
import time

import pytest

from application.research_budget import BudgetSpec, BudgetLedger
from infrastructure.pirc26_worker_control import require_owned_worker
from infrastructure.research_store import ResearchError
from tests.test_pirc26_runtime import prepare, owner_admit


@pytest.mark.parametrize("fault", ["missing-worker", "parent", "hold", "settled", "state", "expired"])
def test_restart_only_worker_refuses_unowned_or_unfunded_process_frames(tmp_path, fault):
    store, value, plugin, _, _, _, _ = prepare(tmp_path, family="M1-R")
    output, receipt = owner_admit(store, value, plugin)
    reservation = BudgetLedger(store).reserve(receipt["attempt_id"], BudgetSpec(10))
    store.transition(receipt["attempt_id"], "RUNNING")
    if fault != "missing-worker":
        store.append("WORKER_STARTED", {"attempt_id": receipt["attempt_id"],
            "pid": os.getppid() + (1 if fault == "parent" else 0),
            "reservation_id": "missing-hold" if fault == "hold" else reservation["reservation_id"],
            "deadline_monotonic": time.monotonic() + (-1 if fault == "expired" else 10)})
    if fault == "settled":
        BudgetLedger(store).settle(reservation["reservation_id"], 1000, outcome="FAILED")
    if fault == "state":
        store.transition(receipt["attempt_id"], "FAILED", error_code="SYNTHETIC_OWNER_FRAME")
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        require_owned_worker(output)


def test_restart_only_soft_request_uses_original_owner_deadline_without_mutating_ledger(tmp_path, monkeypatch):
    store, value, plugin, _, _, _, _ = prepare(tmp_path, family="M1-S")
    output, receipt = owner_admit(store, value, plugin)
    reservation = BudgetLedger(store).reserve(receipt["attempt_id"], BudgetSpec(10))
    store.transition(receipt["attempt_id"], "RUNNING")
    deadline = time.monotonic() + 10
    store.append("WORKER_STARTED", {"attempt_id": receipt["attempt_id"], "pid": os.getppid(),
        "reservation_id": reservation["reservation_id"], "deadline_monotonic": deadline})
    before = store.events()
    ownership = require_owned_worker(output)
    assert ownership.soft_threshold == deadline - 2
    monkeypatch.setattr("infrastructure.pirc26_worker_control.time.monotonic", lambda: deadline - 2.001)
    assert not ownership.requested()
    monkeypatch.setattr("infrastructure.pirc26_worker_control.time.monotonic", lambda: deadline - 2.)
    assert ownership.requested()
    assert store.events() == before
