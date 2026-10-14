"""Explicit soft delivery dates do not become new budget/data permissions."""
from datetime import datetime, timedelta, timezone
import sys

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_continuation import approval
from application.research_supervisor import ResearchSupervisor
from infrastructure.research_store import ResearchError
from tests.test_research_budget import registered
from tests.test_research_continuation import approved, closed


def overdue(store, **changes):
    now = datetime.now(timezone.utc)
    return approved(store, work_started_at=(now-timedelta(hours=2)).isoformat(),
        work_deadline=(now-timedelta(hours=1)).isoformat(),
        work_deadline_policy="report-only", **changes)


def test_explicit_report_only_continuation_retains_deadline_fees_and_restart(tmp_path):
    store, attempts = registered(tmp_path, 2)
    ledger = closed(store, attempts[0])
    decision = overdue(store)
    before = store.events()
    ledger.continue_arm("affine", approval_ref=decision)
    value = approval(store, decision, live=True)
    assert datetime.fromisoformat(value["work_deadline"]) < datetime.now(timezone.utc)
    reservation = ledger.reserve(attempts[1], BudgetSpec(2, allocation_only=True))
    assert reservation["work_deadline"] == value["work_deadline"]
    assert reservation["execution_deadline"] == value["authorization_expires_at"]
    assert ledger.balance("affine")["committed_ms"] == 2500
    assert store.events()[:len(before)] == before
    restarted = BudgetLedger(type(store)(tmp_path, store.store_id))
    assert restarted.balance("affine") == ledger.balance("affine")
    assert restarted._continuation_policy("affine") == value


def test_report_only_does_not_renew_expired_data_or_remove_cumulative_cap(tmp_path):
    store, attempts = registered(tmp_path, 2)
    ledger = closed(store, attempts[0])
    ref = overdue(store, authorization_expires_at=(datetime.now(timezone.utc)-timedelta(minutes=1)).isoformat())
    with pytest.raises(ResearchError, match="expired"):
        ledger.continue_arm("affine", approval_ref=ref)
    ledger.continue_arm("affine", approval_ref=overdue(store))
    reservation = ledger.reserve(attempts[1], BudgetSpec(1, allocation_only=True))
    ledger.settle(reservation["reservation_id"], 86400000-500, outcome="TIMEOUT")
    assert ledger.balance("affine")["closed"]
    assert ledger.balance("affine")["committed_ms"] == 86400000
    with pytest.raises(ResearchError, match="BUDGET_EXHAUSTED"):
        ledger.continue_arm("affine", approval_ref=overdue(store))


def test_actual_worker_can_finish_after_report_deadline(tmp_path):
    store, attempts = registered(tmp_path, 2)
    ledger = closed(store, attempts[0])
    ledger.continue_arm("affine", approval_ref=overdue(store))
    outcome = ResearchSupervisor(store).run(attempts[1], lambda output: [sys.executable, "-c",
        "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('{\"synthetic\":true}')", str(output)],
        BudgetSpec(10, allocation_only=True))
    assert outcome["state"] == "SUCCEEDED", outcome
    assert not ledger.balance("affine")["closed"]


def test_unknown_deadline_policy_is_not_an_implicit_bypass(tmp_path):
    store, _ = registered(tmp_path)
    ref = approved(store, work_deadline_policy="ignore-everything")
    with pytest.raises(ResearchError, match="policy"):
        approval(store, ref, live=True)
