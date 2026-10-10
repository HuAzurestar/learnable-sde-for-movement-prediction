"""Scoped immutable continuations/renewals; isolated stores, no research data."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_continuation import approval, renew_grant, resolve_grant, renewed_source_grant
from infrastructure.research_store import ResearchError, digest
from tests.test_research_budget import registered


def approved(store, **changes):
    now = datetime.now(timezone.utc)
    value = {"schema_version": "pirc25-continuation-approval-v1", "decision": "approved",
             "arm_seconds": 86400, "budget_mode": "cumulative-only-hard", "arm_ids": ["affine"],
             "authorization_ids": ["scope"], "test_authorization": False, "public_export": False,
             "work_started_at": (now - timedelta(minutes=1)).isoformat(),
             "work_deadline": (now + timedelta(hours=1)).isoformat(),
             "authorization_expires_at": (now + timedelta(days=2)).isoformat(), **changes}
    reference = {"manifest_id": "approval-" + digest(value), "sha256": digest(value)}
    store.publish(reference["manifest_id"], value)
    return reference


def closed(store, attempt, *, elapsed=500):
    ledger = BudgetLedger(store)
    reservation = ledger.reserve(attempt, BudgetSpec(1))
    ledger.settle(reservation["reservation_id"], elapsed, outcome="TIMEOUT")
    store.transition(attempt, "TIMEOUT", error_code="TIMEOUT")
    assert ledger.balance("affine")["closed"]
    return ledger


def test_continuation_keeps_cost_and_history_and_soft_timeout_can_continue(tmp_path):
    store, attempts = registered(tmp_path, 3)
    ledger = closed(store, attempts[0])
    before = store.events()
    ref = approved(store)
    continuation = ledger.continue_arm("affine", approval_ref=ref)
    assert store.events()[:len(before)] == before
    assert ledger.balance("affine") == {"arm_id": "affine", "remaining_ms": 86399500,
                                        "committed_ms": 500, "closed": False}
    assert ledger.continue_arm("affine", approval_ref=ref) == continuation
    count = len(store.events())
    assert ledger.continue_arm("affine", approval_ref=ref) == continuation
    assert len(store.events()) == count
    reservation = ledger.reserve(attempts[1], BudgetSpec(1))
    ledger.settle(reservation["reservation_id"], 800, outcome="TIMEOUT")
    store.transition(attempts[1], "TIMEOUT", error_code="TIMEOUT")
    assert not ledger.balance("affine")["closed"]
    assert ledger.balance("affine")["committed_ms"] == 1300
    assert ledger.reserve(attempts[2], BudgetSpec(1))
    restarted = BudgetLedger(type(store)(tmp_path, store.store_id))
    assert restarted.balance("affine") == ledger.balance("affine")


def test_unknown_stop_still_closes_continued_arm_and_replay_does_not_reopen(tmp_path):
    store, attempts = registered(tmp_path, 2)
    ledger = closed(store, attempts[0])
    ref = approved(store)
    ledger.continue_arm("affine", approval_ref=ref)
    reservation = ledger.reserve(attempts[1], BudgetSpec(1))
    ledger.recover_unknown(reservation["reservation_id"])
    assert ledger.balance("affine")["closed"]
    ledger.continue_arm("affine", approval_ref=ref)
    assert ledger.balance("affine")["closed"]


@pytest.mark.parametrize("fault", ["unsettled", "cap", "scope", "hold", "stop", "live"])
def test_continuation_rejects_unsafe_or_unapproved_reopening(tmp_path, monkeypatch, fault):
    store, attempts = registered(tmp_path)
    ledger = BudgetLedger(store)
    reservation = ledger.reserve(attempts[0], BudgetSpec(1))
    if fault != "unsettled":
        ledger.settle(reservation["reservation_id"], 86400000 if fault == "cap" else 500,
                      outcome="TIMEOUT")
        store.transition(attempts[0], "TIMEOUT", error_code="TIMEOUT")
    ref = approved(store, arm_ids=["other"] if fault == "scope" else ["affine"])
    if fault == "hold":
        (store.path / "recovery-hold.json").write_text("{}", encoding="utf8")
    if fault in {"stop", "live"}:
        store.append("WORKER_STARTED", {"reservation_id": reservation["reservation_id"],
                                       "attempt_id": attempts[0], "pid": 123})
        if fault == "live":
            store.append("WORKER_TREE_STOPPED", {"reservation_id": reservation["reservation_id"]})
            monkeypatch.setattr("infrastructure.process_tree.process_may_be_alive", lambda pid: True)
    with pytest.raises(ResearchError):
        ledger.continue_arm("affine", approval_ref=ref)
    assert not any(e["event_kind"] == "ARM_CONTINUATION" for e in store.events())


def test_expired_work_window_cannot_authorize_continuation(tmp_path):
    store, attempts = registered(tmp_path)
    ledger = closed(store, attempts[0])
    now = datetime.now(timezone.utc)
    ref = approved(store, work_started_at=(now - timedelta(hours=2)).isoformat(),
                   work_deadline=(now - timedelta(hours=1)).isoformat())
    with pytest.raises(ResearchError, match="expired"):
        ledger.continue_arm("affine", approval_ref=ref)


def test_soft_allocation_requires_approval_and_cumulative_cap_still_closes(tmp_path):
    store, attempts = registered(tmp_path, 3)
    ledger = BudgetLedger(store)
    with pytest.raises(ResearchError, match="approval|approved"):
        ledger.reserve(attempts[0], BudgetSpec(7201, allocation_only=True))
    ledger = closed(store, attempts[0])
    ledger.continue_arm("affine", approval_ref=approved(store))
    reservation = ledger.reserve(attempts[1], BudgetSpec(7201, allocation_only=True))
    ledger.settle(reservation["reservation_id"], 86400000 - 500, outcome="TIMEOUT")
    assert ledger.balance("affine")["committed_ms"] == 86400000
    assert ledger.balance("affine")["closed"]
    with pytest.raises(ResearchError, match="BUDGET_EXHAUSTED"):
        ledger.reserve(attempts[2], BudgetSpec(1, allocation_only=True))
    with pytest.raises(ResearchError, match="BUDGET_EXHAUSTED"):
        ledger.continue_arm("affine", approval_ref=approved(store))


def grant(store):
    value = {"authorization_id": "scope", "version": "v1", "study_id": "synthetic",
             "purposes": ["evaluate"], "visibilities": ["synthetic"], "block_ids": ["fixture"],
             "protocol_hash": "a" * 64, "test_authorization": False, "evidence_hash": "b" * 64,
             "expires_at": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()}
    store.authorize(value)
    return value


def test_renewal_preserves_old_scope_and_requires_exact_explicit_reference(tmp_path):
    store, _ = registered(tmp_path)
    original = grant(store)
    ref = approved(store)
    renewal = renew_grant(store, "scope", old_version="v1", new_version="v2", approval_ref=ref)
    current = resolve_grant(store, original, renewal)
    assert store.authorization("scope", version="v1") == original
    assert current == {**original, "version": "v2", "evidence_hash": ref["sha256"],
                       "expires_at": approval(store, ref)["authorization_expires_at"]}
    assert renewed_source_grant(store, original, {digest(original): renewal}) == current
    with pytest.raises(ResearchError):
        renewed_source_grant(store, original, {})
    before = store.events()
    assert renew_grant(store, "scope", old_version="v1", new_version="v2", approval_ref=ref) == renewal
    assert store.events() == before


@pytest.mark.parametrize("fault", ["hash", "source", "scope", "unregistered", "test"])
def test_renewal_cannot_change_hash_source_scope_or_test_authority(tmp_path, fault):
    store, _ = registered(tmp_path)
    original = grant(store)
    ref = approved(store, authorization_ids=[] if fault == "scope" else ["scope"])
    if fault == "scope":
        with pytest.raises(ResearchError):
            renew_grant(store, "scope", old_version="v1", new_version="v2", approval_ref=ref)
        return
    renewal = renew_grant(store, "scope", old_version="v1", new_version="v2", approval_ref=ref)
    changed, reference = deepcopy(original), deepcopy(renewal)
    if fault == "hash":
        reference["sha256"] = "f" * 64
    elif fault == "source":
        changed["protocol_hash"] = "f" * 64
    elif fault == "test":
        changed["test_authorization"] = True
    else:
        reference["manifest_id"] = "missing"
    with pytest.raises(ResearchError):
        resolve_grant(store, changed, reference)
