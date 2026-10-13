"""Original interruption review is read-only, never closure adjudication."""

from copy import deepcopy

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_restart_review import inspect_restart_review
from infrastructure.research_store import ResearchStore, ResearchError, digest
from tests.test_research_store import spec


def interrupted(tmp_path, *, cost=47892, stop=True, error="SUPERVISOR_ERROR", outcome="INTERRUPTED",
                stop_cost=None, arm="pirc28-k1"):
    store = ResearchStore(tmp_path, "review-fixture", initialize=True)
    value = spec()
    value["arms"][0]["arm_id"] = arm
    value["cells"][0]["arm_id"] = arm
    store.register(value, digest(value))
    run_id = store.register_run(value["study_id"], value["cells"][0])
    attempt = store.new_attempt(run_id)
    ledger = BudgetLedger(store)
    reservation = ledger.reserve(attempt, BudgetSpec(900, category="smoke"))
    store.transition(attempt, "RUNNING")
    store.append("WORKER_STARTED", {"attempt_id": attempt,
        "reservation_id": reservation["reservation_id"], "pid": 123})
    if stop:
        store.append("WORKER_TREE_STOPPED", {"attempt_id": attempt,
            "reservation_id": reservation["reservation_id"],
            "observed_elapsed_ms": cost if stop_cost is None else stop_cost,
            "confirmation": "native-job-or-process-group-no-running-descendants"})
    ledger.settle(reservation["reservation_id"], cost, outcome=outcome)
    store.transition(attempt, outcome, error_code=error)
    return store, ledger, attempt, run_id


def test_known_stop_and_cost_produce_only_a_bound_review_candidate(tmp_path):
    store, ledger, attempt, _ = interrupted(tmp_path)
    before = store.events()
    balance = ledger.balance("pirc28-k1")
    candidate = inspect_restart_review(store, attempt)
    assert candidate["state"] == "CANDIDATE_FOR_REVIEW"
    assert candidate["execution_permitted"] is False
    assert candidate["history_mutated"] is False
    assert candidate["retained_cost_ms"] == 47892
    assert candidate["remaining_retry_count"] == 2
    assert candidate["recovery_level"] == "restart-only-no-checkpoint"
    assert len(candidate["event_hashes"]) == 6
    assert all(any(event["hash"] == reference for event in before)
               for reference in candidate["event_hashes"].values())
    assert "fresh-native-worker-identity-check" in candidate["requirements"]
    assert store.events() == before
    assert ledger.balance("pirc28-k1") == balance
    assert balance["closed"] and balance["committed_ms"] == 47892


@pytest.mark.parametrize("changes", [
    {"cost": None}, {"cost": 0}, {"cost": 900001}, {"stop": False},
    {"stop_cost": 1}, {"error": "HEARTBEAT_LOST"}, {"outcome": "TIMEOUT"},
    {"outcome": "FAILED"}, {"arm": "pirc26-m0"},
])
def test_unknown_unsafe_or_other_participant_interruptions_refuse_review(tmp_path, changes):
    store, _, attempt, _ = interrupted(tmp_path, **changes)
    before = store.events()
    with pytest.raises(ResearchError):
        inspect_restart_review(store, attempt)
    assert store.events() == before


@pytest.mark.parametrize("fault", ["hold", "other-closure", "checkpoint", "active", "retries", "missing-parent"])
def test_review_cannot_cover_other_closures_or_bypass_retry_and_live_guards(tmp_path, fault):
    store, _, attempt, run_id = interrupted(tmp_path)
    if fault == "hold":
        (store.path / "recovery-hold.json").write_text("{}", encoding="utf-8")
    elif fault == "other-closure":
        store.append("ARM_CLOSED", {"arm_id": "pirc28-k1", "reason": "TIMEOUT"})
    elif fault == "checkpoint":
        store.append("CHECKPOINT", {"attempt_id": attempt, "artifact_id": digest("fixture")})
    elif fault in {"active", "retries"}:
        count = 1 if fault == "active" else 2
        for _ in range(count):
            retry = store.new_attempt(run_id, parent_attempt_id=attempt, reason="isolated infrastructure fixture")
            if fault == "retries":
                store.transition(retry, "PREFLIGHT_FAILED", error_code="fixture")
    else:
        attempt = "missing"
    before = store.events()
    with pytest.raises(ResearchError):
        inspect_restart_review(store, attempt)
    assert store.events() == before


def test_final_completion_rechecks_new_closure_instead_of_reusing_review(tmp_path, monkeypatch):
    store, _, attempt, _ = interrupted(tmp_path)
    original = store._read_completion
    def inject(callback, finish):
        store.append("ARM_CLOSED", {"arm_id": "pirc28-k1", "reason": "TIMEOUT"})
        return original(callback, finish)
    monkeypatch.setattr(store, "_read_completion", inject)
    with pytest.raises(ResearchError, match="closure"):
        inspect_restart_review(store, attempt)
    assert BudgetLedger(store).balance("pirc28-k1")["closed"]
