"""Disposable NON-NATIVE ledger controls; durations are simulated, not research costs."""

import json

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from infrastructure.research_store import ResearchError, ResearchStore, digest
from tests.test_research_budget import registered
from tests.test_research_store import spec


def settled_capacity(tmp_path):
    store, attempts = registered(tmp_path, count=15)
    ledger = BudgetLedger(store)
    # Model 85,800 settled seconds. No OS worker or actual elapsed time exists.
    for index, seconds in enumerate([7200] * 11 + [6600]):
        reservation = ledger.reserve(attempts[index], BudgetSpec(seconds))
        ledger.settle(reservation["reservation_id"], seconds * 1000, outcome="FAILED")
        store.transition(attempts[index], "FAILED")
    assert ledger.balance("affine") == {
        "arm_id": "affine", "remaining_ms": 600000,
        "committed_ms": 85800000, "closed": False,
    }
    return store, attempts, ledger


def independent_attempt(store):
    value = spec()
    value["study_id"] = "independent-study"
    value["arms"] = [{**value["arms"][0], "arm_id": "independent",
                      "model_family_id": "independent-model"}]
    value["cells"] = [{"arm_id": "independent", "seed": 0,
                       "block_id": "independent-fixture", "visibility": "synthetic"}]
    store.register(value, digest(value))
    return store.new_attempt(store.register_run(value["study_id"], value["cells"][0]))


def try_tiny(ledger, attempt):
    try:
        reservation = ledger.reserve(attempt, BudgetSpec(1))
    except ResearchError as exc:
        return {"result": exc.code}
    return {"result": "FUNDED", "reservation_id": reservation["reservation_id"]}


def test_paid_use_refusal_fuses_reopened_arm_without_charging_denied_work(tmp_path):
    store, attempts, ledger = settled_capacity(tmp_path)
    other = independent_attempt(store)
    with pytest.raises(ResearchError) as refusal:
        ledger.reserve(attempts[12], BudgetSpec(7200))
    assert refusal.value.code == "BUDGET_EXHAUSTED"
    assert store.attempts()[attempts[12]]["state"] == "BUDGET_EXHAUSTED"
    rejected_id = digest([store.store_id, attempts[12]])
    assert rejected_id not in ledger._state()[0]
    reopened = BudgetLedger(ResearchStore(tmp_path, store.store_id))
    after_refusal = reopened.balance("affine")
    tiny = try_tiny(reopened, attempts[13])
    independent = try_tiny(reopened, other)
    print(json.dumps({"fixture": str(tmp_path), "simulated_cost_ms": 85800000,
                      "after_refusal": after_refusal, "tiny": tiny,
                      "independent": independent}, sort_keys=True))
    assert independent["result"] == "FUNDED"
    assert after_refusal["committed_ms"] == 85800000
    assert after_refusal["remaining_ms"] == 600000
    assert after_refusal["closed"] and tiny["result"] == "BUDGET_EXHAUSTED"
    assert reopened.balance("affine")["committed_ms"] == 85800000
    events = store.events()
    closes = [e for e in events if e["event_kind"] == "ARM_CLOSED"
              and e["payload"]["arm_id"] == "affine"]
    rejected = [e for e in events if e["event_kind"] == "ATTEMPT"
                and e["payload"].get("attempt_id") == attempts[12]
                and e["payload"].get("state") == "BUDGET_EXHAUSTED"]
    assert len(closes) == 1 and len(rejected) == 1
    assert closes[0]["sequence"] < rejected[0]["sequence"]


def test_refusal_transition_interruption_keeps_durable_arm_fuse(tmp_path, monkeypatch):
    store, attempts, ledger = settled_capacity(tmp_path)
    original = store._transition

    def interrupted(attempt_id, state, **kwargs):
        if attempt_id == attempts[12] and state == "BUDGET_EXHAUSTED":
            raise OSError("simulated rejection transition interruption")
        return original(attempt_id, state, **kwargs)

    monkeypatch.setattr(store, "_transition", interrupted)
    with pytest.raises(OSError, match="simulated rejection transition interruption"):
        ledger.reserve(attempts[12], BudgetSpec(7200))
    reopened = BudgetLedger(ResearchStore(tmp_path, store.store_id))
    after_interruption = reopened.balance("affine")
    tiny = try_tiny(reopened, attempts[13])
    print(json.dumps({"fixture": str(tmp_path), "after_interruption": after_interruption,
                      "tiny": tiny}, sort_keys=True))
    assert after_interruption["committed_ms"] == 85800000
    assert after_interruption["closed"] and tiny["result"] == "BUDGET_EXHAUSTED"
    assert digest([store.store_id, attempts[12]]) not in reopened._state()[0]


def test_unsettled_capacity_busy_is_not_a_hard_fuse_and_releases_normally(tmp_path):
    store, attempts = registered(tmp_path, count=14)
    ledger = BudgetLedger(store)
    reservations = [ledger.reserve(attempt, BudgetSpec(7200)) for attempt in attempts[:12]]
    with pytest.raises(ResearchError) as refusal:
        ledger.reserve(attempts[12], BudgetSpec(7200))
    assert refusal.value.code == "BUDGET_BUSY"
    assert store.attempts()[attempts[12]]["state"] == "PREFLIGHT_FAILED"
    assert digest([store.store_id, attempts[12]]) not in ledger._state()[0]
    assert ledger.balance("affine")["remaining_ms"] == 0
    assert not ledger.balance("affine")["closed"]
    ledger.settle(reservations[0]["reservation_id"], 1000, outcome="FAILED")
    store.transition(attempts[0], "FAILED")
    reopened = BudgetLedger(ResearchStore(tmp_path, store.store_id))
    assert reopened.balance("affine")["remaining_ms"] == 7199000
    assert not reopened.balance("affine")["closed"]
    assert try_tiny(reopened, attempts[13])["result"] == "FUNDED"
    assert not reopened.balance("affine")["closed"]
    assert not any(e["event_kind"] == "ARM_CLOSED" for e in store.events())


def test_positive_leftover_can_fund_small_work_before_any_hard_refusal(tmp_path):
    store, attempts, ledger = settled_capacity(tmp_path)
    reservation = ledger.reserve(attempts[12], BudgetSpec(1))
    ledger.settle(reservation["reservation_id"], 1000, outcome="FAILED")
    store.transition(attempts[12], "FAILED")
    reopened = BudgetLedger(ResearchStore(tmp_path, store.store_id))
    assert reopened.balance("affine")["committed_ms"] == 85801000
    assert reopened.balance("affine")["remaining_ms"] == 599000
    assert not reopened.balance("affine")["closed"]
    assert not any(e["event_kind"] == "ARM_CLOSED" for e in store.events())
