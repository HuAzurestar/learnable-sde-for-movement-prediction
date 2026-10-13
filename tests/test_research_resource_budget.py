"""Opt-in shared accounting, concurrency and crash holds in disposable stores."""

from concurrent.futures import ThreadPoolExecutor
import sys

import pytest

from application.research_budget import BudgetLedger, BudgetSpec, ResourceBudgetSpec
from application.research_supervisor import ResearchSupervisor
from infrastructure.research_store import ResearchStore, ResearchError, digest
from tests.test_research_store import spec
from tests.test_research_execution_policy import policy


def registered(tmp_path, count=2):
    store = ResearchStore(tmp_path, "policy-fixture", initialize=True)
    value = spec()
    value["code_hash"] = "a" * 64
    value["arms"][0]["arm_id"] = "pirc28-k2"
    value["cells"] = [{"arm_id": "pirc28-k2", "seed": i, "block_id": "fixture",
                       "visibility": "synthetic"} for i in range(count)]
    store.register(value, digest(value))
    attempts = [store.new_attempt(store.register_run("synthetic", cell)) for cell in value["cells"]]
    decision = policy()
    decision["resources"].update(minimum_system_free_bytes=1,
        minimum_runtime_free_bytes=1, minimum_available_memory_bytes=1,
        maximum_worker_rss_bytes=2**40)
    reference = {"manifest_id": "policy-" + digest(decision), "sha256": digest(decision)}
    store.publish(reference["manifest_id"], decision)
    return store, attempts, ResourceBudgetSpec(reference, "a" * 64)


def observation(**changes):
    return {"system_free_bytes": 2**40, "runtime_free_bytes": 2**40,
            "available_memory_bytes": 2**40, "worker_rss_bytes": 1024, **changes}


def balance(ledger, budget):
    return ledger.balance("pirc28-k2", execution_policy=budget.execution_policy,
                          source_code_hash=budget.source_code_hash)


def test_actual_cost_beyond_old_caps_preserves_old_charges_and_policy(tmp_path):
    store, attempts, budget = registered(tmp_path)
    ledger = BudgetLedger(store)
    first = ledger.reserve(attempts[0], BudgetSpec(1))
    ledger.settle(first["reservation_id"], 19, outcome="FAILED")
    reservation = ledger.reserve(attempts[1], budget)
    assert reservation["reserved_ms"] is None
    assert balance(ledger, budget)["remaining_ms"] is None
    assert balance(ledger, budget)["committed_ms"] is None  # Running cost unknown, never zero.
    ledger.settle(reservation["reservation_id"], 90_000_000, outcome="SUCCEEDED")
    reopened = BudgetLedger(ResearchStore(tmp_path, "policy-fixture"))
    assert balance(reopened, budget) == {"arm_id": "pirc28-k2", "remaining_ms": None,
                                      "committed_ms": 90_000_019, "closed": False}
    assert reopened.balance("pirc28-k2")["closed"]  # Legacy caller cannot borrow the opt-in.
    assert ledger.reserve(attempts[1], budget)["charged_ms"] == 90_000_000


def test_concurrent_resource_reservations_have_one_durable_winner(tmp_path):
    store, attempts, budget = registered(tmp_path, 4)
    def reserve(attempt):
        try:
            return BudgetLedger(store).reserve(attempt, budget)
        except ResearchError as exc:
            assert exc.code == "BUDGET_BUSY"
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(reserve, attempts))
    assert sum(value is not None for value in results) == 1
    assert len([event for event in store.events() if event["event_kind"] == "RESERVE"]) == 1
    assert all(value["state"] == "REGISTERED" for value in store.attempts().values())


def test_replay_binds_policy_and_cannot_duplicate_reservation(tmp_path):
    store, attempts, budget = registered(tmp_path)
    ledger = BudgetLedger(store)
    reservation = ledger.reserve(attempts[0], budget)
    before = store.events()
    assert ledger.reserve(attempts[0], budget) == reservation
    assert store.events() == before
    with pytest.raises(ResearchError):
        ledger.reserve(attempts[0], ResourceBudgetSpec({**budget.execution_policy, "sha256": "f"*64}, "a"*64))
    assert store.events() == before


@pytest.mark.parametrize("reason", ["INTERRUPTED", "TIMEOUT", "BUDGET_EXHAUSTED"])
def test_policy_does_not_clear_any_historical_closure(tmp_path, reason):
    store, attempts, budget = registered(tmp_path)
    ledger = BudgetLedger(store)
    original = ledger.reserve(attempts[0], BudgetSpec(1))
    ledger.settle(original["reservation_id"], 400, outcome=reason)
    before = store.events()
    with pytest.raises(ResearchError) as error:
        ledger.reserve(attempts[1], budget)
    assert error.value.code == "BUDGET_EXHAUSTED"
    assert store.events() == before and balance(ledger, budget)["closed"]
    assert balance(ledger, budget)["committed_ms"] == 400


def test_unknown_unbounded_cost_cannot_be_settled_at_zero_or_release_worker(tmp_path):
    store, attempts, budget = registered(tmp_path)
    ledger = BudgetLedger(store)
    reservation = ledger.reserve(attempts[0], budget)
    with pytest.raises(ResearchError) as error:
        ledger.settle(reservation["reservation_id"], None, outcome="INTERRUPTED")
    assert error.value.code == "RECOVERY_REQUIRED"
    assert not ledger._state()[0][reservation["reservation_id"]]["settled"]
    assert balance(ledger, budget)["closed"] and balance(ledger, budget)["committed_ms"] is None
    before = store.events()
    with pytest.raises(ResearchError):
        ledger.recover_unknown(reservation["reservation_id"], stop_evidence_hash=digest("stopped"))
    assert store.events() == before


@pytest.mark.parametrize("fault", ["source", "slot", "hold", "missing-publication"])
def test_refusal_precedes_new_funding(tmp_path, fault):
    store, attempts, budget = registered(tmp_path)
    options = {}
    if fault == "source":
        budget = ResourceBudgetSpec(budget.execution_policy, "b"*64)
    elif fault == "slot":
        options["worker_slot"] = True
    elif fault == "hold":
        (store.path / "recovery-hold.json").write_text("{}")
    else:
        budget = ResourceBudgetSpec({"manifest_id": "absent", "sha256": "a"*64}, "a"*64)
    with pytest.raises(ResearchError):
        BudgetLedger(store).reserve(attempts[0], budget, **options)
    # Read events directly with no mutation; hold does not hide authoritative history.
    assert not any(event["event_kind"] == "RESERVE" for event in store.events())


def test_unbounded_policy_cannot_be_used_on_a_sibling_arm(tmp_path):
    from tests.test_research_budget import registered as legacy_registered
    store, attempts = legacy_registered(tmp_path)
    value = policy(store.store_id)
    ref = {"manifest_id": "policy-"+digest(value), "sha256": digest(value)}
    store.publish(ref["manifest_id"], value)
    with pytest.raises(ResearchError) as error:
        BudgetLedger(store).reserve(attempts[0], ResourceBudgetSpec(ref, "a"*64))
    assert error.value.code == "UNAUTHORIZED_DATA"
    assert BudgetLedger(store).balance("affine")["committed_ms"] == 0


def test_supervisor_runs_without_deadline_and_records_actual_cost(tmp_path, monkeypatch):
    store, attempts, budget = registered(tmp_path)
    monkeypatch.setattr("application.research_supervisor.observe_resources", lambda **_: observation())
    result = ResearchSupervisor(store).run(attempts[0], lambda output: [sys.executable, "-c",
        "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('{\"metric\":1}')", str(output)], budget)
    assert result["state"] == "SUCCEEDED" and result["elapsed_ms"] >= 0
    events = store.events()
    started = [event["payload"] for event in events if event["event_kind"] == "WORKER_STARTED"]
    assert len(started) == 1 and started[0]["deadline_monotonic"] is None
    assert started[0]["resource_observation"] == observation()
    settled = [event["payload"] for event in events if event["event_kind"] == "SETTLE"]
    assert len(settled) == 1 and settled[0]["reserved_ms"] is None
    assert settled[0]["charged_ms"] == result["elapsed_ms"]


def test_resource_drop_stops_tree_and_closes_policy_arm(tmp_path, monkeypatch):
    store, attempts, budget = registered(tmp_path)
    calls = 0
    def observe(**_):
        nonlocal calls
        calls += 1
        return observation(system_free_bytes=0) if calls >= 3 else observation()
    monkeypatch.setattr("application.research_supervisor.observe_resources", observe)
    result = ResearchSupervisor(store).run(attempts[0],
        lambda _: [sys.executable, "-c", "import time; time.sleep(30)"], budget)
    assert result["state"] == "INTERRUPTED" and result["elapsed_ms"] < 10_000
    attempt = store.attempts()[attempts[0]]
    assert attempt["error_code"] == "RESOURCE_LIMIT"
    assert balance(BudgetLedger(store), budget)["closed"]
