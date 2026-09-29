"""Budget replay/concurrency and actual OS-contained timeout checks."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_supervisor import ResearchSupervisor
from infrastructure.research_store import ResearchStore, ResearchError, digest
from tests.test_research_store import spec


def registered(tmp_path, count=1):
    store = ResearchStore(tmp_path, "budget-fixture", initialize=True)
    value = spec()
    value["cells"] = [{"arm_id": "affine", "seed": i, "block_id": "fixture", "visibility": "synthetic"} for i in range(count)]
    store.register(value, digest(value))
    attempts = [store.new_attempt(store.register_run("synthetic", cell)) for cell in value["cells"]]
    return store, attempts


def test_concurrent_reservations_never_overallocate_and_restart_keeps_cost(tmp_path):
    store, attempts = registered(tmp_path, 13)
    ledger = BudgetLedger(store)

    def reserve(attempt):
        try:
            return ledger.reserve(attempt, BudgetSpec())
        except ResearchError as exc:
            assert exc.code == "BUDGET_BUSY"
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        reservations = list(pool.map(reserve, attempts))
    assert sum(r is not None for r in reservations) == 12
    assert ledger.balance("affine")["remaining_ms"] == 0
    reopened = BudgetLedger(ResearchStore(tmp_path, "budget-fixture"))
    assert reopened.balance("affine") == ledger.balance("affine")
    first = next(r for r in reservations if r)
    assert ledger.reserve(first["attempt_id"], BudgetSpec()) == first
    ledger.settle(first["reservation_id"], 1000, outcome="FAILED")
    ledger.settle(first["reservation_id"], 1000, outcome="FAILED")
    assert ledger.balance("affine")["remaining_ms"] == 7199000
    with pytest.raises(ResearchError, match="IDENTITY_CONFLICT"):
        ledger.settle(first["reservation_id"], 0, outcome="FAILED")


def test_unknown_worker_charges_upper_bound_and_closes_only_its_arm(tmp_path):
    store, attempts = registered(tmp_path)
    ledger = BudgetLedger(store)
    reservation = ledger.reserve(attempts[0], BudgetSpec(1))
    ledger.recover_unknown(reservation["reservation_id"])
    assert ledger.balance("affine")["committed_ms"] == 1000
    assert ledger.balance("affine")["closed"]
    assert not ledger.balance("unaffected")["closed"]
    assert store.attempts()[attempts[0]]["state"] == "INTERRUPTED"


def test_unknown_recovery_never_releases_a_live_worker_and_requires_stop_evidence(tmp_path):
    store, attempts = registered(tmp_path)
    ledger = BudgetLedger(store)
    reservation = ledger.reserve(attempts[0], BudgetSpec(5))
    store.transition(attempts[0], "RUNNING")
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                               start_new_session=os.name != "nt")
    try:
        store.append("WORKER_STARTED", {"attempt_id": attempts[0], "pid": process.pid,
                                      "reservation_id": reservation["reservation_id"]})
        with pytest.raises(ResearchError, match="WORKER_ACTIVE"):
            ledger.recover_unknown(reservation["reservation_id"], stop_evidence_hash=digest("premature"))
        assert not ledger._state()[0][reservation["reservation_id"]]["settled"]
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=5)
    with pytest.raises(ResearchError, match="RECOVERY_REQUIRED"):
        ledger.recover_unknown(reservation["reservation_id"])
    proof = digest("owned child terminated and reaped; no descendants launched")
    result = ledger.recover_unknown(reservation["reservation_id"], stop_evidence_hash=proof)
    assert result["charged_ms"] == 5000 and ledger.balance("affine")["closed"]
    assert any(e["event_kind"] == "WORKER_STOP_CONFIRMED" and e["payload"]["stop_evidence_hash"] == proof
               for e in store.events())
    assert ledger.recover_unknown(reservation["reservation_id"]) == result


def test_incomplete_launch_metadata_needs_explicit_audited_reconciliation(tmp_path):
    store, attempts = registered(tmp_path)
    ledger = BudgetLedger(store)
    reservation = ledger.reserve(attempts[0], BudgetSpec(5))
    store.transition(attempts[0], "RUNNING")
    with pytest.raises(ResearchError, match="RECOVERY_REQUIRED"):
        ledger.recover_unknown(reservation["reservation_id"])
    ledger.recover_unknown(reservation["reservation_id"], stop_evidence_hash=digest("verified no worker was launched"))
    assert store.attempts()[attempts[0]]["state"] == "INTERRUPTED"
    with pytest.raises(ResearchError, match="terminal"):
        store.transition(attempts[0], "RUNNING")


@pytest.mark.parametrize("budget", [BudgetSpec(7201), BudgetSpec(901, category="smoke"), BudgetSpec(1801, category="pilot"), BudgetSpec(float("nan")), BudgetSpec(1, arm_seconds=86401)])
def test_budget_hard_caps_cannot_be_overridden(budget):
    with pytest.raises(ResearchError):
        budget.validate()


def test_real_hung_worker_is_terminated_and_timeout_persisted(tmp_path):
    store, attempts = registered(tmp_path)
    supervisor = ResearchSupervisor(store)
    result = supervisor.run(attempts[0], lambda output: [sys.executable, "-c", "import time; time.sleep(30)"], BudgetSpec(0.5))
    assert result["state"] == "TIMEOUT" and result["exit_code"] != 0
    assert result["elapsed_ms"] < 10000
    assert store.attempts()[attempts[0]]["state"] == "TIMEOUT"
    assert supervisor.budget.balance("affine")["closed"]


def test_successful_worker_publishes_before_success_and_charges_once(tmp_path):
    store, attempts = registered(tmp_path)
    result = ResearchSupervisor(store).run(attempts[0], lambda output: [sys.executable, "-c",
        "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('{\"metric\":1}')", str(output)], BudgetSpec(5))
    assert result["state"] == "SUCCEEDED"
    assert result["artifact_id"]
    with pytest.raises(ResearchError, match="settled attempt"):
        ResearchSupervisor(store).run(attempts[0], lambda _: [sys.executable], BudgetSpec(5))


def test_timeout_kills_descendants_not_only_worker(tmp_path):
    store, attempts = registered(tmp_path)
    marker = tmp_path / "escaped.txt"
    child = "import time,pathlib,sys; time.sleep(2); pathlib.Path(sys.argv[1]).write_text('escaped')"
    parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]); time.sleep(30)"
    result = ResearchSupervisor(store).run(attempts[0], lambda output: [sys.executable, "-c", parent, child, str(marker)], BudgetSpec(0.8))
    assert result["state"] == "TIMEOUT"
    time.sleep(2)
    assert not marker.exists()


def test_wrapper_watchdog_enforces_deadline_without_parent_polling(tmp_path):
    from infrastructure.process_tree import ProcessTree

    marker = tmp_path / "escaped.txt"
    wrapper = Path(__file__).resolve().parents[1] / "infrastructure/research_worker.py"
    command = "import time,pathlib,sys; time.sleep(3); pathlib.Path(sys.argv[1]).write_text('escaped')"
    process = subprocess.Popen([sys.executable, str(wrapper), str(tmp_path / "heartbeat"),
                                str(time.monotonic() + 0.6), sys.executable, "-c", command, str(marker)],
                               stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               start_new_session=os.name != "nt",
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    tree = ProcessTree(process)
    try:
        process.stdin.write(b"GO\n")
        process.stdin.flush()
        assert process.wait(timeout=5) != 0
        time.sleep(3)
        assert not marker.exists()
    finally:
        tree.terminate()
        tree.close()
        process.stdin.close()


def test_settlement_crash_still_closes_arm_from_authoritative_cost_event(tmp_path, monkeypatch):
    store, attempts = registered(tmp_path)
    ledger = BudgetLedger(store)
    reservation = ledger.reserve(attempts[0], BudgetSpec(1))
    original = store._append

    def append(kind, *args, **kwargs):
        if kind == "ARM_CLOSED":
            raise OSError("simulated interruption")
        return original(kind, *args, **kwargs)

    monkeypatch.setattr(store, "_append", append)
    with pytest.raises(OSError):
        ledger.settle(reservation["reservation_id"], 1000, outcome="TIMEOUT")
    assert BudgetLedger(ResearchStore(tmp_path, "budget-fixture")).balance("affine")["closed"]


def test_fake_clock_deadline_and_completed_settlement_recovery(tmp_path):
    store, attempts = registered(tmp_path)
    value = time.monotonic()

    def clock():
        nonlocal value
        value += 0.2
        return value

    result = ResearchSupervisor(store, monotonic=clock).run(attempts[0],
        lambda _: [sys.executable, "-c", "import time; time.sleep(30)"], BudgetSpec(0.5))
    assert result["state"] == "TIMEOUT"
    assert result["elapsed_ms"] >= 500


def test_preflight_error_is_terminal_and_does_not_charge_execution(tmp_path):
    store, attempts = registered(tmp_path)
    supervisor = ResearchSupervisor(store)
    with pytest.raises(ResearchError, match="command"):
        supervisor.run(attempts[0], lambda _: [], BudgetSpec(1))
    assert store.attempts()[attempts[0]]["state"] == "PREFLIGHT_FAILED"
    assert supervisor.budget.balance("affine")["committed_ms"] == 0


def test_changing_study_or_arm_label_cannot_refresh_same_family_budget(tmp_path):
    store, attempts = registered(tmp_path)
    ledger = BudgetLedger(store)
    reservation = ledger.reserve(attempts[0], BudgetSpec(1))
    ledger.settle(reservation["reservation_id"], 500, outcome="FAILED")
    original = store.manifest("study-synthetic")["spec"]
    renamed = {**original, "study_id": "renamed"}
    renamed["arms"] = [{**original["arms"][0], "arm_id": "fresh-arm"}]
    renamed["cells"] = [{**original["cells"][0], "arm_id": "fresh-arm"}]
    with pytest.raises(ResearchError, match="relabeled"):
        store.register(renamed, digest(renamed))
    second = {**original, "study_id": "second-study"}
    store.register(second, digest(second))
    assert ledger.balance("affine")["committed_ms"] == 500


def test_bad_tail_is_quarantined_without_replenishing_budget(tmp_path):
    store, attempts = registered(tmp_path)
    BudgetLedger(store).reserve(attempts[0], BudgetSpec(1))
    files = sorted((store.path / "events").glob("*.json"))
    files[-1].write_bytes(b"truncated")
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        ResearchStore(tmp_path, "budget-fixture")
    repair = ResearchStore(tmp_path, "budget-fixture", allow_corrupt=True)
    hold = repair.quarantine_tail("fixture power-cut recovery")
    assert (repair.path / hold["quarantine"] / files[-1].name).read_bytes() == b"truncated"
    reopened = ResearchStore(tmp_path, "budget-fixture")
    with pytest.raises(ResearchError, match="RECOVERY_REQUIRED"):
        BudgetLedger(reopened).reserve(attempts[0], BudgetSpec(1))
