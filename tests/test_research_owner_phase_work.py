"""Actual owner proof/save phases retain final integrity and deadline checks."""

import time
from types import SimpleNamespace

import pytest

from application.research_admission import AdmissionGate
from application.research_budget import BudgetLedger, BudgetSpec
import application.research_computation as computation
import application.research_recovery as recovery_module
from application.research_recovery import SharedRecovery
from infrastructure.research_store import ResearchError, digest
from tests.test_research_comparison import comparison_source, compute
from tests.test_research_live_checkpoint import prepared


def observe_events(store, monkeypatch):
    original, reads = store._json, []
    def read(path):
        if path.parent == store.path / "events":
            reads.append(path)
        return original(path)
    monkeypatch.setattr(store, "_json", read)
    return reads


@pytest.fixture(scope="module")
def actual_computation(tmp_path_factory):
    source = comparison_source(tmp_path_factory.mktemp("owner-proof"))
    result = compute(source)
    assert result["state"] == "SUCCEEDED", result
    store = source[0]
    reference = store.manifest("computation-" + result["attempt_id"])["computation_ref"]
    return store, reference


def test_managed_proof_verifies_one_initial_and_one_final_physical_chain(actual_computation, monkeypatch):
    store, reference = actual_computation
    reads = observe_events(store, monkeypatch)
    proof, result = computation.verified_computation(store, reference)
    assert proof["computation_ref"] == result["computation_ref"] == reference
    count = len(reads)
    events = store.events()
    assert 0 < count <= 2 * len(events), (
        f"owner proof reread {count} actual event files for {len(events)} events")
    assert store._read_snapshot() is None


def test_managed_proof_cannot_return_after_last_cost_check_corrupts_chain(actual_computation, monkeypatch):
    store, reference = actual_computation
    path = store.path / "events/0000000000000001.json"
    original_bytes = path.read_bytes()
    original, reached = computation._verify_job_cost, []
    def verify(*args, **kwargs):
        result = original(*args, **kwargs)
        reached.append(True)
        path.write_bytes(b"{}")
        return result
    monkeypatch.setattr(computation, "_verify_job_cost", verify)
    try:
        with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
            computation.verified_computation(store, reference)
        assert reached == [True]
        assert store._read_snapshot() is None
    finally:
        # Restore only this synthetic test fixture for module-scoped controls;
        # this is not a recovery procedure for any real authority store.
        path.write_bytes(original_bytes)


def checkpoint_owner(tmp_path):
    store, value, registry, adapters, _ = prepared(tmp_path / "owner", "exact", "owner")
    cell = value["cells"][0]
    plugin = registry.resolve(cell["plugin_id"], cell["capability"], version="1.0.0")
    attempt = store.new_attempt(store.register_run("owner", cell))
    BudgetLedger(store).reserve(attempt, BudgetSpec(3))
    receipt = AdmissionGate(store).prepare(value, cell, plugin, attempt)
    store.transition(attempt, "RUNNING")
    owner = SharedRecovery(store, registry, adapters)
    state = {"step": 1, "data_position": 1, "method_state": {}, "rng_state": {"fixture": [1]}}
    progress = {"completed_steps": 1, "total_steps": 300, "throughput_per_second": 100, "eta_seconds": 2.99}
    return store, owner, attempt, receipt, state, progress


def save_checkpoint(owner, attempt, receipt, state, progress, phase, deadline):
    if phase == "public":
        return owner.checkpoint(attempt, state, admission_hash=receipt["admission_hash"],
                                progress=progress, deadline=deadline)
    return owner.checkpoint_handler(attempt)(state, progress, receipt, deadline)


@pytest.mark.parametrize("phase", ["public", "handler"])
def test_checkpoint_phase_verifies_one_initial_and_one_final_chain(tmp_path, monkeypatch, phase):
    store, owner, attempt, receipt, state, progress = checkpoint_owner(tmp_path)
    reads = observe_events(store, monkeypatch)
    result = save_checkpoint(owner, attempt, receipt, state, progress, phase, None)
    assert result
    count = len(reads)
    events = store.events()
    assert 0 < count <= 2 * len(events), (
        f"{phase} save reread {count} actual event files for {len(events)} events")
    assert events[-1]["event_kind"] == "CHECKPOINT"
    assert store._read_snapshot() is None


@pytest.mark.parametrize("phase", ["public", "handler"])
def test_checkpoint_cannot_return_after_last_publication_corrupts_chain(tmp_path, monkeypatch, phase):
    store, owner, attempt, receipt, state, progress = checkpoint_owner(tmp_path)
    original, reached = store._append, []
    def append(kind, *args, **kwargs):
        result = original(kind, *args, **kwargs)
        if kind == "CHECKPOINT":
            reached.append(True)
            (store.path / "events/0000000000000001.json").write_bytes(b"{}")
        return result
    monkeypatch.setattr(store, "_append", append)
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        save_checkpoint(owner, attempt, receipt, state, progress, phase, None)
    assert reached == [True]
    assert store._read_snapshot() is None


@pytest.mark.parametrize("phase", ["public", "handler"])
def test_checkpoint_cannot_return_after_final_journal_crosses_deadline(tmp_path, monkeypatch, phase):
    store, owner, attempt, receipt, state, progress = checkpoint_owner(tmp_path)
    original, reached = store._append, []
    now, deadline = time.monotonic, time.monotonic() + 30
    monkeypatch.setattr(recovery_module, "time", SimpleNamespace(
        monotonic=lambda: deadline + 1 if reached else now()))
    def append(kind, *args, **kwargs):
        result = original(kind, *args, **kwargs)
        if kind == "CHECKPOINT":
            reached.append(True)
        return result
    monkeypatch.setattr(store, "_append", append)
    with pytest.raises(ResearchError, match="TIMEOUT"):
        save_checkpoint(owner, attempt, receipt, state, progress, phase, deadline)
    assert reached == [True], "deadline hook must reach the actual durable final journal"
    assert store._read_snapshot() is None
