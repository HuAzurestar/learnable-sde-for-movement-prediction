"""Completed-cell reuse must validate the actual immutable output, not a flag."""

import pytest

from experiments.pirc25.affine import fixture_spec
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_store import ResearchError, ResearchStore, digest


@pytest.mark.parametrize("damage", ["corrupt", "missing"])
def test_reuse_rejects_damaged_completed_result(tmp_path, damage):
    store = ResearchStore(tmp_path, "reuse-integrity", initialize=True)
    value = fixture_spec("fixture", 1, (19,))
    store.register(value, digest(value))
    runner = SharedRunner(store)
    original = runner.run_cell("fixture", digest(value["cells"][0]))
    assert original["state"] == "SUCCEEDED"
    path = store.path / "artifacts" / original["artifact_id"]
    if damage == "corrupt":
        path.write_bytes(b"damaged synthetic fixture output")
    else:
        path.unlink()
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT|MISSING_INPUT"):
        runner.run_cell("fixture", digest(value["cells"][0]))
    assert len(store.attempts()) == 1, "bad output must not silently rerun or overwrite the completed cell"
    assert store.attempts()[original["attempt_id"]]["state"] == "SUCCEEDED"
    assert store.events()[-1]["event_kind"] == "REUSE_CHECK_FAILED"


def completed_plugin(tmp_path):
    from application.research_budget import BudgetSpec
    from tests.test_research_admission_chain import prepared
    store, value, registry, _ = prepared(tmp_path)
    store.register(value, digest(value))
    runner = SharedRunner(store, registry)
    result = runner.run_cell("synthetic", digest(value["cells"][0]), budget=BudgetSpec(10))
    assert result["state"] == "SUCCEEDED"
    return store, value, runner, result


def test_verified_reuse_preserves_attempt_budget_and_output(tmp_path):
    from application.research_budget import BudgetLedger
    store, value, runner, original = completed_plugin(tmp_path)
    content = (store.path / "artifacts" / original["artifact_id"]).read_bytes()
    before = BudgetLedger(store).balance("affine")
    reused = runner.run_cell("synthetic", digest(value["cells"][0]))
    assert reused["reused"] is True and reused["attempt_id"] == original["attempt_id"]
    assert (store.path / "artifacts" / original["artifact_id"]).read_bytes() == content
    assert BudgetLedger(store).balance("affine") == before and len(store.attempts()) == 1
    assert store.events()[-1]["event_kind"] == "REUSE_VERIFIED"


@pytest.mark.parametrize("mutation", ["missing-settlement", "wrong-cost-study", "unknown-cost", "wrong-reservation",
    "wrong-admission-reference", "missing-worker"])
def test_reuse_requires_complete_authoritative_execution_and_cost_bindings(tmp_path, monkeypatch, mutation):
    # Fault-inject a legacy authority reader after a genuine completed job.
    # Valid chain/metadata integrity is independently tested by ResearchStore;
    # the application must also validate cross-object provenance, not trust
    # the fact that a manifest/event happens to exist.
    from copy import deepcopy
    store, value, runner, original = completed_plugin(tmp_path)
    events = deepcopy(store.events())
    if mutation == "missing-settlement":
        events = [event for event in events if event["event_kind"] != "SETTLE"]
    elif mutation == "missing-worker":
        events = [event for event in events if event["event_kind"] != "WORKER_STARTED"]
    elif mutation == "wrong-admission-reference":
        next(event for event in events if event["event_kind"] == "ADMISSION")["payload"]["admission_hash"] = "0" * 64
    else:
        cost = next(event for event in events if event["event_kind"] == "SETTLE")["payload"]
        if mutation == "wrong-cost-study":
            cost["study_id"] = "unrelated"
        elif mutation == "unknown-cost":
            cost["monotonic_elapsed_ms"] = None
        else:
            cost["reservation_id"] = "0" * 64
    monkeypatch.setattr(store, "_events", lambda: events)
    with pytest.raises(ResearchError, match="UNQUALIFIED"):
        runner.run_cell("synthetic", digest(value["cells"][0]))
    assert store._attempts()[original["attempt_id"]]["state"] == "SUCCEEDED"


@pytest.mark.parametrize("mutation", ["role", "block", "completion-hash", "receipt-hash", "receipt-cell", "plugin"])
def test_reuse_rejects_metadata_and_admission_contract_mismatches(tmp_path, monkeypatch, mutation):
    from copy import deepcopy
    store, value, runner, original = completed_plugin(tmp_path)
    resolve = store._manifest
    def legacy_resolve(object_id):
        data = deepcopy(resolve(object_id))
        if object_id == "artifact-" + original["artifact_id"]:
            if mutation == "role":
                data["role"] = "checkpoint"
            elif mutation == "block":
                data["block_ids"] = ["unrelated"]
            elif mutation == "completion-hash":
                data["visibility"] = "restricted"
        if object_id.startswith("admission-"):
            if mutation == "receipt-hash":
                data["admission_hash"] = "0" * 64
            elif mutation == "receipt-cell":
                data["cell"] = {**data["cell"], "seed": 999}
            elif mutation == "plugin":
                data["plugin_hash"] = "0" * 64
        return data
    monkeypatch.setattr(store, "_manifest", legacy_resolve)
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH|UNQUALIFIED"):
        runner.run_cell("synthetic", digest(value["cells"][0]))
    assert len(store._attempts()) == 1
