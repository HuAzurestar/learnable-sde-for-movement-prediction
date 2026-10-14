"""Actual mapped shared MLMC interruption, linked restore and owner receipts."""

import json

from application.research_budget import BudgetLedger
from infrastructure.research_admission_selection import verify_admission_selection
from infrastructure.research_store import ResearchStore, digest
from tests import test_propagation_shared_adapter as adapter


def test_actual_mapped_interrupted_restore_retains_original_package_table_and_arm(tmp_path, monkeypatch):
    original_prepare = adapter.prepare

    def mapped_prepare(*args, **kwargs):
        store, spec, registry = original_prepare(*args, **kwargs)
        reference = spec["admission"].pop("package_hash")
        spec["admission"]["cell_packages"] = {"schema_version": "pirc25-cell-packages-v1", "bindings": [
            {"cell_hash": digest(cell), "package_hash": reference} for cell in spec["cells"]]}
        return store, spec, registry

    monkeypatch.setattr(adapter, "prepare", mapped_prepare)
    # Reuse the complete real soft-save/ACK/reopened-restore/statistics/cost
    # assertions, now with the mapped original spec BEFORE registration.
    adapter.test_actual_chunk_worker_stops_and_reopened_resume_preserves_computation_and_cost(tmp_path, "mlmc")
    store = ResearchStore(tmp_path, "propagation-unit")
    spec = store.manifest("study-propagation-unit")["spec"]
    attempts = list(store.attempts().values())
    failed = next(attempt for attempt in attempts if attempt["state"] == "FAILED")
    succeeded = next(attempt for attempt in attempts if attempt["state"] == "SUCCEEDED")
    assert succeeded["parent_attempt_id"] == failed["attempt_id"]
    result = json.loads((store.path / "artifacts" / succeeded["artifact_id"]).read_bytes())
    receipt = store.manifest("admission-" + result["admission_hash"])
    verify_admission_selection(spec, spec["cells"][0], receipt)
    admission_events = [event for event in store.events() if event["event_kind"] == "ADMISSION"]
    assert len(admission_events) == 2
    original = store.manifest("admission-" + admission_events[0]["payload"]["admission_hash"])
    assert original["admission_selection"] == receipt["admission_selection"]
    assert BudgetLedger(store).balance("affine-method")["committed_ms"] > 0
