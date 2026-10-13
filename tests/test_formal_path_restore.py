"""Actual qualified path/IS save-ACK, reopened linked restore and owner costs.

Disposable synthetic engineering controls only; no official research ledger.
Frozen same-law policies precede reservations. No live deadline extension,
fake clock, forced state, sleep-only worker, rescoped arm or numerical rescue.
"""

from copy import deepcopy
import json
import time

import pytest

from application.path_qualification_admission import validate_formal_path_result
from application.propagation_execution import validate_propagation_cell
from application.research_budget import BudgetLedger, BudgetSpec
from application.research_evidence import export_evidence
from application.research_recovery import RecoveryRegistry, SharedRecovery
from domain.path_qualification import METHODS
from experiments.pirc25.runner import SharedRunner
from experiments.pirc27.path_production_plugin import path_production_recovery_plugin
from inference.propagation_methods import monte_carlo, importance_sampling
from infrastructure.research_store import ResearchError, ResearchStore, digest, encode
from tests.path_paper_helpers import independent_reader, paper_validate
from tests.test_managed_path_admission import prepare_target, METRIC
from tests.test_path_qualification import policies
from tests.test_propagation_shared_adapter import (_path_checkpoint_samples,
    _worker_startup_seconds, _checkpoint_job_seconds)


def actual_kernel(package, request, method, proposal, **kwargs):
    if method == "importance":
        return importance_sampling(package, request, proposal=tuple(proposal), **kwargs)
    return monte_carlo(package, request, solver="additive-heun" if method == "heun" else method, **kwargs)


@pytest.mark.parametrize("method", METHODS, ids=["euler", "heun", "rev", "is"])
def test_actual_formal_path_save_ack_reopened_restore_retains_current_statistics_and_all_costs(tmp_path, method):
    # Timing-only sizing before ANY store/grant/reservation. The same workload
    # and first-job cap as existing real path checkpoint tests, not new authority.
    maximum_resumes = 2
    startup = _worker_startup_seconds()
    package, timing_request, q = policies(method,
        request_changes={"samples": 128, "steps": 128, "chunk_size": 1})
    started = time.monotonic()
    actual_kernel(package, timing_request, method, q.proposal)
    samples = _path_checkpoint_samples(time.monotonic()-started)
    store, spec, registry, grant, _, _ = prepare_target(tmp_path, method,
        source_request_changes={"samples": 512, "steps": 128, "chunk_size": 1, "tolerance": .25},
        target_changes={"samples": samples},
        changes={"maximum_sampling_uncertainty": .2, "maximum_observed_grid_error": .25,
            "maximum_time_bias": .02, "maximum_total_observed_functional_error": .25})
    cell = spec["cells"][0]
    model, request, config, _ = validate_propagation_cell(spec, cell)
    final = []
    # Pure, unprotected synthetic-kernel expected output, not an extra admission
    # or research result. First deadline is sized BEFORE its own reservation.
    started = time.monotonic()
    expected = actual_kernel(model, request, method, q.proposal,
        checkpoint=lambda state, total: final.append(state) if state["data_position"]["level"] == 1 else None).manifest()
    job_seconds = _checkpoint_job_seconds(time.monotonic()-started, startup)
    assert len(final) == 1 and final[0]["step"] == request.samples*request.steps
    arm = cell["arm_id"]
    source_cost = BudgetLedger(store).balance(arm)["committed_ms"]
    assert source_cost > 0
    adapters = RecoveryRegistry()
    adapters.register(path_production_recovery_plugin())
    interrupted = SharedRunner(store, registry, recovery_registry=adapters).run_cell(
        spec["study_id"], digest(cell), budget=BudgetSpec(job_seconds))
    assert interrupted["state"] == "FAILED", {**interrupted,
        "engineering_calibration_seconds": {"startup": startup, "job": job_seconds}}
    parent = store.attempts()[interrupted["attempt_id"]]
    assert parent["error_code"] == "CHECKPOINT_SAVED", interrupted
    saved = interrupted["checkpoint"]
    progress = saved["progress"]
    assert 0 < progress["completed_steps"] < progress["total_steps"] == config["work_steps"]
    assert progress["throughput_per_second"] > 0 and progress["eta_seconds"] > 0
    assert progress["eta_seconds"] == pytest.approx(
        (progress["total_steps"]-progress["completed_steps"])/progress["throughput_per_second"])
    events = [e for e in store.events() if e["payload"].get("attempt_id") == parent["attempt_id"]]
    kinds = [e["event_kind"] for e in events]
    assert kinds.index("CHECKPOINT_REQUESTED") < kinds.index("CHECKPOINT") < kinds.index("CHECKPOINT_SAVED")
    assert kinds.index("CHECKPOINT_SAVED") < kinds.index("WORKER_TREE_STOPPED") < kinds.index("SETTLE")
    requested = next(e["payload"] for e in events if e["event_kind"] == "CHECKPOINT_REQUESTED")
    assert 0 < requested["remaining_seconds"] <= .2*job_seconds and saved["elapsed_ms"] < job_seconds*1000
    saved_value = json.loads((store.path/"artifacts"/saved["artifact_id"]).read_bytes())
    state = saved_value["state"]
    assert len(encode(state)) <= 16384 and state["chunk_complete"] is True
    assert state["method_state"]["request_hash"] == request.request_hash
    assert state["rng_state"]["seed"] == request.seed and state["rng_state"]["coupling_id"] == request.coupling_id
    assert "budget" not in state and "remaining_seconds" not in state
    before = BudgetLedger(store).balance(arm)
    assert before["committed_ms"] == source_cost+interrupted["elapsed_ms"] and not before["closed"]
    reopened = ResearchStore(tmp_path, store.store_id)
    with pytest.raises(ResearchError):
        SharedRecovery(reopened, registry, adapters).prepare(parent["attempt_id"], saved["artifact_id"],
            authorization={**grant, "purposes": ["evaluate"]})
    assert BudgetLedger(reopened).balance(arm) == before
    continuations, interrupted_ids = [], {parent["attempt_id"]}
    previous, checkpoint, completed = parent["attempt_id"], saved["artifact_id"], progress["completed_steps"]
    for _ in range(maximum_resumes):
        reopened = ResearchStore(tmp_path, store.store_id)
        resumed = SharedRecovery(reopened, registry, adapters).resume(previous, checkpoint,
            authorization=grant, budget=BudgetSpec(60))
        continuations.append(resumed)
        current = reopened.attempts()[resumed["attempt_id"]]
        assert current["parent_attempt_id"] == previous and not BudgetLedger(reopened).balance(arm)["closed"]
        if resumed["state"] == "SUCCEEDED":
            break
        assert resumed["state"] == "FAILED" and current["error_code"] == "CHECKPOINT_SAVED", resumed
        next_saved = resumed["checkpoint"]
        assert completed < next_saved["progress"]["completed_steps"] < progress["total_steps"]
        previous, checkpoint, completed = resumed["attempt_id"], next_saved["artifact_id"], next_saved["progress"]["completed_steps"]
        interrupted_ids.add(previous)
    assert resumed["state"] == "SUCCEEDED", resumed
    actual = json.loads((reopened.path/"artifacts"/resumed["artifact_id"]).read_bytes())
    receipt = reopened.manifest("admission-"+actual["admission_hash"])
    assert actual["qualification"] == "qualified" and actual["forecast"]["current_output_qualification"] == "PASSED"
    assert receipt["attempt_id"] == resumed["attempt_id"] and actual["admission_hash"] != saved_value["admission_hash"]
    analysis = actual["forecast"]["path_output_analysis"]
    assert analysis["completed_statistics"] == final[0]
    assert analysis["actual_functional_hash"] == digest(expected)
    assert analysis["kernel_error_budget"] == expected["error_budget"]
    assert 0 < actual["metrics"][METRIC] <= request.tolerance == .25
    assert analysis["sampling_error"]["uncertainty_radius"] > 0
    assert analysis["implementation_roundoff"]["value"] is None and analysis["model_error"]["value"] is None
    proof = receipt["documents"]["propagation_qualification"]
    assert receipt["input_evidence"] and all(proof["completion_event"]["sequence"] < e["sequence"] for e in receipt["input_evidence"])
    assert proof["settlement_event"]["payload"]["charged_ms"] > 0
    assert proof["target_request_hash"] != proof["qualification_policy"]["request_hash"]
    assert proof["policy"] == cell["path_production_policy"] and proof["qualification_policy"] == cell["path_qualification_policy"]
    assert {k: v for k, v in actual["forecast"]["functional"].items() if k != "error_budget"} == json.loads(
        encode({k: v for k, v in expected.items() if k != "error_budget"}))
    validate_formal_path_result(receipt, spec, cell, actual)
    assert BudgetLedger(reopened).balance(arm)["committed_ms"] == before["committed_ms"]+sum(c["elapsed_ms"] for c in continuations)
    bundle = export_evidence(reopened, spec["study_id"], grant)
    row, all_ids = bundle["cells"][0], interrupted_ids | {resumed["attempt_id"]}
    assert {a["attempt_id"] for a in row["history"]} == all_ids
    assert row["cost"]["charged_ms"] == interrupted["elapsed_ms"]+sum(c["elapsed_ms"] for c in continuations)
    assert {e["payload"]["attempt_id"] for e in row["cost"]["sources"]} == all_ids
    assert independent_reader().validate_path_qualification(row["admission"], row) == "PASSED"
    checked = paper_validate(tmp_path, bundle)
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout)["current_passed_cells"] == 1
    for fault in ("missing-parent-charge", "zero-parent-charge"):
        altered = deepcopy(bundle)
        entries = altered["cells"][0]["cost"]["sources"]
        if fault == "missing-parent-charge":
            entries[:] = [e for e in entries if e["payload"]["attempt_id"] != parent["attempt_id"]]
        else:
            entry = next(e for e in entries if e["payload"]["attempt_id"] == parent["attempt_id"])
            entry["payload"]["charged_ms"] = 0
            entry["hash"] = digest({k: v for k, v in entry.items() if k != "hash"})
        altered["bundle_hash"] = digest({k: v for k, v in altered.items() if k != "bundle_hash"})
        assert paper_validate(tmp_path, altered).returncode != 0, fault
