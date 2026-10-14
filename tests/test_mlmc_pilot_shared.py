"""Disposable managed engineering pilots; never research/model acceptance."""

import json
import time

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_recovery import RecoveryRegistry, SharedRecovery
from experiments.pirc25.runner import SharedRunner
from experiments.pirc27.plugin import propagation_plugin, propagation_recovery_plugin
from infrastructure.research_store import ResearchError, ResearchStore, digest
from tests.test_propagation_shared_adapter import (prepare, _mlmc_checkpoint_allocation,
    _worker_startup_seconds, _checkpoint_job_seconds)


def setup(tmp_path, **kwargs):
    return prepare(tmp_path, "mlmc-pilot", recovery=True, changes={"samples": 24},
                   level_samples=(8, 8, 8), **kwargs)


@pytest.mark.parametrize("synthetic", [False, True])
def test_real_managed_pilot_retains_original_arm_unknown_reference_and_ledger_charge(tmp_path, synthetic):
    store, spec, registry = setup(tmp_path, synthetic=synthetic)
    store.register(spec, digest(spec))
    adapters = RecoveryRegistry()
    adapters.register(propagation_recovery_plugin(synthetic=synthetic))
    runner = SharedRunner(store, registry, recovery_registry=adapters)
    outcome = runner.run_cell(spec["study_id"], digest(spec["cells"][0]), budget=BudgetSpec(60, category="pilot"))
    assert outcome["state"] == "SUCCEEDED", outcome
    output = json.loads((store.path / "artifacts" / outcome["artifact_id"]).read_bytes())
    functional, analysis = output["forecast"]["functional"], output["forecast"]["pilot_analysis"]
    assert output["qualification"] == "fixture" and functional["status"] == "PILOT_ONLY"
    assert analysis["status"] == "UNQUALIFIED" and analysis["reference_uncertainty"] is None
    assert "REFERENCE_UNRESOLVED" in analysis["reasons"] and not analysis["automatic_execution"]
    assert spec["arms"][0]["method_family_id"] == "mlmc"
    assert all(ns > 0 for ns in dict(functional["diagnostics"])["level_compute_ns"])
    balance = BudgetLedger(store).balance(spec["arms"][0]["arm_id"])
    assert balance["committed_ms"] > 0 and not balance["closed"]
    settlements = [e["payload"] for e in store.events() if e["event_kind"] == "SETTLE"]
    assert len(settlements) == 1 and settlements[0]["charged_ms"] == balance["committed_ms"]
    reused = runner.run_cell(spec["study_id"], digest(spec["cells"][0]), budget=BudgetSpec(60, category="pilot"))
    assert reused["reused"] and reused["artifact_id"] == outcome["artifact_id"]
    assert BudgetLedger(store).balance(spec["arms"][0]["arm_id"]) == balance


@pytest.mark.parametrize("category", ["smoke", "job"])
def test_wrong_category_is_rejected_before_reservation(tmp_path, category):
    store, spec, registry = setup(tmp_path)
    store.register(spec, digest(spec))
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]),
                                             budget=BudgetSpec(60, category=category))
    assert not any(e["event_kind"] in {"RESERVE", "WORKER_STARTED", "READ_STARTED"} for e in store.events())
    assert next(iter(store.attempts().values()))["state"] == "PREFLIGHT_FAILED"


@pytest.mark.parametrize("case", ["final-eval", "fixture-mode", "formal-mode", "missing-policy",
    "wrong-role", "production-seed-reused", "new-method-family"])
def test_pilot_refuses_unfrozen_policy_or_protected_data_before_any_artifact_read(tmp_path, monkeypatch, case):
    store, spec, registry = setup(tmp_path, formal=case == "final-eval")
    cell = spec["cells"][0]
    if case.endswith("-mode"):
        spec["admission"]["mode"] = case.removesuffix("-mode")
    elif case == "missing-policy":
        del cell["mlmc_pilot_policy"]
    elif case == "wrong-role":
        cell["execution_role"] = "production"
    elif case == "production-seed-reused":
        cell["mlmc_pilot_policy"]["production_seed"] = cell["seed"]
    elif case == "new-method-family":
        spec["arms"][0]["method_family_id"] = "mlmc-pilot"
    def forbidden(*args, **kwargs):
        pytest.fail("pilot refusal read qualification/input artifact")
    monkeypatch.setattr(store, "read_artifact", forbidden)
    store.register(spec, digest(spec))
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA|UNQUALIFIED"):
        SharedRunner(store, registry).run_cell(spec["study_id"], digest(cell), budget=BudgetSpec(60, category="pilot"))
    assert not any(e["event_kind"] in {"READ_STARTED", "WORKER_STARTED", "EXPOSURE_ALLOWED"} for e in store.events())


def test_pilot_does_not_extend_the_shared_stage_cap():
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        BudgetSpec(1801, category="pilot").validate()


def test_actual_pilot_soft_save_and_reopened_resume_preserve_measured_completed_cost(tmp_path):
    from application.propagation_execution import execute_propagation
    from domain.propagation import PropagationRequest
    from experiments.pirc27.oracles import oracle_suite
    from inference.propagation_methods import mlmc_estimate
    # Engineering-only calibration BEFORE creating any store or reservation.
    case = oracle_suite()[0]
    request = PropagationRequest("timing-pilot", case.package.package_hash, case.initial_mean,
        case.initial_covariance, 0., 0., (1.,), "endpoint-x", 11, "paired-root", "affine-method",
        samples=112, steps=128, chunk_size=1)
    started = time.monotonic()
    mlmc_estimate(case.package, request, level_samples=(64, 32, 16), phase=2, pilot=True)
    counts = _mlmc_checkpoint_allocation(time.monotonic()-started)
    startup_seconds = _worker_startup_seconds()
    store, spec, registry = prepare(tmp_path, "mlmc-pilot", recovery=True, level_samples=counts,
        changes={"samples": sum(counts), "steps": 128, "chunk_size": 1})
    cell = spec["cells"][0]
    started = time.monotonic()
    expected = execute_propagation(spec, cell)
    job_seconds = _checkpoint_job_seconds(time.monotonic()-started, startup_seconds)
    store.register(spec, digest(spec))
    adapters = RecoveryRegistry()
    adapters.register(propagation_recovery_plugin())
    interrupted = SharedRunner(store, registry, recovery_registry=adapters).run_cell(
        spec["study_id"], digest(cell), budget=BudgetSpec(job_seconds, category="pilot"))
    assert interrupted["state"] == "FAILED", interrupted
    assert store.attempts()[interrupted["attempt_id"]]["error_code"] == "CHECKPOINT_SAVED"
    saved = interrupted["checkpoint"]
    assert 0 < saved["progress"]["completed_steps"] < saved["progress"]["total_steps"]
    checkpoint = json.loads((store.path / "artifacts" / saved["artifact_id"]).read_bytes())
    assert "budget" not in checkpoint
    saved_costs = [item["compute_ns"] for item in checkpoint["state"]["method_state"]["statistics"]]
    assert any(ns > 0 for ns in saved_costs)
    before = BudgetLedger(store).balance(cell["arm_id"])
    reopened = ResearchStore(tmp_path, "propagation-unit")
    grant = reopened.manifest("authorization-"+spec["admission"]["authorization_id"])
    resumed = SharedRecovery(reopened, registry, adapters).resume(interrupted["attempt_id"], saved["artifact_id"],
        authorization=grant, budget=BudgetSpec(60, category="pilot"))
    assert resumed["state"] == "SUCCEEDED", resumed
    actual = json.loads((reopened.path / "artifacts" / resumed["artifact_id"]).read_bytes())
    actual_functional = actual["forecast"]["functional"]
    expected_functional = expected["forecast"]["functional"]
    actual_diagnostics = dict(actual_functional.pop("diagnostics"))
    expected_diagnostics = dict(expected_functional.pop("diagnostics"))
    costs = actual_diagnostics.pop("level_compute_ns")
    expected_diagnostics.pop("level_compute_ns")
    assert actual_diagnostics == json.loads(json.dumps(expected_diagnostics))
    assert actual_functional == json.loads(json.dumps(expected_functional))
    assert actual["metrics"] == expected["metrics"]
    assert all(ns > 0 for ns in costs)
    assert all(ns >= saved_ns for ns, saved_ns in zip(costs, saved_costs))
    assert actual["forecast"]["pilot_analysis"]["status"] == "UNQUALIFIED"
    after = BudgetLedger(reopened).balance(cell["arm_id"])
    assert after["committed_ms"] > before["committed_ms"] and not after["closed"]
    assert reopened.attempts()[resumed["attempt_id"]]["parent_attempt_id"] == interrupted["attempt_id"]
