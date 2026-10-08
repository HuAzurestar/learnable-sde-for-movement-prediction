"""Disposable charged calibration controls, never an official research study."""

from dataclasses import asdict, replace
import json

import pytest

from application.propagation_execution import execute_propagation, validate_propagation_cell
from application.research_admission import AdmissionGate
from application.research_budget import BudgetLedger, BudgetSpec
from application.research_contracts import CapabilityRegistry
from application.research_execution import execution_binding
from application.research_registry import GLOBAL_LIMITS, plan_resources
from experiments.pirc25.affine import code_hash
from experiments.pirc25.runner import SharedRunner
from experiments.pirc27.calibration_plugin import calibration_config, calibration_plugin
from experiments.pirc27.plugin import execution_inputs
from infrastructure.research_store import ResearchError, ResearchStore, digest
from tests.research_admission_fixtures import admit_fixture
from tests.test_affine_probability_calibration import example


def prepared(root, *, formal=False, policy_changes=None):
    store = ResearchStore(root, "probability-calibration-unit", initialize=True)
    package, request, policy = example()
    policy = replace(policy, maximum_job_seconds=60., **(policy_changes or {}))
    plugin = calibration_plugin()
    config, parameters = calibration_config(request, policy), execution_inputs(request)
    cell = {"arm_id": request.arm_id, "block_id": "generator-v1", "seed": request.seed,
        "horizon": request.horizons[0], "plugin_id": plugin.plugin_id,
        "capability": "exact-transition", "resource_class": "cpu", "visibility": "synthetic",
        "dimensions": 4, "study_role": "secondary", "execution_role": "probability-calibration",
        "propagation_request": json.loads(json.dumps(asdict(request))),
        "frozen_dynamics": package.manifest(), "probability_calibration_policy": policy.manifest(),
        "execution": execution_binding(plugin.registry_entry, config, parameters, matrix_cells=1)}
    spec = {"schema_version": "pirc25-contract-v1", "study_id": "probability-calibration-unit",
        "experiment_id": "probability-calibration-unit", "comparison_family": "synthetic-engineering",
        "code_hash": code_hash(), "protocol_hash": digest("pending"), "data_hash": digest("pending"),
        "feature_hash": digest("none"), "selection_hash": digest("none"),
        "arms": [{"arm_id": request.arm_id, "model_family_id": "affine-stable-v1",
            "method_family_id": "exact", "objective_id": request.functional, "budget_seconds": 86400}],
        "cells": [cell], "runtime_binding": {"root": str(root.resolve()), "store_id": store.store_id}}
    admit_fixture(store, spec, plugin, root, formal=formal, execution_config=config,
        execution_inputs=parameters, legacy_upstream=False)
    spec["admission"]["mode"] = "formal" if formal else "pilot"
    registry = CapabilityRegistry()
    registry.register(plugin)
    return store, spec, registry


def test_declared_work_is_not_paths_or_an_expanded_physical_dimension():
    _, request, policy = example()
    plugin = calibration_plugin()
    plan = plan_resources(plugin.registry_entry, calibration_config(request, policy),
        execution_inputs(request), matrix_cells=1)
    assert plan["counts"]["paths"] == 0
    assert plan["counts"]["steps"] == policy.maximum_operations == 200000
    assert plan["counts"]["state_dim"] == 4
    assert plugin.state_order == ("x", "y", "vx", "vy") and plugin.resume_level == "restart-only"
    pools = [tensor for tensor in plan["tensors"] if "-pool-" in tensor["name"]]
    assert len(pools) == 64 and sum(tensor["bytes"] for tensor in pools) == 64*1024*1024
    assert all(limit <= GLOBAL_LIMITS[name] for name, limit in plan["limits"].items())


@pytest.mark.parametrize("failed", [False, True])
def test_actual_charged_worker_retains_numerical_outcome_and_reuses_without_new_charge(tmp_path, failed):
    changes = {"maximum_relative_probability_error": 1e-30,
        "maximum_relative_probability_width": 1e-50} if failed else {}
    store, spec, registry = prepared(tmp_path, policy_changes=changes)
    store.register(spec, digest(spec))
    cell, arm = spec["cells"][0], spec["arms"][0]["arm_id"]
    runner = SharedRunner(store, registry)
    result = runner.run_cell(spec["study_id"], digest(cell), budget=BudgetSpec(60, category="pilot"))
    assert result["state"] == "SUCCEEDED", result
    artifact = json.loads((store.path/"artifacts"/result["artifact_id"]).read_bytes())
    analysis = artifact["forecast"]["probability_calibration_analysis"]
    assert analysis["status"] == ("FAILED" if failed else "PASSED")
    assert analysis["scientific_qualification"] is analysis["method_qualification"] is False
    assert analysis["admission_status"] == "NOT_ADMITTED" and analysis["cost_status"] == "OWNER_SETTLEMENT_REQUIRED"
    assert artifact["qualification"] == "fixture" and "functional" not in artifact["forecast"]
    assert artifact["metrics"] == {"reference_arithmetic_operations": analysis["reference_arithmetic_operations"]}
    assert artifact["metric_units"] == {"reference_arithmetic_operations": "operations"}
    events = store.events()
    own = [e for e in events if e["payload"].get("attempt_id") == result["attempt_id"]]
    kinds = [e["event_kind"] for e in own]
    assert kinds.index("RESERVE") < kinds.index("WORKER_STARTED") < kinds.index("WORKER_TREE_STOPPED") < kinds.index("SETTLE")
    settlement = next(e["payload"] for e in own if e["event_kind"] == "SETTLE")
    stop = next(e["payload"] for e in own if e["event_kind"] == "WORKER_TREE_STOPPED")
    assert 0 < settlement["charged_ms"] == stop["observed_elapsed_ms"] <= settlement["reserved_ms"] <= 60000
    before = BudgetLedger(store).balance(arm)
    assert before["committed_ms"] == settlement["charged_ms"]
    reused = SharedRunner(ResearchStore(tmp_path, store.store_id), registry).run_cell(spec["study_id"], digest(cell))
    assert reused["reused"] and reused["artifact_id"] == result["artifact_id"]
    assert BudgetLedger(store).balance(arm) == before
    assert len([e for e in store.events() if e["event_kind"] == "WORKER_STARTED"]) == 1


@pytest.mark.parametrize("budget", [BudgetSpec(60, category="smoke"), BudgetSpec(61, category="pilot")])
def test_wrong_budget_refuses_before_reservation_reads_or_worker(tmp_path, budget):
    store, spec, registry = prepared(tmp_path)
    store.register(spec, digest(spec))
    with pytest.raises(ResearchError, match="frozen pilot job budget"):
        SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]), budget=budget)
    assert next(iter(store.attempts().values()))["state"] == "PREFLIGHT_FAILED"
    assert not any(e["event_kind"] in {"RESERVE", "READ_STARTED", "READ_COMPLETED", "WORKER_STARTED"} for e in store.events())


def test_formal_inputs_refuse_before_any_protected_read(tmp_path, monkeypatch):
    store, spec, _ = prepared(tmp_path, formal=True)
    store.register(spec, digest(spec))
    cell = spec["cells"][0]
    attempt = store.new_attempt(store.register_run(spec["study_id"], cell))
    monkeypatch.setattr(store, "authorization", lambda *args, **kwargs: pytest.fail("grant lookup before held-out refusal"))
    with pytest.raises(ResearchError, match="cannot consume test/final-eval"):
        AdmissionGate(store).prepare(spec, cell, calibration_plugin(), attempt)
    assert not any(e["event_kind"] in {"READ_STARTED", "READ_COMPLETED", "WORKER_STARTED"} for e in store.events())


def test_actual_operation_exhaustion_is_a_charged_failed_attempt_not_an_expanded_retry(tmp_path):
    store, spec, registry = prepared(tmp_path, policy_changes={"maximum_operations": 100})
    frozen = json.loads(json.dumps(spec))
    store.register(spec, digest(spec))
    cell = spec["cells"][0]
    result = SharedRunner(store, registry).run_cell(spec["study_id"], digest(cell),
        budget=BudgetSpec(60, category="pilot"))
    assert result["state"] == "FAILED" and result["artifact_id"] is None, result
    assert BudgetLedger(store).balance(cell["arm_id"])["committed_ms"] > 0
    assert spec == frozen and cell["probability_calibration_policy"]["maximum_operations"] == 100
    assert len(store.attempts()) == 1
    assert len([e for e in store.events() if e["event_kind"] == "WORKER_STARTED"]) == 1


@pytest.mark.parametrize("fault", ["role", "arm", "policy", "config"])
def test_role_reference_arm_and_policy_cannot_be_substituted(tmp_path, fault):
    _, spec, _ = prepared(tmp_path)
    cell = spec["cells"][0]
    if fault == "role":
        cell["execution_role"] = "qualification"
    elif fault == "arm":
        spec["arms"][0]["method_family_id"] = "euler"
    elif fault == "policy":
        cell["probability_calibration_policy"]["target_probability"] = .01
    else:
        cell["execution"]["config"]["calibration_policy_hash"] = "0"*64
    with pytest.raises(ResearchError):
        validate_propagation_cell(spec, cell)


def test_calibration_does_not_execute_discarded_generic_estimators(tmp_path, monkeypatch):
    _, spec, _ = prepared(tmp_path)
    import inference.propagation_methods as methods
    import inference.affine_probability_calibration as calibration
    def forbidden(*args, **kwargs):
        pytest.fail("calibration executed a discarded generic estimator")
    monkeypatch.setattr(methods, "analytic_estimate", forbidden)
    monkeypatch.setattr(calibration, "calibrate_affine_halfspace", lambda *args: {
        "reference_arithmetic_operations": 7, "status": "FAILED", "scientific_qualification": False})
    result = execute_propagation(spec, spec["cells"][0])
    assert result["metrics"] == {"reference_arithmetic_operations": 7}
    assert result["qualification"] == "fixture"


def test_restart_only_refuses_state_before_numerics(tmp_path, monkeypatch):
    _, spec, _ = prepared(tmp_path)
    import inference.affine_probability_calibration as calibration
    monkeypatch.setattr(calibration, "calibrate_affine_halfspace", lambda *args: pytest.fail("numerics before refusal"))
    with pytest.raises(ResearchError, match="restart-only"):
        execute_propagation(spec, spec["cells"][0], resume_state={"step": 1})
