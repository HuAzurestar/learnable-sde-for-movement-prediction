"""Actual bounded analytic qualification work; disposable synthetic controls."""

from dataclasses import asdict, replace
import json

import pytest

from application.research_admission import AdmissionGate
from application.research_budget import BudgetLedger, BudgetSpec
from application.research_contracts import CapabilityRegistry
from application.research_execution import execution_binding
from application.research_registry import plan_resources
from domain.affine_qualification import AffineQualificationPolicy
from domain.errors import DataValidationError
from experiments.pirc25.affine import code_hash
from experiments.pirc25.runner import SharedRunner
from experiments.pirc27.plugin import execution_inputs
from experiments.pirc27.qualification_plugin import qualification_config, qualification_plugin
from inference.affine_qualification import analyze_affine_qualification
from inference.propagation_methods import analytic_estimate
from infrastructure.research_store import ResearchError, ResearchStore, digest
from tests.research_admission_fixtures import admit_fixture
from tests.test_propagation_methods import inputs


def policy_for(package, request, active_method, **changes):
    policy = AffineQualificationPolicy(request.request_hash, package.package_hash, code_hash(), active_method,
        maximum_reference_width=1e-20, maximum_functional_roundoff=1e-8,
        maximum_time_bias=1., maximum_scaled_transition_norm=10.,
        state_scales=(1., 1., 1., 1.), maximum_operations=400_001, maximum_job_seconds=60.)
    return replace(policy, **changes)


def prepared(tmp_path, method, *, policy_changes=None, formal=False):
    store = ResearchStore(tmp_path, "analytic-qualification-unit", initialize=True)
    package, request = inputs(samples=2, steps=4)
    policy = policy_for(package, request, method, **(policy_changes or {}))
    plugin = qualification_plugin()
    config, parameters = qualification_config(request, method, policy), execution_inputs(request)
    cell = {"arm_id": request.arm_id, "block_id": "generator-v1", "seed": request.seed,
        "horizon": request.horizons[0], "plugin_id": plugin.plugin_id,
        "capability": "exact-transition" if method == "exact" else "generic-rollout",
        "resource_class": "cpu", "visibility": "synthetic", "dimensions": 4,
        "study_role": "secondary", "execution_role": "qualification",
        "propagation_request": json.loads(json.dumps(asdict(request))),
        "frozen_dynamics": package.manifest(), "affine_qualification_policy": policy.manifest(),
        "execution": execution_binding(plugin.registry_entry, config, parameters, matrix_cells=1)}
    spec = {"schema_version": "pirc25-contract-v1", "study_id": "analytic-qualification-unit",
        "experiment_id": "analytic-qualification-unit", "comparison_family": "synthetic-engineering",
        "code_hash": code_hash(), "protocol_hash": digest("pending"), "data_hash": digest("pending"),
        "feature_hash": digest("none"), "selection_hash": digest("none"),
        "arms": [{"arm_id": request.arm_id, "model_family_id": "affine-stable-v1",
            "method_family_id": method, "objective_id": request.functional, "budget_seconds": 86400}],
        "cells": [cell], "runtime_binding": {"root": str(tmp_path.resolve()), "store_id": store.store_id}}
    admit_fixture(store, spec, plugin, tmp_path, formal=formal, execution_config=config,
        execution_inputs=parameters, legacy_upstream=False)
    spec["admission"]["mode"] = "formal" if formal else "pilot"
    registry = CapabilityRegistry()
    registry.register(plugin)
    return store, spec, registry


@pytest.mark.parametrize("method", ["exact", "gaussian"])
@pytest.mark.parametrize("functional", ["endpoint-x", "endpoint-halfspace"])
def test_actual_numerical_analysis_passes_bound_thresholds_not_a_flag(method, functional):
    package, request = inputs(samples=2, functional=functional)
    policy = policy_for(package, request, method)
    result = analytic_estimate(package, request, discrete=method == "gaussian")
    analysis = analyze_affine_qualification(package, request, method, policy, result)
    assert analysis["status"] == "PASSED" and all(analysis["checks"].values())
    assert analysis["functional_roundoff_upper"] <= policy.maximum_functional_roundoff
    assert analysis["reference_width_upper"] <= policy.maximum_reference_width
    assert analysis["absolute_time_bias_upper"] <= policy.maximum_time_bias
    assert analysis["scientific_qualification"] is False
    assert analysis["model_error"] == {"value": None, "status": "NOT_IDENTIFIABLE"}
    assert analysis["sampling_error"] == {"value": 0, "status": "NOT_APPLICABLE"}
    assert analysis["cost_status"] == "OWNER_SETTLEMENT_REQUIRED"
    assert digest({key: value for key, value in analysis.items() if key != "analysis_hash"}) == analysis["analysis_hash"]


@pytest.mark.parametrize("changes,check", [({"maximum_reference_width": 1e-100}, "reference_width"),
    ({"maximum_functional_roundoff": 1e-100}, "functional_roundoff"),
    ({"maximum_time_bias": 1e-10}, "time_bias"),
    ({"maximum_scaled_transition_norm": .1}, "scaled_transition_growth"),
    ({"maximum_operations": 1}, "arithmetic_operations")])
def test_failed_check_is_retained_without_repair_or_threshold_expansion(changes, check):
    package, request = inputs(samples=2, steps=4)
    policy = policy_for(package, request, "gaussian", **changes)
    result = analytic_estimate(package, request, discrete=True)
    analysis = analyze_affine_qualification(package, request, "gaussian", policy, result)
    assert analysis["status"] == "FAILED" and analysis["checks"][check] is False
    assert analysis["policy_hash"] == policy.policy_hash and analysis["scientific_qualification"] is False


@pytest.mark.parametrize("changes", [{"request_hash": "0"*64}, {"model_package_hash": "0"*64},
    {"code_hash": "0"*64}, {"method": "euler"}, {"maximum_job_seconds": 1801},
    {"maximum_operations": True}, {"maximum_operations": 400_002}, {"state_scales": (1., 0., 1., 1.)},
    {"maximum_reference_width": float("nan")}, {"maximum_time_bias": float("inf")},
    {"maximum_functional_roundoff": 10**400}])
def test_invalid_unbound_or_unbounded_policy_refuses_before_numerics(changes):
    package, request = inputs(samples=2)
    policy = policy_for(package, request, "exact", **changes)
    with pytest.raises(DataValidationError):
        policy.validate(package, request, "exact", code_hash())


def test_caller_cannot_substitute_another_estimate_as_an_analytic_qualification():
    package, request = inputs(samples=2)
    policy = policy_for(package, request, "exact")
    result = analytic_estimate(package, request)
    with pytest.raises(DataValidationError, match="actual bound analytic"):
        analyze_affine_qualification(package, request, "exact", policy, replace(result, estimate=result.estimate+1))


def test_policy_detaches_manifest_and_refuses_missing_fields():
    package, request = inputs(samples=2)
    policy = policy_for(package, request, "exact")
    document = policy.manifest()
    restored = AffineQualificationPolicy.from_manifest(document)
    document["state_scales"][0] = 100
    assert restored == policy
    document = policy.manifest()
    document.pop("maximum_time_bias")
    with pytest.raises(DataValidationError):
        AffineQualificationPolicy.from_manifest(document)


def test_qualification_has_explicit_nonfloat_resource_pool_and_never_changes_physical_dimension():
    package, request = inputs(samples=2)
    policy = policy_for(package, request, "exact")
    plugin = qualification_plugin()
    plan = plan_resources(plugin.registry_entry, qualification_config(request, "exact", policy),
        execution_inputs(request), matrix_cells=1)
    pools = [tensor for tensor in plan["tensors"] if "-pool-" in tensor["name"]]
    assert len(pools) == 64 and sum(tensor["bytes"] for tensor in pools) == 64*1024*1024
    assert plan["tensor_bytes"] >= 64*1024*1024 and plan["counts"]["state_dim"] == 4


@pytest.mark.parametrize("method", ["exact", "gaussian"])
def test_real_shared_worker_generates_qualification_evidence_under_existing_pilot_budget(tmp_path, method):
    store, spec, registry = prepared(tmp_path, method)
    store.register(spec, digest(spec))
    result = SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]),
        budget=BudgetSpec(60, category="pilot"))
    assert result["state"] == "SUCCEEDED", result
    artifact = json.loads((store.path / "artifacts" / result["artifact_id"]).read_bytes())
    analysis = artifact["forecast"]["qualification_analysis"]
    assert analysis["status"] == "PASSED" and artifact["qualification"] == "fixture"
    assert analysis["scientific_qualification"] is False
    events = store.events()
    assert len([e for e in events if e["event_kind"] == "WORKER_STARTED"]) == 1
    settles = [e for e in events if e["event_kind"] == "SETTLE"]
    assert len(settles) == 1 and settles[0]["payload"]["charged_ms"] > 0
    assert BudgetLedger(store).balance(spec["arms"][0]["arm_id"])["committed_ms"] == settles[0]["payload"]["charged_ms"]
    receipt = store.manifest("admission-"+artifact["admission_hash"])
    assert receipt["mode"] == "pilot" and receipt["documents"]["protocol"]["blocks"][0]["split_role"] == "train"
    assert receipt["resource_plan"]["tensor_bytes"] >= 64*1024*1024


def test_real_worker_retains_failed_numerical_qualification_without_claiming_pass(tmp_path):
    store, spec, registry = prepared(tmp_path, "gaussian", policy_changes={"maximum_time_bias": 1e-10})
    store.register(spec, digest(spec))
    result = SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]), budget=BudgetSpec(60, category="pilot"))
    assert result["state"] == "SUCCEEDED", result
    artifact = json.loads((store.path / "artifacts" / result["artifact_id"]).read_bytes())
    assert artifact["forecast"]["qualification_analysis"]["status"] == "FAILED"
    assert artifact["qualification"] == "fixture"
    assert BudgetLedger(store).balance(spec["arms"][0]["arm_id"])["committed_ms"] > 0


@pytest.mark.parametrize("budget", [BudgetSpec(60, category="smoke"), BudgetSpec(61, category="pilot")])
def test_wrong_budget_is_refused_before_any_input_read_or_worker(tmp_path, budget):
    store, spec, registry = prepared(tmp_path, "exact")
    store.register(spec, digest(spec))
    with pytest.raises(ResearchError, match="frozen pilot job budget"):
        SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]), budget=budget)
    assert next(iter(store.attempts().values()))["state"] == "PREFLIGHT_FAILED"
    assert not any(e["event_kind"] in {"READ_COMPLETED", "WORKER_STARTED", "RESERVE"} for e in store.events())


def test_formal_input_cannot_be_consumed_by_qualification_even_with_imported_pass_report(tmp_path):
    store, spec, registry = prepared(tmp_path, "exact", formal=True)
    store.register(spec, digest(spec))
    cell = spec["cells"][0]
    run = store.register_run(spec["study_id"], cell)
    attempt = store.new_attempt(run)
    # Call prepare directly to exercise the protected-input gate, bypassing
    # the earlier configuration/mode check in the run wrapper.
    with pytest.raises(ResearchError, match="cannot consume test/final-eval"):
        AdmissionGate(store).prepare(spec, cell, qualification_plugin(), attempt)
    assert not any(e["event_kind"] in {"READ_COMPLETED", "WORKER_STARTED"} for e in store.events())
