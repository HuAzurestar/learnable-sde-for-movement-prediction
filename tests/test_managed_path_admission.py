"""Actual own settled path/IS pilots and independent targets, not a study."""

from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from application.path_qualification_admission import (METRIC, PAYLOAD_KEY, analysis_values,
    prepare_managed_paths, validate_formal_path_result)
from application.propagation_execution import request_from_manifest
from application.research_budget import BudgetLedger, BudgetSpec
from application.research_contracts import CapabilityRegistry
from application.research_evidence import export_evidence
from application.research_execution import execution_binding
from application.research_registry import plan_resources
from domain.errors import DataValidationError
from domain.frozen_dynamics import FrozenDynamicsPackage
from domain.path_production import PathProductionPolicy
from domain.path_qualification import PathQualificationPolicy, METHODS
from experiments.pirc25.affine import code_hash
from experiments.pirc25.runner import SharedRunner
from experiments.pirc27.path_production_plugin import path_production_plugin, path_production_config
from experiments.pirc27.plugin import execution_inputs, propagation_resume_command
from infrastructure.research_store import ResearchError, digest
from tests.research_admission_fixtures import admit_fixture
from tests.test_path_qualification import policies, prepared


def prepare_target(root, method, *, changes=None, target_changes=None):
    store, source_spec, source_registry = prepared(root, method=method, changes=changes)
    store.register(source_spec, digest(source_spec))
    pilot = SharedRunner(store, source_registry).run_cell(source_spec["study_id"], digest(source_spec["cells"][0]),
        budget=BudgetSpec(60, category="pilot"))
    assert pilot["state"] == "SUCCEEDED", pilot
    consumer = "independent-path-"+method
    original_grant = store.authorization(source_spec["admission"]["authorization_id"])
    source_grant = {**original_grant, "authorization_id": "path-source-consumer", "version": "1",
        "consumer_study_ids": [consumer], "purposes": ["evaluate", "export"]}
    store.authorize(source_grant)
    plugin = path_production_plugin()
    cell = deepcopy(source_spec["cells"][0])
    source_request = request_from_manifest(cell["propagation_request"])
    request = replace(source_request, **{"request_id": "independent-target", "seed": source_request.seed+100,
        "coupling_id": "independent-formal-root", **(target_changes or {})})
    qualification = PathQualificationPolicy.from_manifest(cell["path_qualification_policy"])
    production = PathProductionPolicy(request.request_hash, request.model_package_hash, code_hash(),
        source_request.request_hash, qualification.policy_hash, 60.)
    cell.update(plugin_id=plugin.plugin_id, execution_role="production", study_role="primary",
        block_id="independent-heldout-v1", seed=request.seed,
        propagation_request=json.loads(json.dumps(asdict(request))), path_production_policy=production.manifest())
    config, parameters = path_production_config(request, qualification, production), execution_inputs(request)
    cell["execution"] = execution_binding(plugin.registry_entry, config, parameters, matrix_cells=1)
    spec = {**deepcopy(source_spec), "study_id": consumer, "experiment_id": consumer, "cells": [cell]}
    pointer = {"schema_version": "managed-affine-path-qualification-v1", "policy": production.manifest(),
        "qualification_policy": qualification.manifest(), "source_attempt_id": pilot["attempt_id"],
        "source_artifact_id": pilot["artifact_id"], "source_authorization_id": source_grant["authorization_id"],
        "source_authorization_version": "1"}
    grant = admit_fixture(store, spec, plugin, root, formal=True, legacy_upstream=False,
        execution_config=config, execution_inputs=parameters, recovery_command_builder=propagation_resume_command,
        fixture_prefix=consumer+"-", input_content=("independent synthetic heldout "+consumer).encode(),
        primary_metrics=[METRIC], preregistration_extra={"path_qualification_policies": [qualification.manifest()],
            "path_production_policies": [production.manifest()]}, package_payload={PAYLOAD_KEY: pointer})
    store.register(spec, digest(spec))
    registry = CapabilityRegistry()
    registry.register(plugin)
    package = store.manifest("package-"+spec["admission"]["package_hash"])
    prereg = store.manifest("preregistration-"+package["preregistration_hash"])
    return store, spec, registry, grant, package, prereg


@pytest.fixture(scope="module", params=METHODS, ids=["euler", "heun", "rev", "is"])
def source(request, tmp_path_factory):
    return prepare_target(tmp_path_factory.mktemp("path-owner-"+request.param), request.param)


@pytest.fixture(scope="module")
def completed(source):
    store, spec, registry, grant, _, _ = source
    before = BudgetLedger(store).balance(spec["arms"][0]["arm_id"])["committed_ms"]
    outcome = SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]), budget=BudgetSpec(60))
    assert outcome["state"] == "SUCCEEDED", outcome
    result = json.loads(store.read_artifact(outcome["artifact_id"], purpose="evaluate", authorization=grant))
    return result, store.manifest("admission-"+result["admission_hash"]), outcome, before


def test_actual_independent_formal_target_retains_own_statistics_and_original_arm_cost(source, completed):
    store, spec, registry, grant, _, _ = source
    result, receipt, outcome, before = completed
    evidence = receipt["documents"]["propagation_qualification"]
    source_forecast = evidence["source_result"]["forecast"]
    current = result["forecast"]["path_output_analysis"]
    assert source_forecast["path_qualification_analysis"]["status"] == "PASSED"
    assert result["qualification"] == "qualified" and result["resume_level"] == "chunk"
    assert current["status"] == result["forecast"]["current_output_qualification"] == "PASSED"
    assert current["scientific_qualification"] is False
    assert current["actual_functional_hash"] != source_forecast["path_qualification_analysis"]["actual_functional_hash"]
    assert current["completed_statistics"]["rng_state"]["seed"] != source_forecast["path_qualification_analysis"]["completed_statistics"]["rng_state"]["seed"]
    assert current["request_hash"] != current["reference_request_hash"]
    assert current["reference_recomputation"] == "NONE; bounded saved same-law proof reused"
    assert current["sampling_error"]["uncertainty_radius"] > 0
    assert current["implementation_roundoff"]["value"] is None and current["model_error"]["value"] is None
    assert result["metrics"] == {METRIC: current["total_observed_functional_error_upper"]}
    after = BudgetLedger(store).balance(spec["arms"][0]["arm_id"])["committed_ms"]
    assert after > before > 0 and after-before == outcome["elapsed_ms"]
    reads = [e for e in store.events() if e["event_kind"] == "READ_COMPLETED"
        and e["payload"].get("attempt_id") == outcome["attempt_id"]]
    assert reads and evidence["completion_event"]["sequence"] < reads[0]["sequence"]
    validate_formal_path_result(receipt, spec, spec["cells"][0], result)
    reused = SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]))
    assert reused["reused"] and BudgetLedger(store).balance(spec["arms"][0]["arm_id"])["committed_ms"] == after
    bundle = export_evidence(store, spec["study_id"], grant)
    assert bundle["cells"][0]["result"] == result


@pytest.mark.parametrize("method", METHODS, ids=["euler", "heun", "rev", "is"])
def test_source_pass_does_not_promote_failed_current_sampling_or_ess(tmp_path, method):
    store, spec, registry, grant, _, _ = prepare_target(tmp_path, method,
        changes={"maximum_sampling_uncertainty": .2}, target_changes={"samples": 2})
    before = BudgetLedger(store).balance(spec["arms"][0]["arm_id"])["committed_ms"]
    outcome = SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]), budget=BudgetSpec(60))
    assert outcome["state"] == "SUCCEEDED", outcome  # Computational completion, not numerical approval.
    result = json.loads(store.read_artifact(outcome["artifact_id"], purpose="evaluate", authorization=grant))
    receipt = store.manifest("admission-"+result["admission_hash"])
    assert receipt["documents"]["propagation_qualification"]["source_result"]["forecast"]["path_qualification_analysis"]["status"] == "PASSED"
    current = result["forecast"]["path_output_analysis"]
    assert result["forecast"]["current_output_qualification"] == current["status"] == "FAILED"
    assert not all(current["checks"].values()) and current["scientific_qualification"] is False
    assert result["forecast"]["functional"]["sample_count"] == 2
    after = BudgetLedger(store).balance(spec["arms"][0]["arm_id"])["committed_ms"]
    assert after-before == outcome["elapsed_ms"] > 0
    validate_formal_path_result(receipt, spec, spec["cells"][0], result)
    bundle = export_evidence(store, spec["study_id"], grant)
    assert len(bundle["cells"]) == 1 and bundle["cells"][0]["result"]["forecast"]["current_output_qualification"] == "FAILED"


@pytest.mark.parametrize("fault", ["pointer", "borrowed", "policy", "duplicate", "metric", "artifact", "grant", "arm"])
def test_pre_read_source_gate_refuses_missing_or_wrong_bound_evidence_without_engine_replay(source, monkeypatch, fault):
    store, original, _, _, original_package, original_prereg = source
    spec, package, prereg = deepcopy(original), deepcopy(original_package), deepcopy(original_prereg)
    pointer = package["payload"][PAYLOAD_KEY]
    if fault == "pointer":
        package["payload"].pop(PAYLOAD_KEY)
    elif fault == "borrowed":
        pointer["schema_version"] = "managed-affine-analytic-qualification-v1"
    elif fault == "policy":
        prereg.pop("path_production_policies")
    elif fault == "duplicate":
        prereg["path_qualification_policies"] *= 2
    elif fault == "metric":
        prereg["primary_metrics"] = ["absolute_error_upper_vs_declared_affine_law"]
    elif fault == "artifact":
        pointer["source_artifact_id"] = "0"*64
    elif fault == "grant":
        pointer["source_authorization_id"] = spec["admission"]["authorization_id"]
    else:
        spec["arms"][0]["objective_id"] = "other-functional"
    import inference.path_qualification as module
    for name in ("monte_carlo", "importance_sampling", "bound_affine_reference", "bound_affine_discrete"):
        monkeypatch.setattr(module, name, lambda *a, **k: pytest.fail("owner gate replayed numerical engines"))
    with pytest.raises(ResearchError):
        prepare_managed_paths(store, spec, spec["cells"][0], package, prereg)


@pytest.mark.parametrize("fault", ["width", "error", "time", "sampling", "statistics", "stream", "unknown", "schema", "dimension", "binomial", "false-check"])
def test_resealed_source_values_cannot_replace_own_saved_numeric_meanings(source, completed, fault):
    _, spec, _, _, _, _ = source
    _, receipt, _, _ = completed
    evidence = deepcopy(receipt["documents"]["propagation_qualification"])
    analysis = evidence["source_result"]["forecast"]["path_qualification_analysis"]
    if fault in {"width", "error", "time"}:
        analysis[{"width": "reference_width_upper", "error": "total_observed_functional_error_upper", "time": "absolute_time_bias_upper"}[fault]] += 1
    elif fault == "sampling":
        analysis["sampling_error"]["uncertainty_radius"] = 0.
    elif fault == "statistics":
        analysis["completed_statistics"]["method_state"]["statistics"][0]["n"] -= 1
    elif fault == "stream":
        analysis["completed_statistics"]["rng_state"]["seed"] += 1
    elif fault == "unknown":
        analysis["implementation_roundoff"] = {"value": 0, "status": "BOUNDED"}
    elif fault == "schema":
        analysis["schema_version"] = "affine-analytic-qualification-analysis-v1"
    elif fault == "dimension":
        analysis["physical_dimension"] = 8
    elif fault == "binomial":
        analysis["sampling_error"]["interval_kind"] = "exact-binomial-one-sided-95-formula"
    else:
        analysis["checks"]["sampling_uncertainty"] = 1
    analysis["completed_statistics_hash"] = digest(analysis["completed_statistics"])
    analysis["analysis_hash"] = digest({k: v for k, v in analysis.items() if k != "analysis_hash"})
    sc = evidence["source_run"]["cell"]
    package = FrozenDynamicsPackage.from_manifest(sc["frozen_dynamics"], expected_hash=sc["propagation_request"]["model_package_hash"])
    with pytest.raises((ResearchError, DataValidationError)):
        analysis_values(analysis, PathQualificationPolicy.from_manifest(sc["path_qualification_policy"]), package,
            request_from_manifest(sc["propagation_request"]), evidence["source_result"]["forecast"]["functional"])


@pytest.mark.parametrize("fault", ["estimate", "se", "stats", "classification", "model", "metric", "reference-unit", "time-definition"])
def test_current_result_must_retain_its_own_statistics_error_and_classification(source, completed, fault):
    _, spec, _, _, _, _ = source
    original, receipt, _, _ = completed
    result = deepcopy(original)
    forecast = result["forecast"]
    current = forecast["path_output_analysis"]
    if fault == "estimate":
        forecast["functional"]["estimate"] += .001  # Even inside the coarse tolerance, not the actual output.
    elif fault == "se":
        forecast["functional"]["standard_error"] = 0.
    elif fault == "stats":
        current["completed_statistics"]["rng_state"]["coupling_id"] = "pilot-root"
    elif fault == "classification":
        forecast["current_output_qualification"] = "FAILED"
    elif fault == "model":
        forecast["functional"]["error_budget"]["model"]["value"] = 0.
    elif fault == "metric":
        result["metrics"][METRIC] += 1
    elif fault == "reference-unit":
        forecast["functional"]["error_budget"]["reference"]["units"] = "ms"
    else:
        forecast["functional"]["error_budget"]["time_discretization"]["estimated_by"] = "free floating oracle"
    current["analysis_hash"] = digest({k: v for k, v in current.items() if k != "analysis_hash"})
    with pytest.raises((ResearchError, DataValidationError)):
        validate_formal_path_result(receipt, spec, spec["cells"][0], result)


@pytest.mark.parametrize("changes", [{"request_hash": "0"*64}, {"source_request_hash": "0"*64},
    {"model_package_hash": "0"*64}, {"qualification_policy_hash": "0"*64}, {"code_hash": "0"*64},
    {"maximum_job_seconds": True}, {"maximum_job_seconds": 7201.}, {"maximum_job_seconds": float("nan")},
    {"maximum_job_seconds": []}, {"schema_version": []}])
def test_invalid_production_policy_is_bounded_before_copy_or_sampling(changes):
    package, source_request, qualification = policies()
    target = replace(source_request, request_id="target", seed=19, coupling_id="independent")
    policy = PathProductionPolicy(target.request_hash, package.package_hash, code_hash(),
        source_request.request_hash, qualification.policy_hash, 60.)
    with pytest.raises(DataValidationError):
        replace(policy, **changes).validate(package, target, qualification, code_hash())


@pytest.mark.parametrize("change", [{"seed": 71}, {"coupling_id": "paired-root-v1"},
    {"request_id": "test-endpoint"}, {"horizons": (2.,)}, {"steps": 4}, {"threshold": .5},
    {"normal": (0., 1., 0., 0.)}, {"closed": False}, {"arm_id": "fresh-arm"}])
def test_production_binding_refuses_same_stream_or_changed_law_and_arm(change):
    package, source_request, qualification = policies()
    target = replace(source_request, **{"request_id": "target", "seed": 19, "coupling_id": "independent", **change})
    policy = PathProductionPolicy(target.request_hash, package.package_hash, code_hash(),
        source_request.request_hash, qualification.policy_hash, 60.)
    with pytest.raises(DataValidationError):
        policy.validate_source_request(package, source_request, target, qualification, code_hash())


def test_production_policy_canonical_early_copy_and_saved_only_resource_proxy(monkeypatch):
    import domain.path_production as module
    package, source_request, qualification = policies()
    target = replace(source_request, request_id="target", seed=19, coupling_id="independent", samples=100,
        chunk_size=3, tolerance=2.)
    policy = PathProductionPolicy(target.request_hash, package.package_hash, code_hash(),
        source_request.request_hash, qualification.policy_hash, 60.)
    policy.validate_source_request(package, source_request, target, qualification, code_hash())
    assert PathProductionPolicy.from_manifest(policy.manifest()) == policy
    config = path_production_config(target, qualification, policy)
    plan = plan_resources(path_production_plugin().registry_entry, config, execution_inputs(target), matrix_cells=1)
    assert plan["counts"]["steps"] == target.samples*target.steps
    assert plan["counts"]["state_dim"] == 4 and plan["tensor_bytes"] >= 64*1024*1024
    cycle = []
    cycle.append(cycle)
    value = {**policy.manifest(), "maximum_job_seconds": cycle}
    monkeypatch.setattr(module, "asdict", lambda *a: pytest.fail("malformed fields deep-copied"))
    with pytest.raises(DataValidationError):
        PathProductionPolicy.from_manifest(value)


def test_failed_source_cannot_authorize_target_reads_or_worker(tmp_path):
    store, spec, registry, _, _, _ = prepare_target(tmp_path, "euler", changes={"maximum_sampling_uncertainty": 1e-10})
    before = len(store.events())
    with pytest.raises(ResearchError, match="source numerical"):
        SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]), budget=BudgetSpec(60))
    new = store.events()[before:]
    assert not any(e["event_kind"] == "WORKER_STARTED" for e in new)
    assert not any(e["event_kind"] == "READ_COMPLETED" and e["payload"].get("block_id") == "independent-heldout-v1" for e in new)


def test_target_job_cannot_exceed_frozen_cap_before_reservation(source):
    store, spec, registry, _, _, _ = source
    # Separate never-executed study on the same original arm, not reopening a
    # succeeded cell or refreshing any cumulative allowance.
    from application.research_admission import AdmissionGate
    budget_spec = {**deepcopy(spec), "study_id": spec["study_id"]+"-budget", "experiment_id": spec["experiment_id"]+"-budget"}
    store.register(budget_spec, digest(budget_spec))
    cell = budget_spec["cells"][0]
    attempt = store.new_attempt(store.register_run(budget_spec["study_id"], cell))
    before = len(store.events())
    with pytest.raises(ResearchError, match="frozen independent production job budget"):
        AdmissionGate(store).run(attempt, budget_spec, cell, path_production_plugin(),
            lambda output: pytest.fail("bad budget reached command"), BudgetSpec(61))
    assert not any(e["event_kind"] in {"RESERVE", "READ_COMPLETED", "WORKER_STARTED"} for e in store.events()[before:])
