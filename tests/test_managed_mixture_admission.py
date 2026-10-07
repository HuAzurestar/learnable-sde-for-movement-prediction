"""Actual settled mixture pilot -> formal owner target, disposable controls."""

from copy import deepcopy
import json

import pytest

from application.mixture_qualification_admission import (METRIC, PAYLOAD_KEY,
    analysis_values, prepare_managed_mixture, validate_formal_mixture_result)
from application.propagation_execution import request_from_manifest
from application.research_admission import AdmissionGate
from application.research_budget import BudgetLedger, BudgetSpec
from application.research_contracts import CapabilityRegistry
from application.research_evidence import export_evidence
from application.research_execution import execution_binding
from application.research_registry import plan_resources
from domain.frozen_dynamics import FrozenDynamicsPackage
from domain.mixture import MixturePolicy
from domain.mixture_qualification import MixtureQualificationPolicy
from experiments.pirc25.runner import SharedRunner
from experiments.pirc27.mixture_production_plugin import (mixture_production_plugin,
    mixture_production_config, mixture_production_recovery_plugin)
from experiments.pirc27.plugin import execution_inputs, propagation_resume_command
from infrastructure.research_store import ResearchError, digest, encode
from tests.research_admission_fixtures import admit_fixture
from tests.test_mixture_qualification import prepared


def prepare_target(root, cap, changes=None):
    store, source_spec, source_registry = prepared(root, cap=cap, changes=changes)
    store.register(source_spec, digest(source_spec))
    pilot = SharedRunner(store, source_registry).run_cell(source_spec["study_id"], digest(source_spec["cells"][0]),
        budget=BudgetSpec(60, category="pilot"))
    assert pilot["state"] == "SUCCEEDED", pilot
    consumer = "formal-mixture-cap-"+str(cap)
    original_grant = store.authorization(source_spec["admission"]["authorization_id"])
    source_grant = {**original_grant, "authorization_id": "mixture-source-consumer", "version": "1",
        "consumer_study_ids": [consumer], "purposes": ["evaluate", "export"]}
    store.authorize(source_grant)
    plugin = mixture_production_plugin()
    cell = deepcopy(source_spec["cells"][0])
    cell.update(plugin_id=plugin.plugin_id, execution_role="production", study_role="primary",
        block_id="independent-heldout-v1")
    request = request_from_manifest(cell["propagation_request"])
    mixture = MixturePolicy.from_manifest(cell["mixture_policy"])
    policy = MixtureQualificationPolicy.from_manifest(cell["mixture_qualification_policy"])
    config, inputs = mixture_production_config(request, mixture, policy), execution_inputs(request)
    cell["execution"] = execution_binding(plugin.registry_entry, config, inputs, matrix_cells=1)
    spec = {**deepcopy(source_spec), "study_id": consumer, "experiment_id": consumer, "cells": [cell]}
    pointer = {"schema_version": "managed-affine-mixture-qualification-v1", "policy": policy.manifest(),
        "source_attempt_id": pilot["attempt_id"], "source_artifact_id": pilot["artifact_id"],
        "source_authorization_id": source_grant["authorization_id"], "source_authorization_version": "1"}
    grant = admit_fixture(store, spec, plugin, root, formal=True, legacy_upstream=False,
        execution_config=config, execution_inputs=inputs, recovery_command_builder=propagation_resume_command,
        fixture_prefix=consumer+"-", input_content=("independent heldout synthetic control "+consumer).encode(),
        primary_metrics=[METRIC], preregistration_extra={"mixture_qualification_policies": [policy.manifest()]},
        package_payload={PAYLOAD_KEY: pointer})
    store.register(spec, digest(spec))
    registry = CapabilityRegistry()
    registry.register(plugin)
    execution_package = store.manifest("package-"+spec["admission"]["package_hash"])
    prereg = store.manifest("preregistration-"+execution_package["preregistration_hash"])
    return store, spec, registry, grant, execution_package, prereg


@pytest.fixture(scope="module", params=[1, 4])
def source(request, tmp_path_factory):
    return prepare_target(tmp_path_factory.mktemp("managed-mixture-cap-"+str(request.param)), request.param)


def test_actual_formal_target_requires_settled_pilot_and_retains_error_lineage_and_original_cost(source):
    store, spec, registry, grant, _, _ = source
    arm = spec["arms"][0]["arm_id"]
    before = BudgetLedger(store).balance(arm)["committed_ms"]
    outcome = SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]), budget=BudgetSpec(60))
    assert outcome["state"] == "SUCCEEDED", outcome
    result = json.loads(store.read_artifact(outcome["artifact_id"], purpose="evaluate", authorization=grant))
    receipt = store.manifest("admission-"+result["admission_hash"])
    evidence = receipt["documents"]["propagation_qualification"]
    source_result = evidence["source_result"]
    assert source_result["qualification"] == "fixture" and result["qualification"] == "qualified"
    assert result["forecast"]["functional"]["status"] == "APPROXIMATION_ONLY"
    assert source_result["forecast"]["mixture_qualification_analysis"]["scientific_qualification"] is False
    errors = result["forecast"]["qualified_error_components"]
    assert errors["model_error"] == errors["propagation_approximation"] == errors["implementation_roundoff"] == {
        "value": None, "status": "NOT_IDENTIFIABLE"}
    assert result["metrics"] == {METRIC: errors["total_functional_error_upper"]}
    assert result["forecast"]["functional"]["error_budget"]["reference"]["status"] == "BOUNDED"
    assert errors["mixture_policy_hash"] == digest(spec["cells"][0]["mixture_policy"])
    after = BudgetLedger(store).balance(arm)["committed_ms"]
    assert after > before > 0 and after-before == outcome["elapsed_ms"]
    reads = [e for e in store.events() if e["event_kind"] == "READ_COMPLETED"
        and e["payload"].get("attempt_id") == outcome["attempt_id"]]
    assert reads and evidence["settlement_event"]["sequence"] < reads[0]["sequence"]
    assert evidence["stop_event"]["payload"]["confirmation"] == "native-job-or-process-group-no-running-descendants"
    assert dict(result["forecast"]["functional"]["diagnostics"])["lineage_digest"] == dict(
        source_result["forecast"]["functional"]["diagnostics"])["lineage_digest"]
    reused = SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]))
    assert reused["reused"] and BudgetLedger(store).balance(arm)["committed_ms"] == after
    bundle = export_evidence(store, spec["study_id"], grant)
    assert bundle["cells"][0]["result"] == result
    assert bundle["cells"][0]["admission"]["documents"]["propagation_qualification"] == evidence


@pytest.fixture(scope="module")
def completed(source):
    store, spec, registry, grant, _, _ = source
    outcome = SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]))
    assert outcome["state"] == "SUCCEEDED"
    result = json.loads(store.read_artifact(outcome["artifact_id"], purpose="evaluate", authorization=grant))
    return result, store.manifest("admission-"+result["admission_hash"])


@pytest.mark.parametrize("fault", ["missing-pointer", "borrowed-analytic", "missing-policy", "duplicate-policy",
    "wrong-primary", "wrong-artifact", "wrong-policy", "wrong-arm", "wrong-source-grant"])
def test_missing_or_wrong_source_binding_is_refused_without_numerical_replay(source, monkeypatch, fault):
    store, original_spec, _, _, original_package, original_prereg = source
    spec, package, prereg = deepcopy(original_spec), deepcopy(original_package), deepcopy(original_prereg)
    pointer = package["payload"][PAYLOAD_KEY]
    if fault == "missing-pointer":
        package["payload"].pop(PAYLOAD_KEY)
    elif fault == "borrowed-analytic":
        pointer["schema_version"] = "managed-affine-analytic-qualification-v1"
    elif fault == "missing-policy":
        prereg.pop("mixture_qualification_policies")
    elif fault == "duplicate-policy":
        prereg["mixture_qualification_policies"] *= 2
    elif fault == "wrong-primary":
        prereg["primary_metrics"] = ["functional_estimate"]
    elif fault == "wrong-artifact":
        pointer["source_artifact_id"] = "0"*64
    elif fault == "wrong-policy":
        pointer["policy"]["maximum_retained_functional_error"] = .5
    elif fault == "wrong-arm":
        spec["arms"][0]["objective_id"] = "another-functional"
    else:
        pointer["source_authorization_id"] = spec["admission"]["authorization_id"]
    import inference.mixture_propagation as mixture_module
    import inference.affine_reference as reference_module
    import inference.affine_discrete_reference as grid_module
    def forbidden(*args, **kwargs):
        pytest.fail("pre-read gate replayed numerical engines")
    monkeypatch.setattr(mixture_module, "mixture_estimate", forbidden)
    monkeypatch.setattr(reference_module, "bound_affine_reference", forbidden)
    monkeypatch.setattr(grid_module, "bound_affine_discrete", forbidden)
    with pytest.raises(ResearchError):
        prepare_managed_mixture(store, spec, spec["cells"][0], package, prereg)


@pytest.mark.parametrize("fault", ["reference-width", "retained-error", "total-error", "time-bias", "growth",
    "operations", "zero-closure", "zero-roundoff", "zero-model", "false-check", "gaussian-schema", "wrong-grid",
    "wrong-certificate-scope", "bad-dyadic", "bool-scalar"])
def test_resealed_saved_numeric_substitutions_cannot_pass(completed, source, fault):
    _, receipt = completed
    _, spec, _, _, _, _ = source
    evidence = deepcopy(receipt["documents"]["propagation_qualification"])
    analysis = evidence["source_result"]["forecast"]["mixture_qualification_analysis"]
    functional = evidence["source_result"]["forecast"]["functional"]
    if fault in {"reference-width", "retained-error", "total-error", "time-bias", "growth", "operations"}:
        key = {"reference-width": "reference_width_upper", "retained-error": "retained_functional_error_upper",
            "total-error": "total_functional_error_upper", "time-bias": "absolute_time_bias_upper",
            "growth": "scaled_transition_norm_upper", "operations": "reference_operations"}[fault]
        analysis[key] += 1
    elif fault in {"zero-closure", "zero-roundoff", "zero-model"}:
        analysis[{"zero-closure": "propagation_approximation", "zero-roundoff": "implementation_roundoff",
            "zero-model": "model_error"}[fault]] = {"value": 0, "status": "BOUNDED"}
    elif fault == "false-check":
        analysis["checks"]["total_functional_error"] = False
    elif fault == "gaussian-schema":
        analysis["schema_version"] = "affine-analytic-qualification-analysis-v1"
    elif fault == "wrong-grid":
        analysis["target_certificate"]["grid"]["solver"] = "heun"
    elif fault == "wrong-certificate-scope":
        analysis["continuous_certificate"]["scope"] = "declared-affine-finite-grid-Gaussian-law"
    elif fault == "bad-dyadic":
        analysis["continuous_certificate"]["functional_bounds"][0] = ["1", "3"]
    else:
        functional["estimate"] = True
        analysis["actual_estimate"] = True
    analysis["actual_functional_hash"] = digest(functional)
    for kind in ("continuous", "target"):
        analysis[kind+"_certificate_hash"] = digest(analysis[kind+"_certificate"])
    analysis["analysis_hash"] = digest({k: v for k, v in analysis.items() if k != "analysis_hash"})
    cell = spec["cells"][0]
    package = FrozenDynamicsPackage.from_manifest(cell["frozen_dynamics"], expected_hash=cell["propagation_request"]["model_package_hash"])
    with pytest.raises((ResearchError, ValueError)):
        analysis_values(analysis, MixtureQualificationPolicy.from_manifest(evidence["policy"]),
            MixturePolicy.from_manifest(evidence["mixture_policy"]), package,
            request_from_manifest(cell["propagation_request"]), functional)


@pytest.mark.parametrize("fault", ["scalar-outside-target", "scalar-within-target", "lineage", "metric", "zero-model", "reference"])
def test_formal_current_result_rejects_scalar_lineage_and_error_substitution(completed, source, fault):
    original, receipt = completed
    _, spec, _, _, _, _ = source
    result = deepcopy(original)
    if fault == "scalar-outside-target":
        result["forecast"]["functional"]["estimate"] += 2
    elif fault == "scalar-within-target":
        result["forecast"]["functional"]["estimate"] += .001
    elif fault == "lineage":
        diagnostics = dict(result["forecast"]["functional"]["diagnostics"])
        diagnostics["lineage_digest"] = "0"*64
        result["forecast"]["functional"]["diagnostics"] = list(diagnostics.items())
    elif fault == "metric":
        result["metrics"][METRIC] += 1
    elif fault == "zero-model":
        result["forecast"]["functional"]["error_budget"]["model"]["value"] = 0
    else:
        result["forecast"]["functional"]["error_budget"]["reference"]["value"] += 1
    with pytest.raises(ResearchError):
        validate_formal_mixture_result(receipt, spec, spec["cells"][0], result)


def test_actual_failed_tail_pilot_is_not_promoted_before_heldout_read(tmp_path):
    store, spec, _, _, _, _ = prepare_target(tmp_path, 4, {"maximum_retained_functional_error": .01})
    cell = spec["cells"][0]
    attempt = store.new_attempt(store.register_run(spec["study_id"], cell))
    with pytest.raises(ResearchError, match="numerical checks failed"):
        AdmissionGate(store).prepare(spec, cell, mixture_production_plugin(), attempt)
    assert not any(e["event_kind"] in {"READ_COMPLETED", "WORKER_STARTED"}
        and e["payload"].get("attempt_id") == attempt for e in store.events())


def test_formal_job_cap_rejects_before_reservation_or_input(source):
    store, spec, _, _, _, _ = source
    cell = spec["cells"][0]
    attempt = store.new_attempt(store.register_run(spec["study_id"], cell))
    sequence = len(store.events())
    with pytest.raises(ResearchError, match="frozen qualification job budget"):
        AdmissionGate(store).run(attempt, spec, cell, mixture_production_plugin(),
            lambda output: pytest.fail("invalid job reached command builder"), BudgetSpec(61))
    assert not any(e["event_kind"] in {"RESERVE", "READ_COMPLETED", "WORKER_STARTED"} for e in store.events()[sequence:])


def test_production_registry_preserves_physical_dimensions_saved_proof_and_chunk_limits(source):
    _, spec, _, _, _, _ = source
    cell = spec["cells"][0]
    plugin = mixture_production_plugin()
    plan = plan_resources(plugin.registry_entry, cell["execution"]["config"], cell["execution"]["inputs"], matrix_cells=1)
    assert plan["counts"]["state_dim"] == 4 and plan["tensor_bytes"] >= 64*1024*1024
    assert plan["counts"]["steps"] == cell["propagation_request"]["steps"]*MixturePolicy.from_manifest(cell["mixture_policy"]).work_per_step
    assert plugin.resume_level == mixture_production_recovery_plugin().resume_level == "chunk"
