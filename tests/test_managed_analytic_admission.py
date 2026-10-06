"""Actual qualification pilot -> pre-read owner gate -> formal analytic worker.

Synthetic engineering controls only. No imported numeric pass or fabricated
worker/settlement evidence, new real arm, official store or research permission.
"""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from application.propagation_execution import execute_propagation, request_from_manifest
from application.propagation_qualification_admission import (METRIC, PAYLOAD_KEY,
    _analysis_values, load_worker_admission, prepare_managed_qualification, validate_formal_analytic_result)
from application.research_admission import AdmissionGate
from application.research_budget import BudgetLedger, BudgetSpec
from application.research_contracts import CapabilityRegistry
from application.research_evidence import export_evidence
from application.research_execution import execution_binding
from domain.affine_qualification import AffineQualificationPolicy
from domain.frozen_dynamics import FrozenDynamicsPackage
from experiments.pirc25.runner import SharedRunner
from experiments.pirc27.plugin import propagation_plugin, execution_config, execution_inputs
from infrastructure.research_store import ResearchError, digest, encode
from tests.research_admission_fixtures import admit_fixture
from tests.test_affine_qualification import prepared


@pytest.fixture(scope="module", params=["exact", "gaussian"])
def source(request, tmp_path_factory):
    root = tmp_path_factory.mktemp("managed-analytic-"+request.param)
    store, source_spec, source_registry = prepared(root, request.param)
    store.register(source_spec, digest(source_spec))
    outcome = SharedRunner(store, source_registry).run_cell(source_spec["study_id"], digest(source_spec["cells"][0]),
        budget=BudgetSpec(60, category="pilot"))
    assert outcome["state"] == "SUCCEEDED", outcome
    policy = source_spec["cells"][0]["affine_qualification_policy"]
    consumer_id = "formal-analytic-"+request.param
    original_grant = store.authorization(source_spec["admission"]["authorization_id"])
    source_grant = {**original_grant, "authorization_id": "analytic-source-consumer", "version": "1",
        "consumer_study_ids": [consumer_id], "purposes": ["evaluate", "export"]}
    store.authorize(source_grant)
    pointer = {"schema_version": "managed-affine-analytic-qualification-v1", "policy": policy,
        "source_attempt_id": outcome["attempt_id"], "source_artifact_id": outcome["artifact_id"],
        "source_authorization_id": source_grant["authorization_id"], "source_authorization_version": "1"}
    plugin = propagation_plugin()
    cell = deepcopy(source_spec["cells"][0])
    cell.pop("affine_qualification_policy")
    cell.pop("execution_role")
    cell.update(plugin_id=plugin.plugin_id, study_role="primary")
    propagation_request = request_from_manifest(cell["propagation_request"])
    config, inputs = execution_config(propagation_request, request.param), execution_inputs(propagation_request)
    cell["execution"] = execution_binding(plugin.registry_entry, config, inputs, matrix_cells=1)
    cell["block_id"] = "independent-heldout-v1"
    spec = {**deepcopy(source_spec), "study_id": consumer_id, "experiment_id": consumer_id, "cells": [cell]}
    # Independent content AND source-block identity, not merely a renamed or
    # reseeded copy of the pilot's already-read training data.
    grant = admit_fixture(store, spec, plugin, root, formal=True, legacy_upstream=False,
        execution_config=config, execution_inputs=inputs, fixture_prefix=consumer_id+"-",
        input_content=("independent heldout synthetic control for "+request.param).encode(),
        primary_metrics=[METRIC], preregistration_extra={"propagation_qualification_policies": [policy]},
        package_payload={PAYLOAD_KEY: pointer})
    store.register(spec, digest(spec))
    registry = CapabilityRegistry()
    registry.register(plugin)
    execution_package = store.manifest("package-"+spec["admission"]["package_hash"])
    prereg = store.manifest("preregistration-"+execution_package["preregistration_hash"])
    return store, spec, registry, grant, execution_package, prereg


def test_actual_formal_worker_requires_settled_source_and_retains_error_and_cost_chain(source):
    store, spec, registry, grant, _, _ = source
    before = BudgetLedger(store).balance(spec["arms"][0]["arm_id"])["committed_ms"]
    outcome = SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]), budget=BudgetSpec(60))
    assert outcome["state"] == "SUCCEEDED", outcome
    content = store.read_artifact(outcome["artifact_id"], purpose="evaluate", authorization=grant)
    result = json.loads(content)
    assert result["qualification"] == "qualified" and set(result["metrics"]) == {METRIC}
    receipt = store.manifest("admission-"+result["admission_hash"])
    evidence = receipt["documents"]["propagation_qualification"]
    assert evidence["source_result"]["qualification"] == "fixture"
    assert evidence["source_result"]["forecast"]["qualification_analysis"]["scientific_qualification"] is False
    assert result["forecast"]["qualified_error_components"]["model_error"]["value"] is None
    assert result["forecast"]["functional"]["error_budget"]["reference"]["status"] == "BOUNDED"
    assert result["forecast"]["functional"]["error_budget"]["time_discretization"]["status"] == "BOUNDED"
    after = BudgetLedger(store).balance(spec["arms"][0]["arm_id"])["committed_ms"]
    assert after > before > 0
    assert outcome["elapsed_ms"] == after-before
    events = store.events()
    heldout_reads = [e for e in events if e["event_kind"] == "READ_COMPLETED"
                    and e["payload"].get("attempt_id") == outcome["attempt_id"]]
    assert heldout_reads and evidence["settlement_event"]["sequence"] < heldout_reads[0]["sequence"]
    assert evidence["source_admission"]["mode"] == "pilot"
    charged = after
    reuse = SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]))
    assert reuse["reused"] and BudgetLedger(store).balance(spec["arms"][0]["arm_id"])["committed_ms"] == charged
    bundle = export_evidence(store, spec["study_id"], grant)
    assert bundle["cells"][0]["qualification"] == "qualified"
    assert bundle["cells"][0]["admission"]["documents"]["propagation_qualification"] == evidence


@pytest.mark.parametrize("fault", ["missing-pointer", "unknown-attempt", "wrong-artifact", "source-model",
    "changed-threshold", "absent-prereg-policy", "wrong-metric", "wrong-source-grant"])
def test_unqualified_pointer_policy_or_source_cannot_pass_pre_read_gate(source, fault):
    store, spec, _, _, original, prereg = source
    package, frozen_plan = deepcopy(original), deepcopy(prereg)
    pointer = package["payload"][PAYLOAD_KEY]
    if fault == "missing-pointer":
        package["payload"].pop(PAYLOAD_KEY)
    elif fault == "unknown-attempt":
        pointer["source_attempt_id"] = "0"*32
    elif fault == "wrong-artifact":
        pointer["source_artifact_id"] = "0"*64
    elif fault == "source-model":
        pointer["policy"]["model_package_hash"] = "0"*64
    elif fault == "changed-threshold":
        pointer["policy"]["maximum_time_bias"] *= 2
        frozen_plan["propagation_qualification_policies"] = [pointer["policy"]]
    elif fault == "absent-prereg-policy":
        frozen_plan.pop("propagation_qualification_policies")
    elif fault == "wrong-metric":
        frozen_plan["primary_metrics"] = ["absolute_error_vs_float64_reference"]
    else:
        pointer["source_authorization_id"] = spec["admission"]["authorization_id"]
        pointer["source_authorization_version"] = None
    before = len([e for e in store.events() if e["event_kind"] == "READ_COMPLETED" and e["payload"].get("entrypoint") == "shared-admission"])
    with pytest.raises(ResearchError):
        prepare_managed_qualification(store, spec, spec["cells"][0], package, frozen_plan)
    after = len([e for e in store.events() if e["event_kind"] == "READ_COMPLETED" and e["payload"].get("entrypoint") == "shared-admission"])
    assert before == after


@pytest.mark.parametrize("fault", ["status", "float-error", "checks", "bias", "component-hash", "grid"])
def test_resealed_numeric_analysis_cannot_turn_failed_or_false_values_into_a_pass(source, fault):
    store, spec, _, _, package, _ = source
    pointer = package["payload"][PAYLOAD_KEY]
    result = json.loads(store.read_artifact(pointer["source_artifact_id"], purpose="evaluate",
        authorization=store.authorization(pointer["source_authorization_id"], version="1")))
    analysis = result["forecast"]["qualification_analysis"]
    if fault == "status":
        analysis["status"] = "FAILED"
    elif fault == "float-error":
        analysis["functional_roundoff_upper"] = 0
    elif fault == "checks":
        analysis["checks"]["arithmetic_operations"] = False
    elif fault == "bias":
        analysis["signed_time_bias_bounds"] = [["7", "1"], ["7", "1"]]
    elif fault == "component-hash":
        analysis["target_certificate_hash"] = "0"*64
    else:
        analysis["target_grid"] = {"solver": "euler", "steps": 999}
    analysis["analysis_hash"] = digest({k: v for k, v in analysis.items() if k != "analysis_hash"})
    request = request_from_manifest(spec["cells"][0]["propagation_request"])
    model = FrozenDynamicsPackage.from_manifest(spec["cells"][0]["frozen_dynamics"], expected_hash=request.model_package_hash)
    policy = AffineQualificationPolicy.from_manifest(pointer["policy"])
    with pytest.raises(ResearchError):
        _analysis_values(analysis, policy, model, request, policy.method, result["forecast"]["functional"]["estimate"])


def test_direct_formal_computation_cannot_promote_itself_without_owner_admission(source):
    _, spec, _, _, _, _ = source
    with pytest.raises(ResearchError, match="owner admission"):
        execute_propagation(spec, spec["cells"][0])


def test_actual_owner_result_validator_rejects_false_metrics_and_missing_provenance(source):
    store, spec, registry, grant, _, _ = source
    outcome = SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]))
    result = json.loads(store.read_artifact(outcome["artifact_id"], purpose="evaluate", authorization=grant))
    receipt = store.manifest("admission-"+result["admission_hash"])
    for fault in ("metric", "model", "components"):
        forged = deepcopy(result)
        if fault == "metric":
            forged["metrics"][METRIC] += 1
        elif fault == "model":
            forged["forecast"]["functional"]["error_budget"]["model"]["value"] = 0
        else:
            forged["forecast"].pop("qualified_error_components")
        with pytest.raises(ResearchError):
            validate_formal_analytic_result(receipt, spec, spec["cells"][0], forged)
    with pytest.raises(ResearchError, match="current owner attempt"):
        load_worker_admission(store, receipt["admission_hash"], spec, spec["cells"][0], Path("other-result.json"))
