"""Actual disposable cubature source/owner/target/export/Paper chain, not science."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from application.cubature_qualification_admission import (
    METRIC, PAYLOAD_KEY, _analysis_values, prepare_managed_cubature,
    validate_formal_cubature_result)
from application.propagation_execution import execute_propagation, validate_propagation_cell
from application.research_budget import BudgetLedger, BudgetSpec
from application.research_contracts import CapabilityRegistry
from application.research_evidence import export_evidence
from domain.cubature_qualification import CubatureQualificationPolicy
from experiments.pirc25.runner import SharedRunner
from experiments.pirc27.cubature_plugin import cubature_plugin, cubature_config
from experiments.pirc27.plugin import execution_inputs
from infrastructure.research_store import ResearchError, digest, encode
from tests.research_admission_fixtures import admit_fixture
from tests.test_propagation_shared_adapter import prepare


@pytest.fixture(scope="module", params=["endpoint-x", "endpoint-halfspace"])
def source(request, tmp_path_factory):
    root = tmp_path_factory.mktemp("managed-cubature-"+request.param)
    store, pilot, registry = prepare(root, "cubature", cubature_qualification=True,
        changes={"functional": request.param, "threshold": .6})
    store.register(pilot, digest(pilot))
    outcome = SharedRunner(store, registry).run_cell(pilot["study_id"], digest(pilot["cells"][0]),
        budget=BudgetSpec(60, category="pilot"))
    assert outcome["state"] == "SUCCEEDED", outcome
    consumer_id = "formal-cubature-"+request.param
    source_grant = {**store.authorization(pilot["admission"]["authorization_id"]),
        "authorization_id": "cubature-source-consumer", "version": "1",
        "consumer_study_ids": [consumer_id], "purposes": ["evaluate", "export"]}
    store.authorize(source_grant)
    policy = pilot["cells"][0]["cubature_qualification_policy"]
    pointer = {"schema_version": "managed-affine-cubature-qualification-v1", "policy": policy,
        "source_attempt_id": outcome["attempt_id"], "source_artifact_id": outcome["artifact_id"],
        "source_authorization_id": source_grant["authorization_id"], "source_authorization_version": "1"}
    cell = deepcopy(pilot["cells"][0])
    cell.pop("cubature_qualification_policy")
    cell.pop("execution_role")
    plugin = cubature_plugin()
    cell.update(plugin_id=plugin.plugin_id, study_role="primary", block_id="independent-cubature-heldout")
    spec = {**deepcopy(pilot), "study_id": consumer_id, "experiment_id": consumer_id, "cells": [cell]}
    _, target, _, _ = validate_propagation_cell(pilot, pilot["cells"][0])
    grant = admit_fixture(store, spec, plugin, root, formal=True, legacy_upstream=False,
        execution_config=cubature_config(target), execution_inputs=execution_inputs(target),
        fixture_prefix=consumer_id+"-", input_content=("independent heldout cubature control "+request.param).encode(),
        primary_metrics=[METRIC], preregistration_extra={"cubature_qualification_policies": [policy]},
        package_payload={PAYLOAD_KEY: pointer})
    store.register(spec, digest(spec))
    registry = CapabilityRegistry()
    registry.register(plugin)
    package = store.manifest("package-"+spec["admission"]["package_hash"])
    prereg = store.manifest("preregistration-"+package["preregistration_hash"])
    return store, spec, registry, grant, package, prereg


@pytest.fixture(scope="module")
def exported(source):
    store, spec, registry, grant, _, _ = source
    before = BudgetLedger(store).balance(spec["arms"][0]["arm_id"])["committed_ms"]
    outcome = SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]), budget=BudgetSpec(60))
    assert outcome["state"] == "SUCCEEDED", outcome
    after = BudgetLedger(store).balance(spec["arms"][0]["arm_id"])["committed_ms"]
    assert after > before > 0 and after-before == outcome["elapsed_ms"]
    reuse = SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]))
    assert reuse["reused"] and BudgetLedger(store).balance(spec["arms"][0]["arm_id"])["committed_ms"] == after
    return export_evidence(store, spec["study_id"], grant)


def test_actual_formal_cubature_preserves_own_method_and_owner_cost_chain(exported):
    row = exported["cells"][0]
    result, receipt = row["result"], row["admission"]
    evidence = receipt["documents"]["propagation_qualification"]
    assert result["qualification"] == "qualified"
    assert result["forecast"]["functional"]["estimator_id"] == "gaussian-cubature-euler-v1"
    assert result["forecast"]["functional"]["sample_count"] == 0
    assert result["resume_level"] == "restart-only"
    assert evidence["source_result"]["forecast"]["cubature_qualification_analysis"]["scientific_qualification"] is False
    assert evidence["source_cell_hash"] == digest(evidence["source_run"]["cell"])
    assert evidence["completion_event"]["sequence"] < min(r["sequence"] for r in receipt["input_evidence"])
    assert result["forecast"]["qualified_error_components"]["model_error"] == {"value": None, "status": "NOT_IDENTIFIABLE"}
    assert result["forecast"]["functional"]["error_budget"]["propagation_approximation"]["value"] == 0


def paper_validate(tmp_path, bundle):
    paper = Path(__file__).resolve().parents[2]/"TSDE-SDE"
    path = tmp_path/"cubature-bundle.json"
    path.write_bytes(encode(bundle))
    return subprocess.run([sys.executable, "-B", str(paper/"scripts/pirc25/validate_cubature.py"),
        str(path), "--expected-hash", bundle["bundle_hash"]], cwd=paper,
        capture_output=True, text=True, timeout=30)


def test_actual_export_is_verified_independently_by_paper_cli(exported, tmp_path):
    checked = paper_validate(tmp_path, exported)
    assert checked.returncode == 0, checked.stderr
    value = json.loads(checked.stdout)
    assert value["verified_cubature_cells"] == value["expected_cells"] == 1
    assert value["source_bundle_hash"] == exported["bundle_hash"]
    assert "no statistical adjudication or model qualification" in value["scope"]


@pytest.mark.parametrize("field,value", [("point_count", 4), ("point_weight", .25),
    ("covariance_projection", True), ("cubature_point_updates", 0),
    ("output_replay_point_updates", 0), ("scope", "nonlinear-closure-qualified"),
    ("reference_arithmetic_operations", 0), ("status", "FAILED"),
    ("schema_version", "affine-analytic-qualification-analysis-v1")])
def test_owner_recomputes_saved_cube_analysis_even_after_rehash(source, field, value):
    store, spec, _, _, package, _ = source
    pointer = package["payload"][PAYLOAD_KEY]
    artifact = json.loads((store.path/"artifacts"/pointer["source_artifact_id"]).read_bytes())
    analysis = deepcopy(artifact["forecast"]["cubature_qualification_analysis"])
    analysis[field] = value
    analysis["analysis_hash"] = digest({k: v for k, v in analysis.items() if k != "analysis_hash"})
    model, request, _, _ = validate_propagation_cell(spec, spec["cells"][0])
    with pytest.raises(ResearchError, match="UNQUALIFIED"):
        _analysis_values(analysis, CubatureQualificationPolicy.from_manifest(pointer["policy"]),
            model, request, artifact["forecast"]["functional"]["estimate"])


def reseal_target(bundle):
    row = bundle["cells"][0]
    receipt = row["admission"]
    receipt["admission_hash"] = digest({k: v for k, v in receipt.items() if k != "admission_hash"})
    row["admission_hash"] = receipt["admission_hash"]
    result = row["result"]
    result["admission_hash"] = receipt["admission_hash"]
    result["output_hash"] = digest({k: result[k] for k in ("metrics", "forecast", "fit", "source_schema")})
    row["result_artifact"].update(artifact_id=digest(result), sha256=digest(result), size_bytes=len(encode(result)))
    row["artifact_id"] = digest(result)
    bundle["bundle_hash"] = digest({k: v for k, v in bundle.items() if k != "bundle_hash"})


@pytest.mark.parametrize("fault", ["missing-proof", "changed-estimate", "analytic-estimator", "projected",
    "false-error", "claimed-model", "missing-sampling-count", "unexpected-calibration"])
def test_resealed_current_target_never_falls_back_to_generic_paper_pass(exported, tmp_path, fault):
    bundle = deepcopy(exported)
    row = bundle["cells"][0]
    result = row["result"]
    functional = result["forecast"]["functional"]
    if fault == "missing-proof":
        row["admission"]["documents"].pop("propagation_qualification")
    elif fault == "changed-estimate":
        functional["estimate"] += 1
    elif fault == "analytic-estimator":
        functional["estimator_id"] = "gaussian-discrete-v1"
    elif fault == "projected":
        next(pair for pair in functional["diagnostics"] if pair[0] == "covariance_projection")[1] = True
    elif fault == "false-error":
        result["metrics"][METRIC] += 1
        row["metrics"] = deepcopy(result["metrics"])
    elif fault == "claimed-model":
        functional["error_budget"]["model"] = {"value": 0, "status": "IDENTIFIED", "units": "m"}
    elif fault == "unexpected-calibration":
        row["admission"]["documents"]["probability_calibration"] = {"opaque_pass": True}
    else:
        functional.pop("sample_count")
    reseal_target(bundle)
    checked = paper_validate(tmp_path, bundle)
    assert checked.returncode != 0
    assert ("calibration" if fault == "unexpected-calibration" else "cubature") in checked.stderr.lower()


def test_formal_target_does_not_replay_reference_kernels(exported, monkeypatch):
    row = exported["cells"][0]
    receipt = row["admission"]
    def forbidden(*args, **kwargs):
        raise AssertionError("formal target replayed reference kernel")
    monkeypatch.setattr("inference.propagation_methods.analytic_estimate", forbidden)
    monkeypatch.setattr("inference.affine_reference.bound_affine_reference", forbidden)
    monkeypatch.setattr("inference.affine_discrete_reference.bound_affine_discrete", forbidden)
    result = execute_propagation(receipt["spec"], receipt["cell"], admission=receipt)
    validate_formal_cubature_result(receipt, receipt["spec"], receipt["cell"], result)


@pytest.mark.parametrize("fault", ["missing-pointer", "analytic-pointer", "wrong-source", "wrong-policy",
    "unfrozen-policy", "wrong-metric"])
def test_owner_refuses_before_any_new_heldout_read(source, fault):
    store, spec, _, _, package, prereg = source
    package, prereg = deepcopy(package), deepcopy(prereg)
    pointer = package["payload"][PAYLOAD_KEY]
    if fault == "missing-pointer":
        package["payload"].pop(PAYLOAD_KEY)
    elif fault == "analytic-pointer":
        pointer["schema_version"] = "managed-affine-analytic-qualification-v1"
    elif fault == "wrong-source":
        pointer["source_attempt_id"] = "missing-source-attempt"
    elif fault == "wrong-policy":
        pointer["policy"]["request_hash"] = "0"*64
    elif fault == "unfrozen-policy":
        prereg["cubature_qualification_policies"] = []
    else:
        prereg["primary_metrics"] = ["generic-operator-pass"]
    count = len(store.events())
    with pytest.raises(ResearchError, match="UNQUALIFIED"):
        prepare_managed_cubature(store, spec, spec["cells"][0], package, prereg)
    assert not any(e["event_kind"] in {"READ_STARTED", "WORKER_STARTED"} for e in store.events()[count:])
