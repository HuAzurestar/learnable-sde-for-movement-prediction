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


def prepare_target(root, cap, changes=None, request_changes=None):
    store, source_spec, source_registry = prepared(root, cap=cap, changes=changes, request_changes=request_changes)
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


@pytest.mark.parametrize("fault", ["scalar-outside-target", "scalar-within-target", "lineage", "metric", "zero-model",
    "reference", "reference-unit", "reference-definition", "time-unit", "time-definition"])
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
    elif fault in {"reference-unit", "reference-definition", "time-unit", "time-definition"}:
        key = "reference" if fault.startswith("reference-") else "time_discretization"
        result["forecast"]["functional"]["error_budget"][key]["units" if fault.endswith("unit") else "estimated_by"] = "substituted"
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
    store, original_spec, registry, _, _, _ = source
    # Separate invalid engineering control, SAME original cumulative arm.
    # Never try to reopen/reset the completed positive cell.
    spec = deepcopy(original_spec)
    spec.update(study_id=spec["study_id"]+"-over-cap", experiment_id="mixture-over-cap-control")
    store.register(spec, digest(spec))
    cell = spec["cells"][0]
    sequence = len(store.events())
    with pytest.raises(ResearchError, match="frozen qualification job budget"):
        SharedRunner(store, registry).run_cell(spec["study_id"], digest(cell), budget=BudgetSpec(61))
    assert not any(e["event_kind"] in {"RESERVE", "READ_COMPLETED", "WORKER_STARTED"} for e in store.events()[sequence:])
    assert any(e["event_kind"] == "ATTEMPT" and e["payload"]["state"] == "PREFLIGHT_FAILED" for e in store.events()[sequence:])


@pytest.mark.parametrize("fault", ["missing-stop", "unconfirmed-stop", "wrong-stop-reservation", "wrong-stop-cost",
    "zero-charge", "over-cap-charge", "stop-after-settle", "missing-worker", "missing-admission", "missing-completion"])
def test_actual_source_native_stop_cost_and_order_cannot_be_substituted(source, monkeypatch, fault):
    store, spec, _, _, package, prereg = source
    source_attempt = package["payload"][PAYLOAD_KEY]["source_attempt_id"]
    events = deepcopy(store.events())
    source_events = {e["event_kind"]: e for e in events if e["payload"].get("attempt_id") == source_attempt}
    if fault.startswith("missing-"):
        kind = {"missing-stop": "WORKER_TREE_STOPPED", "missing-worker": "WORKER_STARTED",
            "missing-admission": "ADMISSION", "missing-completion": "ATTEMPT"}[fault]
        events = [e for e in events if not (e["event_kind"] == kind and e["payload"].get("attempt_id") == source_attempt)]
    elif fault == "unconfirmed-stop":
        source_events["WORKER_TREE_STOPPED"]["payload"]["confirmation"] = "unverified"
    elif fault == "wrong-stop-reservation":
        source_events["WORKER_TREE_STOPPED"]["payload"]["reservation_id"] = "0"*64
    elif fault == "wrong-stop-cost":
        source_events["WORKER_TREE_STOPPED"]["payload"]["observed_elapsed_ms"] += 1
    elif fault == "zero-charge":
        source_events["SETTLE"]["payload"]["charged_ms"] = 0
    elif fault == "over-cap-charge":
        source_events["SETTLE"]["payload"].update(charged_ms=60001, reserved_ms=60001)
    else:
        source_events["WORKER_TREE_STOPPED"]["sequence"] = source_events["SETTLE"]["sequence"]+1
    # Transient read view only; never rewrite the actual owner journal/grants.
    monkeypatch.setattr(store, "events", lambda: events)
    with pytest.raises(ResearchError):
        prepare_managed_mixture(store, spec, spec["cells"][0], package, prereg)


@pytest.mark.parametrize("fault", ["expired", "wrong-consumer", "missing-evaluate", "wrong-protocol"])
def test_source_consumer_permission_is_not_inferred_from_target_grant(source, monkeypatch, fault):
    store, spec, _, _, package, prereg = source
    source_id = package["payload"][PAYLOAD_KEY]["source_authorization_id"]
    original = store.authorization
    def substituted(authorization_id, *, version=None):
        grant = deepcopy(original(authorization_id, version=version))
        if authorization_id == source_id:
            if fault == "expired":
                grant["expires_at"] = "2000-01-01T00:00:00+00:00"
            elif fault == "wrong-consumer":
                grant["consumer_study_ids"] = []
            elif fault == "missing-evaluate":
                grant["purposes"] = ["export"]
            else:
                grant["protocol_hash"] = "0"*64
        return grant
    monkeypatch.setattr(store, "authorization", substituted)
    with pytest.raises(ResearchError, match="source grant does not authorize consumer"):
        prepare_managed_mixture(store, spec, spec["cells"][0], package, prereg)


def test_production_registry_preserves_physical_dimensions_saved_proof_and_chunk_limits(source):
    _, spec, _, _, _, _ = source
    cell = spec["cells"][0]
    plugin = mixture_production_plugin()
    plan = plan_resources(plugin.registry_entry, cell["execution"]["config"], cell["execution"]["inputs"], matrix_cells=1)
    assert plan["counts"]["state_dim"] == 4 and plan["tensor_bytes"] >= 64*1024*1024
    assert plan["counts"]["steps"] == cell["propagation_request"]["steps"]*MixturePolicy.from_manifest(cell["mixture_policy"]).work_per_step
    assert plugin.resume_level == mixture_production_recovery_plugin().resume_level == "chunk"


@pytest.fixture(scope="module")
def exported(source, completed):
    store, spec, _, grant, _, _ = source
    return export_evidence(store, spec["study_id"], grant)


def test_actual_formal_mixture_save_ack_reopened_resume_retains_current_receipt_lineage_and_all_costs(tmp_path):
    import time
    from application.propagation_execution import validate_propagation_cell
    from application.research_recovery import RecoveryRegistry, SharedRecovery
    from inference.mixture_propagation import mixture_estimate
    from infrastructure.research_store import ResearchStore
    from tests.test_mixture_shared_adapter import _mixture_checkpoint_job_seconds, _mixture_worker_startup_seconds

    # Frozen engineering request/caps before the source reservation. The fixed
    # worst-case reference allocation remains400001, not the pass threshold.
    # At most two fixed60s linked continuations; never retry a hard fuse/failure.
    maximum_resumes = 2
    steps = (1_000_000-400001)//(8*(1+1+4**3))
    startup = _mixture_worker_startup_seconds()
    store, spec, registry, grant, _, _ = prepare_target(tmp_path, 1, request_changes={"steps": steps,
        "chunk_size": 1, "initial_covariance": tuple(tuple(.25 if i == j else 0. for j in range(4)) for i in range(4))})
    cell = spec["cells"][0]
    model, request, config, _ = validate_propagation_cell(spec, cell)
    mixture = MixturePolicy.from_manifest(cell["mixture_policy"])
    # Unprotected pure-kernel unit control, not a formal result or extra grant.
    # Size the first disposable job BEFORE its reservation, never a live deadline.
    started = time.monotonic()
    expected = mixture_estimate(model, request, mixture).manifest()
    job_seconds = _mixture_checkpoint_job_seconds(time.monotonic()-started, startup)
    arm = cell["arm_id"]
    pilot_cost = BudgetLedger(store).balance(arm)["committed_ms"]
    adapters = RecoveryRegistry()
    adapters.register(mixture_production_recovery_plugin())
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
    assert progress["eta_seconds"] == pytest.approx((progress["total_steps"]-progress["completed_steps"])/progress["throughput_per_second"])
    events = [e for e in store.events() if e["payload"].get("attempt_id") == parent["attempt_id"]]
    kinds = [e["event_kind"] for e in events]
    assert kinds.index("CHECKPOINT_REQUESTED") < kinds.index("CHECKPOINT") < kinds.index("CHECKPOINT_SAVED")
    assert kinds.index("CHECKPOINT_SAVED") < kinds.index("WORKER_TREE_STOPPED") < kinds.index("SETTLE")
    requested = next(e["payload"] for e in events if e["event_kind"] == "CHECKPOINT_REQUESTED")
    assert 0 < requested["remaining_seconds"] <= .2*job_seconds and saved["elapsed_ms"] < job_seconds*1000
    saved_value = json.loads((store.path/"artifacts"/saved["artifact_id"]).read_bytes())
    state = saved_value["state"]
    assert len(encode(state)) <= 65536
    assert state["method_state"]["policy_hash"] == mixture.policy_hash
    assert state["method_state"]["request_hash"] == request.request_hash
    assert state["rng_state"]["scheme"] == "deterministic-cubature-no-sampled-rng-v1"
    assert state["chunk_complete"] is True and "budget" not in state and "remaining_seconds" not in state
    before = BudgetLedger(store).balance(arm)
    assert before["committed_ms"] == pilot_cost+interrupted["elapsed_ms"] and not before["closed"]
    reopened = ResearchStore(tmp_path, store.store_id)
    with pytest.raises(ResearchError):
        SharedRecovery(reopened, registry, adapters).prepare(parent["attempt_id"], saved["artifact_id"],
            authorization={**grant, "purposes": ["evaluate"]})
    assert BudgetLedger(reopened).balance(arm) == before
    continuations, failed_ids = [], {parent["attempt_id"]}
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
        failed_ids.add(previous)
    assert resumed["state"] == "SUCCEEDED", resumed
    actual = json.loads((reopened.path/"artifacts"/resumed["artifact_id"]).read_bytes())
    receipt = reopened.manifest("admission-"+actual["admission_hash"])
    assert actual["qualification"] == "qualified" and actual["forecast"]["functional"]["status"] == "APPROXIMATION_ONLY"
    assert receipt["attempt_id"] == resumed["attempt_id"] and actual["admission_hash"] != saved_value["admission_hash"]
    proof = receipt["documents"]["propagation_qualification"]
    assert proof["policy"] == cell["mixture_qualification_policy"] and proof["mixture_policy"] == cell["mixture_policy"]
    assert receipt["input_evidence"] and all(proof["completion_event"]["sequence"] < event["sequence"]
        for event in receipt["input_evidence"])
    assert proof["settlement_event"]["payload"]["charged_ms"] > 0
    assert {k: v for k, v in actual["forecast"]["functional"].items() if k != "error_budget"} == json.loads(
        encode({k: v for k, v in expected.items() if k != "error_budget"}))
    validate_formal_mixture_result(receipt, spec, cell, actual)
    assert BudgetLedger(reopened).balance(arm)["committed_ms"] == before["committed_ms"]+sum(c["elapsed_ms"] for c in continuations)
    bundle = export_evidence(reopened, spec["study_id"], grant)
    row, target_ids = bundle["cells"][0], failed_ids | {resumed["attempt_id"]}
    assert {a["attempt_id"] for a in row["history"]} == target_ids
    assert row["cost"]["charged_ms"] == interrupted["elapsed_ms"]+sum(c["elapsed_ms"] for c in continuations)
    assert {e["payload"]["attempt_id"] for e in row["cost"]["sources"]} == target_ids
    independent_reader().validate_mixture_qualification(row["admission"], row)
    checked = paper_validate(tmp_path, bundle)
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout)["verified_mixture_cells"] == 1
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


def paper_validate(tmp_path, bundle):
    from pathlib import Path
    import subprocess
    import sys
    paper = Path(__file__).resolve().parents[2]/"TSDE-SDE"
    path = tmp_path/"mixture-bundle.json"
    path.write_bytes(encode(bundle))
    return subprocess.run([sys.executable, "-B", str(paper/"scripts/pirc25/validate_mixture.py"), str(path),
        "--expected-hash", bundle["bundle_hash"]], cwd=paper, capture_output=True, text=True, timeout=30)


def independent_reader():
    from pathlib import Path
    import importlib.util
    import sys
    path = Path(__file__).resolve().parents[2]/"TSDE-SDE/scripts/pirc25/mixture_qualification.py"
    sys.path.insert(0, str(path.parent))
    try:
        entry = importlib.util.spec_from_file_location("independent_mixture_reader", path)
        reader = importlib.util.module_from_spec(entry)
        entry.loader.exec_module(reader)
    finally:
        sys.path.pop(0)
    return reader


def test_real_mixture_owner_export_passes_independent_reader_and_cli(exported, tmp_path):
    row = exported["cells"][0]
    independent_reader().validate_mixture_qualification(row["admission"], row)
    checked = paper_validate(tmp_path, exported)
    assert checked.returncode == 0, checked.stderr
    verified = json.loads(checked.stdout)
    assert verified["source_bundle_hash"] == exported["bundle_hash"]
    assert verified["verified_mixture_cells"] == verified["expected_cells"] == 1
    assert "no statistical adjudication, full-distribution or scientific model qualification" in verified["scope"]


def reseal_transport(bundle):
    """Alter only disposable exported copies, never actual owner journal/grants."""
    row = bundle["cells"][0]
    receipt = row["admission"]
    evidence = receipt["documents"].get("propagation_qualification")
    if evidence:
        source = evidence["source_result"]
        analysis = source["forecast"]["mixture_qualification_analysis"]
        for key in ("continuous", "target"):
            analysis[key+"_certificate_hash"] = digest(analysis[key+"_certificate"])
        analysis["analysis_hash"] = digest({k: v for k, v in analysis.items() if k != "analysis_hash"})
        source["output_hash"] = digest({k: source[k] for k in ("metrics", "forecast", "fit", "source_schema")})
        metadata = evidence["source_artifact"]
        metadata.update(artifact_id=digest(source), sha256=digest(source), size_bytes=len(encode(source)))
        evidence["source_attempt"].update(artifact_id=metadata["artifact_id"], artifact_manifest_hash=digest(metadata))
        receipt["documents"]["package"]["payload"][PAYLOAD_KEY]["source_artifact_id"] = metadata["artifact_id"]
        for key in ("reservation_event", "worker_event", "stop_event", "settlement_event", "admission_event", "completion_event"):
            if key == "completion_event":
                evidence[key]["payload"] = deepcopy(evidence["source_attempt"])
            evidence[key]["hash"] = digest({k: v for k, v in evidence[key].items() if k != "hash"})
        evidence["evidence_hash"] = digest({k: v for k, v in evidence.items() if k != "evidence_hash"})
    receipt["admission_hash"] = digest({k: v for k, v in receipt.items() if k != "admission_hash"})
    row["admission_hash"] = receipt["admission_hash"]
    result = row["result"]
    result["admission_hash"] = receipt["admission_hash"]
    if evidence:
        result["forecast"]["qualified_error_components"]["qualification_evidence_hash"] = evidence["evidence_hash"]
    result["output_hash"] = digest({k: result[k] for k in ("metrics", "forecast", "fit", "source_schema")})
    row["result_artifact"].update(artifact_id=digest(result), sha256=digest(result), size_bytes=len(encode(result)))
    row["artifact_id"] = digest(result)
    bundle["bundle_hash"] = digest({k: v for k, v in bundle.items() if k != "bundle_hash"})


@pytest.mark.parametrize("fault", ["width", "retained", "total", "bias", "norm", "operations", "grid", "scope", "scientific",
    "checks", "zero-closure", "zero-roundoff", "zero-model", "gaussian-schema", "source-cost", "native-stop",
    "completion-order", "source-heldout", "source-grant", "source-root", "target-scalar", "target-inbounds-scalar",
    "target-lineage", "target-metric", "target-unit", "target-reference-unit", "target-reference-definition", "target-time-definition"])
def test_independent_reader_refuses_resealed_numeric_cost_permission_and_current_output_substitutions(exported, fault):
    bundle = deepcopy(exported)
    row = bundle["cells"][0]
    evidence = row["admission"]["documents"]["propagation_qualification"]
    analysis = evidence["source_result"]["forecast"]["mixture_qualification_analysis"]
    numeric = {"width": "reference_width_upper", "retained": "retained_functional_error_upper",
        "total": "total_functional_error_upper", "bias": "absolute_time_bias_upper", "norm": "scaled_transition_norm_upper",
        "operations": "reference_operations"}
    if fault in numeric:
        analysis[numeric[fault]] += 1
    elif fault == "grid":
        analysis["target_certificate"]["grid"]["steps"] *= 2
    elif fault == "scope":
        analysis["scope"] = "qualified-entire-distribution"
    elif fault == "scientific":
        analysis["scientific_qualification"] = True
    elif fault == "checks":
        analysis["checks"]["retained_functional_error"] = 1
    elif fault.startswith("zero-"):
        key = {"zero-closure": "propagation_approximation", "zero-roundoff": "implementation_roundoff", "zero-model": "model_error"}[fault]
        analysis[key] = {"value": 0, "status": "BOUNDED"}
    elif fault == "gaussian-schema":
        analysis["schema_version"] = "affine-analytic-qualification-analysis-v1"
    elif fault == "source-cost":
        evidence["settlement_event"]["payload"]["charged_ms"] = 0
    elif fault == "native-stop":
        evidence["stop_event"]["payload"]["confirmation"] = "not-stopped"
    elif fault == "completion-order":
        evidence["completion_event"]["sequence"] = row["admission"]["input_evidence"][0]["sequence"]+1
    elif fault == "source-heldout":
        evidence["source_admission"]["documents"]["protocol"]["blocks"][0]["split_role"] = "test"
    elif fault == "source-grant":
        evidence["authorization"]["consumer_study_ids"] = []
    elif fault == "source-root":
        evidence["source_admission"]["spec"]["runtime_binding"]["store_id"] = "another-root"
    elif fault in {"target-scalar", "target-inbounds-scalar"}:
        row["result"]["forecast"]["functional"]["estimate"] += 2 if fault == "target-scalar" else .001
    elif fault == "target-lineage":
        functional = row["result"]["forecast"]["functional"]
        functional["diagnostics"] = [[k, "0"*64 if k == "lineage_digest" else v] for k, v in functional["diagnostics"]]
    elif fault == "target-metric":
        row["result"]["metrics"][METRIC] += 1
        row["metrics"] = deepcopy(row["result"]["metrics"])
    elif fault == "target-unit":
        row["metric_units"][METRIC] = row["result"]["metric_units"][METRIC] = "other"
    else:
        budget = row["result"]["forecast"]["functional"]["error_budget"]
        key = "time_discretization" if fault == "target-time-definition" else "reference"
        field = "units" if fault == "target-reference-unit" else "estimated_by"
        budget[key][field] = "other"
    reseal_transport(bundle)
    with pytest.raises(ValueError):
        independent_reader().validate_mixture_qualification(row["admission"], row)


@pytest.mark.parametrize("fault", ["missing-numeric", "missing-pointer", "target-false-metric"])
def test_mixture_cli_has_no_generic_operator_pass_fallback(exported, tmp_path, fault):
    bundle = deepcopy(exported)
    row = bundle["cells"][0]
    if fault in {"missing-numeric", "missing-pointer"}:
        row["admission"]["documents"].pop("propagation_qualification")
        if fault == "missing-pointer":
            row["admission"]["documents"]["package"]["payload"].pop(PAYLOAD_KEY)
    else:
        row["result"]["metrics"][METRIC] += 1
        row["metrics"] = deepcopy(row["result"]["metrics"])
    reseal_transport(bundle)
    assert paper_validate(tmp_path, bundle).returncode != 0
