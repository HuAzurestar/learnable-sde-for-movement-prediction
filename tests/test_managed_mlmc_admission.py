"""Actual pilot -> owner pre-read gate -> independent production controls.

Disposable synthetic grants/stores only; not an official research or model
approval. The paper admission reader checks the saved evidence independently;
statistical adjudication and actual research remain separate, still required.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from application.mlmc_qualification_admission import (METRIC, PAYLOAD_KEY, prepare_managed_mlmc,
    source_analysis_values, functional_result, validate_formal_mlmc_result)
from application.propagation_execution import execute_propagation, request_from_manifest, validate_propagation_cell
from application.propagation_qualification_admission import load_worker_admission
from application.research_budget import BudgetLedger, BudgetSpec
from application.research_contracts import CapabilityRegistry
from application.research_evidence import export_evidence
from application.research_execution import execution_binding
from application.research_recovery import RecoveryRegistry
from domain.affine_mlmc_qualification import AffineMLMCReferencePolicy
from domain.frozen_dynamics import FrozenDynamicsPackage
from domain.mlmc_pilot import MLMCPilotPolicy
from domain.mlmc_production import MLMCProductionPolicy
from domain.errors import DataValidationError
from experiments.pirc25.affine import code_hash
from experiments.pirc25.runner import SharedRunner
from experiments.pirc27.mlmc_production_plugin import (mlmc_production_plugin,
    mlmc_production_config, mlmc_production_recovery_plugin)
from experiments.pirc27.plugin import propagation_resume_command, propagation_plugin
from infrastructure.research_store import ResearchError, digest, encode
from tests.research_admission_fixtures import admit_fixture
from tests.test_affine_mlmc_qualification import managed


def prepared_source(root, **pilot_options):
    store, pilot_spec, registry, recovery = managed(root, **pilot_options)
    store.register(pilot_spec, digest(pilot_spec))
    outcome = SharedRunner(store, registry, recovery_registry=recovery).run_cell(
        pilot_spec["study_id"], digest(pilot_spec["cells"][0]), budget=BudgetSpec(60, category="pilot"))
    assert outcome["state"] == "SUCCEEDED", outcome
    result = json.loads((store.path/"artifacts"/outcome["artifact_id"]).read_bytes())
    analysis = result["forecast"]["bounded_mlmc_pilot_analysis"]
    assert analysis["status"] == "NUMERICAL_READY"
    sc = pilot_spec["cells"][0]
    pilot = MLMCPilotPolicy(**sc["mlmc_pilot_policy"])
    counts = tuple(analysis["empirical_pilot_analysis"]["sampling_only_proposal"])
    production_request = replace(request_from_manifest(sc["propagation_request"]), request_id="independent-production-v1",
        seed=pilot.production_seed, coupling_id=pilot.production_coupling_id, samples=sum(counts))
    production = MLMCProductionPolicy(production_request.request_hash, production_request.model_package_hash, code_hash(),
        pilot.pilot_request_hash, pilot.policy_hash, digest(sc["affine_mlmc_reference_policy"]), counts, 60.)
    consumer_id = "formal-independent-mlmc"
    grant = {**store.authorization(pilot_spec["admission"]["authorization_id"]), "authorization_id": "mlmc-source-consumer",
        "version": "1", "consumer_study_ids": [consumer_id], "purposes": ["evaluate", "export"]}
    store.authorize(grant)
    pointer = {"schema_version": "managed-affine-mlmc-qualification-v1", "production_policy": production.manifest(),
        "source_attempt_id": outcome["attempt_id"], "source_artifact_id": outcome["artifact_id"],
        "source_authorization_id": grant["authorization_id"], "source_authorization_version": "1"}
    plugin = mlmc_production_plugin()
    cell = deepcopy(sc)
    for key in ("affine_mlmc_reference_policy", "mlmc_pilot_policy"):
        cell.pop(key)
    cell.update(plugin_id=plugin.plugin_id, execution_role="production", study_role="primary",
        block_id="independent-heldout-mlmc-v1", seed=production_request.seed,
        propagation_request=json.loads(encode(asdict(production_request))), mlmc_production_policy=production.manifest())
    config = mlmc_production_config(production_request, production)
    from experiments.pirc27.plugin import execution_inputs
    inputs = execution_inputs(production_request)
    cell["execution"] = execution_binding(plugin.registry_entry, config, inputs, matrix_cells=1)
    spec = {**deepcopy(pilot_spec), "study_id": consumer_id, "experiment_id": consumer_id, "cells": [cell]}
    target_grant = admit_fixture(store, spec, plugin, root, formal=True, legacy_upstream=False,
        execution_config=config, execution_inputs=inputs, recovery_command_builder=propagation_resume_command,
        fixture_prefix="formal-mlmc-", input_content=b"independent heldout MLMC synthetic control, not pilot data",
        primary_metrics=[METRIC], preregistration_extra={"mlmc_production_policies": [production.manifest()]},
        package_payload={PAYLOAD_KEY: pointer})
    store.register(spec, digest(spec))
    registry, recovery = CapabilityRegistry(), RecoveryRegistry()
    registry.register(plugin)
    recovery.register(mlmc_production_recovery_plugin())
    package = store.manifest("package-"+spec["admission"]["package_hash"])
    prereg = store.manifest("preregistration-"+package["preregistration_hash"])
    return store, spec, registry, recovery, target_grant, package, prereg


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return prepared_source(tmp_path_factory.mktemp("managed-mlmc-production"))


def test_actual_formal_mlmc_save_ack_reopened_resume_retains_stream_receipt_and_all_costs(tmp_path):
    import time
    from application.research_recovery import SharedRecovery
    from inference.propagation_methods import mlmc_estimate
    from infrastructure.research_store import ResearchStore
    from tests.test_propagation_shared_adapter import _checkpoint_job_seconds, _worker_startup_seconds

    # Fixed engineering-only pilot policy before any reservation. The actual
    # pilot's measured allocation, not an inflated or hand-written allocation,
    # supplies the independently registered production workload.
    # Predeclare at most four linked continuations: a correctly budget-stopped
    # resume may itself save at80%, never requiring a longer live deadline.
    maximum_resumes = 4
    startup_seconds = _worker_startup_seconds()
    store, spec, registry, recovery, grant, _, _ = prepared_source(tmp_path,
        sampling_tolerance=.001, counts=(1024, 1024, 1024), request_changes={"chunk_size": 1})
    cell = spec["cells"][0]
    model, request, config, _ = validate_propagation_cell(spec, cell)
    counts = tuple(cell["mlmc_production_policy"]["level_samples"])
    # Direct pure-kernel unit control, not a formal output or protected read.
    # Size this disposable test's first job before its target reservation;
    # neither its frozen60s cap nor a running deadline can change afterward.
    started = time.monotonic()
    expected = mlmc_estimate(model, request, level_samples=counts)
    job_seconds = _checkpoint_job_seconds(time.monotonic()-started, startup_seconds)
    assert expected.standard_error <= .001
    arm = cell["arm_id"]
    pilot_cost = BudgetLedger(store).balance(arm)["committed_ms"]
    interrupted = SharedRunner(store, registry, recovery_registry=recovery).run_cell(
        spec["study_id"], digest(cell), budget=BudgetSpec(job_seconds))
    assert interrupted["state"] == "FAILED", interrupted
    parent = store.attempts()[interrupted["attempt_id"]]
    assert parent["error_code"] == "CHECKPOINT_SAVED"
    saved = interrupted["checkpoint"]
    assert saved["resume_level"] == "chunk"
    assert 0 < saved["progress"]["completed_steps"] < saved["progress"]["total_steps"] == config["work_steps"]
    progress = saved["progress"]
    assert progress["throughput_per_second"] > 0 and progress["eta_seconds"] > 0
    assert progress["eta_seconds"] == pytest.approx(
        (progress["total_steps"]-progress["completed_steps"])/progress["throughput_per_second"])
    events = [e for e in store.events() if e["payload"].get("attempt_id") == parent["attempt_id"]]
    kinds = [e["event_kind"] for e in events]
    assert kinds.index("CHECKPOINT_REQUESTED") < kinds.index("CHECKPOINT") < kinds.index("CHECKPOINT_SAVED")
    assert kinds.index("CHECKPOINT_SAVED") < kinds.index("WORKER_TREE_STOPPED") < kinds.index("SETTLE")
    requested = next(e["payload"] for e in events if e["event_kind"] == "CHECKPOINT_REQUESTED")
    assert 0 < requested["remaining_seconds"] <= job_seconds*.2
    assert saved["elapsed_ms"] < job_seconds*1000
    value = json.loads((store.path/"artifacts"/saved["artifact_id"]).read_bytes())
    state = value["state"]
    assert state["rng_state"]["phase"] == 1 and state["rng_state"]["seed"] == request.seed
    assert state["rng_state"]["coupling_id"] == request.coupling_id
    assert state["method_state"]["request_hash"] == request.request_hash and state["chunk_complete"] is True
    assert "budget" not in state and "remaining_seconds" not in state
    before = BudgetLedger(store).balance(arm)
    assert before["committed_ms"] == pilot_cost+interrupted["elapsed_ms"] and not before["closed"]

    reopened = ResearchStore(tmp_path, store.store_id)
    restored = SharedRecovery(reopened, registry, recovery)
    # A supplied evaluate permission cannot borrow the separate resume grant.
    # This refused read does not revoke/replace the genuine operator test grant.
    denied = {**grant, "purposes": ["evaluate"]}
    with pytest.raises(ResearchError):
        restored.prepare(parent["attempt_id"], saved["artifact_id"], authorization=denied)
    assert BudgetLedger(reopened).balance(arm) == before
    continued = []
    failed_ids = {parent["attempt_id"]}
    previous = parent["attempt_id"]
    checkpoint = saved["artifact_id"]
    completed = saved["progress"]["completed_steps"]
    for _ in range(maximum_resumes):
        reopened = ResearchStore(tmp_path, store.store_id)
        resumed = SharedRecovery(reopened, registry, recovery).resume(previous, checkpoint,
            authorization=grant, budget=BudgetSpec(60))
        continued.append(resumed)
        current = reopened.attempts()[resumed["attempt_id"]]
        assert current["parent_attempt_id"] == previous
        assert not BudgetLedger(reopened).balance(arm)["closed"]
        if resumed["state"] == "SUCCEEDED":
            break
        # Never retry a timeout, hard fuse or numerical/admission failure.
        # Only an actual verified new completed-boundary save can continue.
        assert resumed["state"] == "FAILED" and current["error_code"] == "CHECKPOINT_SAVED", resumed
        next_saved = resumed["checkpoint"]
        assert completed < next_saved["progress"]["completed_steps"] < config["work_steps"]
        completed = next_saved["progress"]["completed_steps"]
        previous, checkpoint = resumed["attempt_id"], next_saved["artifact_id"]
        failed_ids.add(previous)
    assert resumed["state"] == "SUCCEEDED", resumed
    current = reopened.attempts()[resumed["attempt_id"]]
    assert current["parent_attempt_id"] in failed_ids
    actual = json.loads((reopened.path/"artifacts"/resumed["artifact_id"]).read_bytes())
    receipt = reopened.manifest("admission-"+actual["admission_hash"])
    assert receipt["attempt_id"] == resumed["attempt_id"] and actual["admission_hash"] != value["admission_hash"]
    assert receipt["documents"]["propagation_qualification"]["production_policy"] == cell["mlmc_production_policy"]
    # Only owner-qualified error components replace the pure-kernel unknown
    # reference/time values. All actual signed-level statistics must be exact.
    unmodified = {k: v for k, v in actual["forecast"]["functional"].items() if k != "error_budget"}
    assert unmodified == json.loads(encode({k: v for k, v in expected.manifest().items() if k != "error_budget"}))
    validate_formal_mlmc_result(receipt, spec, cell, actual)
    after = BudgetLedger(reopened).balance(arm)
    assert after["committed_ms"] == before["committed_ms"]+sum(a["elapsed_ms"] for a in continued) and not after["closed"]
    bundle = export_evidence(reopened, spec["study_id"], grant)
    row = bundle["cells"][0]
    target_ids = failed_ids | {resumed["attempt_id"]}
    assert {a["attempt_id"] for a in row["history"]} == target_ids
    assert row["cost"]["charged_ms"] == interrupted["elapsed_ms"]+sum(a["elapsed_ms"] for a in continued)
    assert {e["payload"]["attempt_id"] for e in row["cost"]["sources"]} == target_ids
    checked = paper_validate(tmp_path, bundle)
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout)["verified_mlmc_cells"] == 1
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
        refused = paper_validate(tmp_path, altered)
        assert refused.returncode != 0, fault


def test_actual_formal_mlmc_worker_requires_settled_pilot_then_independent_stream(source):
    store, spec, registry, recovery, grant, _, _ = source
    before = BudgetLedger(store).balance(spec["arms"][0]["arm_id"])["committed_ms"]
    outcome = SharedRunner(store, registry, recovery_registry=recovery).run_cell(
        spec["study_id"], digest(spec["cells"][0]), budget=BudgetSpec(60))
    assert outcome["state"] == "SUCCEEDED", outcome
    result = json.loads((store.path/"artifacts"/outcome["artifact_id"]).read_bytes())
    receipt = store.manifest("admission-"+result["admission_hash"])
    evidence = receipt["documents"]["propagation_qualification"]
    assert evidence["source_result"]["qualification"] == "fixture" and result["qualification"] == "qualified"
    components = result["forecast"]["qualified_error_components"]
    assert components["stream"] == {"phase": 1, "seed": spec["cells"][0]["seed"],
        "coupling_id": spec["cells"][0]["propagation_request"]["coupling_id"],
        "level_samples": spec["cells"][0]["mlmc_production_policy"]["level_samples"]}
    assert components["sampler_roundoff"] == components["model_error"] == {"value": None, "status": "NOT_IDENTIFIABLE"}
    assert components["sampling_error"]["status"] == "ESTIMATED"
    assert components["confidence_scope"] == "normal-approximation-not-rigorous-coverage"
    assert result["forecast"]["functional"]["error_budget"]["time_discretization"]["status"] == "BOUNDED"
    after = BudgetLedger(store).balance(spec["arms"][0]["arm_id"])["committed_ms"]
    assert after > before > 0 and outcome["elapsed_ms"] == after-before
    reads = [e for e in store.events() if e["event_kind"] == "READ_COMPLETED" and e["payload"].get("attempt_id") == outcome["attempt_id"]]
    assert reads and evidence["completion_event"]["sequence"] < reads[0]["sequence"]
    reused = SharedRunner(store, registry, recovery_registry=recovery).run_cell(spec["study_id"], digest(spec["cells"][0]))
    assert reused["reused"] and BudgetLedger(store).balance(spec["arms"][0]["arm_id"])["committed_ms"] == after
    bundle = export_evidence(store, spec["study_id"], grant)
    assert bundle["cells"][0]["result"] == result
    assert bundle["cells"][0]["admission"]["documents"]["propagation_qualification"] == evidence


@pytest.mark.parametrize("fault", ["pointer", "source-attempt", "source-artifact", "source-grant", "prereg", "metric", "policy"])
def test_missing_substituted_or_unregistered_pilot_refuses_before_target_input(source, fault):
    store, spec, _, _, _, package, prereg = source
    package, prereg = deepcopy(package), deepcopy(prereg)
    pointer = package["payload"][PAYLOAD_KEY]
    if fault == "pointer":
        package["payload"].pop(PAYLOAD_KEY)
    elif fault == "source-attempt":
        pointer["source_attempt_id"] = "missing"
    elif fault == "source-artifact":
        pointer["source_artifact_id"] = "0"*64
    elif fault == "source-grant":
        pointer["source_authorization_id"] = "missing"
    elif fault == "prereg":
        prereg.pop("mlmc_production_policies")
    elif fault == "metric":
        prereg["primary_metrics"] = ["error"]
    else:
        pointer["production_policy"]["maximum_job_seconds"] = 61
    target_reads = [e for e in store.events() if e["event_kind"] == "READ_COMPLETED" and e["payload"].get("study_id") == spec["study_id"]]
    with pytest.raises(ResearchError):
        prepare_managed_mlmc(store, spec, spec["cells"][0], package, prereg)
    assert target_reads == [e for e in store.events() if e["event_kind"] == "READ_COMPLETED" and e["payload"].get("study_id") == spec["study_id"]]


@pytest.mark.parametrize("fault", ["seed", "coupling", "bias-grid", "horizon", "model", "allocation", "arm", "role", "mode"])
def test_substituted_production_identity_or_policy_cannot_borrow_pilot(source, fault):
    store, spec, _, _, _, package, prereg = source
    spec, package, prereg = deepcopy(spec), deepcopy(package), deepcopy(prereg)
    cell = spec["cells"][0]
    if fault in {"seed", "coupling", "bias-grid", "horizon"}:
        key = {"seed": "seed", "coupling": "coupling_id", "bias-grid": "steps", "horizon": "horizons"}[fault]
        cell["propagation_request"][key] = {"seed": 11, "coupling": "pilot-root", "bias-grid": 8, "horizon": [2.]}[fault]
    elif fault == "model":
        cell["frozen_dynamics"]["package_id"] += "changed"
    elif fault == "allocation":
        cell["mlmc_production_policy"]["level_samples"][0] += 1
    elif fault == "arm":
        spec["arms"][0]["method_family_id"] = "mlmc-new-budget"
    elif fault == "role":
        cell["execution_role"] = "pilot"
    else:
        spec["admission"]["mode"] = "pilot"
    with pytest.raises((ResearchError, DataValidationError)):
        prepare_managed_mlmc(store, spec, cell, package, prereg)


@pytest.mark.parametrize("fault", ["pass", "bias", "width", "roundoff", "unknown-model", "cost", "certificate", "phase", "proposal"])
def test_resealed_false_numeric_pilot_is_not_qualification(source, fault):
    store, spec, _, _, _, package, prereg = source
    evidence = prepare_managed_mlmc(store, spec, spec["cells"][0], package, prereg)
    analysis = deepcopy(evidence["source_result"]["forecast"]["bounded_mlmc_pilot_analysis"])
    value = deepcopy(evidence["source_result"]["forecast"]["functional"])
    if fault == "pass":
        analysis["checks"]["finest_grid_bias"] = False
    elif fault == "bias":
        analysis["finest_grid_bias_absolute_upper"] = 0
    elif fault == "width":
        analysis["reference_width_upper"] += 1
    elif fault == "roundoff":
        analysis["sampler_roundoff"] = {"value": 0, "status": "BOUNDED"}
    elif fault == "unknown-model":
        analysis["model_error"] = {"value": 0, "status": "BOUNDED"}
    elif fault == "cost":
        analysis["cost_status"] = "free"
    elif fault == "certificate":
        analysis["finest_certificate"]["grid"]["steps"] *= 2
        analysis["finest_certificate_hash"] = digest(analysis["finest_certificate"])
    elif fault == "phase":
        value["diagnostics"] = [[k, 1 if k == "pilot_phase" else v] for k, v in value["diagnostics"]]
    else:
        analysis["empirical_pilot_analysis"]["sampling_only_proposal"][0] += 1
    analysis["analysis_hash"] = digest({k: v for k, v in analysis.items() if k != "analysis_hash"})
    cell = evidence["source_run"]["cell"]
    model = FrozenDynamicsPackage.from_manifest(cell["frozen_dynamics"],
        expected_hash=cell["propagation_request"]["model_package_hash"])
    with pytest.raises((ResearchError, DataValidationError)):
        source_analysis_values(analysis, AffineMLMCReferencePolicy.from_manifest(evidence["reference_policy"]),
            MLMCPilotPolicy(**evidence["pilot_policy"]), model, request_from_manifest(cell["propagation_request"]), functional_result(value))


def test_direct_formal_computation_has_no_self_promotion_or_stale_worker_receipt(source, tmp_path):
    store, spec, _, _, _, _, _ = source
    with pytest.raises(ResearchError, match="owner admission"):
        execute_propagation(spec, spec["cells"][0])
    outcome = next(a for a in store.attempts().values() if store.manifest("run-"+a["run_id"])["study_id"] == spec["study_id"])
    result = json.loads((store.path/"artifacts"/outcome["artifact_id"]).read_bytes())
    with pytest.raises(ResearchError, match="current owner attempt"):
        load_worker_admission(store, result["admission_hash"], spec, spec["cells"][0], tmp_path/"outside.json")


@pytest.mark.parametrize("fault", ["metric", "se", "estimate", "interval", "model-error", "stream", "sample-count", "unit", "zero-variance", "sampling-target"])
def test_owner_rechecks_actual_current_stochastic_output_not_just_pilot(source, fault):
    store, spec, registry, recovery, _, _, _ = source
    outcome = SharedRunner(store, registry, recovery_registry=recovery).run_cell(spec["study_id"], digest(spec["cells"][0]))
    result = json.loads((store.path/"artifacts"/outcome["artifact_id"]).read_bytes())
    receipt = store.manifest("admission-"+result["admission_hash"])
    if fault == "metric":
        result["metrics"][METRIC] += 1
    elif fault in {"se", "estimate", "interval", "sample-count"}:
        key = {"se": "standard_error", "estimate": "estimate", "interval": "interval", "sample-count": "sample_count"}[fault]
        result["forecast"]["functional"][key] = [0, 0] if fault == "interval" else result["forecast"]["functional"][key]+1
    elif fault == "model-error":
        result["forecast"]["functional"]["error_budget"]["model"]["value"] = 0
    elif fault == "stream":
        result["forecast"]["qualified_error_components"]["stream"]["phase"] = 2
    elif fault in {"zero-variance", "sampling-target"}:
        value = result["forecast"]["functional"]
        variance = 0. if fault == "zero-variance" else 1_000_000.
        counts = cell_counts = spec["cells"][0]["mlmc_production_policy"]["level_samples"]
        value["diagnostics"] = [[k, [variance]*len(cell_counts) if k == "level_variances" else v] for k, v in value["diagnostics"]]
        import math
        se = math.sqrt(math.fsum(variance/n for n in counts))
        value["standard_error"] = se
        value["interval"] = [value["estimate"]-1.959963984540054*se, value["estimate"]+1.959963984540054*se]
    else:
        result["metric_units"][METRIC] = "other"
    with pytest.raises(ResearchError):
        validate_formal_mlmc_result(receipt, spec, spec["cells"][0], result)


@pytest.mark.parametrize("fault", ["counts-list", "hash", "cap", "missing", "extra"])
def test_complete_production_policy_is_strict_and_detached(source, fault):
    _, spec, _, _, _, _, _ = source
    value = deepcopy(spec["cells"][0]["mlmc_production_policy"])
    model, request, _, _ = validate_propagation_cell(spec, spec["cells"][0])
    if fault == "counts-list":
        policy = replace(MLMCProductionPolicy.from_manifest(value), level_samples=value["level_samples"])
    else:
        if fault == "hash":
            value["code_hash"] = "0"*64
        elif fault == "cap":
            value["maximum_job_seconds"] = 7201
        elif fault == "missing":
            value.pop("schema_version")
        else:
            value["extra"] = True
        with pytest.raises(DataValidationError):
            MLMCProductionPolicy.from_manifest(value).validate(model, request, code_hash())
        return
    with pytest.raises(DataValidationError):
        policy.validate(model, request, code_hash())


def test_old_generic_mlmc_adapter_does_not_borrow_analytic_or_new_mlmc_gate(source):
    store, spec, _, _, _, package, prereg = source
    spec = deepcopy(spec)
    spec["cells"][0]["plugin_id"] = propagation_plugin(recovery=True).plugin_id
    with pytest.raises(ResearchError):
        prepare_managed_mlmc(store, spec, spec["cells"][0], package, prereg)


def test_actual_production_job_cap_refuses_before_reservation_or_input(source):
    store, spec, registry, recovery, _, _, _ = source
    spec = deepcopy(spec)
    spec.update(study_id=spec["study_id"]+"-over-cap", experiment_id="over-cap-control")
    store.register(spec, digest(spec))
    before = store.events()
    with pytest.raises(ResearchError, match="frozen job budget"):
        SharedRunner(store, registry, recovery_registry=recovery).run_cell(spec["study_id"], digest(spec["cells"][0]), budget=BudgetSpec(61))
    added = store.events()[len(before):]
    assert not any(e["event_kind"] in {"RESERVE", "WORKER_STARTED", "READ_STARTED", "READ_COMPLETED"} for e in added)
    assert any(e["event_kind"] == "ATTEMPT" and e["payload"]["state"] == "PREFLIGHT_FAILED" for e in added)


@pytest.fixture(scope="module")
def exported(source):
    store, spec, registry, recovery, grant, _, _ = source
    outcome = SharedRunner(store, registry, recovery_registry=recovery).run_cell(spec["study_id"], digest(spec["cells"][0]))
    assert outcome["state"] == "SUCCEEDED"
    return export_evidence(store, spec["study_id"], grant)


def paper_validate(tmp_path, bundle):
    from pathlib import Path
    import subprocess
    import sys
    paper = Path(__file__).resolve().parents[2]/"TSDE-SDE"
    path = tmp_path/"mlmc-bundle.json"
    path.write_bytes(encode(bundle))
    return subprocess.run([sys.executable, "-B", str(paper/"scripts/pirc25/validate_mlmc.py"), str(path),
        "--expected-hash", bundle["bundle_hash"]], cwd=paper, capture_output=True, text=True, timeout=30)


def test_real_pilot_and_production_export_passes_independent_paper_cli(exported, tmp_path):
    checked = paper_validate(tmp_path, exported)
    assert checked.returncode == 0, checked.stderr
    verified = json.loads(checked.stdout)
    assert verified["source_bundle_hash"] == exported["bundle_hash"]
    assert verified["verified_mlmc_cells"] == verified["expected_cells"] == 1
    assert "no statistical adjudication, rigorous coverage or model qualification" in verified["scope"]


def reseal_transport(bundle):
    """Disposable altered export only, never source store/permissions/ledger."""
    row = bundle["cells"][0]
    receipt, evidence = row["admission"], row["admission"]["documents"].get("propagation_qualification")
    if evidence:
        source = evidence["source_result"]
        analysis = source["forecast"]["bounded_mlmc_pilot_analysis"]
        for key in ("continuous", "finest"):
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


@pytest.mark.parametrize("fault", ["width", "bias", "finest-grid", "empirical-cost", "proposal", "normal-as-exact",
    "unknown-model", "source-phase", "source-producer", "source-cost", "native-stop", "completion-order", "source-heldout",
    "target-se", "target-interval", "target-estimate", "target-stream", "target-metric", "target-roundoff", "target-unit"])
def test_independent_reader_rejects_resealed_stochastic_numeric_and_cost_substitutions(exported, tmp_path, fault):
    from pathlib import Path
    import importlib.util
    module_path = Path(__file__).resolve().parents[2]/"TSDE-SDE/scripts/pirc25/mlmc_qualification.py"
    import sys
    sys.path.insert(0, str(module_path.parent))
    try:
        entry = importlib.util.spec_from_file_location("independent_mlmc_reader", module_path)
        reader = importlib.util.module_from_spec(entry)
        entry.loader.exec_module(reader)
    finally:
        sys.path.pop(0)
    bundle = deepcopy(exported)
    row = bundle["cells"][0]
    evidence = row["admission"]["documents"]["propagation_qualification"]
    analysis = evidence["source_result"]["forecast"]["bounded_mlmc_pilot_analysis"]
    if fault == "width":
        analysis["reference_width_upper"] += 1
    elif fault == "bias":
        analysis["finest_grid_bias_absolute_upper"] = 0
    elif fault == "finest-grid":
        analysis["finest_certificate"]["grid"]["steps"] *= 2
    elif fault == "empirical-cost":
        analysis["empirical_pilot_analysis"]["level_cost_ratios"][0] = 0
    elif fault == "proposal":
        analysis["empirical_pilot_analysis"]["sampling_only_proposal"][0] += 1
    elif fault == "normal-as-exact":
        analysis["sampling_error"]["interval_kind"] = "exact-guaranteed"
    elif fault == "unknown-model":
        analysis["model_error"] = {"value": 0, "status": "BOUNDED"}
    elif fault == "source-phase":
        value = evidence["source_result"]["forecast"]["functional"]
        value["diagnostics"] = [[k, 1 if k == "pilot_phase" else v] for k, v in value["diagnostics"]]
    elif fault == "source-producer":
        evidence["source_run"]["cell"]["plugin_id"] = "affine-propagation-qualification"
    elif fault == "source-cost":
        evidence["settlement_event"]["payload"]["charged_ms"] = 0
    elif fault == "native-stop":
        evidence["stop_event"]["payload"]["confirmation"] = "not stopped"
    elif fault == "completion-order":
        evidence["completion_event"]["sequence"] = row["admission"]["input_evidence"][0]["sequence"]+1
    elif fault == "source-heldout":
        evidence["source_admission"]["documents"]["protocol"]["blocks"][0]["split_role"] = "test"
    elif fault in {"target-se", "target-interval", "target-estimate"}:
        f = row["result"]["forecast"]["functional"]
        key = {"target-se": "standard_error", "target-interval": "interval", "target-estimate": "estimate"}[fault]
        f[key] = [0, 0] if key == "interval" else f[key]+1
    elif fault == "target-stream":
        row["result"]["forecast"]["qualified_error_components"]["stream"]["phase"] = 2
    elif fault == "target-metric":
        row["result"]["metrics"][METRIC] += 1
        row["metrics"] = deepcopy(row["result"]["metrics"])
    elif fault == "target-roundoff":
        row["result"]["forecast"]["qualified_error_components"]["sampler_roundoff"] = {"value": 0, "status": "BOUNDED"}
    else:
        row["metric_units"][METRIC] = row["result"]["metric_units"][METRIC] = "other"
    reseal_transport(bundle)
    with pytest.raises(ValueError):
        reader.validate_mlmc_qualification(row["admission"], row)


@pytest.mark.parametrize("fault", ["missing-numeric", "missing-pointer", "target-false-metric"])
def test_full_mlmc_cli_has_no_generic_operator_pass_fallback(exported, tmp_path, fault):
    bundle = deepcopy(exported)
    row = bundle["cells"][0]
    if fault == "missing-numeric":
        row["admission"]["documents"].pop("propagation_qualification")
    elif fault == "missing-pointer":
        # The dedicated producer also forces validation when pointer is absent.
        row["admission"]["documents"]["package"]["payload"].pop(PAYLOAD_KEY)
        row["admission"]["documents"].pop("propagation_qualification")
    else:
        row["result"]["metrics"][METRIC] += 1
        row["metrics"] = deepcopy(row["result"]["metrics"])
    reseal_transport(bundle)
    checked = paper_validate(tmp_path, bundle)
    assert checked.returncode != 0
