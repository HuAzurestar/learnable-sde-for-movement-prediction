"""Own path/IS evidence and actual disposable charged pilots, not research."""

from dataclasses import asdict, replace
from fractions import Fraction
import json
import math

import pytest

from application.research_admission import AdmissionGate
from application.research_budget import BudgetLedger, BudgetSpec
from application.research_contracts import CapabilityRegistry
from application.research_execution import execution_binding
from application.research_registry import plan_resources
from domain.errors import DataValidationError
from domain.path_qualification import PathQualificationPolicy, METHODS, REFERENCE_WORK_CAP
from experiments.pirc25.affine import code_hash
from experiments.pirc25.runner import SharedRunner
from experiments.pirc27.path_qualification_plugin import path_qualification_config, path_qualification_plugin
from experiments.pirc27.plugin import execution_inputs
from inference.affine_reference import _parse_interval
from inference.path_qualification import qualify_affine_paths
from infrastructure.research_store import ResearchError, ResearchStore, digest
from tests.research_admission_fixtures import admit_fixture
from tests.test_propagation_methods import inputs


def policies(method="euler", *, changes=None, request_changes=None):
    # Coarse explicit engineering control, not a default scientific target.
    settings = {"samples": 64, "steps": 2, "chunk_size": 7, "tolerance": 1.,
        "functional": "endpoint-halfspace", "threshold": .25, "initial_mean": (0.,)*4,
        "initial_covariance": tuple(tuple(.25 if i == j else 0. for j in range(4)) for i in range(4)),
        "arm_id": method+"-original-arm"}
    package, request = inputs(**{**settings, **(request_changes or {})})
    policy = PathQualificationPolicy(request.request_hash, package.package_hash, code_hash(), method,
        (.5, 0.) if method == "importance" else (0., 0.), (1.,)*4,
        1e-20, 1., 1., 1., 1., 2., 100., 400_001, 60.)
    return package, request, replace(policy, **(changes or {}))


@pytest.mark.parametrize("method,functional", [(m, f) for m in METHODS
    for f in ("endpoint-x", "endpoint-halfspace") if m != "importance" or f == "endpoint-halfspace"])
def test_actual_paths_get_own_saved_statistics_and_correct_grid(method, functional):
    package, request, policy = policies(method, request_changes={"functional": functional})
    result, analysis = qualify_affine_paths(package, request, policy)
    assert analysis["status"] == "PASSED" and all(analysis["checks"].values())
    assert result.status == "SUCCEEDED" and analysis["scientific_qualification"] is False
    assert analysis["actual_functional_hash"] == digest(result.manifest())
    assert analysis["completed_statistics_hash"] == digest(analysis["completed_statistics"])
    assert analysis["target_grid"] == {"solver": "euler" if method == "importance" else method, "steps": request.steps}
    assert analysis["physical_dimension"] == 4
    assert analysis["auxiliary_dimension"] == (4 if method == "reversible-heun" else 0)
    state = analysis["completed_statistics"]
    assert state["data_position"] == {"level": 1, "next_sample": 0}
    assert state["step"] == request.samples*request.steps and state["chunk_complete"] is True
    assert state["rng_state"]["seed"] == request.seed and state["rng_state"]["phase"] == 0
    assert state["rng_state"]["coupling_id"] == request.coupling_id
    assert len(json.dumps(state).encode()) <= 16384
    assert analysis["implementation_roundoff"]["value"] is None
    assert analysis["propagation_approximation"]["value"] is None
    assert analysis["model_error"]["value"] is None
    assert analysis["sampling_error"]["uncertainty_radius"] > 0
    assert analysis["cost_status"] == "OWNER_SETTLEMENT_REQUIRED"
    for key, error in (("continuous_certificate", "total_observed_functional_error_upper"),
                       ("target_certificate", "observed_grid_error_upper")):
        bounds = _parse_interval(analysis[key]["functional_bounds"])
        required = max(abs(Fraction(result.estimate)-bounds.lo), abs(Fraction(result.estimate)-bounds.hi))
        assert Fraction(analysis[error]) >= required
    assert digest({k: v for k, v in analysis.items() if k != "analysis_hash"}) == analysis["analysis_hash"]
    if method == "importance":
        assert analysis["sampling_error"]["self_normalized"] is False
        assert analysis["sampling_error"]["interval_kind"] == "iid-weighted-normal-approximation-95"
        assert analysis["proposal_hash"] == dict(result.diagnostics)["proposal_hash"]
        assert analysis["sampling_error"]["effective_sample_size"] >= policy.minimum_effective_sample_size


@pytest.mark.parametrize("threshold,hits", [(100., 0), (-100., 64)])
@pytest.mark.parametrize("method", ["euler", "heun", "reversible-heun"])
def test_zero_or_all_mc_hits_have_positive_one_sided_uncertainty(method, threshold, hits):
    package, request, policy = policies(method, request_changes={"threshold": threshold})
    result, analysis = qualify_affine_paths(package, request, policy)
    sampling = analysis["sampling_error"]
    assert sampling["hits"] == hits and sampling["uncertainty_radius"] > 0
    assert sampling["uncertainty_radius"] == -math.expm1(math.log(.05)/request.samples)
    assert sampling["interval_kind"] == "exact-binomial-one-sided-95-formula"
    assert "ideal iid Bernoulli" in sampling["coverage_scope"] and "not certified" in sampling["coverage_scope"]
    assert result.standard_error == (None if hits == 0 else 0.)
    strict = replace(policy, maximum_sampling_uncertainty=1e-10)
    _, failed = qualify_affine_paths(package, request, strict)
    assert failed["status"] == "FAILED" and failed["checks"]["sampling_uncertainty"] is False


def test_zero_weighted_hits_do_not_borrow_binomial_mc_bound():
    package, request, policy = policies("importance", request_changes={"threshold": 100.})
    result, analysis = qualify_affine_paths(package, request, policy)
    assert result.status == "INSUFFICIENT_EVENTS" and result.standard_error is None
    assert analysis["status"] == "FAILED" and analysis["checks"]["sampler_status"] is False
    assert analysis["sampling_error"]["uncertainty_radius"] is None
    assert analysis["sampling_error"]["interval_kind"] == "unavailable-no-weighted-hits"
    assert analysis["sampling_error"]["self_normalized"] is False


@pytest.mark.parametrize("samples", [2, 5, 64])
def test_weighted_zero_variance_is_not_a_zero_sampling_error_proof(samples):
    package, request, policy = policies("importance", changes={"proposal": (0., 0.)},
        request_changes={"threshold": -100., "samples": samples})
    result, analysis = qualify_affine_paths(package, request, policy)
    # Log-sum roundoff can be platform-sensitive; actual output stays untouched.
    assert result.status == "SUCCEEDED" and 0. <= result.standard_error < 1e-6
    assert analysis["status"] == "FAILED" and analysis["checks"]["sampling_uncertainty"] is False
    assert analysis["sampling_error"]["status"] == "NOT_IDENTIFIABLE"
    assert analysis["sampling_error"]["uncertainty_radius"] is None
    assert analysis["sampling_error"]["interval"] is None
    assert analysis["sampling_error"]["interval_kind"] == "unavailable-degenerate-weighted-event-statistics"


def test_frozen_ess_threshold_rejects_completed_is_without_reallocation():
    package, request, policy = policies("importance", changes={"minimum_effective_sample_size": 64.})
    result, analysis = qualify_affine_paths(package, request, policy)
    assert result.status == "SUCCEEDED" and analysis["status"] == "FAILED"
    assert analysis["checks"]["effective_sample_size"] is False
    assert analysis["sampling_error"]["sample_count"] == request.samples
    assert analysis["proposal"] == list(policy.proposal)


def test_actual_low_ess_status_is_retained_without_proposal_rescue():
    package, request, policy = policies("importance", changes={"proposal": (12., 0.)},
        request_changes={"threshold": -100.})
    result, analysis = qualify_affine_paths(package, request, policy)
    assert result.status == "LOW_ESS" and analysis["sampling_error"]["hits"] == request.samples
    assert analysis["status"] == "FAILED" and analysis["checks"]["sampler_status"] is False
    assert analysis["checks"]["effective_sample_size"] is False
    assert analysis["proposal"] == [12., 0.]


def test_zero_observed_endpoint_variance_is_not_qualified_as_zero_error():
    package, request, policy = policies(request_changes={"functional": "endpoint-x", "steps": 1,
        "initial_covariance": ((0.,)*4,)*4})
    result, analysis = qualify_affine_paths(package, request, policy)
    assert result.standard_error == 0. and result.status == "SUCCEEDED"
    assert analysis["status"] == "FAILED" and analysis["checks"]["sampling_uncertainty"] is False
    assert analysis["sampling_error"]["uncertainty_radius"] is None
    assert analysis["implementation_roundoff"]["value"] is None


@pytest.mark.parametrize("changes,check", [({"maximum_reference_width": 1e-100}, "reference_width"),
    ({"maximum_observed_grid_error": 1e-100}, "observed_grid_error"),
    ({"maximum_time_bias": 1e-100}, "time_bias"),
    ({"maximum_total_observed_functional_error": 1e-100}, "total_observed_functional_error"),
    ({"maximum_sampling_uncertainty": 1e-100}, "sampling_uncertainty"),
    ({"maximum_scaled_transition_norm": .1}, "scaled_transition_growth"),
    ({"maximum_reference_operations": 1}, "reference_operations")])
def test_failed_thresholds_preserve_actual_output_and_policy(changes, check):
    package, request, policy = policies(changes=changes)
    result, analysis = qualify_affine_paths(package, request, policy)
    assert analysis["status"] == "FAILED" and analysis["checks"][check] is False
    assert analysis["policy_hash"] == policy.policy_hash
    assert analysis["actual_functional_hash"] == digest(result.manifest())


@pytest.mark.parametrize("changes", [{"request_hash": "0"*64}, {"model_package_hash": "0"*64},
    {"code_hash": "0"*64}, {"method": "gaussian"}, {"method": []},
    {"proposal": (1., 0.)}, {"proposal": (float("inf"), 0.)}, {"state_scales": (1.,)*8},
    {"state_scales": (True, 1., 1., 1.)}, {"maximum_reference_width": [1.]*100},
    {"maximum_reference_operations": True}, {"maximum_reference_operations": 400_002},
    {"maximum_total_observed_functional_error": 1.01}, {"maximum_sampling_uncertainty": True},
    {"maximum_sampling_uncertainty": float("nan")}, {"minimum_effective_sample_size": 65.},
    {"minimum_effective_sample_size": 1.}, {"maximum_job_seconds": 1801.}, {"schema_version": []}])
def test_invalid_policy_refuses_before_sampler_or_reference(monkeypatch, changes):
    import inference.path_qualification as module
    package, request, policy = policies(changes=changes)
    def forbidden(*args, **kwargs):
        pytest.fail("invalid policy reached numerical work")
    for name in ("monte_carlo", "importance_sampling", "bound_affine_reference", "bound_affine_discrete"):
        monkeypatch.setattr(module, name, forbidden)
    with pytest.raises(DataValidationError):
        qualify_affine_paths(package, request, policy)


@pytest.mark.parametrize("field", ["proposal", "state_scales", "maximum_reference_width"])
def test_cyclic_inputs_are_rejected_before_deepcopy(monkeypatch, field):
    import domain.path_qualification as module
    _, _, policy = policies()
    assert PathQualificationPolicy.from_manifest(policy.manifest()) == policy
    cycle = []
    cycle.append(cycle)
    value = {**policy.manifest(), field: cycle}
    monkeypatch.setattr(module, "asdict", lambda *a: pytest.fail("invalid policy deep-copied"))
    with pytest.raises(DataValidationError):
        PathQualificationPolicy.from_manifest(value)


def test_one_actual_sampler_run_no_replay_or_caller_supplied_analytic_value(monkeypatch):
    import inference.path_qualification as module
    package, request, policy = policies()
    original, calls = module.monte_carlo, []
    def once(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "monte_carlo", once)
    result, analysis = qualify_affine_paths(package, request, policy)
    assert len(calls) == 1 and result.estimate == analysis["actual_estimate"]
    monkeypatch.setattr(module, "monte_carlo", lambda *a, **k: replace(original(*a, **k), estimate=.123))
    with pytest.raises(DataValidationError, match="sufficient statistics"):
        qualify_affine_paths(package, request, policy)


@pytest.mark.parametrize("mutation", ["missing", "partial", "seed", "count", "mean"])
def test_missing_or_tampered_final_statistics_refused(monkeypatch, mutation):
    import inference.path_qualification as module
    package, request, policy = policies()
    original = module.monte_carlo
    def corrupt(*args, **kwargs):
        callback = kwargs["checkpoint"]
        def bad(state, total):
            if mutation == "missing":
                return
            if mutation == "partial":
                state["data_position"] = {"level": 0, "next_sample": 0}
            if mutation == "seed":
                state["rng_state"]["seed"] += 1
            if mutation == "count":
                state["method_state"]["statistics"][0]["n"] = 2
            if mutation == "mean":
                state["method_state"]["statistics"][0]["mean"] = .123
            callback(state, total)
        return original(*args, **{**kwargs, "checkpoint": bad})
    monkeypatch.setattr(module, "monte_carlo", corrupt)
    with pytest.raises(DataValidationError):
        qualify_affine_paths(package, request, policy)


def prepared(root, *, method="euler", changes=None, formal=False, request_changes=None):
    package, request, policy = policies(method, changes=changes, request_changes=request_changes)
    plugin = path_qualification_plugin()
    config, parameters = path_qualification_config(request, policy), execution_inputs(request)
    store = ResearchStore(root, "path-qualification-unit", initialize=True)
    cell = {"arm_id": request.arm_id, "block_id": "generator-v1", "seed": request.seed,
        "horizon": request.horizons[0], "plugin_id": plugin.plugin_id,
        "capability": "rare-event" if method == "importance" else "generic-rollout",
        "resource_class": "cpu", "visibility": "synthetic", "dimensions": 4,
        "study_role": "secondary", "execution_role": "qualification",
        "propagation_request": json.loads(json.dumps(asdict(request))), "frozen_dynamics": package.manifest(),
        "path_qualification_policy": policy.manifest(),
        "execution": execution_binding(plugin.registry_entry, config, parameters, matrix_cells=1)}
    spec = {"schema_version": "pirc25-contract-v1", "study_id": "path-qualification-unit",
        "experiment_id": "path-qualification-unit", "comparison_family": "synthetic-engineering",
        "code_hash": code_hash(), "protocol_hash": digest("pending"), "data_hash": digest("pending"),
        "feature_hash": digest("none"), "selection_hash": digest("none"),
        "arms": [{"arm_id": request.arm_id, "model_family_id": "affine-stable-v1",
            "method_family_id": method, "objective_id": request.functional, "budget_seconds": 86400}],
        "cells": [cell], "runtime_binding": {"root": str(root.resolve()), "store_id": store.store_id}}
    admit_fixture(store, spec, plugin, root, formal=formal, execution_config=config,
        execution_inputs=parameters, legacy_upstream=False, fixture_prefix="path-qualification-")
    spec["admission"]["mode"] = "formal" if formal else "pilot"
    registry = CapabilityRegistry()
    registry.register(plugin)
    return store, spec, registry


@pytest.mark.parametrize("method,changes,request_changes,expected", [(m, None, None, "PASSED") for m in METHODS]
    + [("euler", {"maximum_sampling_uncertainty": 1e-10}, None, "FAILED"),
       ("importance", None, {"threshold": 100.}, "FAILED"),
       ("importance", {"minimum_effective_sample_size": 64.}, None, "FAILED")], ids=[
           "euler", "heun", "rev", "is", "mc-fail", "is-zero", "is-ess"])
def test_actual_charged_pilots_keep_method_specific_positive_and_negative_evidence(
        tmp_path, method, changes, request_changes, expected):
    store, spec, registry = prepared(tmp_path, method=method, changes=changes, request_changes=request_changes)
    store.register(spec, digest(spec))
    result = SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]),
        budget=BudgetSpec(60, category="pilot"))
    assert result["state"] == "SUCCEEDED", result
    artifact = json.loads((store.path/"artifacts"/result["artifact_id"]).read_bytes())
    analysis = artifact["forecast"]["path_qualification_analysis"]
    assert analysis["status"] == expected and artifact["qualification"] == "fixture"
    assert artifact["resume_level"] == "restart-only" and analysis["scientific_qualification"] is False
    assert analysis["actual_functional_hash"] == digest(artifact["forecast"]["functional"])
    settlements = [e for e in store.events() if e["event_kind"] == "SETTLE"]
    assert len(settlements) == 1 and settlements[0]["payload"]["charged_ms"] > 0
    assert BudgetLedger(store).balance(spec["arms"][0]["arm_id"])["committed_ms"] == result["elapsed_ms"]
    receipt = store.manifest("admission-"+artifact["admission_hash"])
    assert receipt["mode"] == "pilot" and receipt["documents"]["protocol"]["blocks"][0]["split_role"] == "train"
    assert receipt["resource_plan"]["tensor_bytes"] >= 64*1024*1024


@pytest.mark.parametrize("budget", [BudgetSpec(60, category="smoke"), BudgetSpec(61, category="pilot")])
def test_wrong_budget_refused_before_reservation_or_inputs(tmp_path, budget):
    store, spec, registry = prepared(tmp_path)
    store.register(spec, digest(spec))
    with pytest.raises(ResearchError, match="frozen pilot job budget"):
        SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]), budget=budget)
    assert not any(e["event_kind"] in {"RESERVE", "READ_COMPLETED", "WORKER_STARTED"} for e in store.events())


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("mode", ["formal", "pilot"])
def test_generic_operator_report_cannot_make_source_formal_or_heldout(tmp_path, method, mode):
    store, spec, _ = prepared(tmp_path, method=method, formal=True)
    spec["admission"]["mode"] = mode
    store.register(spec, digest(spec))
    cell = spec["cells"][0]
    attempt = store.new_attempt(store.register_run(spec["study_id"], cell))
    with pytest.raises(ResearchError, match="cannot consume test/final-eval"):
        AdmissionGate(store).prepare(spec, cell, path_qualification_plugin(), attempt)
    assert not any(e["event_kind"] in {"READ_COMPLETED", "WORKER_STARTED"} for e in store.events())


@pytest.mark.parametrize("method", METHODS)
def test_resources_keep_physical_four_and_full_fixed_reference_work(method):
    package, request, policy = policies(method)
    plugin = path_qualification_plugin()
    plan = plan_resources(plugin.registry_entry, path_qualification_config(request, policy),
        execution_inputs(request), matrix_cells=1)
    pools = [t for t in plan["tensors"] if "-pool-" in t["name"]]
    assert len(pools) == 64 and sum(t["bytes"] for t in pools) == 64*1024*1024
    assert plan["counts"]["steps"] == request.samples*request.steps+REFERENCE_WORK_CAP
    assert plan["counts"]["state_dim"] == 4
    strict = replace(policy, maximum_reference_operations=1)
    assert path_qualification_config(request, strict)["work_steps"] == plan["counts"]["steps"]
    large = replace(request, samples=300_000)
    large_policy = replace(policy, request_hash=large.request_hash)
    with pytest.raises(DataValidationError, match="work binding"):
        large_policy.validate(package, large, code_hash())
    with pytest.raises(DataValidationError, match="quota"):
        path_qualification_config(large, large_policy)
    assert plugin.resume_level == "restart-only"
