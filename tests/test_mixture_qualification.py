"""Dedicated actual mixture evidence; disposable controls, not research grants."""

from dataclasses import asdict, replace
from fractions import Fraction
import json

import pytest

from application.research_admission import AdmissionGate
from application.research_budget import BudgetLedger, BudgetSpec
from application.research_contracts import CapabilityRegistry
from application.research_execution import execution_binding
from application.research_registry import plan_resources
from domain.errors import DataValidationError
from domain.mixture import MixtureSettings
from domain.mixture_qualification import MixtureQualificationPolicy, REFERENCE_WORK_CAP
from experiments.pirc25.affine import code_hash
from experiments.pirc25.runner import SharedRunner
from experiments.pirc27.mixture_qualification_plugin import mixture_qualification_config, mixture_qualification_plugin
from experiments.pirc27.plugin import execution_inputs
from inference.affine_reference import _parse_interval
from inference.mixture_qualification import qualify_affine_mixture
from infrastructure.research_store import ResearchError, ResearchStore, digest
from tests.research_admission_fixtures import admit_fixture
from tests.test_propagation_methods import inputs


def policies(*, cap=1, functional="endpoint-halfspace", changes=None):
    # Explicit coarse engineering target for positive/negative mechanism checks;
    # never a scientific default or an approved study accuracy target.
    package, request = inputs(samples=2, steps=1, tolerance=1., functional=functional,
        initial_mean=(0.,)*4,
        initial_covariance=((.25, 0., 0., 0.), (0.,)*4, (0.,)*4, (0.,)*4))
    mixture = MixtureSettings(cap, 0., 0., (1.,)*4, 0., 1_000_000, 60.).bind(package, request, code_hash())
    policy = MixtureQualificationPolicy(request.request_hash, package.package_hash,
        code_hash(), mixture.policy_hash, 1e-20, 1., 1., 1., 10., 400_001, 60.)
    return package, request, mixture, replace(policy, **(changes or {}))


@pytest.mark.parametrize("cap", [1, 4])
@pytest.mark.parametrize("functional", ["endpoint-x", "endpoint-halfspace"])
def test_actual_mixture_gets_own_continuous_and_grid_error_not_gaussian_approval(cap, functional):
    package, request, mixture, policy = policies(cap=cap, functional=functional)
    result, analysis = qualify_affine_mixture(package, request, mixture, policy)
    assert analysis["status"] == "PASSED" and all(analysis["checks"].values())
    assert analysis["actual_estimate"] == result.estimate
    assert analysis["actual_functional_hash"] == digest(result.manifest())
    assert analysis["mixture_policy_hash"] == mixture.policy_hash
    assert analysis["scientific_qualification"] is False and result.status == "APPROXIMATION_ONLY"
    assert analysis["propagation_approximation"] == analysis["implementation_roundoff"] == {
        "value": None, "status": "NOT_IDENTIFIABLE"}
    assert analysis["model_error"]["value"] is None and analysis["sampling_error"]["value"] == 0
    assert analysis["cost_status"] == "OWNER_SETTLEMENT_REQUIRED"
    for key, error in (("continuous_certificate", "total_functional_error_upper"),
                       ("target_certificate", "retained_functional_error_upper")):
        interval = _parse_interval(analysis[key]["functional_bounds"])
        required = max(abs(Fraction(result.estimate)-interval.lo), abs(Fraction(result.estimate)-interval.hi))
        assert Fraction(analysis[error]) >= required
    assert digest({k: v for k, v in analysis.items() if k != "analysis_hash"}) == analysis["analysis_hash"]
    if cap == 4 and functional == "endpoint-halfspace":
        assert analysis["retained_functional_error_upper"] > .3
        assert len(dict(result.diagnostics)["components"]) > 1


@pytest.mark.parametrize("changes,check", [({"maximum_reference_width": 1e-100}, "reference_width"),
    ({"maximum_retained_functional_error": 1e-100}, "retained_functional_error"),
    ({"maximum_time_bias": 1e-20}, "time_bias"),
    ({"maximum_total_functional_error": 1e-20}, "total_functional_error"),
    ({"maximum_scaled_transition_norm": .1}, "scaled_transition_growth"),
    ({"maximum_reference_operations": 1}, "reference_operations")])
def test_negative_qualification_is_saved_without_adjusting_thresholds(changes, check):
    package, request, mixture, policy = policies(cap=4, changes=changes)
    _, analysis = qualify_affine_mixture(package, request, mixture, policy)
    assert analysis["status"] == "FAILED" and analysis["checks"][check] is False
    assert analysis["policy_hash"] == policy.policy_hash and analysis["scientific_qualification"] is False


@pytest.mark.parametrize("changes", [{"code_hash": "0"*64}, {"request_hash": "0"*64},
    {"model_package_hash": "0"*64}, {"mixture_policy_hash": "0"*64},
    {"maximum_job_seconds": 61}, {"maximum_job_seconds": 1801},
    {"maximum_reference_operations": True}, {"maximum_reference_operations": 400_002},
    {"maximum_total_functional_error": 1.01},
    {"maximum_total_functional_error": float("inf")}, {"maximum_time_bias": True},
    {"maximum_reference_width": [1.]*100}])
def test_invalid_policy_refuses_before_mixture_or_reference_work(monkeypatch, changes):
    import inference.mixture_qualification as module
    package, request, mixture, policy = policies(changes=changes)
    def forbidden(*args, **kwargs):
        pytest.fail("invalid policy reached numerical work")
    monkeypatch.setattr(module, "mixture_estimate", forbidden)
    monkeypatch.setattr(module, "bound_affine_reference", forbidden)
    with pytest.raises(DataValidationError):
        qualify_affine_mixture(package, request, mixture, policy)


def test_policy_early_canonical_copy_and_combined_work_bounds(monkeypatch):
    import domain.mixture_qualification as module
    package, request, mixture, policy = policies()
    assert MixtureQualificationPolicy.from_manifest(policy.manifest()) == policy
    cyclic = []
    cyclic.append(cyclic)
    document = {**policy.manifest(), "maximum_reference_width": cyclic}
    monkeypatch.setattr(module, "asdict", lambda *args: pytest.fail("invalid fields copied"))
    with pytest.raises(DataValidationError):
        MixtureQualificationPolicy.from_manifest(document)
    large_request = replace(request, steps=1893)
    large_mixture = MixtureSettings(1, 0., 0., (1.,)*4, 0., 1_000_000, 60.).bind(
        package, large_request, code_hash())
    large_policy = replace(policy, request_hash=large_request.request_hash,
        mixture_policy_hash=large_mixture.policy_hash)
    with pytest.raises(DataValidationError, match="work binding"):
        large_policy.validate(package, large_request, large_mixture, code_hash())


@pytest.mark.parametrize("cap,expected", [(1, "PASSED"), (4, "FAILED")])
def test_actual_tail_accuracy_target_is_not_inferred_from_preserved_moments(cap, expected):
    package, request, mixture, policy = policies(cap=cap)
    request = replace(request, tolerance=.1)
    mixture = replace(mixture, request_hash=request.request_hash)
    policy = replace(policy, request_hash=request.request_hash, mixture_policy_hash=mixture.policy_hash,
        maximum_total_functional_error=request.tolerance)
    _, analysis = qualify_affine_mixture(package, request, mixture, policy)
    assert analysis["status"] == expected
    assert analysis["checks"]["total_functional_error"] is (expected == "PASSED")
    with pytest.raises(DataValidationError):
        replace(policy, maximum_total_functional_error=.1001).validate(package, request, mixture, code_hash())


def prepared(root, *, cap=1, changes=None, formal=False):
    package, request, mixture, policy = policies(cap=cap, changes=changes)
    plugin = mixture_qualification_plugin()
    config, parameters = mixture_qualification_config(request, mixture, policy), execution_inputs(request)
    store = ResearchStore(root, "mixture-qualification-unit", initialize=True)
    cell = {"arm_id": request.arm_id, "block_id": "generator-v1", "seed": request.seed,
        "horizon": request.horizons[0], "plugin_id": plugin.plugin_id, "capability": "generic-rollout",
        "resource_class": "cpu", "visibility": "synthetic", "dimensions": 4,
        "study_role": "secondary", "execution_role": "qualification",
        "propagation_request": json.loads(json.dumps(asdict(request))), "frozen_dynamics": package.manifest(),
        "mixture_policy": mixture.manifest(), "mixture_qualification_policy": policy.manifest(),
        "execution": execution_binding(plugin.registry_entry, config, parameters, matrix_cells=1)}
    spec = {"schema_version": "pirc25-contract-v1", "study_id": "mixture-qualification-unit",
        "experiment_id": "mixture-qualification-unit", "comparison_family": "synthetic-engineering",
        "code_hash": code_hash(), "protocol_hash": digest("pending"), "data_hash": digest("pending"),
        "feature_hash": digest("none"), "selection_hash": digest("none"),
        "arms": [{"arm_id": request.arm_id, "model_family_id": "affine-stable-v1",
            "method_family_id": "mixture", "objective_id": request.functional, "budget_seconds": 86400}],
        "cells": [cell], "runtime_binding": {"root": str(root.resolve()), "store_id": store.store_id}}
    admit_fixture(store, spec, plugin, root, formal=formal, execution_config=config,
        execution_inputs=parameters, legacy_upstream=False, fixture_prefix="mixture-qualification-")
    spec["admission"]["mode"] = "formal" if formal else "pilot"
    registry = CapabilityRegistry()
    registry.register(plugin)
    return store, spec, registry


@pytest.mark.parametrize("cap,changes,expected", [(1, None, "PASSED"), (4, None, "PASSED"),
    (4, {"maximum_retained_functional_error": .01}, "FAILED")])
def test_actual_charged_pilot_retains_positive_and_negative_method_evidence(tmp_path, cap, changes, expected):
    store, spec, registry = prepared(tmp_path, cap=cap, changes=changes)
    store.register(spec, digest(spec))
    result = SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]),
        budget=BudgetSpec(60, category="pilot"))
    assert result["state"] == "SUCCEEDED", result
    artifact = json.loads((store.path/"artifacts"/result["artifact_id"]).read_bytes())
    analysis = artifact["forecast"]["mixture_qualification_analysis"]
    assert analysis["status"] == expected and artifact["qualification"] == "fixture"
    assert artifact["resume_level"] == "restart-only" and analysis["scientific_qualification"] is False
    assert analysis["actual_functional_hash"] == digest(artifact["forecast"]["functional"])
    settlement = [e for e in store.events() if e["event_kind"] == "SETTLE"]
    assert len(settlement) == 1 and settlement[0]["payload"]["charged_ms"] > 0
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


def test_direct_admission_cannot_consume_heldout_even_with_generic_report(tmp_path):
    store, spec, _ = prepared(tmp_path, formal=True)
    store.register(spec, digest(spec))
    cell = spec["cells"][0]
    attempt = store.new_attempt(store.register_run(spec["study_id"], cell))
    with pytest.raises(ResearchError, match="cannot consume test/final-eval"):
        AdmissionGate(store).prepare(spec, cell, mixture_qualification_plugin(), attempt)
    assert not any(e["event_kind"] in {"READ_COMPLETED", "WORKER_STARTED"} for e in store.events())


def test_qualification_resources_include_kernel_and_bounded_rational_pools():
    _, request, mixture, policy = policies(cap=4)
    plugin = mixture_qualification_plugin()
    plan = plan_resources(plugin.registry_entry, mixture_qualification_config(request, mixture, policy),
        execution_inputs(request), matrix_cells=1)
    pools = [t for t in plan["tensors"] if "-pool-" in t["name"]]
    assert len(pools) == 64 and sum(t["bytes"] for t in pools) == 64*1024*1024
    assert plan["counts"]["steps"] == request.steps*mixture.work_per_step+policy.maximum_reference_operations
    assert plan["counts"]["state_dim"] == 4 and plan["counts"]["components"] == 4
    from inference.affine_reference import algorithm_manifest
    assert REFERENCE_WORK_CAP == 2*algorithm_manifest()["operations"]+1
    strict = replace(policy, maximum_reference_operations=1)
    assert mixture_qualification_config(request, mixture, strict)["work_steps"] == plan["counts"]["steps"]
