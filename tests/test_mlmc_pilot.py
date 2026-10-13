"""Engineering-only independent pilot diagnostics, not research qualification."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json

import numpy as np
import pytest

from domain.errors import DataValidationError
from domain.mlmc_pilot import MLMCPilotPolicy
from inference.mlmc_pilot import analyze_mlmc_pilot
from inference.propagation_methods import mlmc_estimate
from tests.test_propagation_methods import inputs


COUNTS = (8, 8, 8)


def policy_for(request, **changes):
    # Explicit unit-fixture thresholds; never business/research defaults.
    return replace(MLMCPilotPolicy(request.request_hash, request.seed+1, "production-root",
        1.0, 1.0, 0.75, 100.0, 2, 1000, 100000), **changes)


def pilot(**changes):
    package, request = inputs(samples=sum(COUNTS), steps=4, chunk_size=4, **changes)
    result = mlmc_estimate(package, request, level_samples=COUNTS, phase=2, pilot=True)
    return package, request, result


def diagnostics(result, **changes):
    values = dict(result.diagnostics)
    values.update(changes)
    return replace(result, diagnostics=tuple(values.items()))


def test_actual_pilot_measures_each_level_without_changing_global_rng_or_production():
    before = np.random.get_state()
    package, request, result = pilot()
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    production = mlmc_estimate(package, request, level_samples=COUNTS)
    assert result.status == "PILOT_ONLY" and production.status == "SUCCEEDED"
    assert result.estimate != production.estimate
    assert "level_compute_ns" not in dict(production.diagnostics)
    assert all(ns > 0 for ns in dict(result.diagnostics)["level_compute_ns"])
    assert dict(result.diagnostics)["level_work_per_sample"] == (4, 12, 24)
    analysis = analyze_mlmc_pilot(request, result, policy_for(request))
    assert analysis["status"] == "UNQUALIFIED" and "REFERENCE_UNRESOLVED" in analysis["reasons"]
    assert analysis["reference_uncertainty"] is None
    assert analysis["automatic_execution"] is False
    assert analysis["production_stream"]["seed"] != request.seed
    json.dumps(analysis, allow_nan=False)


@pytest.mark.parametrize("changes", [{"production_seed": 71}, {"production_seed": True},
    {"pilot_request_hash": "substituted"}, {"sampling_tolerance": float("nan")},
    {"sampling_tolerance": 10**1000}, {"minimum_level_samples": 9},
    {"maximum_samples": 23}, {"maximum_work_steps": 319},
    {"maximum_variance_ratio": 1}, {"maximum_cost_ratio": 1},
    {"production_coupling_id": "bad\nstream"}])
def test_policy_refuses_mutated_or_unbounded_allocation(changes):
    _, request = inputs(samples=24, steps=4)
    with pytest.raises(DataValidationError):
        policy_for(request, **changes).validate(request, COUNTS)


def test_policy_is_immutable_content_bound_and_requires_three_levels():
    _, request = inputs(samples=24, steps=4)
    policy = policy_for(request)
    with pytest.raises(FrozenInstanceError):
        policy.maximum_samples = 2000
    assert policy.policy_hash != replace(policy, maximum_samples=2000).policy_hash
    with pytest.raises(DataValidationError):
        policy.validate(request, (12, 12))
    with pytest.raises(DataValidationError):
        policy.validate(request, list(COUNTS))


@pytest.mark.parametrize("phase", [0, 1, 2.0, True])
def test_pilot_refuses_nonindependent_phase(phase):
    package, request = inputs(samples=24)
    with pytest.raises(DataValidationError):
        mlmc_estimate(package, request, level_samples=COUNTS, phase=phase, pilot=True)


def test_sampling_proposal_uses_measured_cost_but_never_approves_unknown_reference():
    _, request, result = pilot()
    controlled = diagnostics(result, level_variances=(1.0, 0.25, 0.0625), level_compute_ns=(80, 160, 320))
    analysis = analyze_mlmc_pilot(request, controlled, policy_for(request))
    allocation = analysis["sampling_only_proposal"]
    assert allocation[0] > allocation[1] >= allocation[2] >= 2
    assert sum(v/n for v, n in zip((1, .25, .0625), allocation)) <= 1.0
    assert analysis["level_variance_ratios"] == (0.25,)
    assert analysis["level_cost_ratios"] == (2.0, 2.0)
    assert analysis["status"] == "UNQUALIFIED" and analysis["requires_separate_production_registration"]
    assert "ENGINEERING_ONLY" in analysis["reasons"]
    assert "shared-ledger-remains-sole-budget-authority" in analysis["cost_scope"]


@pytest.mark.parametrize("variance,cost,reason", [
    ((1., 1., 1.), (80, 160, 320), "VARIANCE_DECAY_FAILED"),
    ((1., 0., 0.), (80, 160, 320), "UNRESOLVED_LEVEL_VARIANCE"),
    ((1., 1e-320, 1e300), (80, 160, 320), "UNRESOLVED_LEVEL_VARIANCE"),
    ((1., .25, .0625), (80, 160, 32000), "COST_GROWTH_FAILED")])
def test_failed_or_unresolved_rates_have_no_automatic_allocation(variance, cost, reason):
    _, request, result = pilot()
    analysis = analyze_mlmc_pilot(request, diagnostics(result, level_variances=variance, level_compute_ns=cost), policy_for(request))
    assert reason in analysis["reasons"] and analysis["sampling_only_proposal"] is None
    json.dumps(analysis, allow_nan=False)


def test_allocation_refuses_policy_cap_without_expanding_work():
    _, request, result = pilot()
    result = diagnostics(result, level_variances=(1., .25, .0625), level_compute_ns=(80, 160, 320))
    analysis = analyze_mlmc_pilot(request, result, policy_for(request, sampling_tolerance=1e-6))
    assert "ALLOCATION_INFEASIBLE" in analysis["reasons"]
    assert analysis["sampling_only_proposal"] is None


@pytest.mark.parametrize("changes", [{"level_compute_ns": (0, 1, 1)},
    {"level_compute_ns": (1, 1, 2**63)}, {"level_variances": (1., float("nan"), 1.)},
    {"pilot_phase": 1}, {"pilot_phase": 2.0}, {"production_phase": True},
    {"cost_scope": "ledger-charge"}, {"level_work_per_sample": (4, 8, 16)}])
def test_substituted_pilot_evidence_is_rejected(changes):
    _, request, result = pilot()
    with pytest.raises(DataValidationError):
        analyze_mlmc_pilot(request, diagnostics(result, **changes), policy_for(request))


def test_completed_cost_checkpoint_resumes_only_remaining_samples():
    package, request, expected = pilot()
    saved = []
    class StopAfterChunk(Exception):
        pass
    def stop(state, total):
        saved.append(state)
        raise StopAfterChunk
    with pytest.raises(StopAfterChunk):
        mlmc_estimate(package, request, level_samples=COUNTS, phase=2, pilot=True, checkpoint=stop)
    checkpoint = saved[0]
    old_cost = checkpoint["method_state"]["statistics"][0]["compute_ns"]
    assert checkpoint["step"] == 16 and old_cost > 0
    resumed = mlmc_estimate(package, request, level_samples=COUNTS, phase=2, pilot=True, resume_state=checkpoint)
    actual_diagnostics, expected_diagnostics = dict(resumed.diagnostics), dict(expected.diagnostics)
    actual_costs = actual_diagnostics.pop("level_compute_ns")
    expected_diagnostics.pop("level_compute_ns")
    assert actual_diagnostics == expected_diagnostics
    assert actual_costs[0] > old_cost and resumed.estimate == expected.estimate
    assert resumed.error_budget == expected.error_budget
    assert "budget" not in checkpoint and len(json.dumps(checkpoint)) < 16384
    with pytest.raises(DataValidationError):
        mlmc_estimate(package, request, level_samples=COUNTS, phase=2, resume_state=checkpoint)
    for cost in [-1, 0, 2**63, True]:
        mutated = deepcopy(checkpoint)
        mutated["method_state"]["statistics"][0]["compute_ns"] = cost
        with pytest.raises(DataValidationError):
            mlmc_estimate(package, request, level_samples=COUNTS, phase=2, pilot=True, resume_state=mutated)


def test_zero_hit_pilot_retains_unknown_sampling_and_bias_components():
    _, request, result = pilot(functional="endpoint-halfspace", threshold=100)
    analysis = analyze_mlmc_pilot(request, result, policy_for(request))
    assert result.status == "UNRESOLVED_SAMPLING" and result.standard_error is None
    assert "SAMPLING_UNRESOLVED" in analysis["reasons"]
    assert analysis["sampling_only_proposal"] is None


@pytest.mark.parametrize("synthetic", [False, True])
def test_restart_registry_does_not_advertise_or_admit_pilot(synthetic):
    from experiments.pirc27.plugin import execution_config, propagation_plugin
    from infrastructure.research_store import ResearchError
    _, request = inputs(samples=24, steps=4)
    assert "mlmc-pilot" not in propagation_plugin(synthetic=synthetic).registry_entry.config_schema["properties"]["method"]["enum"]
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        execution_config(request, "mlmc-pilot", level_samples=COUNTS, synthetic=synthetic)
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        execution_config(replace(request, samples=16), "mlmc-pilot", level_samples=(8, 8), recovery=True, synthetic=synthetic)
