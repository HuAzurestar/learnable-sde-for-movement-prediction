"""Numerical qualification of bounded affine endpoint methods and coupling."""

from dataclasses import replace
import json

import numpy as np
import pytest

from domain.errors import DataValidationError
from domain.propagation import PropagationRequest
from experiments.pirc27.oracles import oracle_suite
from inference.propagation_methods import (allocate_mlmc, analytic_estimate, discrete_moments,
    endpoint_chunks, importance_sampling, mlmc_estimate, monte_carlo)


def inputs(**changes):
    case = oracle_suite()[0]
    request = PropagationRequest("test-endpoint", case.package.package_hash, case.initial_mean,
        case.initial_covariance, 0.0, 0.0, (1.0,), "endpoint-x", 71, "paired-root-v1", "affine-mc",
        samples=512, steps=16)
    return case.package, replace(request, **changes)


def endpoints(package, request, **kwargs):
    return np.concatenate([fine for _, _, fine, _, _ in endpoint_chunks(package, request, **kwargs)])


def test_chunk_size_and_resumed_sample_ranges_replay_the_same_paths():
    package, request = inputs(samples=37, steps=8, chunk_size=9)
    expected = endpoints(package, request)
    np.testing.assert_array_equal(expected, endpoints(package, replace(request, chunk_size=1)))
    resumed = np.concatenate([endpoints(package, request, sample_count=11),
                              endpoints(package, request, start_sample=11, sample_count=26)])
    np.testing.assert_array_equal(expected, resumed)


def test_coupled_coarse_is_the_sum_of_adjacent_fine_brownian_increments():
    package, request = inputs(samples=6, steps=2)
    fine, coarse = next(endpoint_chunks(package, request, level=1, paired=True))[2:4]
    from inference.propagation_methods import _inputs, _rng
    A, b, L, mean, root = _inputs(package, request)
    for sample in range(6):
        rng = _rng(request, sample, 1, 0)
        x = mean+root@rng.standard_normal(4)
        increments = rng.standard_normal((4, 2))*0.5
        f = x.copy()
        for noise in increments:
            f = f+(A@f+b)*0.25+L@noise
        c = x.copy()
        for noise in increments.reshape(2, 2, 2).sum(axis=1):
            c = c+(A@c+b)*0.5+L@noise
        np.testing.assert_allclose(fine[sample], f, rtol=0, atol=1e-15)
        np.testing.assert_allclose(coarse[sample], c, rtol=0, atol=1e-15)


@pytest.mark.parametrize("solver", ["euler", "additive-heun"])
def test_path_mc_moments_agree_with_discrete_gaussian_distribution(solver):
    package, request = inputs(samples=12000, steps=8)
    samples = endpoints(package, request, solver=solver)
    mean, covariance = discrete_moments(package, request, solver=solver)
    mean_error = np.abs(samples.mean(axis=0)-mean)
    assert np.all(mean_error <= 5*np.sqrt(np.diag(covariance)/len(samples)))
    # Gaussian sample covariance has a known Wishart standard deviation, also
    # for zero cross-covariances where relative tolerances cannot be meaningful.
    covariance_se = np.sqrt((np.outer(np.diag(covariance), np.diag(covariance)) + covariance**2)/(len(samples)-1))
    assert np.all(np.abs(np.cov(samples.T)-covariance) <= 5*covariance_se)


def test_heun_discrete_expectation_has_second_order_convergence_on_affine_fixture():
    package, request = inputs()
    target = analytic_estimate(package, request).estimate
    errors = []
    for steps in (8, 16, 32):
        value = analytic_estimate(package, replace(request, steps=steps), discrete=True, solver="additive-heun")
        errors.append(abs(value.estimate-target))
    assert errors[0]/errors[1] > 3.5
    assert errors[1]/errors[2] > 3.5


def test_zero_hit_mc_has_exact_positive_binomial_upper_bound_and_unknown_se():
    package, request = inputs(functional="endpoint-halfspace", threshold=100, samples=64)
    result = monte_carlo(package, request)
    assert result.estimate == 0 and result.standard_error is None
    assert result.interval == pytest.approx((0, 1-0.05**(1/64)))
    assert result.interval_kind == "exact-binomial-one-sided-95"
    assert result.error_budget.sampling.value is None


def test_mlmc_telescope_targets_finest_distribution_with_independent_levels():
    package, request = inputs(samples=1024, steps=2)
    result = mlmc_estimate(package, request, level_samples=(4000, 2000, 1000))
    target = float(discrete_moments(package, request, steps=8)[0][0])
    assert abs(result.estimate-target) < 5*result.standard_error
    diagnostics = dict(result.diagnostics)
    assert result.estimate == pytest.approx(sum(diagnostics["level_means"]))
    assert len(diagnostics["level_variances"]) == 3
    assert diagnostics["level_variances"][2] < diagnostics["level_variances"][1]


def test_importance_zero_shift_matches_mc_and_does_not_self_normalize():
    package, request = inputs(functional="endpoint-halfspace", threshold=0.7, samples=1024)
    mc = monte_carlo(package, request)
    importance = importance_sampling(package, request, proposal=(0.0, 0.0))
    assert importance.estimate == pytest.approx(mc.estimate, rel=1e-13)
    assert importance.standard_error == pytest.approx(mc.standard_error, rel=1e-12)
    assert dict(importance.diagnostics)["ess"] == pytest.approx(request.samples)
    assert dict(importance.diagnostics)["self_normalized"] is False


def test_importance_velocity_shift_agrees_with_target_and_preserves_weight_diagnostics():
    package, request = inputs(functional="endpoint-halfspace", threshold=1.1, samples=12000, steps=8)
    result = importance_sampling(package, request, proposal=(1.2, 0.0))
    target = analytic_estimate(package, request, discrete=True).estimate
    assert abs(result.estimate-target) < 5*result.standard_error
    diagnostics = dict(result.diagnostics)
    assert 1 < diagnostics["ess"] <= request.samples*(1+1e-12)
    assert len(diagnostics["proposal_hash"]) == 64
    assert abs(np.exp(diagnostics["log_sum_weights"])/request.samples-1) < 0.1
    json.dumps(result.manifest(), allow_nan=False)


def test_importance_no_hits_does_not_claim_zero_sampling_error_or_binomial_interval():
    package, request = inputs(functional="endpoint-halfspace", threshold=100, samples=32)
    result = importance_sampling(package, request, proposal=(1.0, 0.0))
    assert result.status == "INSUFFICIENT_EVENTS"
    assert result.standard_error is None and result.interval is None
    assert result.error_budget.sampling.status == "NOT_IDENTIFIABLE"


def test_mlmc_pilot_allocation_is_bounded_and_has_variance_cost_scaling():
    allocation = allocate_mlmc((1.0, 0.25, 0.0625), (1.0, 2.0, 4.0), 0.1)
    assert allocation[0] > allocation[1] > allocation[2]
    assert sum(v/n for v, n in zip((1, 0.25, 0.0625), allocation)) <= 0.1**2
    with pytest.raises(DataValidationError):
        allocate_mlmc((1.0,), (1.0,), 1e-10)
    with pytest.raises(DataValidationError):
        allocate_mlmc((1.0,), (1.0,), 1e-300)


@pytest.mark.parametrize("changes", [{"history_cutoff": 1}, {"horizons": (1.0, 2.0)},
    {"samples": True}, {"steps": 8193}, {"functional": "first-passage"}, {"normal": (0, 0, 0, 0)},
    {"model_package_hash": "0"*64}])
def test_invalid_request_or_model_binding_is_refused(changes):
    package, request = inputs(**changes)
    with pytest.raises(DataValidationError):
        monte_carlo(package, request)


def test_numerical_error_components_remain_separate_with_unknown_model_reference_bounds():
    package, request = inputs()
    result = monte_carlo(package, request)
    target = analytic_estimate(package, request)
    discrete = analytic_estimate(package, request, discrete=True)
    assert result.error_budget.time_discretization.value == pytest.approx(discrete.estimate-target.estimate)
    assert result.error_budget.model.value is None
    assert result.error_budget.reference.value is None
    assert result.error_budget.propagation_approximation.value == 0
    assert result.error_budget.sampling.value > 0
