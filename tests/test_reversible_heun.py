"""Additive recurrence checks, not a preregistered scientific comparison."""

from dataclasses import replace
import json

import numpy as np
import pytest

from domain.errors import DataValidationError, NumericalError
from inference.propagation_methods import (discrete_moments, endpoint_chunks,
    monte_carlo, _inputs, _rng)
from inference.reversible_heun import affine_extended_scheme, reversible_step
from tests.test_nonlinear_propagation import stress_inputs
from tests.test_propagation_methods import inputs
from tests.test_propagation_recovery import Stopped


@pytest.mark.parametrize("nonlinear", [False, True])
def test_extended_pair_reverses_with_same_negated_noise_without_iteration(nonlinear):
    state = np.array([[0.2, -0.4, 0.8, -0.3], [0.5, 0.1, -0.6, 0.2]])
    auxiliary = state + 0.07
    noise = np.array([[0., 0., 0.1, -0.04], [0., 0., -0.09, 0.07]])
    def drift(x):
        return -0.3*x + (np.tanh(x) if nonlinear else 0.2)
    forward, aux, cached = reversible_step(state, auxiliary, drift, noise, 0.125)
    backward, original_aux, _ = reversible_step(forward, aux, drift, -noise, -0.125, cached)
    np.testing.assert_allclose(backward, state, rtol=0, atol=3e-16)
    np.testing.assert_allclose(original_aux, auxiliary, rtol=0, atol=3e-16)
    assert not np.array_equal(forward, aux)


def test_cached_auxiliary_drift_uses_one_new_evaluation_per_step():
    calls = []
    def drift(x):
        calls.append(x.copy())
        return -x
    state = np.ones((1, 4))
    aux, cached = state.copy(), drift(state)
    for _ in range(9):
        state, aux, cached = reversible_step(state, aux, drift, np.zeros_like(state), 0.05, cached)
    assert len(calls) == 10


def test_affine_augmented_recurrence_matches_direct_extended_step():
    package, request = inputs(samples=2)
    A, b, L, _, _ = _inputs(package, request)
    y = np.array([0.3, -0.2, 0.7, 0.1])
    z = y+np.array([0.1, 0.03, -0.01, 0.2])
    dw, dt = np.array([0.07, -0.02]), 0.125
    F, offset, noise = affine_extended_scheme(A, b, L, dt)
    actual = reversible_step(y, z, lambda x: A@x+b, L@dw, dt)
    np.testing.assert_allclose(F@np.concatenate((y, z))+offset+noise@dw,
                               np.concatenate(actual[:2]), rtol=0, atol=3e-16)


@pytest.mark.parametrize("nonlinear", [False, True])
def test_chunk_and_uncommitted_sample_ranges_preserve_extended_paths(nonlinear):
    package, request = stress_inputs() if nonlinear else inputs(samples=37, steps=8, chunk_size=4)
    def paths(req, **kwargs):
        return np.concatenate([row[2] for row in endpoint_chunks(package, req, solver="reversible-heun", **kwargs)])
    expected = paths(request)
    np.testing.assert_array_equal(paths(replace(request, chunk_size=1)), expected)
    np.testing.assert_array_equal(np.concatenate((paths(request, sample_count=11),
        paths(request, start_sample=11, sample_count=26))), expected)


@pytest.mark.parametrize("nonlinear", [False, True])
def test_adjacent_increment_coupling_preserves_each_own_auxiliary_path(nonlinear):
    package, request = stress_inputs() if nonlinear else inputs()
    request = replace(request, samples=3, steps=2)
    fine, coarse = next(endpoint_chunks(package, request, solver="reversible-heun", level=1, paired=True))[2:4]
    A, b, L, mean, root = _inputs(package, request)
    params = package.manifest()["parameters"]
    def drift(x):
        if nonlinear:
            from inference.nonlinear_propagation import nonlinear_drift
            return nonlinear_drift(A, b, params["amplitude"], params["length_scale"], x[None, :])[0]
        return A@x+b
    for sample in range(3):
        rng = _rng(request, sample, 1, 0)
        initial = mean+root@rng.standard_normal(4)
        increments = rng.standard_normal((4, 2))*0.5
        def evolve(noises, dt):
            y, z = initial.copy(), initial.copy()
            for dw in noises:
                # Independent direct scalar recurrence, without solver helper.
                first = drift(z)
                z = 2*y-z+dt*first+L@dw
                y = y+0.5*dt*(first+drift(z))+L@dw
            return y
        np.testing.assert_allclose(fine[sample], evolve(increments, 0.25), rtol=0, atol=2e-15)
        np.testing.assert_allclose(coarse[sample], evolve(increments.reshape(2, 2, 2).sum(axis=1), 0.5), rtol=0, atol=2e-15)


def test_physical_gaussian_marginal_matches_sampled_paths_with_correlated_initial_pair():
    from experiments.pirc27.oracles import oracle_suite
    case = oracle_suite()[3]
    _, request = inputs(samples=6000, steps=8)
    request = replace(request, model_package_hash=case.package.package_hash,
        initial_mean=case.initial_mean, initial_covariance=case.initial_covariance)
    samples = np.concatenate([row[2] for row in endpoint_chunks(case.package, request, solver="reversible-heun")])
    mean, covariance = discrete_moments(case.package, request, solver="reversible-heun")
    assert mean.shape == (4,) and covariance.shape == (4, 4)
    assert np.all(abs(samples.mean(axis=0)-mean) <= 5*np.sqrt(np.diag(covariance)/len(samples)))
    se = np.sqrt((np.outer(np.diag(covariance), np.diag(covariance))+covariance**2)/(len(samples)-1))
    assert np.all(abs(np.cov(samples.T)-covariance) <= 5*se)


@pytest.mark.parametrize("nonlinear", [False, True])
@pytest.mark.parametrize("boundary", [1, 4, 10])
def test_completed_chunks_restore_actual_statistics_not_partial_auxiliary_state(nonlinear, boundary):
    package, request = stress_inputs() if nonlinear else inputs(samples=37, steps=8, chunk_size=4)
    saved = []
    def stop(state, total):
        saved.append(state)
        if len(saved) == boundary:
            raise Stopped
    with pytest.raises(Stopped):
        monte_carlo(package, request, solver="reversible-heun", checkpoint=stop)
    checkpoint = json.loads(json.dumps(saved[-1], allow_nan=False))
    actual = monte_carlo(package, request, solver="reversible-heun", resume_state=checkpoint)
    assert actual.manifest() == monte_carlo(package, request, solver="reversible-heun").manifest()
    assert "auxiliary" not in checkpoint["method_state"]
    with pytest.raises(DataValidationError):
        monte_carlo(package, request, solver="additive-heun", resume_state=checkpoint)


@pytest.mark.parametrize("nonlinear", [False, True])
def test_unknown_reference_and_model_errors_and_method_limitations_remain_visible(nonlinear):
    package, request = stress_inputs() if nonlinear else inputs(samples=32)
    result = monte_carlo(package, request, solver="reversible-heun")
    diagnostic = dict(result.diagnostics)
    assert result.estimator_id == "reversible-heun-path-mc-v1"
    assert diagnostic["drift_evaluations_per_path"] == request.steps+1
    assert "not-A-stable" in diagnostic["stability"]
    assert result.error_budget.reference.value is None and result.error_budget.model.value is None
    if nonlinear:
        assert result.error_budget.time_discretization.value is None
    else:
        assert diagnostic["extended_spectral_radius"] > 1
    json.dumps(result.manifest(), allow_nan=False)


def test_unstable_extended_moments_fail_without_psd_projection_or_automatic_step_increase():
    package, request = inputs(horizons=(1000.,), steps=512, samples=2)
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(NumericalError, match="nonfinite"):
            discrete_moments(package, request, solver="reversible-heun")


@pytest.mark.parametrize("nonlinear", [False, True])
def test_unknown_solver_refuses_instead_of_falling_back_to_heun(nonlinear):
    package, request = stress_inputs() if nonlinear else inputs()
    with pytest.raises(DataValidationError):
        next(endpoint_chunks(package, request, solver="pretend-reversible"))


@pytest.mark.parametrize("nonlinear", [False, True])
@pytest.mark.parametrize("recovery", [False, True])
def test_adapter_reserves_bounded_auxiliary_workspace_and_keeps_four_physical_states(nonlinear, recovery):
    from application.research_registry import plan_resources
    from experiments.pirc27.plugin import execution_config, execution_inputs, propagation_plugin
    package, request = stress_inputs() if nonlinear else inputs(samples=37, steps=8, chunk_size=4)
    plugin = propagation_plugin(synthetic=nonlinear, recovery=recovery)
    config = execution_config(request, "reversible-heun", synthetic=nonlinear, recovery=recovery)
    plan = plan_resources(plugin.registry_entry, config, execution_inputs(request), matrix_cells=1)
    assert plugin.state_order == ("x", "y", "vx", "vy")
    assert plan["counts"]["state_dim"] == 4
    expected_rows = max(8, request.chunk_size) if nonlinear else request.chunk_size
    tensors = {item["name"]: item["shape"] for item in plan["tensors"]}
    for name in ("fine-auxiliary", "coarse-auxiliary", "fine-cached-drift", "coarse-cached-drift"):
        assert tensors[name] == [expected_rows, 4]
    assert tensors["augmented-gaussian-workspace"] == [4]*5
    assert config["samples"]*config["steps"] <= 1_000_000
    assert plugin.registry_entry.version == "1.2.0"
