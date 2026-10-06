"""Synthetic method-capability qualification, not a research comparison."""

from dataclasses import replace
import json
import math

import numpy as np
import pytest

from application.propagation_inputs import validate_nonlinear_input
from domain.errors import DataValidationError, NumericalError
from domain.frozen_dynamics import content_hash
from domain.nonlinear_dynamics import FrozenNonlinearPackage
from experiments.pirc27.nonlinear import nonlinear_package
from experiments.pirc27.oracles import affine_package
from inference.affine_oracle import exact_transition
from inference.nonlinear_propagation import cubature_estimate, cubature_moments, nonlinear_drift, _strict_root
from inference.propagation_methods import (analytic_estimate, discrete_moments, endpoint_chunks,
    importance_sampling, mlmc_estimate, monte_carlo, _rng)
from tests.test_propagation_methods import inputs
from tests.test_propagation_recovery import Stopped


def stress_inputs(**changes):
    package = nonlinear_package()
    _, request = inputs()
    request = replace(request, model_package_hash=package.package_hash, initial_mean=(0., 0., 0.2, 0.),
        initial_covariance=tuple(tuple(0.1 if i == j else 0. for j in range(4)) for i in range(4)),
        samples=37, steps=8, chunk_size=4, **changes)
    return package, request


def test_stress_package_is_detached_content_bound_and_cannot_claim_affine_oracle():
    package, request = stress_inputs()
    manifest = package.manifest()
    manifest["parameters"]["amplitude"][0] = 100
    assert package.manifest()["parameters"]["amplitude"][0] == 3
    with pytest.raises(DataValidationError):
        FrozenNonlinearPackage.from_manifest(manifest, expected_hash=package.package_hash)
    validate_nonlinear_input(package, expected_package_hash=package.package_hash)
    with pytest.raises(DataValidationError):
        exact_transition(package, 1.)
    with pytest.raises(DataValidationError):
        analytic_estimate(package, request)
    with pytest.raises(DataValidationError):
        discrete_moments(package, request)


@pytest.mark.parametrize("field,value", [("noise_convention", "stratonovich"), ("capabilities", ["affine-exact"]),
    ("qualification", {"scope": "stress-fixture", "scientific_qualification": True}), ("units", ["km", "m", "m/s", "m/s"])])
def test_invalid_stress_claims_are_refused_even_with_a_matching_content_hash(field, value):
    document = nonlinear_package().manifest()
    document[field] = value
    with pytest.raises(DataValidationError):
        FrozenNonlinearPackage.from_manifest(document, expected_hash=content_hash(document))


def test_stress_parameter_and_current_code_bindings_are_not_metadata_only():
    document = nonlinear_package().manifest()
    document["parameters"]["length_scale"][0] = 0
    document["parameter_hash"] = content_hash(document["parameters"])
    with pytest.raises(DataValidationError):
        FrozenNonlinearPackage.from_manifest(document, expected_hash=content_hash(document))
    document = nonlinear_package().manifest()
    document["code_hash"] = "0"*64
    package = FrozenNonlinearPackage.from_manifest(document, expected_hash=content_hash(document))
    with pytest.raises(DataValidationError):
        validate_nonlinear_input(package, expected_package_hash=package.package_hash)


@pytest.mark.parametrize("solver", ["euler", "additive-heun"])
def test_nonlinear_chunk_and_sample_resume_replay_identical_endpoints(solver):
    package, request = stress_inputs()
    def paths(request, **kwargs):
        return np.concatenate([row[2] for row in endpoint_chunks(package, request, solver=solver, **kwargs)])
    expected = paths(request)
    np.testing.assert_array_equal(paths(replace(request, chunk_size=1)), expected)
    resumed = np.concatenate([paths(request, sample_count=11), paths(request, start_sample=11, sample_count=26)])
    np.testing.assert_array_equal(resumed, expected)


def test_nonlinear_coarse_path_uses_the_actual_adjacent_fine_increment_sums():
    package, request = stress_inputs()
    request = replace(request, samples=3, steps=2)
    fine, coarse = next(endpoint_chunks(package, request, level=1, paired=True))[2:4]
    parameters = package.manifest()["parameters"]
    A, b, L = (np.array(parameters[key]) for key in ("A", "b", "L"))
    root = np.eye(4)*np.sqrt(0.1)
    for sample in range(3):
        rng = _rng(request, sample, 1, 0)
        initial = np.array(request.initial_mean)+root@rng.standard_normal(4)
        increments = rng.standard_normal((4, 2))*0.5
        def evolve(noises, dt):
            state = initial[None, :]
            for noise in noises:
                state = state + dt*nonlinear_drift(A, b, parameters["amplitude"], parameters["length_scale"], state)+noise@L.T
            return state[0]
        np.testing.assert_allclose(fine[sample], evolve(increments, 0.25), rtol=0, atol=1e-15)
        np.testing.assert_allclose(coarse[sample], evolve(increments.reshape(2, 2, 2).sum(axis=1), 0.5), rtol=0, atol=1e-15)


@pytest.mark.parametrize("method", ["mc", "heun", "mlmc", "importance"])
def test_actual_nonlinear_statistics_continue_without_false_reference_bounds(method):
    package, request = stress_inputs()
    request = replace(request, functional="endpoint-halfspace")
    def estimate(**kwargs):
        if method == "mlmc":
            return mlmc_estimate(package, request, level_samples=(17, 11, 9), **kwargs)
        if method == "importance":
            return importance_sampling(package, request, proposal=(1., 0.), **kwargs)
        return monte_carlo(package, request, solver="additive-heun" if method == "heun" else "euler", **kwargs)
    saved = []
    def stop(state, total):
        saved.append(state)
        raise Stopped
    with pytest.raises(Stopped):
        estimate(checkpoint=stop)
    actual = estimate(resume_state=json.loads(json.dumps(saved[0])))
    assert actual.manifest() == estimate().manifest()
    assert actual.error_budget.reference.value is None and actual.error_budget.time_discretization.value is None
    assert actual.error_budget.propagation_approximation.value == 0


def test_cubature_reproduces_affine_discrete_moments_in_the_zero_nonlinearity_limit():
    package = nonlinear_package(amplitude=(0., 0.))
    _, request = stress_inputs()
    request = replace(request, model_package_hash=package.package_hash)
    parameters = package.manifest()["parameters"]
    affine = affine_package("same-linear-limit", parameters["A"], parameters["b"], parameters["L"])
    target = discrete_moments(affine, replace(request, model_package_hash=affine.package_hash))
    actual = cubature_moments(package, request)
    np.testing.assert_allclose(actual[0], target[0], rtol=0, atol=1e-14)
    np.testing.assert_allclose(actual[1], target[1], rtol=0, atol=1e-14)


def test_cubature_retains_unknown_closure_error_and_fixed_point_complexity():
    package, request = stress_inputs()
    result = cubature_estimate(package, request)
    assert result.error_budget.propagation_approximation.value is None
    assert result.error_budget.time_discretization.value is None
    assert result.status == "APPROXIMATION_ONLY" and result.sample_count == 0
    assert dict(result.diagnostics)["point_count"] == 8 and dict(result.diagnostics)["point_weight"] == 0.125
    with pytest.raises((DataValidationError, NumericalError)):
        _strict_root(np.diag([1, 1, 1, -1e-15]))


def test_nonlinear_zero_noise_integrators_converge_against_an_independent_scalar_rk4_reference():
    package = nonlinear_package(diffusion=0.)
    _, request = stress_inputs()
    request = replace(request, model_package_hash=package.package_hash, samples=2,
        initial_mean=(0.5, 0., 0.2, 0.), initial_covariance=tuple((0.,)*4 for _ in range(4)))
    def drift(x, v):
        return v, -0.5*x-v+3*math.tanh(x)
    x, v, dt = 0.5, 0.2, 1/4096
    for _ in range(4096):
        k1 = drift(x, v)
        k2 = drift(x+dt*k1[0]/2, v+dt*k1[1]/2)
        k3 = drift(x+dt*k2[0]/2, v+dt*k2[1]/2)
        k4 = drift(x+dt*k3[0], v+dt*k3[1])
        x += dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6
        v += dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6
    for solver, minimum_ratio in (("euler", 1.8), ("additive-heun", 3.5)):
        errors = [abs(next(endpoint_chunks(package, replace(request, steps=steps), solver=solver))[2][0, 0]-x)
                  for steps in (8, 16, 32)]
        assert errors[0]/errors[1] > minimum_ratio and errors[1]/errors[2] > minimum_ratio


def test_two_well_toy_distribution_exposes_moment_matched_gaussian_central_mass_distortion():
    # Prescribed tiny regression counterexample, not a preregistered research
    # applicability result. In particular no fitted Gaussian is scientific truth.
    package = nonlinear_package(diffusion=0.)
    _, request = stress_inputs()
    request = replace(request, model_package_hash=package.package_hash, samples=128, steps=128, horizons=(8.,),
        initial_mean=(0.,)*4, initial_covariance=((0.25, 0., 0., 0.), (0.,)*4, (0.,)*4, (0.,)*4))
    x = np.concatenate([chunk[2][:, 0] for chunk in endpoint_chunks(package, request)])
    assert np.count_nonzero(x < -0.5) > 40 and np.count_nonzero(x > 0.5) > 40
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    central_gaussian = 0.5*(math.erf((0.5-mean)/(math.sqrt(2)*sd))-math.erf((-0.5-mean)/(math.sqrt(2)*sd)))
    central_observed = float(np.mean(abs(x) < 0.5))
    assert central_gaussian-central_observed > 0.03
