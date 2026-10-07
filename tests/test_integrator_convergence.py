"""Bounded manufactured-law checks, not a study or a universal order theorem.

References use scalar closed forms, not the production continuous/discrete
moment recurrences. Strong checks call the actual endpoint sampler and compare
its paired paths with the exact integrated-Brownian endpoint conditional on the
same increments, plus an independent Gaussian Brownian-bridge residual. Different
resolution ensembles measure L2 errors; they are not independent study blocks.
"""

from dataclasses import replace
import hashlib
import json
import math

import numpy as np
import pytest

from domain.propagation import PropagationRequest
from experiments.pirc27.oracles import affine_package, oracle_suite
from inference.propagation_methods import analytic_estimate, discrete_moments, endpoint_chunks


SOLVERS = ("euler", "additive-heun", "reversible-heun")
GRIDS = (8, 16, 32)
SIGMAS = np.array([0.4, 0.2])


def integrated_inputs(horizon, **changes):
    package = affine_package("integrator-convergence-brownian-v1",
        [[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        [0, 0, 0, 0], [[0, 0], [0, 0], [0.4, 0], [0, 0.2]])
    request = PropagationRequest("integrator-convergence", package.package_hash,
        (0., 0., 0., 0.), ((0.,)*4,)*4, 0., 0., (horizon,),
        "endpoint-halfspace", 219, "manufactured-paired-root-v1", "original-method",
        steps=8, samples=512, chunk_size=128)
    return package, replace(request, **changes)


def independent_increments(request, sample, level, steps):
    # Reproduce only the documented entropy contract, not _rng/_inputs or a
    # production recurrence. Even zero initial covariance consumes four normals.
    encoded = json.dumps({"coupling_id": request.coupling_id,
        "scheme": "per-sample-seedsequence-v1"}, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    root = hashlib.sha256(encoded).hexdigest()
    words = [int(root[start:start+8], 16) for start in range(0, 32, 8)]
    stream = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [request.seed, sample, level, 0, *words])))
    stream.standard_normal(4)
    dt = request.horizons[0]/steps
    return stream.standard_normal((steps, 2))*math.sqrt(dt)


def conditional_exact(request, increments, sample):
    steps = len(increments)
    horizon = request.horizons[0]
    dt = horizon/steps
    midpoint_remaining = horizon-(np.arange(steps)+0.5)*dt
    # On each interval, integral W(s) ds conditional on dW has mean h*dW/2
    # and independent variance h^3/12. Sum over intervals: T*h^2/12.
    bridge_stream = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [request.seed, sample, 0x42524944, steps])))
    residual = math.sqrt(horizon*dt**2/12)*bridge_stream.standard_normal(2)
    position = SIGMAS*(midpoint_remaining@increments+residual)
    velocity = SIGMAS*increments.sum(axis=0)
    return np.concatenate((position, velocity))


def discrete_from_increments(increments, horizon, solver):
    steps = len(increments)
    dt = horizon/steps
    offset = 1.0 if solver == "euler" else 0.5
    # For this nilpotent drift, both Heun variants have the same physical
    # trapezoidal endpoint; this identity is not assumed for other models.
    position = SIGMAS*((horizon-(np.arange(steps)+offset)*dt)@increments)
    velocity = SIGMAS*increments.sum(axis=0)
    return np.concatenate((position, velocity))


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("horizon", [1., 10.])
def test_actual_noisy_paired_paths_have_first_order_strong_l2_error(solver, horizon):
    package, template = integrated_inputs(horizon)
    fine_rms = []
    for steps in GRIDS:
        # The sampler's level=1 generates the fine grid and the actual adjacent
        # increment sums for its coarse grid, each with its own solver state.
        request = replace(template, steps=steps//2)
        rows = list(endpoint_chunks(package, request, solver=solver, level=1, paired=True))
        fine = np.concatenate([row[2] for row in rows])
        coarse = np.concatenate([row[3] for row in rows])
        exact = []
        for sample in range(request.samples):
            increments = independent_increments(request, sample, 1, steps)
            np.testing.assert_allclose(fine[sample],
                discrete_from_increments(increments, horizon, solver), rtol=2e-14, atol=2e-13)
            np.testing.assert_allclose(coarse[sample], discrete_from_increments(
                increments.reshape(steps//2, 2, 2).sum(axis=1), horizon, solver),
                rtol=2e-14, atol=2e-13)
            exact.append(conditional_exact(request, increments, sample))
        exact = np.array(exact)
        # Velocity has no discretization error on this law. Do not mistake it
        # for zero strong error of the four-state process: position is noisy.
        np.testing.assert_allclose(fine[:, 2:], exact[:, 2:], rtol=2e-14, atol=2e-13)
        np.testing.assert_allclose(coarse[:, 2:], exact[:, 2:], rtol=2e-14, atol=2e-13)
        rms = []
        for paths, dt in ((fine, horizon/steps), (coarse, 2*horizon/steps)):
            error = paths[:, :2]-exact[:, :2]
            # Exact E|X_h-X|^2: sigma^2*T*h^2/3 (Euler) or /12 (Heun).
            variance = SIGMAS**2*horizon*dt**2/(3 if solver == "euler" else 12)
            observed_mse = np.mean(error**2, axis=0)
            assert np.all(observed_mse > 0.65*variance)
            assert np.all(observed_mse < 1.35*variance)
            assert np.all(abs(error.mean(axis=0)) < 5*np.sqrt(variance/request.samples))
            rms.append(np.sqrt(observed_mse))
        assert np.all(rms[1]/rms[0] > 1.6)
        assert np.all(rms[1]/rms[0] < 2.4)
        fine_rms.append(rms[0])
    ratios = np.array(fine_rms[:-1])/np.array(fine_rms[1:])
    assert np.all(ratios > 1.6) and np.all(ratios < 2.4)
    assert package.manifest()["qualification"]["scientific_qualification"] is False


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("horizon", [1., 10.])
def test_halfspace_functional_weak_rate_matches_independent_closed_form(solver, horizon):
    continuous_variance = SIGMAS[0]**2*horizon**3/3
    package, template = integrated_inputs(horizon, threshold=math.sqrt(continuous_variance))
    target = 0.5*math.erfc(1/math.sqrt(2))
    errors = []
    for steps in GRIDS:
        request = replace(template, steps=steps)
        dt = horizon/steps
        mean, covariance = discrete_moments(package, request, solver=solver)
        expected = np.zeros((4, 4))
        for p, v, sigma in ((0, 2, SIGMAS[0]), (1, 3, SIGMAS[1])):
            expected[p, p] = sigma**2*(horizon**3/3-horizon**2*dt/2+horizon*dt**2/6
                if solver == "euler" else horizon**3/3-horizon*dt**2/12)
            expected[p, v] = expected[v, p] = sigma**2*(horizon**2/2-
                (horizon*dt/2 if solver == "euler" else 0))
            expected[v, v] = sigma**2*horizon
        np.testing.assert_allclose(mean, np.zeros(4), rtol=0, atol=1e-13)
        np.testing.assert_allclose(covariance, expected, rtol=2e-13, atol=1e-13)
        probability = 0.5*math.erfc(request.threshold/math.sqrt(2*expected[0, 0]))
        actual = analytic_estimate(package, request, discrete=True, solver=solver)
        assert actual.estimate == pytest.approx(probability, rel=0, abs=2e-14)
        errors.append(abs(actual.estimate-target))
    ratios = np.array(errors[:-1])/np.array(errors[1:])
    assert np.all(ratios > (1.8 if solver == "euler" else 3.8))
    assert np.all(ratios < (2.3 if solver == "euler" else 4.2))


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("horizon", [1., 2.])
def test_damped_oscillator_expectation_rate_uses_scalar_closed_form(solver, horizon):
    case = oracle_suite()[0]
    # Solve x''+0.8*x'+0.4*x=0.05 with x(0)=0, x'(0)=1.
    gamma, stiffness, forcing = 0.4, 0.4, 0.05
    frequency = math.sqrt(stiffness-gamma**2)
    equilibrium = forcing/stiffness
    target = equilibrium+math.exp(-gamma*horizon)*(-equilibrium*math.cos(frequency*horizon)+
        (1-gamma*equilibrium)*math.sin(frequency*horizon)/frequency)
    errors = []
    for steps in GRIDS:
        request = PropagationRequest("oscillator-convergence", case.package.package_hash,
            case.initial_mean, case.initial_covariance, 0., 0., (horizon,), "endpoint-x",
            219, "manufactured-root", "original-method", steps=steps, samples=2)
        actual = analytic_estimate(case.package, request, discrete=True, solver=solver)
        errors.append(abs(actual.estimate-target))
    ratios = np.array(errors[:-1])/np.array(errors[1:])
    assert np.all(ratios > (1.6 if solver == "euler" else 3.2))
    assert np.all(ratios < (2.5 if solver == "euler" else 4.8))
