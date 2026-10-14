"""Independent closed-form and quadrature checks for frozen affine references."""

from dataclasses import FrozenInstanceError
import math

import numpy as np
import pytest
import torch

from domain.errors import DataValidationError, NumericalError
from application.propagation_inputs import validate_oracle_input
from domain.frozen_dynamics import FrozenDynamicsPackage, content_hash
from experiments.pirc27.oracles import affine_package, matrix_cardinality, oracle_suite
from inference.affine_oracle import exact_moments, exact_transition, endpoint_halfspace_probability


def integrated_brownian():
    return affine_package("integrated-brownian-v1", [[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
                          [0, 0, 0.3, -0.1], [[0, 0], [0, 0], [0.4, 0], [0, 0.2]])


@pytest.mark.parametrize("horizon", [0.0, 1e-8, 1.0, 10.0, 1000.0])
def test_integrated_brownian_matches_closed_form(horizon):
    result = exact_transition(integrated_brownian(), horizon)
    expected_F = torch.eye(4, dtype=torch.float64)
    expected_F[0, 2] = expected_F[1, 3] = horizon
    expected_Q = torch.zeros((4, 4), dtype=torch.float64)
    for p, v, sigma in ((0, 2, 0.4), (1, 3, 0.2)):
        expected_Q[p, p] = sigma**2 * horizon**3 / 3
        expected_Q[p, v] = expected_Q[v, p] = sigma**2 * horizon**2 / 2
        expected_Q[v, v] = sigma**2 * horizon
    torch.testing.assert_close(result.F, expected_F, rtol=2e-12, atol=1e-12)
    torch.testing.assert_close(result.offset, torch.tensor([0.15*horizon**2, -0.05*horizon**2,
                               0.3*horizon, -0.1*horizon], dtype=torch.float64), rtol=2e-12, atol=1e-12)
    torch.testing.assert_close(result.covariance, expected_Q, rtol=2e-12, atol=1e-12)


@pytest.mark.parametrize("case", oracle_suite(), ids=lambda c: c.case_id)
def test_affine_covariance_matches_independent_gauss_legendre_quadrature(case):
    p = case.package.manifest()["parameters"]
    A, L = torch.tensor(p["A"], dtype=torch.float64), torch.tensor(p["L"], dtype=torch.float64)
    nodes, weights = np.polynomial.legendre.leggauss(48)
    horizon = 3.7
    integral = torch.zeros((4, 4), dtype=torch.float64)
    for node, weight in zip(nodes, weights):
        E = torch.linalg.matrix_exp(A * (horizon * (node + 1) / 2))
        integral += weight * (E @ L @ L.T @ E.T) * horizon / 2
    torch.testing.assert_close(exact_transition(case.package, horizon).covariance, integral, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("case", oracle_suite(), ids=lambda c: c.case_id)
def test_long_horizon_semigroup_and_initial_covariance(case):
    first, second, whole = [exact_transition(case.package, t) for t in (350.0, 650.0, 1000.0)]
    torch.testing.assert_close(whole.F, second.F @ first.F, rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(whole.offset, second.F @ first.offset + second.offset, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(whole.covariance, second.F @ first.covariance @ second.F.T + second.covariance,
                               rtol=1e-9, atol=1e-10)
    mean, covariance = case.moments(1000.0)
    assert torch.isfinite(mean).all() and torch.isfinite(covariance).all()
    assert torch.linalg.eigvalsh(covariance).min() >= -1e-10
    if case.case_id == "stable-v1":
        # Stationary damped oscillator: Var(v)=sigma²/(2 gamma), Var(x)=Var(v)/k.
        torch.testing.assert_close(covariance.diag(), torch.tensor([0.25, 0.25, 0.1, 0.1], dtype=torch.float64),
                                   rtol=1e-10, atol=1e-10)


def test_initial_covariance_is_propagated_and_input_arrays_are_not_modified():
    package = integrated_brownian()
    mean, covariance = torch.ones(4, dtype=torch.float64), torch.eye(4, dtype=torch.float64)
    original_mean, original_cov = mean.clone(), covariance.clone()
    actual_mean, actual_cov = exact_moments(package, mean, covariance, 2.0)
    torch.testing.assert_close(mean, original_mean)
    torch.testing.assert_close(covariance, original_cov)
    assert actual_cov[0, 0] == pytest.approx(5 + 0.16*8/3)
    assert actual_mean[0] == pytest.approx(3.6)


def test_endpoint_probability_tail_and_degenerate_boundary():
    mean, cov = torch.zeros(4, dtype=torch.float64), torch.eye(4, dtype=torch.float64)
    assert endpoint_halfspace_probability(mean, cov, [1, 0, 0, 0], 0) == 0.5
    assert endpoint_halfspace_probability(mean, cov, [1, 0, 0, 0], 8) == pytest.approx(6.220960574271819e-16, rel=1e-12, abs=0)
    assert endpoint_halfspace_probability(mean, cov, [2, 0, 0, 0], 16) == endpoint_halfspace_probability(mean, cov, [1, 0, 0, 0], 8)
    cov.zero_()
    assert endpoint_halfspace_probability(mean, cov, [1, 0, 0, 0], 0, closed=True) == 1
    assert endpoint_halfspace_probability(mean, cov, [1, 0, 0, 0], 0, closed=False) == 0


@pytest.mark.parametrize("field,value", [("units", ["km", "m", "m/s", "m/s"]),
    ("family", "real"), ("frozen", False), ("noise_convention", "stratonovich"),
    ("qualification", {"scope": "formal", "scientific_qualification": True})])
def test_package_rejects_incompatible_manifest_even_with_new_content_hash(field, value):
    document = integrated_brownian().manifest()
    document[field] = value
    with pytest.raises(DataValidationError):
        FrozenDynamicsPackage.from_manifest(document, expected_hash=content_hash(document))


def test_package_content_is_detached_immutable_and_detects_tampering():
    package = integrated_brownian()
    document = package.manifest()
    document["parameters"]["A"][2][2] = 1.0
    with pytest.raises(DataValidationError):
        FrozenDynamicsPackage.from_manifest(document, expected_hash=package.package_hash)
    assert package.manifest()["parameters"]["A"][2][2] == 0
    with pytest.raises(FrozenInstanceError):
        package._document = b"{}"
    document["parameter_hash"] = content_hash(document["parameters"])
    assert FrozenDynamicsPackage.from_manifest(document, expected_hash=content_hash(document)).package_hash != package.package_hash


@pytest.mark.parametrize("dt", [-1, math.inf, math.nan, True])
def test_invalid_horizon_refused(dt):
    with pytest.raises(DataValidationError):
        exact_transition(integrated_brownian(), dt)


def test_invalid_initial_covariance_refused_without_projection():
    covariance = torch.eye(4, dtype=torch.float64)
    covariance[0, 0] = -0.1
    with pytest.raises(DataValidationError):
        exact_moments(integrated_brownian(), [0]*4, covariance, 1)
    covariance = torch.eye(4, dtype=torch.float64)
    covariance[0, 1] = 0.1
    with pytest.raises(DataValidationError):
        exact_moments(integrated_brownian(), [0]*4, covariance, 1)


def test_actual_overflow_is_numerical_failure():
    document = integrated_brownian().manifest()
    document["parameters"]["A"][2][2] = 10
    document["parameter_hash"] = content_hash(document["parameters"])
    package = FrozenDynamicsPackage.from_manifest(document, expected_hash=content_hash(document))
    with pytest.raises(NumericalError):
        exact_transition(package, 1000)


def test_matrix_limit_is_checked_without_materializing_cells():
    assert matrix_cardinality([4, 5, 5, 4, 5]) == 2000
    assert matrix_cardinality([100, 100]) == 10000
    for axes in ([100, 101], [True], [0], [10**100], []):
        with pytest.raises(DataValidationError):
            matrix_cardinality(axes)


def test_live_oracle_binding_rejects_stale_code_or_different_package():
    package = integrated_brownian()
    assert validate_oracle_input(package, expected_package_hash=package.package_hash) is package
    with pytest.raises(DataValidationError):
        validate_oracle_input(package, expected_package_hash="0"*64)
    document = package.manifest()
    document["code_hash"] = "0"*64
    stale = FrozenDynamicsPackage.from_manifest(document, expected_hash=content_hash(document))
    with pytest.raises(DataValidationError):
        validate_oracle_input(stale, expected_package_hash=stale.package_hash)


@pytest.mark.parametrize("edit", ["position-drift", "position-noise", "nonfinite", "oversized-int"])
def test_affine_kinematics_and_nonfinite_coefficients_refused(edit):
    document = integrated_brownian().manifest()
    if edit == "position-drift":
        document["parameters"]["b"][0] = 1
    elif edit == "position-noise":
        document["parameters"]["L"][0][0] = 1
    elif edit == "nonfinite":
        document["parameters"]["A"][2][0] = math.inf
    else:
        document["parameters"]["A"][2][0] = 10**400
    with pytest.raises(DataValidationError):
        FrozenDynamicsPackage.from_manifest(document, expected_hash="0"*64)


def test_oracle_calls_preserve_global_random_stream():
    state = torch.random.get_rng_state().clone()
    for case in oracle_suite():
        case.moments(10.0)
    assert torch.equal(state, torch.random.get_rng_state())
