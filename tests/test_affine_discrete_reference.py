"""Exact rational counterchecks independent of interval matrix powering."""

from dataclasses import replace
from fractions import Fraction as Q
import math

import pytest

from domain.errors import DataValidationError, NumericalError
from domain.frozen_dynamics import _encode
from experiments.pirc27.oracles import affine_package, oracle_suite
from inference.affine_discrete_reference import (bound_affine_discrete,
    bound_affine_time_bias, verify_affine_discrete, verify_affine_time_bias)
from inference.affine_reference import AffineReferenceCertificate, _parse_interval
from tests.test_propagation_methods import inputs
from tests.test_propagation_oracles import integrated_brownian


def contains(interval, value):
    assert interval.lo <= value <= interval.hi


def exact_path_moments(package, request, solver, steps):
    """Propagate formal independent noise coefficients, not a covariance FQF'.

    Each coordinate is [constant, four initial random coordinates, 2*N unit
    variance increments]. Exact rational arithmetic, no production helper.
    """
    p = package.manifest()["parameters"]
    width = 5+2*steps
    h = Q(request.horizons[0])/steps
    y = [[Q(request.initial_mean[i])]+[Q(i == j) for j in range(4)]+[Q(0)]*(2*steps) for i in range(4)]
    z = [row[:] for row in y]

    def drift(state):
        return [[sum(Q(p["A"][i][j])*state[j][k] for j in range(4))
                 + (Q(p["b"][i]) if k == 0 else 0) for k in range(width)] for i in range(4)]

    for step in range(steps):
        noise = [[Q(p["L"][i][k-5-2*step]) if 5+2*step <= k < 7+2*step else Q(0)
                  for k in range(width)] for i in range(4)]
        first = drift(z if solver == "reversible-heun" else y)
        if solver == "euler":
            y = [[y[i][k]+h*first[i][k]+noise[i][k] for k in range(width)] for i in range(4)]
        else:
            predictor = [[(2*y[i][k]-z[i][k] if solver == "reversible-heun" else y[i][k])
                          +h*first[i][k]+noise[i][k] for k in range(width)] for i in range(4)]
            second = drift(predictor)
            y = [[y[i][k]+h*(first[i][k]+second[i][k])/2+noise[i][k]
                  for k in range(width)] for i in range(4)]
            z = predictor
    mean = [row[0] for row in y]
    covariance = [[sum(y[i][k+1]*Q(request.initial_covariance[k][l])*y[j][l+1]
                         for k in range(4) for l in range(4))
                   +h*sum(y[i][k]*y[j][k] for k in range(5, width))
                   for j in range(4)] for i in range(4)]
    return mean, covariance


@pytest.mark.parametrize("solver", ["euler", "heun", "reversible-heun"])
@pytest.mark.parametrize("steps", [1, 3, 5, 8])
@pytest.mark.parametrize("case_index", [2, 3])
def test_matrix_powering_encloses_independent_formal_noise_recurrence(solver, steps, case_index):
    case = oracle_suite()[case_index]
    _, request = inputs(horizons=(1.3,), samples=2)
    request = replace(request, model_package_hash=case.package.package_hash,
        initial_mean=case.initial_mean, initial_covariance=case.initial_covariance)
    certificate = bound_affine_discrete(case.package, request, solver=solver, steps=steps)
    mean, covariance = exact_path_moments(case.package, request, solver, steps)
    doc = certificate.manifest()
    for i in range(4):
        contains(_parse_interval(doc["mean_bounds"][i]), mean[i])
        for j in range(4):
            contains(_parse_interval(doc["covariance_bounds"][i][j]), covariance[i][j])
    contains(certificate.functional_bounds(), mean[0])
    assert certificate.functional_bounds().hi-certificate.functional_bounds().lo < Q(1, 10**35)
    assert doc["physical_dimension"] == 4
    assert doc["auxiliary_dimension"] == (4 if solver == "reversible-heun" else 0)
    assert doc["operations"] <= doc["algorithm"]["operations"]
    assert doc["scientific_qualification"] is False


@pytest.mark.parametrize("solver", ["euler", "heun", "reversible-heun"])
@pytest.mark.parametrize("steps", [1, 3, 8192])
def test_maximum_grid_is_bounded_binary_powering_not_a_path_loop(solver, steps):
    package = integrated_brownian()
    _, request = inputs(samples=2)
    request = replace(request, model_package_hash=package.package_hash)
    certificate = bound_affine_discrete(package, request, solver=solver, steps=steps)
    doc = certificate.manifest()
    assert doc["compositions"] <= 26
    assert doc["operations"] <= 200_000
    # For A^2=0, constant acceleration's endpoint mean bias is explicit.
    b = Q(package.manifest()["parameters"]["b"][2])
    continuous_mean = Q(request.initial_mean[0])+Q(request.initial_mean[2])+b/2
    expected = continuous_mean-b/(2*steps) if solver == "euler" else continuous_mean
    contains(certificate.functional_bounds(), expected)
    assert len(certificate._document) <= 128*1024


@pytest.mark.parametrize("solver", ["euler", "heun", "reversible-heun"])
def test_signed_bias_contains_exact_polynomial_and_binds_both_certificates(solver):
    package = integrated_brownian()
    _, request = inputs(samples=2)
    request = replace(request, model_package_hash=package.package_hash)
    certificate = bound_affine_time_bias(package, request, solver=solver, steps=3)
    expected = -Q(package.manifest()["parameters"]["b"][2])/6 if solver == "euler" else Q(0)
    contains(certificate.functional_bounds(), expected)
    doc = certificate.manifest()
    assert doc["status"] == "BOUNDED" and doc["scientific_qualification"] is False
    assert doc["operations"] == sum(doc["component_operations"].values())
    assert len(doc["continuous_certificate_hash"]) == len(doc["discrete_certificate_hash"]) == 64
    assert verify_affine_time_bias(certificate, package, request, solver=solver, steps=3) == certificate
    if solver == "euler":
        assert certificate.functional_bounds().hi < 0  # Do not clip the signed bias.


@pytest.mark.parametrize("closed", [True, False])
@pytest.mark.parametrize("solver", ["euler", "heun", "reversible-heun"])
def test_true_deterministic_boundary_and_centered_gaussian_halfspace(solver, closed):
    _, request = inputs(samples=2, initial_mean=(0.,)*4, functional="endpoint-halfspace", closed=closed)
    p = integrated_brownian().manifest()["parameters"]
    for L, expected in ((p["L"], Q(1, 2)), ([[0, 0]]*4, Q(int(closed)))):
        package = affine_package("centered", p["A"], [0]*4, L)
        changed = replace(request, model_package_hash=package.package_hash)
        certificate = bound_affine_discrete(package, changed, solver=solver, steps=3)
        bounds = certificate.functional_bounds()
        assert bounds.lo == bounds.hi == expected


@pytest.mark.parametrize("solver,steps", [("other", 3), (None, 3), ("euler", 0),
    ("heun", 8193), ("euler", True), ("euler", 2.0), ("euler", 10**200)])
def test_grid_has_no_defaults_or_automatic_expansion(solver, steps):
    package, request = inputs()
    with pytest.raises(DataValidationError):
        bound_affine_discrete(package, request, solver=solver, steps=steps)


def test_resealed_false_bounds_and_changed_grid_request_source_are_refused():
    package, request = inputs(samples=2)
    certificate = bound_affine_discrete(package, request, solver="heun", steps=3)
    assert verify_affine_discrete(certificate, package, request, solver="heun", steps=3) == certificate
    for key, value in (("functional_bounds", [["0", "1"], ["0", "1"]]),
                       ("code_hash", "0"*64), ("scientific_qualification", True)):
        doc = certificate.manifest()
        doc[key] = value
        with pytest.raises(DataValidationError, match="recomputed"):
            verify_affine_discrete(AffineReferenceCertificate(_encode(doc)), package, request, solver="heun", steps=3)
    for solver, steps, target in (("euler", 3, request), ("heun", 4, request),
                                 ("heun", 3, replace(request, seed=request.seed+1))):
        with pytest.raises(DataValidationError, match="recomputed"):
            verify_affine_discrete(certificate, package, target, solver=solver, steps=steps)


def test_continuous_certificate_cannot_be_used_as_a_discrete_or_time_bias_proof():
    from inference.affine_reference import bound_affine_reference
    package, request = inputs(samples=2)
    certificate = bound_affine_reference(package, request)
    with pytest.raises(DataValidationError, match="recomputed"):
        verify_affine_discrete(certificate, package, request, solver="euler", steps=3)
    with pytest.raises(DataValidationError, match="recomputed"):
        verify_affine_time_bias(certificate, package, request, solver="euler", steps=3)


def test_strict_covariance_and_model_bindings_are_not_relaxed():
    package, request = inputs(samples=2)
    for changes in ({"model_package_hash": "0"*64},
                    {"initial_covariance": ((-1e-100, 0., 0., 0.),)+((0.,)*4,)*3}):
        with pytest.raises(DataValidationError):
            bound_affine_discrete(package, replace(request, **changes), solver="euler", steps=3)


def test_unstable_grid_refuses_fixed_integer_quota_instead_of_repairing_steps():
    p = integrated_brownian().manifest()["parameters"]
    p["A"][2][2] = 1000
    package = affine_package("exploding-grid", p["A"], p["b"], p["L"])
    _, request = inputs(samples=2, horizons=(1000.,))
    request = replace(request, model_package_hash=package.package_hash)
    with pytest.raises(NumericalError, match="quota"):
        bound_affine_discrete(package, request, solver="reversible-heun", steps=8192)


def test_probability_bias_encloses_independent_exact_cdf_series_not_clipped_estimates():
    p = integrated_brownian().manifest()["parameters"]
    package = affine_package("unit-brownian", p["A"], [0]*4,
        [[0, 0], [0, 0], [1, 0], [0, 1]])
    _, request = inputs(samples=2, horizons=(3.,), initial_mean=(0., 0., 1., 0.),
        functional="endpoint-halfspace")
    request = replace(request, model_package_hash=package.package_hash)
    # Continuous x has mean3 and variance9: Phi(1). One Euler step has
    # deterministic x=3, probability1. Independent exact CDF enclosure below.
    def atan(q):
        partial = sum(((-1)**k*q**(2*k+1)/Q(2*k+1) for k in range(40)), Q(0))
        return partial, partial+q**81/81
    left, right = atan(Q(1, 5)), atan(Q(1, 239))
    pi_lo, pi_hi = 16*left[0]-4*right[1], 16*left[1]-4*right[0]
    grid = 1 << 256
    denominator_lo = Q(math.isqrt((2*pi_lo).numerator*grid*grid//(2*pi_lo).denominator), grid)
    denominator_hi = Q(math.isqrt((2*pi_hi).numerator*grid*grid//(2*pi_hi).denominator)+1, grid)
    integral = sum((Q((-1)**k, 2**k*math.factorial(k)*(2*k+1)) for k in range(256)), Q(0))
    remainder = Q(1, 2**256*math.factorial(256)*513)
    exact_bias_lo = Q(1, 2)-(integral+remainder)/denominator_lo
    exact_bias_hi = Q(1, 2)-integral/denominator_hi
    certificate = bound_affine_time_bias(package, request, solver="euler", steps=1)
    bounds = certificate.functional_bounds()
    assert 0 < bounds.lo <= exact_bias_lo <= exact_bias_hi <= bounds.hi < Q(1, 2)
    assert bounds.hi-bounds.lo < Q(1, 10**35)


def test_time_bias_refuses_component_source_change_and_resealed_bias(monkeypatch):
    import inference.affine_discrete_reference as module
    package, request = inputs(samples=2)
    certificate = bound_affine_time_bias(package, request, solver="heun", steps=3)
    doc = certificate.manifest()
    doc["continuous_certificate_hash"] = "0"*64
    with pytest.raises(DataValidationError, match="recomputed"):
        verify_affine_time_bias(AffineReferenceCertificate(_encode(doc)), package, request, solver="heun", steps=3)
    original = module.bound_affine_reference
    def changed_source(*args):
        continuous = original(*args).manifest()
        continuous["code_hash"] = "0"*64
        return AffineReferenceCertificate(_encode(continuous))
    monkeypatch.setattr(module, "bound_affine_reference", changed_source)
    with pytest.raises(DataValidationError, match="different source"):
        bound_affine_time_bias(package, request, solver="heun", steps=3)
