"""Exact rational counterchecks for reference enclosure arithmetic and scope."""

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext
from fractions import Fraction
import json
import math

import pytest

from domain.errors import DataValidationError, NumericalError
from domain.frozen_dynamics import _encode, content_hash
from experiments.pirc27.oracles import affine_package, oracle_suite
from inference.affine_reference import (AffineReferenceCertificate, Arithmetic, GRID,
    bound_affine_reference, reference_error_bound, verify_affine_reference,
    _cdf, _negative_exponential, _parse_interval, _pi, _short_exponential)
from inference.propagation_methods import analytic_estimate
from tests.test_propagation_methods import inputs
from tests.test_propagation_oracles import integrated_brownian


def contained(interval, value):
    assert interval.lo <= value <= interval.hi


@pytest.mark.parametrize("left,right", [(Fraction(1, 3), Fraction(2, 7)),
    (Fraction(-9, 5), Fraction(1, 11)), (Fraction(1, 10**70), Fraction(5, 9)),
    (Fraction(10**40), Fraction(-11, 13))])
def test_each_arithmetic_operation_encloses_exact_rational_value(left, right):
    a = Arithmetic()
    x, y = a.box(left), a.box(right)
    contained(x, left)
    contained(a.add(x, y), left+right)
    contained(a.sub(x, y), left-right)
    contained(a.mul(x, y), left*right)
    contained(a.div(x, y), left/right)


def test_interval_products_cover_all_signed_corners_and_zero_division_refuses():
    a = Arithmetic()
    x, y = a.box(-2, 3), a.box(-5, 7)
    product = a.mul(x, y)
    for left in (-2, 0, 3):
        for right in (-5, 0, 7):
            contained(product, left*right)
    with pytest.raises(NumericalError, match="contains zero"):
        a.div(a.box(1), x)


@pytest.mark.parametrize("value", [Fraction(0), Fraction(4), Fraction(2),
    Fraction(1, 3), Fraction(1, 10**100), Fraction(10**100)])
def test_integer_sqrt_bounds_cover_exact_root_without_floating_rounding(value):
    a = Arithmetic()
    result = a.sqrt(a.box(value))
    assert result.lo >= 0 and result.lo**2 <= value <= result.hi**2


@pytest.mark.parametrize("value", [Fraction(0), Fraction(1, 4), Fraction(-1, 4)])
def test_exponential_encloses_high_precision_independent_scalar_exp(value):
    # Decimal is a countercheck, not the enclosure's proof or implementation.
    a = Arithmetic()
    result = _short_exponential(a, [[a.box(value)]])[0][0]
    with localcontext() as ctx:
        ctx.prec = 80
        truth = Fraction((Decimal(value.numerator)/Decimal(value.denominator)).exp())
    contained(result, truth)
    with pytest.raises(NumericalError, match="scaled"):
        _short_exponential(a, [[a.box(1)]])


def test_pi_comes_from_rational_arctan_bounds_not_machine_pi():
    result = _pi(Arithmetic())
    # Known decimal prefix brackets pi; independent of this implementation.
    # Use a decimal bracket wider than the fixed dyadic rounding grid. A
    # 49-digit bracket is narrower than the allowed outward rounding itself.
    prefix = Fraction("3.141592653589793238462643383279502884197169")
    assert prefix <= result.lo <= result.hi <= prefix+Fraction(1, 10**42)
    assert result.hi-result.lo < Fraction(1, 10**45)
    assert not (result.lo <= Fraction(math.pi) <= result.hi)


@pytest.mark.parametrize("value", [-1, -32, -512])
def test_negative_exponential_tail_bounds_do_not_underflow_to_false_zero(value):
    result = _negative_exponential(Arithmetic(), Fraction(value))
    with localcontext() as ctx:
        ctx.prec = 100
        truth = Fraction(Decimal(value).exp())
    contained(result, truth)
    assert result.hi > 0


@pytest.mark.parametrize("x", [0, 1, 4, 8, 9, 16, 40])
def test_cdf_symmetry_monotonicity_and_explicit_positive_rare_upper_bounds(x):
    a = Arithmetic()
    denominator = a.sqrt(a.mul(a.box(2), _pi(a)))
    positive = _cdf(a, Fraction(x), denominator)
    negative = _cdf(a, Fraction(-x), denominator)
    assert positive.lo == 1-negative.hi and positive.hi == 1-negative.lo
    assert 0 <= negative.lo <= negative.hi <= Fraction(1, 2)
    assert negative.hi > 0
    if x == 0:
        assert positive.lo == positive.hi == Fraction(1, 2)
    if x <= 8:
        # Ordinary erfc is only an independent floating countercheck; do not
        # pretend its rounded value is the mathematical truth inside our box.
        assert float((negative.lo+negative.hi)/2) == pytest.approx(0.5*math.erfc(x/math.sqrt(2)), abs=2e-15)
    else:
        assert negative.lo == 0 and negative.hi < Fraction(1, 10**15)


@pytest.mark.parametrize("horizon", [Fraction(1, 2), Fraction(1), Fraction(10), Fraction(1000)])
def test_integrated_brownian_transition_and_initial_moments_enclose_exact_polynomials(horizon):
    package = integrated_brownian()
    _, request = inputs(samples=2)
    request = replace(request, model_package_hash=package.package_hash, horizons=(float(horizon),),
        initial_mean=(1., 2., 0.5, -0.25), initial_covariance=tuple(tuple(float(i == j) for j in range(4)) for i in range(4)))
    document = bound_affine_reference(package, request).manifest()
    expected_F = [[Fraction(i == j) for j in range(4)] for i in range(4)]
    expected_F[0][2] = expected_F[1][3] = horizon
    b = [Fraction(value) for value in package.manifest()["parameters"]["b"]]
    offsets = [b[2]*horizon**2/2, b[3]*horizon**2/2, b[2]*horizon, b[3]*horizon]
    expected_Q = [[Fraction(0) for _ in range(4)] for _ in range(4)]
    for p, v, sigma in ((0, 2, Fraction(0.4)), (1, 3, Fraction(0.2))):
        expected_Q[p][p] = sigma**2*horizon**3/3
        expected_Q[p][v] = expected_Q[v][p] = sigma**2*horizon**2/2
        expected_Q[v][v] = sigma**2*horizon
    for i in range(4):
        contained(_parse_interval(document["transition_bounds"]["offset"][i]), offsets[i])
        mean = offsets[i]+sum(expected_F[i][j]*Fraction(request.initial_mean[j]) for j in range(4))
        contained(_parse_interval(document["mean_bounds"][i]), mean)
        for j in range(4):
            contained(_parse_interval(document["transition_bounds"]["F"][i][j]), expected_F[i][j])
            contained(_parse_interval(document["transition_bounds"]["covariance"][i][j]), expected_Q[i][j])
            covariance = expected_Q[i][j]+sum(expected_F[i][k]*expected_F[j][k] for k in range(4))
            contained(_parse_interval(document["covariance_bounds"][i][j]), covariance)


@pytest.mark.parametrize("case", oracle_suite(), ids=lambda case: case.case_id)
@pytest.mark.parametrize("horizon", [1., 1000.])
def test_registered_affine_cases_have_finite_narrow_bounded_reference_and_recompute(case, horizon):
    _, request = inputs(samples=2)
    request = replace(request, model_package_hash=case.package.package_hash, horizons=(horizon,),
        initial_mean=case.initial_mean, initial_covariance=case.initial_covariance)
    certificate = bound_affine_reference(case.package, request)
    bounds = certificate.functional_bounds()
    assert bounds.hi-bounds.lo < Fraction(1, 10**20)
    target = analytic_estimate(case.package, request).estimate
    error = reference_error_bound(certificate, case.package, request, target)
    assert 0 <= error < 1e-6
    assert abs(Fraction(target)-bounds.lo) <= Fraction(error)
    assert abs(Fraction(target)-bounds.hi) <= Fraction(error)
    document = certificate.manifest()
    assert document["scientific_qualification"] is False and document["status"] == "BOUNDED"
    assert document["request_hash"] == request.request_hash
    assert document["model_package_hash"] == case.package.package_hash
    assert document["operations"] <= document["algorithm"]["operations"]
    assert len(certificate._document) <= document["algorithm"]["certificate_bytes"]
    json.dumps(document, allow_nan=False)


@pytest.mark.parametrize("closed", [False, True])
def test_deterministic_boundary_is_resolved_only_when_exact_enclosure_proves_it(closed):
    package = affine_package("zero-point", [[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        [0]*4, [[0, 0]]*4)
    _, request = inputs(samples=2)
    request = replace(request, model_package_hash=package.package_hash, initial_mean=(0.,)*4,
        initial_covariance=((0.,)*4,)*4, functional="endpoint-halfspace", closed=closed)
    certificate = bound_affine_reference(package, request)
    bounds = certificate.functional_bounds()
    assert bounds.lo == bounds.hi == Fraction(int(closed))
    assert certificate.manifest()["status"] == "BOUNDED"


def test_degenerate_tie_without_proof_returns_full_probability_interval():
    package = affine_package("zero-noise-point", [[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, -.8, 0], [0, 0, 0, -.8]],
        [0]*4, [[0, 0]]*4)
    _, request = inputs(samples=2)
    # Tiny exact input is below the fixed dyadic precision. Repeated-variable
    # dependency can leave the boundary unresolved; no automatic precision lift.
    request = replace(request, model_package_hash=package.package_hash, initial_mean=(1e-100, 0., 0., 0.),
        initial_covariance=((0.,)*4,)*4, functional="endpoint-halfspace", threshold=1e-100)
    certificate = bound_affine_reference(package, request)
    assert certificate.manifest()["status"] == "UNRESOLVED_DEGENERATE_BOUNDARY"
    assert certificate.functional_bounds().lo == 0 and certificate.functional_bounds().hi == 1


@pytest.mark.parametrize("covariance", [((-1e-15, 0., 0., 0.),)+((0.,)*4,)*3,
    ((0., 1e-15, 0., 0.), (1e-15, 0., 0., 0.))+((0.,)*4,)*2,
    ((0., 1e-15, 0., 0.),)+((0.,)*4,)*3])
def test_reference_refuses_any_exact_indefiniteness_or_asymmetry_without_projection(covariance):
    package, request = inputs(samples=2, initial_covariance=covariance)
    with pytest.raises(DataValidationError):
        bound_affine_reference(package, request)


def test_certificate_is_detached_and_resealed_zero_error_or_scope_substitution_fails():
    package, request = inputs(samples=2)
    original = bound_affine_reference(package, request)
    document = original.manifest()
    document["functional_bounds"] = [["0", "1"], ["0", "1"]]
    modified = AffineReferenceCertificate(_encode(document))
    assert modified.certificate_hash == content_hash(document)
    with pytest.raises(DataValidationError, match="recomputed"):
        verify_affine_reference(modified, package, request)
    with pytest.raises(DataValidationError):
        verify_affine_reference(original, package, replace(request, horizons=(2.,)))
    assert original.manifest()["functional_bounds"] != document["functional_bounds"]
    with pytest.raises(FrozenInstanceError):
        original._document = b"{}"


def test_bound_precision_scaling_and_operation_quotas_are_not_expandable_retries():
    a = Arithmetic()
    with pytest.raises(NumericalError, match="integer quota"):
        a.box(1 << 4096)
    with pytest.raises(NumericalError, match="bit quota"):
        a.box(Fraction(1, 1 << 17000))
    a.operations = 200_000
    with pytest.raises(NumericalError, match="operation quota"):
        a.box(0)
    package, request = inputs(samples=2, horizons=(1e30,))
    with pytest.raises(NumericalError, match="scaling quota"):
        bound_affine_reference(package, request)


@pytest.mark.parametrize("value", [[["1", "0"], ["1", "1"]], [["00", "1"], ["1", "1"]],
    [["0", "1"], ["1", "3"]], [["2", "1"], ["1", "1"]], [[True, "1"], ["1", "1"]]])
def test_untrusted_interval_accessor_rejects_noncanonical_or_non_grid_bounds(value):
    with pytest.raises(DataValidationError):
        _parse_interval(value)


def test_nonlinear_package_never_receives_an_affine_reference_certificate():
    from tests.test_nonlinear_propagation import stress_inputs
    package, request = stress_inputs()
    with pytest.raises(DataValidationError, match="affine"):
        bound_affine_reference(package, request)


def test_structural_zero_and_nilpotent_exponentials_retain_exact_entries():
    a = Arithmetic()
    result = _short_exponential(a, [[a.box(0), a.box(Fraction(1, 4))], [a.box(0), a.box(0)]])
    for i in range(2):
        for j in range(2):
            expected = Fraction(1, 4) if (i, j) == (0, 1) else Fraction(i == j)
            assert result[i][j].lo == result[i][j].hi == expected


def test_symmetric_nondegenerate_endpoint_has_proven_half_probability_and_zero_reference_error():
    package = affine_package("symmetric-point", [[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        [0]*4, [[0, 0], [0, 0], [0.4, 0], [0, 0.2]])
    _, request = inputs(samples=2)
    request = replace(request, model_package_hash=package.package_hash, initial_mean=(0.,)*4,
        initial_covariance=((0.,)*4,)*4, functional="endpoint-halfspace")
    certificate = bound_affine_reference(package, request)
    bounds = certificate.functional_bounds()
    assert bounds.lo == bounds.hi == Fraction(1, 2)
    assert reference_error_bound(certificate, package, request, 0.5) == 0
    assert reference_error_bound(certificate, package, request, 0.6) > 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "0.5", 10**1000],
    ids=["nan", "infinity", "boolean", "string", "oversized-integer"])
def test_nonfinite_or_ill_typed_approximation_refuses_before_any_recomputation(value):
    with pytest.raises(DataValidationError, match="finite"):
        reference_error_bound(None, None, None, value)


@pytest.mark.parametrize("document", [b" "*131073, b"{}\n", _encode({"x": [0]*3000}),
    _encode({"x": "a"*2000}), b'{"x":'+b"["*1500+b"0"+b"]"*1500+b"}"],
    # pytest exports its current node ID; raw byte IDs exceed Windows'32767
    # character environment-variable limit before the test can run.
    ids=["byte-quota", "noncanonical", "node-quota", "string-quota", "depth-quota"])
def test_untrusted_certificate_bytes_and_json_structure_are_bounded(document):
    with pytest.raises(DataValidationError):
        AffineReferenceCertificate(document).manifest()


@pytest.mark.parametrize("scale", [2., 1e300])
def test_positive_exact_geometry_scaling_preserves_probability_bounds_not_request_identity(scale):
    package, request = inputs(samples=2, functional="endpoint-halfspace", threshold=0.)
    original = bound_affine_reference(package, request)
    changed = bound_affine_reference(package, replace(request, normal=(scale, 0., 0., 0.)))
    assert original.functional_bounds() == changed.functional_bounds()
    assert original.manifest()["request_hash"] != changed.manifest()["request_hash"]


@pytest.mark.parametrize("x", [1, 4, 8])
def test_cdf_box_contains_independent_exact_series_and_high_precision_root_enclosure(x):
    # Different computation: exact closed-form terms, no interval recurrence,
    # no Arithmetic operations and a finer256-bit integer square-root grid.
    def atan(q):
        partial = sum(((-1)**k*q**(2*k+1)/Fraction(2*k+1) for k in range(40)), Fraction(0))
        return partial, partial+q**81/81
    left, right = atan(Fraction(1, 5)), atan(Fraction(1, 239))
    pi_lo, pi_hi = 16*left[0]-4*right[1], 16*left[1]-4*right[0]
    grid = 1 << 256
    def root_floor(q):
        return math.isqrt(q.numerator*grid*grid//q.denominator)
    denominator_lo = Fraction(root_floor(2*pi_lo), grid)
    denominator_hi = Fraction(root_floor(2*pi_hi)+1, grid)
    integral = sum((Fraction((-1)**k*x**(2*k+1), 2**k*math.factorial(k)*(2*k+1))
                    for k in range(256)), Fraction(0))
    remainder = Fraction(x**513, 2**256*math.factorial(256)*513)
    independent_lo = Fraction(1, 2)-(integral+remainder)/denominator_lo
    independent_hi = Fraction(1, 2)-integral/denominator_hi
    assert independent_lo > 0  # Including the actual8-sigma rare tail.
    a = Arithmetic()
    actual = _cdf(a, Fraction(-x), a.sqrt(a.mul(a.box(2), _pi(a))))
    assert actual.lo <= independent_lo <= independent_hi <= actual.hi
