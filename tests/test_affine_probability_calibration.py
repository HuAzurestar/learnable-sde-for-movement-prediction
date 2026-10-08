"""Synthetic kernel tests, not managed producer or observed-model qualification."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import math

import pytest

from domain.errors import DataValidationError, NumericalError
from domain.probability_calibration import AffineHalfspaceCalibrationPolicy
from experiments.pirc25.affine import code_hash
from experiments.pirc27.oracles import affine_package, oracle_suite
from inference.affine_probability_calibration import calibrate_affine_halfspace
from inference.affine_reference import (AffineReferenceCertificate, Arithmetic,
    MAX_OPERATIONS, _bound_affine_reference, _encode, _parse_interval, bound_affine_reference,
    verify_affine_reference)
from tests.test_propagation_methods import inputs


def example(probability=.000001, **changes):
    package, request = inputs(samples=2, functional="endpoint-halfspace")
    policy = AffineHalfspaceCalibrationPolicy(request.request_hash,
        package.package_hash, code_hash(), probability, 1e-8, 1e-10,
        MAX_OPERATIONS, 1800.)
    return package, request, replace(policy, **changes)


@pytest.mark.parametrize("probability", [.01, .0001, .000001])
@pytest.mark.parametrize("horizon", [1., 1000.])
def test_actual_float_threshold_has_recomputable_rare_probability_bounds(probability, horizon):
    for case in oracle_suite():
        _, template, _ = example()
        request = replace(template, model_package_hash=case.package.package_hash,
            initial_mean=case.initial_mean, initial_covariance=case.initial_covariance,
            horizons=(horizon,))
        policy = AffineHalfspaceCalibrationPolicy(request.request_hash,
            case.package.package_hash, code_hash(), probability, 1e-8, 1e-10,
            MAX_OPERATIONS, 1800.)
        result = calibrate_affine_halfspace(case.package, request, policy)
        assert result["status"] == "PASSED"
        actual_request = replace(request, threshold=result["threshold"])
        certificate = AffineReferenceCertificate(_encode(result["probability_certificate"]))
        verified = verify_affine_reference(certificate, case.package, actual_request)
        bounds, target = verified.functional_bounds(), Fraction(probability)
        assert max(abs(bounds.lo-target), abs(bounds.hi-target))/target <= Fraction(1e-8)
        assert (bounds.hi-bounds.lo)/target <= Fraction(1e-10)
        assert result["calibrated_request_hash"] == actual_request.request_hash
        assert result["reference_arithmetic_operations"] == sum(result["operation_counts"].values())
        assert result["reference_arithmetic_operations"] <= policy.maximum_operations <= MAX_OPERATIONS
        assert result["scientific_qualification"] is False
        assert result["method_qualification"] is False
        assert result["admission_status"] == "NOT_ADMITTED"
        assert result["cost_status"] == "OWNER_SETTLEMENT_REQUIRED"
        assert result["threshold_units"] == "m"
        assert result["executed_quantile_bisections"] == result["maximum_quantile_bisections"] == 48
        quantile = _parse_interval(result["quantile_bounds"])
        left, right = map(_parse_interval, result["quantile_endpoint_tail_bounds"])
        assert quantile.hi-quantile.lo == Fraction(8, 2**48)
        assert left.lo >= target >= right.hi


def test_frozen_scalar_policy_roundtrips_and_rejects_cycles_before_copying():
    package, request, policy = example()
    assert AffineHalfspaceCalibrationPolicy.from_manifest(policy.manifest()) == policy
    with pytest.raises(FrozenInstanceError):
        policy.target_probability = .1
    manifest = policy.manifest()
    manifest["target_probability"] = manifest
    with pytest.raises(DataValidationError, match="scalar"):
        AffineHalfspaceCalibrationPolicy.from_manifest(manifest)
    with pytest.raises(DataValidationError):
        replace(policy, target_probability=manifest).manifest()
    with pytest.raises(DataValidationError):
        replace(policy, target_probability=manifest).validate(package, request, code_hash())


@pytest.mark.parametrize("changes", [
    {"target_probability": True}, {"target_probability": 0.},
    {"target_probability": 1e-9}, {"target_probability": .5},
    {"target_probability": math.nan}, {"maximum_relative_probability_error": 0.},
    {"maximum_relative_probability_width": math.inf},
    {"maximum_operations": True}, {"maximum_operations": MAX_OPERATIONS+1},
    {"maximum_job_seconds": 1801.}, {"maximum_job_seconds": True},
    {"source_request_hash": "0"*64}, {"model_package_hash": "0"*64},
    {"code_hash": "0"*64}])
def test_policy_refuses_invalid_limits_and_source_bindings(changes):
    package, request, policy = example(**changes)
    with pytest.raises(DataValidationError):
        calibrate_affine_halfspace(package, request, policy)


@pytest.mark.parametrize("changes", [
    {"functional": "endpoint-x"}, {"threshold": 1.},
    {"normal": (1., 0., 1., 0.)}, {"normal": (0., 0., 0., 1.)}])
def test_only_zero_template_position_halfspaces_have_metre_calibration(changes):
    package, request, policy = example()
    request = replace(request, **changes)
    policy = replace(policy, source_request_hash=request.request_hash)
    with pytest.raises(DataValidationError, match="position halfspace"):
        calibrate_affine_halfspace(package, request, policy)


def test_changed_initial_law_cutoff_horizon_or_geometry_is_not_same_policy():
    package, request, policy = example()
    for changes in ({"horizons": (10.,)}, {"origin": 1.},
            {"history_cutoff": -1.}, {"initial_mean": (1., 0., 1., -.5)},
            {"normal": (0., 1., 0., 0.)}, {"closed": False}):
        with pytest.raises(DataValidationError):
            calibrate_affine_halfspace(package, replace(request, **changes), policy)


def test_unsatisfied_numerical_controls_retain_candidate_and_probability_evidence():
    package, request, policy = example(maximum_relative_probability_error=1e-30,
        maximum_relative_probability_width=1e-50)
    result = calibrate_affine_halfspace(package, request, policy)
    assert result["status"] == "FAILED"
    assert math.isfinite(result["threshold"])
    assert result["probability_certificate"]["status"] == "BOUNDED"
    assert result["checks"]["relative_probability_error"] is False
    assert result["checks"]["relative_probability_width"] is False
    assert result["admission_status"] == "NOT_ADMITTED"


def test_shared_counter_refuses_before_exceeding_explicit_operations():
    package, request, policy = example(maximum_operations=100)
    with pytest.raises(NumericalError, match="operation quota"):
        calibrate_affine_halfspace(package, request, policy)
    a = Arithmetic(maximum_operations=2)
    a.box(0)
    a.box(1)
    with pytest.raises(NumericalError):
        a.box(2)
    assert a.operations == 2


def test_shared_reference_counter_does_not_change_canonical_certificate():
    package, request, _ = example()
    counter = Arithmetic()
    for _ in range(7):
        counter.box(0)
    expected = bound_affine_reference(package, request)
    actual = _bound_affine_reference(package, request, counter)
    assert actual == expected
    assert counter.operations == 7 + actual.manifest()["operations"]


def test_geometry_does_not_mutate_template_request_or_policy():
    package, request, policy = example()
    original_request_hash, original_policy_hash = request.request_hash, policy.policy_hash
    calibrate_affine_halfspace(package, request, policy)
    assert request.threshold == 0.
    assert request.request_hash == original_request_hash
    assert policy.policy_hash == original_policy_hash


@pytest.mark.parametrize("normal", [(0., 1., 0., 0.), (-3., 2., 0., 0.), (4., 0., 0., 0.)])
def test_position_rotation_sign_and_scaling_use_actual_event_probability(normal):
    package, request, policy = example()
    request = replace(request, normal=normal)
    policy = replace(policy, source_request_hash=request.request_hash)
    result = calibrate_affine_halfspace(package, request, policy)
    assert result["status"] == "PASSED"
    actual_request = replace(request, threshold=result["threshold"])
    certificate = AffineReferenceCertificate(_encode(result["probability_certificate"]))
    verify_affine_reference(certificate, package, actual_request)
    assert result["normal"] == list(normal)


def test_zero_variance_retains_moments_without_search_or_threshold():
    package = affine_package("deterministic-calibration-test",
        [[0., 0., 1., 0.], [0., 0., 0., 1.], [0., 0., 0., 0.], [0., 0., 0., 0.]],
        [0.]*4, [[0., 0.] for _ in range(4)])
    _, request, policy = example()
    request = replace(request, model_package_hash=package.package_hash)
    policy = replace(policy, model_package_hash=package.package_hash, source_request_hash=request.request_hash)
    result = calibrate_affine_halfspace(package, request, policy)
    assert result["status"] == "FAILED"
    assert result["failure_reason"] == "NON_POSITIVE_OR_UNRESOLVED_PROJECTED_VARIANCE"
    assert result["threshold"] is result["probability_certificate"] is None
    assert result["executed_quantile_bisections"] == 0
    assert result["operation_counts"]["actual_threshold_reference"] == 0
    assert result["moment_certificate"]["status"] == "BOUNDED"


def test_invalid_initial_covariance_is_not_repaired():
    package, request, policy = example()
    covariance = tuple(tuple(-.1 if i == j == 0 else 0. for j in range(4)) for i in range(4))
    request = replace(request, initial_covariance=covariance)
    policy = replace(policy, source_request_hash=request.request_hash)
    with pytest.raises(DataValidationError, match="PSD projection"):
        calibrate_affine_halfspace(package, request, policy)


def test_total_quota_is_shared_through_quantile_not_reset_after_moments(monkeypatch):
    import inference.affine_probability_calibration as kernel
    package, request, policy = example(maximum_operations=60_000)
    original_quantile, counters = kernel._quantile, []
    def recording_quantile(a, target):
        counters.append(a)
        assert a.operations > 0
        return original_quantile(a, target)
    monkeypatch.setattr(kernel, "_quantile", recording_quantile)
    with pytest.raises(NumericalError, match="operation quota"):
        calibrate_affine_halfspace(package, request, policy)
    assert len(counters) == 1 and counters[0].operations == 60_000


def test_late_source_change_cannot_issue_a_passed_calibration(monkeypatch):
    import experiments.pirc25.affine as source_module
    package, request, policy = example()
    hashes = iter([policy.code_hash]*3 + ["0"*64])
    monkeypatch.setattr(source_module, "code_hash", lambda: next(hashes))
    with pytest.raises(DataValidationError, match="source changed"):
        calibrate_affine_halfspace(package, request, policy)


@pytest.mark.parametrize("limit", [True, 0, MAX_OPERATIONS+1])
def test_counter_cannot_enlarge_original_quota(limit):
    with pytest.raises(DataValidationError):
        Arithmetic(maximum_operations=limit)
