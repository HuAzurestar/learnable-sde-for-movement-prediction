"""Bounded worker-side affine spatial calibration kernel, not an admission API.

No compiler, ledger, process or data reads. Integration must run this entire
kernel within an original charged pilot worker before freezing test geometry.
"""

from dataclasses import replace
from fractions import Fraction
import math

from domain.errors import DataValidationError, NumericalError
from domain.frozen_dynamics import content_hash, _encode
from domain.probability_calibration import AffineHalfspaceCalibrationPolicy
from inference.affine_reference import (Arithmetic, Interval, MAX_BYTES,
    _bound_affine_reference, _cdf, _dot, _interval_manifest, _matmul,
    _parse_interval, _pi)


QUANTILE_BISECTIONS = 48


def _quantile(a, target):
    denominator = a.sqrt(a.mul(a.box(2), _pi(a)))
    lo, hi = Fraction(0), Fraction(8)
    left, right = _cdf(a, -lo, denominator), _cdf(a, -hi, denominator)
    if left.lo < target or right.hi > target:
        raise NumericalError("calibration quantile is outside the fixed certified bracket")
    for _ in range(QUANTILE_BISECTIONS):
        mid = (lo+hi)/2
        tail = _cdf(a, -mid, denominator)
        if tail.lo > target:
            lo, left = mid, tail
        elif tail.hi < target:
            hi, right = mid, tail
        else:
            raise NumericalError("calibration quantile comparison is unresolved at fixed precision")
    return Interval(lo, hi), left, right


def _outward_float(value):
    try:
        result = float(value)
    except OverflowError as exc:
        raise NumericalError("calibration diagnostic exceeds finite float range") from exc
    if not math.isfinite(result):
        raise NumericalError("calibration diagnostic exceeds finite float range")
    if Fraction(result) < value:
        result = math.nextafter(result, math.inf)
    if not math.isfinite(result):
        raise NumericalError("calibration outward diagnostic exceeds finite float range")
    return result


def calibrate_affine_halfspace(package, request, policy):
    """Propose and certify the actual rounded threshold under a frozen policy.

    Two reference computations, projection and quantile search share one
    counter; no adaptive precision, extra quota, float inverse CDF or PSD repair.
    A PASSED candidate still lacks owner settlement and consumer admission.
    """
    from experiments.pirc25.affine import code_hash
    source = code_hash()
    if type(policy) is not AffineHalfspaceCalibrationPolicy:
        raise DataValidationError("explicit affine probability calibration policy required")
    policy.validate(package, request, source)
    a = Arithmetic(maximum_operations=policy.maximum_operations)
    moment_certificate = _bound_affine_reference(package,
        replace(request, functional="endpoint-x"), a)
    moment_document = moment_certificate.manifest()
    moment_operations = a.operations
    mean = [_parse_interval(value) for value in moment_document["mean_bounds"]]
    covariance = [[_parse_interval(value) for value in row]
        for row in moment_document["covariance_bounds"]]
    exact_normal = [Fraction(value) for value in request.normal]
    scale = max(abs(value) for value in exact_normal)
    normal = [a.box(value/scale) for value in exact_normal]
    projected_mean = _dot(a, normal, mean)
    variance = _dot(a, normal, [row[0] for row in
        _matmul(a, covariance, [[value] for value in normal])])
    document = {"schema_version": "affine-halfspace-calibration-analysis-v1",
        "scope": "declared-affine-Gaussian-spatial-endpoint-law-for-one-template",
        "policy_hash": policy.policy_hash, "policy": policy.manifest(), "code_hash": source,
        "model_package_hash": package.package_hash, "source_request_hash": request.request_hash,
        "target_probability": policy.target_probability, "normal": list(request.normal),
        "threshold_units": "m", "threshold": None, "calibrated_request_hash": None,
        "scientific_qualification": False, "method_qualification": False,
        "cost_status": "OWNER_SETTLEMENT_REQUIRED", "admission_status": "NOT_ADMITTED",
        "maximum_quantile_bisections": QUANTILE_BISECTIONS, "executed_quantile_bisections": 0,
        "quantile_bounds": None,
        "quantile_endpoint_tail_bounds": None, "ideal_threshold_bounds": None,
        "maximum_job_seconds": policy.maximum_job_seconds,
        "moment_certificate": moment_document, "moment_certificate_hash": moment_certificate.certificate_hash,
        "projected_mean_bounds": _interval_manifest(projected_mean),
        "normalized_projected_variance_bounds": _interval_manifest(variance),
        "probability_certificate": None, "probability_certificate_hash": None,
        "relative_probability_error_upper": None, "relative_probability_width_upper": None,
        "checks": {"positive_projected_variance": variance.lo > 0,
            "resolved_probability": False, "relative_probability_error": False,
            "relative_probability_width": False}}
    if variance.lo <= 0:
        document["failure_reason"] = "NON_POSITIVE_OR_UNRESOLVED_PROJECTED_VARIANCE"
        final_operations = 0
        calibration_operations = a.operations-moment_operations
    else:
        target = Fraction(policy.target_probability)
        quantile, left, right = _quantile(a, target)
        ideal_threshold = a.mul(a.add(projected_mean,
            a.mul(quantile, a.sqrt(variance))), a.box(scale))
        try:
            threshold = float((ideal_threshold.lo+ideal_threshold.hi)/2)
        except OverflowError as exc:
            raise NumericalError("calibrated threshold exceeds finite float range") from exc
        if not math.isfinite(threshold):
            raise NumericalError("calibrated threshold exceeds finite float range")
        calibrated = replace(request, threshold=threshold)
        calibration_operations = a.operations-moment_operations
        final_certificate = _bound_affine_reference(package, calibrated, a)
        final_document = final_certificate.manifest()
        final_operations = final_document["operations"]
        bounds = final_certificate.functional_bounds()
        error = max(abs(bounds.lo-target), abs(bounds.hi-target))/target
        width = (bounds.hi-bounds.lo)/target
        document.update({"threshold": threshold, "calibrated_request_hash": calibrated.request_hash,
            "executed_quantile_bisections": QUANTILE_BISECTIONS,
            "quantile_bounds": _interval_manifest(quantile),
            "quantile_endpoint_tail_bounds": [_interval_manifest(left), _interval_manifest(right)],
            "ideal_threshold_bounds": _interval_manifest(ideal_threshold),
            "probability_certificate": final_document,
            "probability_certificate_hash": final_certificate.certificate_hash,
            "relative_probability_error_upper": _outward_float(error),
            "relative_probability_width_upper": _outward_float(width)})
        document["checks"].update({"resolved_probability": final_document["status"] == "BOUNDED",
            "relative_probability_error": error <= Fraction(policy.maximum_relative_probability_error),
            "relative_probability_width": width <= Fraction(policy.maximum_relative_probability_width)})
    if code_hash() != source or moment_document["code_hash"] != source or (
            document["probability_certificate"] is not None
            and document["probability_certificate"]["code_hash"] != source):
        raise DataValidationError("probability calibration source changed during numerical work")
    document["operation_counts"] = {"moments": moment_operations,
        "projection_and_quantile": calibration_operations, "actual_threshold_reference": final_operations}
    document["reference_arithmetic_operations"] = a.operations
    document["status"] = "PASSED" if all(document["checks"].values()) else "FAILED"
    document["analysis_hash"] = content_hash(document)
    if len(_encode(document)) > MAX_BYTES:
        raise NumericalError("calibration analysis exceeds original reference byte quota")
    return document
