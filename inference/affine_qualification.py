"""Actual deterministic analytic qualification work; supervisor use required.

Thresholds are frozen before execution. Failed checks remain numerical evidence,
never trigger grid/tolerance expansion and never qualify a stochastic sampler.
"""

from fractions import Fraction
import math

from domain.errors import DataValidationError, NumericalError
from domain.affine_qualification import AffineQualificationPolicy
from domain.frozen_dynamics import content_hash
from inference.affine_reference import (Arithmetic, _interval_manifest,
    _parse_interval, bound_affine_reference)
from inference.affine_discrete_reference import bound_affine_discrete


def _outward_float(value):
    try:
        result = float(value)
    except OverflowError as exc:
        raise NumericalError("qualification bound exceeds finite float range") from exc
    if not math.isfinite(result):
        raise NumericalError("qualification bound exceeds finite float range")
    if Fraction(result) < value:
        result = math.nextafter(result, math.inf)
    if not math.isfinite(result):
        raise NumericalError("qualification outward bound exceeds finite float range")
    return result


def _scaled_norm(document, scales):
    matrix = document["transition_bounds"]["F"]
    return max(sum(max(abs(bound.lo), abs(bound.hi))*Fraction(scales[j])/Fraction(scales[i])
                   for j, value in enumerate(row) for bound in [_parse_interval(value)])
               for i, row in enumerate(matrix))


def analyze_affine_qualification(package, request, method, policy, functional_result):
    from experiments.pirc25.affine import code_hash
    from inference.propagation_methods import analytic_estimate
    source = code_hash()
    if type(policy) is not AffineQualificationPolicy:
        raise DataValidationError("explicit immutable analytic qualification policy required")
    policy.validate(package, request, method, source)
    # Do not trust an arbitrary finite value carrying another estimator's name.
    actual = analytic_estimate(package, request, discrete=method == "gaussian")
    if functional_result != actual:
        raise DataValidationError("qualification result differs from actual bound analytic estimator")
    continuous = bound_affine_reference(package, request)
    cm = continuous.manifest()
    target = continuous if method == "exact" else bound_affine_discrete(package, request,
        solver="euler", steps=request.steps)
    tm = target.manifest()
    if cm["code_hash"] != tm["code_hash"] or cm["code_hash"] != source or code_hash() != source:
        raise DataValidationError("qualification source changed during numerical evidence generation")
    bounds, reference = target.functional_bounds(), continuous.functional_bounds()
    value = Fraction(actual.estimate)
    roundoff = max(abs(value-bounds.lo), abs(value-bounds.hi))
    reference_width = reference.hi-reference.lo
    a = Arithmetic()
    bias = a.box(0) if method == "exact" else a.sub(bounds, reference)
    bias_absolute = max(abs(bias.lo), abs(bias.hi))
    norm = max(_scaled_norm(cm, policy.state_scales), _scaled_norm(tm, policy.state_scales))
    operations = cm["operations"]+(tm["operations"] if method != "exact" else 0)+a.operations
    checks = {
        "resolved_reference": cm["status"] == "BOUNDED" and tm["status"] == "BOUNDED",
        "reference_width": reference_width <= Fraction(policy.maximum_reference_width),
        "functional_roundoff": roundoff <= Fraction(policy.maximum_functional_roundoff),
        "time_bias": bias_absolute <= Fraction(policy.maximum_time_bias),
        "scaled_transition_growth": norm <= Fraction(policy.maximum_scaled_transition_norm),
        "arithmetic_operations": operations <= policy.maximum_operations}
    document = {"schema_version": "affine-analytic-qualification-analysis-v1",
        "scope": "declared-affine-analytic-endpoint-functional-for-one-request",
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "scientific_qualification": False, "policy_hash": policy.policy_hash,
        "code_hash": source, "model_package_hash": package.package_hash,
        "request_hash": request.request_hash, "method": method,
        "target_grid": None if method == "exact" else {"solver": "euler", "steps": request.steps},
        "checks": checks, "reference_width_upper": _outward_float(reference_width),
        "functional_roundoff_upper": _outward_float(roundoff),
        "signed_time_bias_bounds": _interval_manifest(bias),
        "absolute_time_bias_upper": _outward_float(bias_absolute),
        "scaled_transition_norm_upper": _outward_float(norm),
        "scaled_transition_norm_definition": "max row sum |F_ij| scale_j/scale_i; whole declared horizon",
        "state_scale_units": ["m", "m", "m/s", "m/s"],
        "operations": operations, "sampling_error": {"status": "NOT_APPLICABLE", "value": 0},
        "model_error": {"status": "NOT_IDENTIFIABLE", "value": None},
        "cost_status": "OWNER_SETTLEMENT_REQUIRED", "maximum_job_seconds": policy.maximum_job_seconds,
        "continuous_certificate_hash": continuous.certificate_hash,
        "target_certificate_hash": target.certificate_hash,
        "continuous_certificate": cm, "target_certificate": tm if method != "exact" else None}
    document["analysis_hash"] = content_hash(document)
    return document
