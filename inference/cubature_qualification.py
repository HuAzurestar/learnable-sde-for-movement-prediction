"""Charged affine cubature numerical checks, not nonlinear or formal proof."""

from fractions import Fraction

from domain.cubature_qualification import CubatureQualificationPolicy
from domain.errors import DataValidationError
from domain.frozen_dynamics import content_hash
from inference.affine_reference import Arithmetic, _interval_manifest, bound_affine_reference
from inference.affine_discrete_reference import bound_affine_discrete
from inference.affine_qualification import _outward_float, _scaled_norm
from inference.nonlinear_propagation import cubature_estimate


def analyze_affine_cubature(package, request, policy, functional_result):
    """Enclose the actual eight-point output against independent Gaussian laws.

    For affine Euler pushforward, the mathematical eight-point mean/covariance
    rule reproduces the finite-grid Gaussian moments. These per-functional checks
    measure its floating implementation error, not a nonlinear closure theorem.
    Both certifiers and replay below belong inside the charged pilot worker.
    """
    from experiments.pirc25.affine import code_hash
    source = code_hash()
    if type(policy) is not CubatureQualificationPolicy:
        raise DataValidationError("explicit affine cubature numerical policy required")
    policy.validate(package, request, source)
    actual = cubature_estimate(package, request)
    if actual != functional_result:
        raise DataValidationError("cubature qualification requires the actual bound estimator output")
    continuous = bound_affine_reference(package, request)
    target = bound_affine_discrete(package, request, solver="euler", steps=request.steps)
    cm, tm = continuous.manifest(), target.manifest()
    if cm["code_hash"] != source or tm["code_hash"] != source or code_hash() != source:
        raise DataValidationError("cubature qualification source changed during numerical work")
    cb, tb = continuous.functional_bounds(), target.functional_bounds()
    value = Fraction(actual.estimate)
    roundoff = max(abs(value-tb.lo), abs(value-tb.hi))
    width = cb.hi-cb.lo
    arithmetic = Arithmetic()
    bias = arithmetic.sub(tb, cb)
    bias_absolute = max(abs(bias.lo), abs(bias.hi))
    norm = max(_scaled_norm(cm, policy.state_scales), _scaled_norm(tm, policy.state_scales))
    operations = cm["operations"]+tm["operations"]+arithmetic.operations
    checks = {"resolved_reference": cm["status"] == "BOUNDED" and tm["status"] == "BOUNDED",
        "reference_width": width <= Fraction(policy.maximum_reference_width),
        "functional_roundoff": roundoff <= Fraction(policy.maximum_functional_roundoff),
        "time_bias": bias_absolute <= Fraction(policy.maximum_time_bias),
        "scaled_transition_growth": norm <= Fraction(policy.maximum_scaled_transition_norm),
        "reference_arithmetic_operations": operations <= policy.maximum_operations}
    document = {"schema_version": "affine-cubature-qualification-analysis-v1",
        "scope": "declared-affine-eight-point-Euler-endpoint-functional-for-one-request",
        "status": "PASSED" if all(checks.values()) else "FAILED", "scientific_qualification": False,
        "policy_hash": policy.policy_hash, "code_hash": source,
        "model_package_hash": package.package_hash, "request_hash": request.request_hash,
        "target_grid": {"solver": "euler", "steps": request.steps}, "point_count": 8,
        "point_weight": .125, "covariance_projection": False,
        "cubature_point_updates": 8*request.steps,
        "output_replay_point_updates": 8*request.steps,
        "checks": checks, "reference_width_upper": _outward_float(width),
        "functional_roundoff_upper": _outward_float(roundoff),
        "signed_time_bias_bounds": _interval_manifest(bias),
        "absolute_time_bias_upper": _outward_float(bias_absolute),
        "scaled_transition_norm_upper": _outward_float(norm),
        "reference_arithmetic_operations": operations,
        "sampling_error": {"value": 0, "status": "NOT_APPLICABLE"},
        "model_error": {"value": None, "status": "NOT_IDENTIFIABLE"},
        "cost_status": "OWNER_SETTLEMENT_REQUIRED", "maximum_job_seconds": policy.maximum_job_seconds,
        "continuous_certificate_hash": continuous.certificate_hash,
        "target_certificate_hash": target.certificate_hash,
        "continuous_certificate": cm, "target_certificate": tm}
    document["analysis_hash"] = content_hash(document)
    return document
