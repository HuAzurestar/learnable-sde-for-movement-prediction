"""Actual mixture output and bounded affine references, not borrowed approval."""

from fractions import Fraction

from domain.errors import DataValidationError
from domain.frozen_dynamics import content_hash
from domain.mixture_qualification import MixtureQualificationPolicy
from inference.affine_reference import Arithmetic, _interval_manifest, bound_affine_reference
from inference.affine_discrete_reference import bound_affine_discrete
from inference.affine_qualification import _outward_float, _scaled_norm
from inference.mixture_propagation import mixture_estimate


def qualify_affine_mixture(package, request, mixture, policy):
    """Compute once inside a supervised pilot; return output and saved analysis.

    Distance of this actual retained estimate to an enclosed Euler expectation
    includes closure, pruning/normalization AND implementation roundoff. These
    are not separately identified. No distribution/KL/unseen-functional claim.
    """
    from experiments.pirc25.affine import code_hash
    if type(policy) is not MixtureQualificationPolicy:
        raise DataValidationError("explicit immutable mixture qualification policy required")
    source = code_hash()
    policy.validate(package, request, mixture, source)
    result = mixture_estimate(package, request, mixture)
    continuous = bound_affine_reference(package, request)
    target = bound_affine_discrete(package, request, solver="euler", steps=request.steps)
    cm, tm = continuous.manifest(), target.manifest()
    if cm["code_hash"] != source or tm["code_hash"] != source or code_hash() != source:
        raise DataValidationError("source changed during mixture qualification")
    cb, tb = continuous.functional_bounds(), target.functional_bounds()
    value = Fraction(result.estimate)
    retained_error = max(abs(value-tb.lo), abs(value-tb.hi))
    total_error = max(abs(value-cb.lo), abs(value-cb.hi))
    width = max(cb.hi-cb.lo, tb.hi-tb.lo)
    arithmetic = Arithmetic()
    bias = arithmetic.sub(tb, cb)
    absolute_bias = max(abs(bias.lo), abs(bias.hi))
    norm = max(_scaled_norm(cm, mixture.state_scales), _scaled_norm(tm, mixture.state_scales))
    operations = cm["operations"]+tm["operations"]+arithmetic.operations
    checks = {"resolved_reference": cm["status"] == tm["status"] == "BOUNDED",
        "reference_width": width <= Fraction(policy.maximum_reference_width),
        "retained_functional_error": retained_error <= Fraction(policy.maximum_retained_functional_error),
        "time_bias": absolute_bias <= Fraction(policy.maximum_time_bias),
        "total_functional_error": total_error <= Fraction(policy.maximum_total_functional_error),
        "scaled_transition_growth": norm <= Fraction(policy.maximum_scaled_transition_norm),
        "reference_operations": operations <= policy.maximum_reference_operations}
    analysis = {"schema_version": "affine-mixture-functional-qualification-analysis-v1",
        "scope": "one-retained-mixture-functional-vs-declared-affine-continuous-and-Euler-law",
        "status": "PASSED" if all(checks.values()) else "FAILED", "scientific_qualification": False,
        "code_hash": source, "request_hash": request.request_hash, "model_package_hash": package.package_hash,
        "policy_hash": policy.policy_hash, "mixture_policy_hash": mixture.policy_hash,
        "actual_functional_hash": content_hash(result.manifest()), "actual_estimate": result.estimate,
        "target_grid": {"solver": "euler", "steps": request.steps}, "checks": checks,
        "reference_width_upper": _outward_float(width),
        "retained_functional_error_upper": _outward_float(retained_error),
        "retained_error_scope": "closure-pruning-normalization-and-implementation-roundoff-inseparable",
        "propagation_approximation": {"value": None, "status": "NOT_IDENTIFIABLE"},
        "implementation_roundoff": {"value": None, "status": "NOT_IDENTIFIABLE"},
        "signed_time_bias_bounds": _interval_manifest(bias),
        "absolute_time_bias_upper": _outward_float(absolute_bias),
        "total_functional_error_upper": _outward_float(total_error),
        "scaled_transition_norm_upper": _outward_float(norm),
        "scaled_transition_norm_definition": "max row sum |F_ij| scale_j/scale_i; declared horizon",
        "reference_operations": operations, "mixture_work_units": request.steps*mixture.work_per_step,
        "sampling_error": {"value": 0, "status": "NOT_APPLICABLE"},
        "model_error": {"value": None, "status": "NOT_IDENTIFIABLE"},
        "cost_status": "OWNER_SETTLEMENT_REQUIRED", "maximum_job_seconds": policy.maximum_job_seconds,
        "continuous_certificate_hash": continuous.certificate_hash,
        "target_certificate_hash": target.certificate_hash,
        "continuous_certificate": cm, "target_certificate": tm}
    analysis["analysis_hash"] = content_hash(analysis)
    return result, analysis
