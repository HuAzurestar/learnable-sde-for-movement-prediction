"""Charged-worker affine finest-grid MLMC bias/reference checks.

Empirical variance/cost evidence is not an asymptotic theorem, ideal sampler
roundoff bound or scientific/model approval. No store, ledger, grant or execution.
"""

from fractions import Fraction
import math

from domain.affine_mlmc_qualification import AffineMLMCReferencePolicy
from domain.errors import DataValidationError, NumericalError
from domain.frozen_dynamics import FrozenDynamicsPackage, content_hash, _encode
from inference.affine_discrete_reference import bound_affine_discrete
from inference.affine_reference import Arithmetic, bound_affine_reference, _interval_manifest
from inference.affine_qualification import _outward_float, _scaled_norm
from inference.mlmc_pilot import analyze_mlmc_pilot


def analyze_bounded_mlmc_pilot(package, request, result, pilot_policy, reference_policy):
    from experiments.pirc25.affine import code_hash
    if type(package) is not FrozenDynamicsPackage or type(reference_policy) is not AffineMLMCReferencePolicy:
        raise DataValidationError("frozen affine law and explicit MLMC reference policy required")
    source = code_hash()
    reference_policy.validate(package, request, pilot_policy, source)
    package.validate()
    empirical = analyze_mlmc_pilot(request, result, pilot_policy)
    diagnostics = dict(result.diagnostics)
    if tuple(diagnostics["level_samples"]) != reference_policy.level_samples:
        raise DataValidationError("pilot sample allocation differs from frozen reference policy")
    estimate = result.estimate
    if type(estimate) not in (float, int) or not math.isfinite(estimate):
        raise DataValidationError("finite actual MLMC pilot output required")
    means, variances = diagnostics.get("level_means"), diagnostics["level_variances"]
    if (type(means) is not tuple or len(means) != len(reference_policy.level_samples)
            or any(type(v) not in (float, int) or not math.isfinite(v) for v in means)
            or math.fsum(means) != estimate):
        raise DataValidationError("pilot scalar differs from recorded level statistics")
    standard_error = math.sqrt(math.fsum(v/n for v, n in zip(variances, reference_policy.level_samples)))
    unresolved = request.functional == "endpoint-halfspace" and standard_error == 0
    interval = None if unresolved else (estimate-1.959963984540054*standard_error, estimate+1.959963984540054*standard_error)
    if (result.kind != "functional_estimate"
            or result.interval_kind != ("unavailable-zero-observed-level-variance" if unresolved else "independent-level-normal-approximation-95")
            or result.status != ("UNRESOLVED_SAMPLING" if unresolved else "PILOT_ONLY")
            or result.standard_error != (None if unresolved else standard_error) or result.interval != interval):
        raise DataValidationError("pilot sampling interval differs from recorded variance")
    finest_steps = request.steps*2**(len(reference_policy.level_samples)-1)
    continuous = bound_affine_reference(package, request)
    finest = bound_affine_discrete(package, request, solver="euler", steps=finest_steps)
    cm, fm = continuous.manifest(), finest.manifest()
    if cm["code_hash"] != source or fm["code_hash"] != source or code_hash() != source:
        raise DataValidationError("source changed during bounded MLMC pilot analysis")
    cb, fb = continuous.functional_bounds(), finest.functional_bounds()
    arithmetic = Arithmetic()
    bias = arithmetic.sub(fb, cb)
    width, absolute_bias = cb.hi-cb.lo, max(abs(bias.lo), abs(bias.hi))
    norm = max(_scaled_norm(cm, reference_policy.state_scales), _scaled_norm(fm, reference_policy.state_scales))
    operations = cm["operations"]+fm["operations"]+arithmetic.operations
    # This observed pilot scalar error includes random sampling AND sampler
    # implementation effects. It is not mislabeled as a roundoff bound or SE.
    observed_error = max(abs(Fraction(estimate)-cb.lo), abs(Fraction(estimate)-cb.hi))
    numeric_checks = {"resolved_reference": cm["status"] == fm["status"] == "BOUNDED",
        "reference_width": width <= Fraction(reference_policy.maximum_reference_width),
        "finest_grid_bias": absolute_bias <= Fraction(pilot_policy.bias_tolerance),
        "scaled_transition_growth": norm <= Fraction(reference_policy.maximum_scaled_transition_norm),
        "arithmetic_operations": operations <= reference_policy.maximum_operations}
    sampling_failures = {"UNRESOLVED_LEVEL_VARIANCE", "VARIANCE_DECAY_FAILED", "COST_GROWTH_FAILED",
        "SAMPLING_UNRESOLVED", "ALLOCATION_INFEASIBLE"}
    empirical_failures = sorted(sampling_failures & set(empirical["reasons"]))
    checks = {**numeric_checks, "empirical_sampling_and_cost": not empirical_failures}
    document = {"schema_version": "affine-mlmc-reference-pilot-analysis-v1",
        "scope": "necessary-affine-finest-grid-and-empirical-pilot-checks-only",
        "status": "NUMERICAL_READY" if all(checks.values()) else "NUMERICAL_FAILED",
        "qualification": "UNQUALIFIED", "scientific_qualification": False,
        "reference_policy_hash": reference_policy.policy_hash, "pilot_policy_hash": pilot_policy.policy_hash,
        "request_hash": request.request_hash, "model_package_hash": package.package_hash, "code_hash": source,
        "functional_unit": "m" if request.functional == "endpoint-x" else "1",
        "checks": checks, "empirical_failures": empirical_failures,
        "finest_grid": {"solver": "euler", "base_steps": request.steps,
            "level_count": len(reference_policy.level_samples), "steps": finest_steps},
        "reference_width_upper": _outward_float(width),
        "signed_finest_grid_bias_bounds": _interval_manifest(bias),
        "finest_grid_bias_absolute_upper": _outward_float(absolute_bias),
        "scaled_transition_norm_upper": _outward_float(norm), "state_scales": list(reference_policy.state_scales),
        "state_scale_units": ["m", "m", "m/s", "m/s"],
        "operations": operations,
        "observed_pilot_error_upper_vs_continuous_law": _outward_float(observed_error),
        "observed_error_scope": "current scalar only; sampling and implementation effects not separated",
        "sampling_error": {"value": result.standard_error, "status": "ESTIMATED" if result.standard_error is not None else "NOT_IDENTIFIABLE",
            "interval_kind": result.interval_kind, "confidence_interval": result.interval},
        "sampler_roundoff": {"value": None, "status": "NOT_IDENTIFIABLE"},
        "model_error": {"value": None, "status": "NOT_IDENTIFIABLE"},
        "empirical_pilot_analysis": empirical,
        "production_stream": empirical["production_stream"],
        "requires_separate_production_registration": True, "automatic_execution": False,
        "cost_status": "OWNER_SETTLEMENT_REQUIRED", "maximum_job_seconds": reference_policy.maximum_job_seconds,
        "continuous_certificate_hash": continuous.certificate_hash, "finest_certificate_hash": finest.certificate_hash,
        "continuous_certificate": cm, "finest_certificate": fm}
    document["analysis_hash"] = content_hash(document)
    if len(_encode(document)) > 384*1024:
        raise NumericalError("MLMC pilot numeric evidence exceeds fixed byte quota")
    return document
