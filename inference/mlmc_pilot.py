"""Bounded empirical diagnostics and sampling-only proposals, never approval."""

import math

from domain.errors import DataValidationError
from domain.mlmc_pilot import MLMCPilotPolicy
from .propagation_methods import allocate_mlmc


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def analyze_mlmc_pilot(request, result, policy):
    if type(policy) is not MLMCPilotPolicy:
        raise DataValidationError("explicit frozen MLMC pilot policy required")
    diagnostics = dict(result.diagnostics)
    if len(diagnostics) != len(result.diagnostics):
        raise DataValidationError("duplicate pilot diagnostic keys")
    counts = diagnostics.get("level_samples")
    policy.validate(request, counts)
    variance = diagnostics.get("level_variances")
    elapsed = diagnostics.get("level_compute_ns")
    work = diagnostics.get("level_work_per_sample")
    expected_work = tuple(request.steps*2**level + (request.steps*2**(level-1) if level else 0)
                          for level in range(len(counts)))
    if (result.request_hash != request.request_hash or result.estimator_id != "coupled-euler-mlmc-pilot-v1"
            or result.sample_count != sum(counts) or type(diagnostics.get("pilot_phase")) is not int
            or diagnostics.get("pilot_phase") != 2 or type(diagnostics.get("production_phase")) is not int
            or diagnostics.get("production_phase") != 1 or work != expected_work
            or diagnostics.get("cost_scope") != "compute-only-excludes-checkpoint-ACK-not-budget-charge"
            or diagnostics.get("coupling") != "coarse-increment=sum(two-fine-increments)"
            or type(variance) is not tuple or len(variance) != len(counts)
            or any(not _finite(v) or v < 0 for v in variance)
            or type(elapsed) is not tuple or len(elapsed) != len(counts)
            or any(type(ns) is not int or not 0 < ns < 2**63 for ns in elapsed)):
        raise DataValidationError("pilot diagnostics/stream/cost binding differs")
    unit_cost = tuple(ns/n for ns, n in zip(elapsed, counts))
    # Level zero is P0, not a correction; compare only correction-level pairs.
    variance_ratios = tuple(variance[level]/variance[level-1] if variance[level-1] > 0 else None
                           for level in range(2, len(counts)))
    variance_ratios = tuple(ratio if _finite(ratio) else None for ratio in variance_ratios)
    cost_ratios = tuple(unit_cost[level]/unit_cost[level-1] for level in range(1, len(counts)))
    reasons = []
    if any(ratio is None or not math.isfinite(ratio) for ratio in variance_ratios) or any(v == 0 for v in variance[1:]):
        reasons.append("UNRESOLVED_LEVEL_VARIANCE")
    elif any(ratio > policy.maximum_variance_ratio for ratio in variance_ratios):
        reasons.append("VARIANCE_DECAY_FAILED")
    if any(not math.isfinite(ratio) or ratio > policy.maximum_cost_ratio for ratio in cost_ratios):
        reasons.append("COST_GROWTH_FAILED")
    bias = result.error_budget.time_discretization.value
    reference = result.error_budget.reference.value
    if ((bias is not None and not _finite(bias))
            or (reference is not None and (not _finite(reference) or reference < 0))):
        raise DataValidationError("pilot error components must be finite or explicitly unknown")
    if bias is None:
        reasons.append("BIAS_UNRESOLVED")
    elif abs(bias) > policy.bias_tolerance:
        reasons.append("BIAS_TOLERANCE_FAILED")
    if reference is None:
        reasons.append("REFERENCE_UNRESOLVED")
    if result.status not in {"PILOT_ONLY"}:
        reasons.append("SAMPLING_UNRESOLVED")
    proposal = None
    sampling_failures = {"UNRESOLVED_LEVEL_VARIANCE", "VARIANCE_DECAY_FAILED", "COST_GROWTH_FAILED", "SAMPLING_UNRESOLVED"}
    if not sampling_failures & set(reasons):
        try:
            candidate = allocate_mlmc(variance, unit_cost, policy.sampling_tolerance, maximum_samples=policy.maximum_samples)
            if sum(n*cost for n, cost in zip(candidate, work)) > policy.maximum_work_steps:
                raise DataValidationError("sampling proposal exceeds frozen total work quota")
            proposal = candidate
        except DataValidationError:
            reasons.append("ALLOCATION_INFEASIBLE")
    reasons.append("ENGINEERING_ONLY")
    return {"schema_version": "endpoint-mlmc-pilot-analysis-v1", "request_hash": request.request_hash,
        "policy_hash": policy.policy_hash, "status": "UNQUALIFIED", "reasons": reasons,
        "qualification_scope": "empirical-engineering-pilot-not-scientific-approval",
        "level_variance_ratios": variance_ratios, "level_cost_ratios": cost_ratios,
        "level_unit_compute_ns": unit_cost, "signed_discretization_bias": bias,
        "reference_uncertainty": reference, "sampling_only_proposal": proposal,
        "production_stream": {"seed": policy.production_seed, "coupling_id": policy.production_coupling_id, "phase": 1},
        "requires_separate_production_registration": True, "automatic_execution": False,
        "cost_scope": "compute-only;shared-ledger-remains-sole-budget-authority"}
