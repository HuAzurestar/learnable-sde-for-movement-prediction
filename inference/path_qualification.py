"""One actual path pilot, own saved statistics and bounded affine references.

Observed scalar errors are not deterministic sampling bounds. Normal intervals
are estimates, not coverage proofs for float64/PRNG paths. No numerical replay,
adaptive allocation, self-normalized IS or borrowed analytic sampler approval.
"""

from fractions import Fraction
import math

from domain.errors import DataValidationError
from domain.frozen_dynamics import content_hash
from domain.path_qualification import PathQualificationPolicy
from inference.affine_reference import Arithmetic, _interval_manifest, _parse_interval, bound_affine_reference
from inference.affine_discrete_reference import bound_affine_discrete
from inference.affine_qualification import _outward_float, _scaled_norm
from inference.propagation_methods import monte_carlo, importance_sampling
from inference.propagation_recovery import ChunkState


SOLVERS = {"euler": "euler", "heun": "additive-heun", "reversible-heun": "reversible-heun"}


def _sampling_evidence(request, method, result, state):
    """Check saved sufficient statistics against the unchanged actual output."""
    stat = state["method_state"]["statistics"][0]
    n, hits = stat["n"], stat["hits"]
    if n != request.samples or result.sample_count != n or result.request_hash != request.request_hash:
        raise DataValidationError("actual path output/statistics request/count differs")
    weighted = method == "importance"
    if weighted:
        estimate = math.exp(stat["log_event"]-math.log(n)) if hits else 0.
        second = math.exp(stat["log_event2"]-math.log(n)) if hits else 0.
        variance = max(0., (second-estimate**2)*n/(n-1))
        se = math.sqrt(variance/n) if hits else None
        ess = math.exp(2*stat["log_w"]-stat["log_w2"])
        expected_status = "INSUFFICIENT_EVENTS" if not hits else "LOW_ESS" if ess < 2 else "SUCCEEDED"
        interval = (estimate-1.959963984540054*se, estimate+1.959963984540054*se) if hits else None
        interval_kind = "iid-weighted-normal-approximation-95" if hits else "unavailable-no-weighted-hits"
        estimator, kind = "velocity-drift-is-unnormalized-v1", "weighted"
        diagnostics = dict(result.diagnostics)
        if (diagnostics.get("ess") != ess or diagnostics.get("hits") != hits
                or diagnostics.get("self_normalized") is not False
                or tuple(diagnostics.get("proposal", ())) != tuple(state["_proposal"])
                or diagnostics.get("proposal_hash") != content_hash(state["_proposal"])
                or diagnostics.get("log_sum_weights") != stat["log_w"]
                or diagnostics.get("log_sum_squared_weights") != stat["log_w2"]
                or diagnostics.get("max_log_weight") != stat["max_log_w"]):
            raise DataValidationError("actual unnormalized IS diagnostics differ from saved statistics")
        # With zero proposal every likelihood is exactly one by construction.
        # Log-sum rounding can nevertheless produce tiny positive variance for
        # an all-hit event. Never treat that artifact as identified uncertainty.
        degenerate = hits == n and tuple(state["_proposal"]) == (0., 0.)
        radius = 1.959963984540054*se if se is not None and se > 0 and not degenerate else None
        evidence_interval, evidence_kind = interval, interval_kind
        coverage = "estimated-only; finite-variance iid weighted normal approximation; no guaranteed coverage"
        if hits and radius is None:
            evidence_interval, evidence_kind = None, "unavailable-degenerate-weighted-event-statistics"
            coverage = "no sampling uncertainty identified from degenerate weighted event statistics; no coverage guarantee"
    else:
        diagnostics = dict(result.diagnostics)
        if (diagnostics.get("hits") != hits or diagnostics.get("coupling_id") != request.coupling_id
                or diagnostics.get("brownian_scheme") != "per-sample-seedsequence-v1"):
            raise DataValidationError("actual MC diagnostics differ from saved statistics/stream")
        estimate = stat["mean"]
        se = math.sqrt(stat["m2"]/(n-1)/n)
        interval = (estimate-1.959963984540054*se, estimate+1.959963984540054*se)
        interval_kind = "normal-approximation-95"
        estimator, kind, expected_status, ess = SOLVERS[method]+"-path-mc-v1", "functional_estimate", "SUCCEEDED", None
        if request.functional == "endpoint-halfspace" and hits == 0:
            se = None
            interval = (0., -math.expm1(math.log(.05)/n))
            interval_kind = "exact-binomial-one-sided-95"
        evidence_interval, evidence_kind = interval, interval_kind
        if request.functional == "endpoint-halfspace" and hits in (0, n):
            radius = -math.expm1(math.log(.05)/n)
            evidence_interval = (0., radius) if hits == 0 else (1.-radius, 1.)
            evidence_kind = "exact-binomial-one-sided-95-formula"
            coverage = "one-sided95 under ideal iid Bernoulli law only; float64 formula/PRNG coverage not certified"
        else:
            radius = 1.959963984540054*se if se > 0 else None
            coverage = "estimated-only; iid normal approximation; no guaranteed coverage"
    if (result.estimate != estimate or result.standard_error != se or result.interval != interval
            or result.interval_kind != interval_kind or result.estimator_id != estimator
            or result.kind != kind or result.status != expected_status):
        raise DataValidationError("actual sampler output differs from saved sufficient statistics")
    return {"status": "ESTIMATED" if radius is not None else "NOT_IDENTIFIABLE",
        "standard_error": se, "uncertainty_radius": radius,
        "interval": list(evidence_interval) if evidence_interval is not None else None,
        "interval_kind": evidence_kind, "coverage_scope": coverage,
        "sample_count": n, "hits": hits, "effective_sample_size": ess,
        "self_normalized": False if weighted else None,
        "zero_observed_variance_is_not_a_zero_error_certificate": True}


def qualify_affine_paths(package, request, policy):
    """Run once inside the shared supervised pilot; return output and analysis."""
    from experiments.pirc25.affine import code_hash
    if type(policy) is not PathQualificationPolicy:
        raise DataValidationError("explicit immutable path qualification policy required")
    source = code_hash()
    policy.validate(package, request, source)
    last = None
    def capture(state, total):
        nonlocal last
        if total != request.samples*request.steps:
            raise DataValidationError("path statistic work proxy differs")
        last = state  # Already detached/bounded by ChunkState, no raw paths.
    if policy.method == "importance":
        result = importance_sampling(package, request, proposal=policy.proposal, checkpoint=capture)
        method, proposal = "importance", policy.proposal
    else:
        method, proposal = SOLVERS[policy.method], ()
        result = monte_carlo(package, request, solver=method, checkpoint=capture)
    if last is None or last["data_position"] != {"level": 1, "next_sample": 0}:
        raise DataValidationError("completed final sampler statistics required")
    ChunkState(request, method, (request.samples,), (request.steps,), proposal=proposal, restored=last)
    sampling = _sampling_evidence(request, policy.method, result, {**last, "_proposal": proposal})
    continuous = bound_affine_reference(package, request)
    solver = "euler" if policy.method == "importance" else policy.method
    target = bound_affine_discrete(package, request, solver=solver, steps=request.steps)
    cm, tm = continuous.manifest(), target.manifest()
    if cm["code_hash"] != source or tm["code_hash"] != source or code_hash() != source:
        raise DataValidationError("source changed during path qualification")
    return result, saved_path_analysis(package, request, policy, result, last, cm, tm)


def saved_path_analysis(package, request, policy, result, last, cm, tm):
    """Bounded saved-state arithmetic only; caller verifies certificate/law binding.

    No sampler, matrix exponential or CDF/reference engine is invoked here.
    The owner may use the same declared law for a separately frozen target.
    """
    method = "importance" if policy.method == "importance" else SOLVERS[policy.method]
    proposal = policy.proposal if policy.method == "importance" else ()
    if type(last) is not dict or last.get("data_position") != {"level": 1, "next_sample": 0}:
        raise DataValidationError("completed final sampler statistics required")
    ChunkState(request, method, (request.samples,), (request.steps,), proposal=proposal, restored=last)
    sampling = _sampling_evidence(request, policy.method, result, {**last, "_proposal": proposal})
    solver = "euler" if policy.method == "importance" else policy.method
    cb, tb = _parse_interval(cm["functional_bounds"]), _parse_interval(tm["functional_bounds"])
    value = Fraction(result.estimate)
    grid_error = max(abs(value-tb.lo), abs(value-tb.hi))
    total_error = max(abs(value-cb.lo), abs(value-cb.hi))
    width = max(cb.hi-cb.lo, tb.hi-tb.lo)
    arithmetic = Arithmetic()
    bias = arithmetic.sub(tb, cb)
    absolute_bias = max(abs(bias.lo), abs(bias.hi))
    target_scales = policy.state_scales*2 if policy.method == "reversible-heun" else policy.state_scales
    norm = max(_scaled_norm(cm, policy.state_scales), _scaled_norm(tm, target_scales))
    operations = cm["operations"]+tm["operations"]+arithmetic.operations
    checks = {"resolved_reference": cm["status"] == tm["status"] == "BOUNDED",
        "sampler_status": result.status == "SUCCEEDED",
        "sampling_uncertainty": sampling["uncertainty_radius"] is not None
            and sampling["uncertainty_radius"] <= policy.maximum_sampling_uncertainty,
        "effective_sample_size": policy.method != "importance"
            or sampling["effective_sample_size"] >= policy.minimum_effective_sample_size,
        "reference_width": width <= Fraction(policy.maximum_reference_width),
        "observed_grid_error": grid_error <= Fraction(policy.maximum_observed_grid_error),
        "time_bias": absolute_bias <= Fraction(policy.maximum_time_bias),
        "total_observed_functional_error": total_error <= Fraction(policy.maximum_total_observed_functional_error),
        "scaled_transition_growth": norm <= Fraction(policy.maximum_scaled_transition_norm),
        "reference_operations": operations <= policy.maximum_reference_operations}
    analysis = {"schema_version": "affine-path-functional-qualification-analysis-v1",
        "scope": "one-realized-path-estimate-vs-declared-affine-continuous-and-finite-grid-functional",
        "status": "PASSED" if all(checks.values()) else "FAILED", "scientific_qualification": False,
        "code_hash": policy.code_hash, "request_hash": request.request_hash, "model_package_hash": package.package_hash,
        "policy_hash": policy.policy_hash, "method": policy.method, "proposal": list(policy.proposal),
        "proposal_hash": content_hash(policy.proposal),
        "actual_functional_hash": content_hash(result.manifest()), "actual_estimate": result.estimate,
        "completed_statistics": last, "completed_statistics_hash": content_hash(last),
        "target_grid": {"solver": solver, "steps": request.steps}, "checks": checks,
        "physical_dimension": 4, "auxiliary_dimension": 4 if policy.method == "reversible-heun" else 0,
        "reference_width_upper": _outward_float(width),
        "observed_grid_error_upper": _outward_float(grid_error),
        "observed_grid_error_scope": "realized-sampling-and-implementation-roundoff-inseparable",
        "implementation_roundoff": {"value": None, "status": "NOT_IDENTIFIABLE"},
        "propagation_approximation": {"value": None, "status": "NOT_SEPARATELY_IDENTIFIABLE"},
        "signed_time_bias_bounds": _interval_manifest(bias),
        "absolute_time_bias_upper": _outward_float(absolute_bias),
        "total_observed_functional_error_upper": _outward_float(total_error),
        "total_observed_error_scope": "actual scalar distance; not a stochastic coverage or predictive-distribution bound",
        "scaled_transition_norm_upper": _outward_float(norm),
        "scaled_transition_norm_definition": "max row sum |F_ij| scale_j/scale_i; declared horizon; auxiliaries repeat physical scales",
        "state_scale_units": ["m", "m", "m/s", "m/s"],
        "reference_operations": operations, "path_work_units": request.samples*request.steps,
        "sampling_error": sampling, "model_error": {"value": None, "status": "NOT_IDENTIFIABLE"},
        "cost_status": "OWNER_SETTLEMENT_REQUIRED", "maximum_job_seconds": policy.maximum_job_seconds,
        "continuous_certificate_hash": content_hash(cm),
        "target_certificate_hash": content_hash(tm),
        "continuous_certificate": cm, "target_certificate": tm}
    analysis["analysis_hash"] = content_hash(analysis)
    return analysis
