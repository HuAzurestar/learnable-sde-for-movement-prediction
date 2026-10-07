"""Owner gate for actual settled mixture pilots, never a borrowed pass report.

Only bounded arithmetic on saved enclosures. No uncharged mixture/reference
replay, new grants, stores, arms or ledger. Scope is the complete affine request.
"""

from datetime import datetime, timezone
from fractions import Fraction
import json
import math

from domain.errors import DataValidationError
from domain.mixture_qualification import MixtureQualificationPolicy
from inference.affine_reference import AffineReferenceCertificate, _parse_interval, algorithm_manifest
from inference.affine_qualification import _outward_float, _scaled_norm
from infrastructure.research_store import ResearchError, digest, encode


PAYLOAD_KEY = "managed_mixture_qualification"
METRIC = "absolute_error_upper_vs_declared_affine_law"


def require(condition, detail):
    if not condition:
        raise ResearchError("UNQUALIFIED", "managed mixture qualification: "+detail)


def analysis_values(analysis, policy, mixture, package, request, functional):
    """Recheck saved numeric meanings, not just their transport hashes."""
    estimate = functional["estimate"]
    require(type(estimate) in (float, int) and math.isfinite(estimate)
        and functional["kind"] == "functional_estimate"
        and functional["estimator_id"] == "retained-cubature-mixture-euler-v1"
        and functional["request_hash"] == request.request_hash
        and type(functional["sample_count"]) is int and functional["sample_count"] == 0
        and functional["standard_error"] == 0 and functional["interval"] is None
        and functional["interval_kind"] == "deterministic-mixture-no-sampling"
        and functional["status"] == "APPROXIMATION_ONLY", "actual source functional differs")
    require(analysis["schema_version"] == "affine-mixture-functional-qualification-analysis-v1"
        and analysis["scope"] == "one-retained-mixture-functional-vs-declared-affine-continuous-and-Euler-law"
        and analysis["scientific_qualification"] is False
        and analysis["request_hash"] == request.request_hash and analysis["model_package_hash"] == package.package_hash
        and analysis["code_hash"] == policy.code_hash and analysis["policy_hash"] == policy.policy_hash
        and analysis["mixture_policy_hash"] == mixture.policy_hash
        and analysis["actual_functional_hash"] == digest(functional) and analysis["actual_estimate"] == estimate
        and analysis["analysis_hash"] == digest({k: v for k, v in analysis.items() if k != "analysis_hash"}),
        "source analysis binding differs")
    cm, tm = analysis["continuous_certificate"], analysis["target_certificate"]
    for document, discrete in ((cm, False), (tm, True)):
        AffineReferenceCertificate(encode(document)).manifest()
        require(document["schema_version"] == ("affine-discrete-certificate-v1" if discrete else "affine-reference-certificate-v1")
            and document["scope"] == ("declared-affine-finite-grid-Gaussian-law" if discrete else "declared-affine-Gaussian-endpoint-law")
            and document["scientific_qualification"] is False and document["algorithm"] == algorithm_manifest()
            and document["code_hash"] == policy.code_hash and document["request_hash"] == request.request_hash
            and document["model_package_hash"] == package.package_hash
            and type(document["operations"]) is int and 0 < document["operations"] <= 200_000,
            "saved component certificate differs")
    require(analysis["continuous_certificate_hash"] == digest(cm) and analysis["target_certificate_hash"] == digest(tm)
        and analysis["target_grid"] == {"solver": "euler", "steps": request.steps}
        and tm["grid"] == {"solver": "euler", "steps": request.steps, "time_step": "exact-horizon-rational/steps",
            "noise": "independent-centered-Gaussian-increments", "roundoff_scope": "mathematical-recurrence-only"},
        "saved reference/grid hashes differ")
    cb, tb = _parse_interval(cm["functional_bounds"]), _parse_interval(tm["functional_bounds"])
    bias = _parse_interval(analysis["signed_time_bias_bounds"])
    require((bias.lo, bias.hi) == (tb.lo-cb.hi, tb.hi-cb.lo), "signed time bias differs")
    retained = max(abs(Fraction(estimate)-tb.lo), abs(Fraction(estimate)-tb.hi))
    total = max(abs(Fraction(estimate)-cb.lo), abs(Fraction(estimate)-cb.hi))
    width, absolute_bias = max(cb.hi-cb.lo, tb.hi-tb.lo), max(abs(bias.lo), abs(bias.hi))
    norm = max(_scaled_norm(cm, mixture.state_scales), _scaled_norm(tm, mixture.state_scales))
    operations = cm["operations"]+tm["operations"]+1
    checks = {"resolved_reference": cm["status"] == tm["status"] == "BOUNDED",
        "reference_width": width <= Fraction(policy.maximum_reference_width),
        "retained_functional_error": retained <= Fraction(policy.maximum_retained_functional_error),
        "time_bias": absolute_bias <= Fraction(policy.maximum_time_bias),
        "total_functional_error": total <= Fraction(policy.maximum_total_functional_error),
        "scaled_transition_growth": norm <= Fraction(policy.maximum_scaled_transition_norm),
        "reference_operations": operations <= policy.maximum_reference_operations}
    require(analysis["checks"] == checks and all(type(v) is bool for v in analysis["checks"].values())
        and analysis["status"] == "PASSED" and all(checks.values()), "actual numerical checks failed")
    unknown = {"value": None, "status": "NOT_IDENTIFIABLE"}
    expected = {"reference_width_upper": _outward_float(width), "retained_functional_error_upper": _outward_float(retained),
        "total_functional_error_upper": _outward_float(total), "absolute_time_bias_upper": _outward_float(absolute_bias),
        "scaled_transition_norm_upper": _outward_float(norm), "reference_operations": operations,
        "mixture_work_units": request.steps*mixture.work_per_step,
        "retained_error_scope": "closure-pruning-normalization-and-implementation-roundoff-inseparable",
        "propagation_approximation": unknown, "implementation_roundoff": unknown, "model_error": unknown,
        "sampling_error": {"value": 0, "status": "NOT_APPLICABLE"}, "cost_status": "OWNER_SETTLEMENT_REQUIRED",
        "maximum_job_seconds": policy.maximum_job_seconds,
        "scaled_transition_norm_definition": "max row sum |F_ij| scale_j/scale_i; declared horizon"}
    require(all(encode(analysis.get(k)) == encode(v) for k, v in expected.items()), "saved numeric values/unknowns differ")
    return cb, tb, bias


def prepare_managed_mixture(store, spec, cell, execution_package, prereg):
    from .propagation_execution import validate_propagation_cell
    from .research_reuse import verified_reuse
    from experiments.pirc25.affine import code_hash
    from experiments.pirc27.mixture_production_plugin import production_policies
    from experiments.pirc27.mixture_qualification_plugin import mixture_qualification_plugin
    try:
        package, request, config, _ = validate_propagation_cell(spec, cell)
        mixture, policy = production_policies(spec, cell, package, request, config)
        pointer = execution_package["payload"][PAYLOAD_KEY]
        require(type(pointer) is dict and set(pointer) == {"schema_version", "policy", "source_attempt_id",
            "source_artifact_id", "source_authorization_id", "source_authorization_version"}
            and pointer["schema_version"] == "managed-affine-mixture-qualification-v1"
            and pointer["policy"] == policy.manifest(), "explicit bound source pointer required")
        policies = prereg.get("mixture_qualification_policies")
        require(type(policies) is list and 0 < len(policies) <= 10000
            and policies.count(policy.manifest()) == 1 and len({digest(p) for p in policies}) == len(policies)
            and prereg["primary_metrics"] == [METRIC], "policy/metric not frozen before held-out input")
        attempt = store.attempts()[pointer["source_attempt_id"]]
        require(attempt["state"] == "SUCCEEDED" and attempt["artifact_id"] == pointer["source_artifact_id"], "source not completed")
        run = store.manifest("run-"+attempt["run_id"])
        source_spec = store.manifest("study-"+run["study_id"])["spec"]
        sc = run["cell"]
        require(source_spec["runtime_binding"] == spec["runtime_binding"]
            and source_spec["code_hash"] == spec["code_hash"] == code_hash()
            and sc["plugin_id"] == "affine-mixture-qualification" and sc["execution_role"] == "qualification"
            and all(sc[k] == cell[k] for k in ("frozen_dynamics", "propagation_request", "mixture_policy",
                "mixture_qualification_policy", "arm_id")), "source producer/model/full request/policies/root/arm differ")
        require({a["arm_id"]: a for a in source_spec["arms"]}[cell["arm_id"]]
            == {a["arm_id"]: a for a in spec["arms"]}[cell["arm_id"]], "original cumulative family arm differs")
        grant = store.authorization(pointer["source_authorization_id"], version=pointer["source_authorization_version"])
        require(grant["study_id"] == source_spec["study_id"] and grant["protocol_hash"] == source_spec["protocol_hash"]
            and (grant["study_id"] == spec["study_id"] or spec["study_id"] in grant.get("consumer_study_ids", []))
            and "evaluate" in grant["purposes"] and datetime.fromisoformat(grant["expires_at"]) > datetime.now(timezone.utc),
            "source grant does not authorize consumer")
        metadata = store.manifest("artifact-"+attempt["artifact_id"])
        require(metadata["size_bytes"] <= 512*1024, "source result byte quota")
        content = store.read_artifact(attempt["artifact_id"], purpose="evaluate", authorization=grant)
        result = json.loads(content)
        require(content == encode(result), "canonical actual source required")
        verified_reuse(store, attempt, source_spec, sc, mixture_qualification_plugin())
        receipt = store.manifest("admission-"+result["admission_hash"])
        blocks = [b for b in receipt["documents"]["protocol"]["blocks"] if b["block_id"] == sc["block_id"]]
        require(receipt["mode"] == "pilot" and len(blocks) == 1 and blocks[0]["split_role"] in {"train", "validation"},
            "source consumed held-out input")
        analysis_values(result["forecast"]["mixture_qualification_analysis"], policy, mixture, package, request,
            result["forecast"]["functional"])
        events = store.events()
        def one(kind):
            matches = [e for e in events if e["event_kind"] == kind and e["payload"].get("attempt_id") == attempt["attempt_id"]]
            require(len(matches) == 1, "source lacks unique "+kind)
            return matches[0]
        reservation, worker, stop, settlement, admission_event = (one(k) for k in
            ("RESERVE", "WORKER_STARTED", "WORKER_TREE_STOPPED", "SETTLE", "ADMISSION"))
        completions = [e for e in events if e["event_kind"] == "ATTEMPT" and e["payload"] == attempt]
        cost = settlement["payload"]
        require(len(completions) == 1 and admission_event["payload"]["admission_hash"] == receipt["admission_hash"]
            and reservation["sequence"] < worker["sequence"] < stop["sequence"] < settlement["sequence"] < completions[0]["sequence"]
            and admission_event["sequence"] < worker["sequence"]
            and type(cost["charged_ms"]) is int and 0 < cost["charged_ms"] <= cost["reserved_ms"] <= Fraction(policy.maximum_job_seconds)*1000
            and stop["payload"]["reservation_id"] == cost["reservation_id"]
            and stop["payload"]["observed_elapsed_ms"] == cost["charged_ms"]
            and stop["payload"]["confirmation"] == "native-job-or-process-group-no-running-descendants", "native source stop/cost/order differs")
        evidence = {"schema_version": "managed-affine-mixture-admission-evidence-v1", "policy": policy.manifest(),
            "mixture_policy": mixture.manifest(), "source_attempt": attempt, "source_run": run, "source_artifact": metadata,
            "source_result": result, "source_admission": receipt, "authorization": grant,
            "reservation_event": reservation, "worker_event": worker, "stop_event": stop, "settlement_event": settlement,
            "admission_event": admission_event, "completion_event": completions[0], "target_request_hash": request.request_hash,
            "target_cell_hash": digest(cell), "target_spec_hash": digest(spec), "preregistration_hash": digest(prereg)}
        evidence["evidence_hash"] = digest(evidence)
        require(len(encode(evidence)) <= 2*1024*1024, "saved owner evidence byte quota")
        store.verify_artifact_read(attempt["artifact_id"], purpose="evaluate", authorization=grant)
        return evidence
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError, DataValidationError) as exc:
        if isinstance(exc, ResearchError):
            raise
        raise ResearchError("UNQUALIFIED", "managed mixture evidence is missing or malformed") from exc


def qualified_mixture_forecast(spec, cell, package, request, functional, receipt):
    from experiments.pirc27.mixture_production_plugin import production_policies
    require(receipt["spec_hash"] == digest(spec) and receipt["cell_hash"] == digest(cell)
        and receipt["mode"] == "formal" and receipt["qualification"] == "qualified", "current formal owner receipt required")
    evidence = receipt["documents"]["propagation_qualification"]
    require(evidence["schema_version"] == "managed-affine-mixture-admission-evidence-v1"
        and evidence["evidence_hash"] == digest({k: v for k, v in evidence.items() if k != "evidence_hash"})
        and evidence["target_spec_hash"] == digest(spec) and evidence["target_cell_hash"] == digest(cell), "current target evidence differs")
    mixture, policy = production_policies(spec, cell, package, request, cell["execution"]["config"])
    require(evidence["policy"] == policy.manifest() and evidence["mixture_policy"] == mixture.manifest(), "current policies differ")
    source = evidence["source_result"]["forecast"]["functional"]
    analysis = evidence["source_result"]["forecast"]["mixture_qualification_analysis"]
    continuous, target, bias = analysis_values(analysis, policy, mixture, package, request, source)
    estimate = functional["estimate"]
    require(type(estimate) in (int, float) and math.isfinite(estimate), "current scalar must be finite")
    retained = max(abs(Fraction(estimate)-target.lo), abs(Fraction(estimate)-target.hi))
    total = max(abs(Fraction(estimate)-continuous.lo), abs(Fraction(estimate)-continuous.hi))
    require(retained <= Fraction(policy.maximum_retained_functional_error)
        and total <= Fraction(policy.maximum_total_functional_error) <= Fraction(request.tolerance), "current output exceeds frozen accuracy target")
    # Same full deterministic request: require the actual source lineage/result,
    # not merely an imported descriptor or a scalar slipped into its interval.
    require(digest(functional) == analysis["actual_functional_hash"], "current deterministic functional/lineage differs from actual pilot")
    components = {"reference_width_upper": _outward_float(max(continuous.hi-continuous.lo, target.hi-target.lo)),
        "retained_functional_error_upper": _outward_float(retained), "total_functional_error_upper": _outward_float(total),
        "time_bias_absolute_upper": _outward_float(max(abs(bias.lo), abs(bias.hi))),
        "signed_time_bias_bounds": analysis["signed_time_bias_bounds"], "propagation_approximation": analysis["propagation_approximation"],
        "implementation_roundoff": analysis["implementation_roundoff"], "model_error": analysis["model_error"],
        "retained_error_scope": analysis["retained_error_scope"], "scope": "one-functional-of-declared-affine-law-only",
        "policy_hash": policy.policy_hash, "mixture_policy_hash": mixture.policy_hash,
        "qualification_evidence_hash": evidence["evidence_hash"], "qualification_source_attempt_id": evidence["source_attempt"]["attempt_id"]}
    return _outward_float(total), components


def validate_formal_mixture_result(receipt, spec, cell, result):
    from .propagation_execution import validate_propagation_cell
    package, request, _, _ = validate_propagation_cell(spec, cell)
    functional = dict(result["forecast"]["functional"])
    budget = functional["error_budget"]
    # The adapter replaces only these two qualified components after computing
    # the actual kernel result; recover that original for full lineage binding.
    functional["error_budget"] = {**budget, **{k: receipt["documents"]["propagation_qualification"]["source_result"]
        ["forecast"]["functional"]["error_budget"][k] for k in ("reference", "time_discretization")}}
    error, components = qualified_mixture_forecast(spec, cell, package, request, functional, receipt)
    unit = "m" if request.functional == "endpoint-x" else "1"
    require(result["metrics"] == {METRIC: error} and result["metric_units"] == {METRIC: unit}
        and result["forecast"]["model_package_hash"] == package.package_hash and result["forecast"]["request_hash"] == request.request_hash
        and result["forecast"]["qualified_error_components"] == components, "current scalar metrics/provenance differ")
    require(budget["reference"]["status"] == budget["time_discretization"]["status"] == "BOUNDED"
        and budget["reference"]["value"] == components["reference_width_upper"]
        and budget["time_discretization"]["value"] == components["time_bias_absolute_upper"], "qualified separated components differ")
