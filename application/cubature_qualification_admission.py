"""Owner consumption of settled affine eight-point evidence, never a grant.

Only saved-enclosure arithmetic runs here; matrix/CDF work belongs to the
charged source worker. The Gaussian closure identity is affine finite-grid
only, not nonlinear, observed-model or continuous-time qualification.
"""

from datetime import datetime, timezone
from fractions import Fraction
import json
import math

from domain.cubature_qualification import CubatureQualificationPolicy
from domain.errors import DataValidationError
from inference.affine_reference import AffineReferenceCertificate, _parse_interval, algorithm_manifest
from inference.affine_qualification import _outward_float, _scaled_norm
from infrastructure.research_store import ResearchError, digest, encode


PAYLOAD_KEY = "managed_cubature_qualification"
METRIC = "absolute_error_upper_vs_declared_affine_law"
SCOPE = "declared-affine-eight-point-Euler-endpoint-functional-for-one-request"


def _require(condition, detail):
    if not condition:
        raise ResearchError("UNQUALIFIED", "managed cubature qualification: "+detail)


def _analysis_values(analysis, policy, package, request, estimate):
    _require(type(estimate) in (float, int) and math.isfinite(estimate), "finite source scalar required")
    _require(len(encode(analysis)) <= 512*1024
        and analysis["schema_version"] == "affine-cubature-qualification-analysis-v1"
        and analysis["scope"] == SCOPE and analysis["scientific_qualification"] is False
        and analysis["policy_hash"] == policy.policy_hash
        and analysis["code_hash"] == policy.code_hash and analysis["model_package_hash"] == package.package_hash
        and analysis["request_hash"] == request.request_hash
        and analysis["analysis_hash"] == digest({k: v for k, v in analysis.items() if k != "analysis_hash"})
        and type(analysis["point_count"]) is int and analysis["point_count"] == 8
        and type(analysis["point_weight"]) is float and analysis["point_weight"] == .125
        and analysis["covariance_projection"] is False
        and type(analysis["cubature_point_updates"]) is int and analysis["cubature_point_updates"] == 8*request.steps
        and type(analysis["output_replay_point_updates"]) is int and analysis["output_replay_point_updates"] == 8*request.steps,
        "analysis identity/eight-point work differs")
    cm, tm = analysis["continuous_certificate"], analysis["target_certificate"]
    for document, schema, scope in ((cm, "affine-reference-certificate-v1", "declared-affine-Gaussian-endpoint-law"),
        (tm, "affine-discrete-certificate-v1", "declared-affine-finite-grid-Gaussian-law")):
        AffineReferenceCertificate(encode(document)).manifest()
        _require(document["schema_version"] == schema and document["scope"] == scope
            and document["scientific_qualification"] is False and document["algorithm"] == algorithm_manifest()
            and document["code_hash"] == policy.code_hash and document["request_hash"] == request.request_hash
            and document["model_package_hash"] == package.package_hash
            and type(document["operations"]) is int and 0 < document["operations"] <= 200_000,
            "bounded component certificate differs")
    _require(analysis["continuous_certificate_hash"] == digest(cm)
        and analysis["target_certificate_hash"] == digest(tm)
        and analysis["target_grid"] == {"solver": "euler", "steps": request.steps}
        and tm["grid"] == {"solver": "euler", "steps": request.steps,
            "time_step": "exact-horizon-rational/steps", "noise": "independent-centered-Gaussian-increments",
            "roundoff_scope": "mathematical-recurrence-only"}, "component hash/grid differs")
    cb, tb = _parse_interval(cm["functional_bounds"]), _parse_interval(tm["functional_bounds"])
    bias = _parse_interval(analysis["signed_time_bias_bounds"])
    _require((bias.lo, bias.hi) == (tb.lo-cb.hi, tb.hi-cb.lo), "signed grid bias differs")
    width = cb.hi-cb.lo
    error = max(abs(Fraction(estimate)-tb.lo), abs(Fraction(estimate)-tb.hi))
    absolute_bias = max(abs(bias.lo), abs(bias.hi))
    norm = max(_scaled_norm(cm, policy.state_scales), _scaled_norm(tm, policy.state_scales))
    operations = cm["operations"]+tm["operations"]+1
    checks = {"resolved_reference": cm["status"] == "BOUNDED" and tm["status"] == "BOUNDED",
        "reference_width": width <= Fraction(policy.maximum_reference_width),
        "functional_roundoff": error <= Fraction(policy.maximum_functional_roundoff),
        "time_bias": absolute_bias <= Fraction(policy.maximum_time_bias),
        "scaled_transition_growth": norm <= Fraction(policy.maximum_scaled_transition_norm),
        "reference_arithmetic_operations": operations <= policy.maximum_operations}
    _require(analysis["status"] == "PASSED" and analysis["checks"] == checks
        and all(type(v) is bool for v in analysis["checks"].values()) and all(checks.values()),
        "actual saved numerical checks did not pass")
    expected = {"reference_width_upper": _outward_float(width), "functional_roundoff_upper": _outward_float(error),
        "absolute_time_bias_upper": _outward_float(absolute_bias), "scaled_transition_norm_upper": _outward_float(norm),
        "reference_arithmetic_operations": operations, "sampling_error": {"value": 0, "status": "NOT_APPLICABLE"},
        "model_error": {"value": None, "status": "NOT_IDENTIFIABLE"},
        "cost_status": "OWNER_SETTLEMENT_REQUIRED", "maximum_job_seconds": policy.maximum_job_seconds}
    _require(all(analysis.get(k) == v for k, v in expected.items()), "saved numerical values differ")
    return cb, tb, bias


def _functional_values(functional, request):
    _require(type(functional["estimate"]) in (int, float) and math.isfinite(functional["estimate"])
        and functional["request_hash"] == request.request_hash
        and functional["estimator_id"] == "gaussian-cubature-euler-v1" and functional["kind"] == "functional_estimate"
        and type(functional["sample_count"]) is int and functional["sample_count"] == 0
        and functional["standard_error"] == 0 and functional["interval"] is None
        and functional["interval_kind"] == "deterministic-closure-no-sampling"
        and functional["status"] == "APPROXIMATION_ONLY", "own deterministic estimator identity differs")
    pairs = functional["diagnostics"]
    _require(type(pairs) in (list, tuple) and len(pairs) == 6
        and all(type(pair) in (list, tuple) and len(pair) == 2 and type(pair[0]) is str for pair in pairs),
        "diagnostics shape differs")
    diagnostics = dict(pairs)
    _require(len(diagnostics) == len(pairs) and diagnostics["point_count"] == 8
        and type(diagnostics["point_count"]) is int and diagnostics["point_weight"] == .125
        and diagnostics["covariance_projection"] is False, "eight-point/no-projection diagnostics differ")
    mean, covariance = diagnostics["mean"], diagnostics["covariance"]
    _require(type(mean) in (list, tuple) and len(mean) == 4
        and type(covariance) in (list, tuple) and len(covariance) == 4
        and all(type(row) in (list, tuple) and len(row) == 4 for row in covariance)
        and all(type(v) in (float, int) and math.isfinite(v) for v in [*mean, *(v for row in covariance for v in row)]),
        "finite four-state moments required")


def prepare_managed_cubature(store, spec, cell, execution_package, prereg):
    from application.propagation_execution import validate_propagation_cell
    from application.research_reuse import verified_reuse
    from experiments.pirc25.affine import code_hash
    from experiments.pirc27.cubature_plugin import cubature_plugin
    try:
        package, request, config, _ = validate_propagation_cell(spec, cell)
        _require(cell["plugin_id"] == "affine-cubature" and config["method"] == "cubature", "dedicated affine target required")
        pointer = execution_package["payload"][PAYLOAD_KEY]
        _require(type(pointer) is dict and set(pointer) == {"schema_version", "policy", "source_attempt_id",
            "source_artifact_id", "source_authorization_id", "source_authorization_version"}
            and pointer["schema_version"] == "managed-affine-cubature-qualification-v1", "explicit cubature pointer required")
        policy = CubatureQualificationPolicy.from_manifest(pointer["policy"])
        policy.validate(package, request, code_hash())
        policies = prereg.get("cubature_qualification_policies")
        _require(type(policies) is list and 0 < len(policies) <= 10000 and policies.count(policy.manifest()) == 1
            and len({digest(p) for p in policies}) == len(policies) and prereg["primary_metrics"] == [METRIC],
            "dedicated policy/metric not frozen in preregistration")
        attempt = store.attempts()[pointer["source_attempt_id"]]
        _require(attempt["state"] == "SUCCEEDED" and attempt["artifact_id"] == pointer["source_artifact_id"], "source is not settled success")
        run = store.manifest("run-"+attempt["run_id"])
        source_spec = store.manifest("study-"+run["study_id"])["spec"]
        source_cell = run["cell"]
        _require(source_spec["runtime_binding"] == spec["runtime_binding"] and source_spec["code_hash"] == spec["code_hash"]
            and source_cell["plugin_id"] == "affine-cubature-qualification" and source_cell["execution_role"] == "qualification"
            and source_cell["frozen_dynamics"] == cell["frozen_dynamics"]
            and source_cell["propagation_request"] == cell["propagation_request"]
            and source_cell["cubature_qualification_policy"] == policy.manifest()
            and source_cell["arm_id"] == cell["arm_id"], "source code/model/request/arm/policy/root differs")
        def arm(value):
            matches = [a for a in value["arms"] if a["arm_id"] == cell["arm_id"]]
            _require(len(matches) == 1, "unique original arm required")
            return matches[0]
        _require(arm(source_spec) == arm(spec) and arm(spec)["method_family_id"] == "cubature", "source cumulative cubature arm differs")
        grant = store.authorization(pointer["source_authorization_id"], version=pointer["source_authorization_version"])
        _require(grant["study_id"] == source_spec["study_id"] and grant["protocol_hash"] == source_spec["protocol_hash"]
            and (grant["study_id"] == spec["study_id"] or spec["study_id"] in grant.get("consumer_study_ids", []))
            and "evaluate" in grant["purposes"]
            and datetime.fromisoformat(grant["expires_at"]) > datetime.now(timezone.utc), "source consumer permission differs")
        metadata = store.manifest("artifact-"+attempt["artifact_id"])
        _require(type(metadata["size_bytes"]) is int and metadata["size_bytes"] <= 512*1024, "source artifact byte quota")
        content = store.read_artifact(attempt["artifact_id"], purpose="evaluate", authorization=grant)
        result = json.loads(content)
        _require(content == encode(result), "canonical source result required")
        verified_reuse(store, attempt, source_spec, source_cell, cubature_plugin(qualification=True))
        receipt = store.manifest("admission-"+result["admission_hash"])
        blocks = [b for b in receipt["documents"]["protocol"]["blocks"] if b["block_id"] == source_cell["block_id"]]
        _require(receipt["mode"] == "pilot" and len(blocks) == 1 and blocks[0]["split_role"] in {"train", "validation"},
            "source must never consume held-out target")
        _functional_values(result["forecast"]["functional"], request)
        _analysis_values(result["forecast"]["cubature_qualification_analysis"], policy, package, request,
            result["forecast"]["functional"]["estimate"])
        events = store.events()
        def one(kind):
            values = [e for e in events if e["event_kind"] == kind and e["payload"].get("attempt_id") == attempt["attempt_id"]]
            _require(len(values) == 1, "source lacks unique "+kind)
            return values[0]
        reservation, worker, stop, settlement, admitted = [one(k) for k in
            ("RESERVE", "WORKER_STARTED", "WORKER_TREE_STOPPED", "SETTLE", "ADMISSION")]
        completions = [e for e in events if e["event_kind"] == "ATTEMPT" and e["payload"] == attempt]
        cost = settlement["payload"]
        _require(len(completions) == 1 and admitted["payload"]["admission_hash"] == receipt["admission_hash"]
            and admitted["sequence"] < worker["sequence"]
            and reservation["sequence"] < worker["sequence"] < stop["sequence"] < settlement["sequence"] < completions[0]["sequence"]
            and type(cost["charged_ms"]) is int and type(cost["reserved_ms"]) is int
            and 0 < cost["charged_ms"] <= cost["reserved_ms"] <= Fraction(policy.maximum_job_seconds)*1000
            and cost["settled"] is True and cost["outcome"] == "SUCCEEDED"
            and stop["payload"]["reservation_id"] == cost["reservation_id"]
            and stop["payload"]["observed_elapsed_ms"] == cost["charged_ms"], "source native settlement/order/caps differ")
        evidence = {"schema_version": "managed-affine-cubature-admission-evidence-v1", "policy": policy.manifest(),
            "source_attempt": attempt, "source_run": run, "source_cell_hash": digest(source_cell),
            "source_artifact": metadata, "source_result": result, "source_admission": receipt,
            "authorization": grant, "reservation_event": reservation, "worker_event": worker,
            "stop_event": stop, "settlement_event": settlement, "admission_event": admitted,
            "completion_event": completions[0], "target_request_hash": request.request_hash,
            "target_cell_hash": digest(cell), "target_spec_hash": digest(spec), "preregistration_hash": digest(prereg)}
        evidence["evidence_hash"] = digest(evidence)
        _require(len(encode(evidence)) <= 2*1024*1024, "owner evidence byte quota")
        store.verify_artifact_read(attempt["artifact_id"], purpose="evaluate", authorization=grant)
        return evidence
    except (KeyError, TypeError, ValueError, DataValidationError, OverflowError) as exc:
        if isinstance(exc, ResearchError):
            raise
        raise ResearchError("UNQUALIFIED", "managed cubature evidence missing or malformed") from exc


def qualified_cubature_forecast(spec, cell, package, request, functional, receipt):
    _require(receipt["spec_hash"] == digest(spec) and receipt["cell_hash"] == digest(cell)
        and receipt["mode"] == "formal" and receipt["qualification"] == "qualified", "bound formal owner receipt required")
    evidence = receipt["documents"]["propagation_qualification"]
    _require(evidence["schema_version"] == "managed-affine-cubature-admission-evidence-v1"
        and evidence["evidence_hash"] == digest({k: v for k, v in evidence.items() if k != "evidence_hash"})
        and evidence["target_spec_hash"] == digest(spec) and evidence["target_cell_hash"] == digest(cell)
        and evidence["target_request_hash"] == request.request_hash, "dedicated target evidence binding differs")
    policy = CubatureQualificationPolicy.from_manifest(evidence["policy"])
    from experiments.pirc25.affine import code_hash
    policy.validate(package, request, code_hash())
    analysis = evidence["source_result"]["forecast"]["cubature_qualification_analysis"]
    cb, tb, bias = _analysis_values(analysis, policy, package, request,
        evidence["source_result"]["forecast"]["functional"]["estimate"])
    _functional_values(functional, request)
    roundoff = max(abs(Fraction(functional["estimate"])-tb.lo), abs(Fraction(functional["estimate"])-tb.hi))
    _require(roundoff <= Fraction(policy.maximum_functional_roundoff), "current output exceeds frozen roundoff tolerance")
    components = {"reference_width_upper": _outward_float(cb.hi-cb.lo), "implementation_roundoff_upper": _outward_float(roundoff),
        "time_bias_absolute_upper": _outward_float(max(abs(bias.lo), abs(bias.hi))),
        "signed_time_bias_bounds": analysis["signed_time_bias_bounds"], "affine_finite_grid_closure_error": 0,
        "model_error": {"value": None, "status": "NOT_IDENTIFIABLE"}, "scope": SCOPE,
        "qualification_evidence_hash": evidence["evidence_hash"],
        "qualification_source_attempt_id": evidence["source_attempt"]["attempt_id"], "policy_hash": policy.policy_hash}
    return _outward_float(max(abs(Fraction(functional["estimate"])-cb.lo), abs(Fraction(functional["estimate"])-cb.hi))), components


def validate_formal_cubature_result(receipt, spec, cell, result):
    from application.propagation_execution import validate_propagation_cell
    package, request, _, _ = validate_propagation_cell(spec, cell)
    forecast = result["forecast"]
    _require(cell["plugin_id"] == "affine-cubature" and forecast["kind"] == "functional_estimate"
        and forecast["model_package_hash"] == package.package_hash and forecast["request_hash"] == request.request_hash
        and forecast["horizons"] == list(request.horizons), "current forecast differs")
    error, components = qualified_cubature_forecast(spec, cell, package, request, forecast["functional"], receipt)
    unit = "m" if request.functional == "endpoint-x" else "1"
    _require(result["metrics"] == {METRIC: error} and result["metric_units"] == {METRIC: unit}
        and forecast["qualified_error_components"] == components, "current scalar error/provenance differs")
    budget = forecast["functional"]["error_budget"]
    for name, value, status in (("reference", components["reference_width_upper"], "BOUNDED"),
        ("time_discretization", components["time_bias_absolute_upper"], "BOUNDED"),
        ("propagation_approximation", 0, "IDENTIFIED"), ("sampling", 0, "NOT_APPLICABLE")):
        _require(budget[name]["value"] == value and budget[name]["status"] == status and budget[name]["units"] == unit,
            "separated bounded/affine-only error differs")
    _require(budget["model"]["value"] is None and budget["model"]["status"] == "NOT_IDENTIFIABLE", "model error remains unknown")
