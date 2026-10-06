"""Owner admission of actual settled affine analytic qualification evidence.

No grant/store/arm creation and no uncharged exponential/certificate replay.
Trust is the current shared owner journal/source, not an imported pass flag.
"""

from datetime import datetime, timezone
from fractions import Fraction
import json
import math
from pathlib import Path
from types import SimpleNamespace

from domain.affine_qualification import AffineQualificationPolicy
from domain.errors import DataValidationError
from inference.affine_reference import AffineReferenceCertificate, _parse_interval
from inference.affine_qualification import _outward_float, _scaled_norm
from infrastructure.research_store import ResearchError, digest, encode


PAYLOAD_KEY = "managed_analytic_qualification"
METRIC = "absolute_error_upper_vs_declared_affine_law"
MAX_EVIDENCE_BYTES = 2*1024*1024


def _require(condition, detail):
    if not condition:
        raise ResearchError("UNQUALIFIED", "managed analytic qualification: " + detail)


def _analysis_values(analysis, policy, package, request, method, estimate):
    """Bounded arithmetic over saved enclosures, never matrix/CDF recomputation."""
    from inference.affine_reference import algorithm_manifest
    _require(analysis["schema_version"] == "affine-analytic-qualification-analysis-v1"
        and analysis["scientific_qualification"] is False
        and analysis["policy_hash"] == policy.policy_hash
        and analysis["code_hash"] == policy.code_hash
        and analysis["model_package_hash"] == package.package_hash
        and analysis["request_hash"] == request.request_hash and analysis["method"] == method
        and analysis["analysis_hash"] == digest({k: v for k, v in analysis.items() if k != "analysis_hash"}),
        "analysis binding differs")
    cm = analysis["continuous_certificate"]
    tm = cm if method == "exact" else analysis["target_certificate"]
    for document, schema, scope in ((cm, "affine-reference-certificate-v1", "declared-affine-Gaussian-endpoint-law"),
        (tm, "affine-reference-certificate-v1" if method == "exact" else "affine-discrete-certificate-v1",
            "declared-affine-Gaussian-endpoint-law" if method == "exact" else "declared-affine-finite-grid-Gaussian-law")):
        AffineReferenceCertificate(encode(document)).manifest()
        _require(document["schema_version"] == schema and document["scope"] == scope
            and document["scientific_qualification"] is False and document["algorithm"] == algorithm_manifest()
            and document["code_hash"] == policy.code_hash and document["request_hash"] == request.request_hash
            and document["model_package_hash"] == package.package_hash
            and type(document["operations"]) is int and 0 < document["operations"] <= 200_000,
            "component certificate binding differs")
    _require(digest(cm) == analysis["continuous_certificate_hash"]
        and digest(tm) == analysis["target_certificate_hash"], "component hash differs")
    _require(analysis["target_grid"] == (None if method == "exact" else {"solver": "euler", "steps": request.steps}),
        "target grid differs")
    if method != "exact":
        _require(tm["grid"] == {"solver": "euler", "steps": request.steps,
            "time_step": "exact-horizon-rational/steps", "noise": "independent-centered-Gaussian-increments",
            "roundoff_scope": "mathematical-recurrence-only"}, "discrete component grid differs")
    cb, tb = _parse_interval(cm["functional_bounds"]), _parse_interval(tm["functional_bounds"])
    bias = _parse_interval(analysis["signed_time_bias_bounds"])
    expected_bias = (Fraction(0), Fraction(0)) if method == "exact" else (tb.lo-cb.hi, tb.hi-cb.lo)
    _require((bias.lo, bias.hi) == expected_bias, "signed bias differs")
    width, error = cb.hi-cb.lo, max(abs(Fraction(estimate)-tb.lo), abs(Fraction(estimate)-tb.hi))
    absolute_bias = max(abs(bias.lo), abs(bias.hi))
    norm = max(_scaled_norm(cm, policy.state_scales), _scaled_norm(tm, policy.state_scales))
    operations = cm["operations"]+(tm["operations"] if method != "exact" else 0)+1
    checks = {"resolved_reference": cm["status"] == "BOUNDED" and tm["status"] == "BOUNDED",
        "reference_width": width <= Fraction(policy.maximum_reference_width),
        "functional_roundoff": error <= Fraction(policy.maximum_functional_roundoff),
        "time_bias": absolute_bias <= Fraction(policy.maximum_time_bias),
        "scaled_transition_growth": norm <= Fraction(policy.maximum_scaled_transition_norm),
        "arithmetic_operations": operations <= policy.maximum_operations}
    _require(analysis["checks"] == checks and all(type(v) is bool for v in analysis["checks"].values())
        and analysis["status"] == "PASSED" and all(checks.values()), "actual numerical checks did not pass")
    expected = {"reference_width_upper": _outward_float(width), "functional_roundoff_upper": _outward_float(error),
        "absolute_time_bias_upper": _outward_float(absolute_bias), "scaled_transition_norm_upper": _outward_float(norm),
        "operations": operations, "sampling_error": {"status": "NOT_APPLICABLE", "value": 0},
        "model_error": {"status": "NOT_IDENTIFIABLE", "value": None},
        "cost_status": "OWNER_SETTLEMENT_REQUIRED", "maximum_job_seconds": policy.maximum_job_seconds}
    _require(all(analysis.get(k) == v for k, v in expected.items()), "analysis numerical value differs")
    return cb, tb, bias


def prepare_managed_qualification(store, spec, cell, execution_package, prereg):
    from application.propagation_execution import validate_propagation_cell
    from application.research_reuse import verified_reuse
    from experiments.pirc25.affine import code_hash
    from experiments.pirc27.qualification_plugin import qualification_plugin
    try:
        package, request, config, _ = validate_propagation_cell(spec, cell)
        method = config["method"]
        _require(cell["plugin_id"] == "affine-propagation" and method in {"exact", "gaussian"},
            "this analytic proof cannot qualify another adapter or method")
        pointer = execution_package["payload"][PAYLOAD_KEY]
        _require(type(pointer) is dict and set(pointer) == {"schema_version", "policy", "source_attempt_id",
            "source_artifact_id", "source_authorization_id", "source_authorization_version"}
            and pointer["schema_version"] == "managed-affine-analytic-qualification-v1", "explicit source pointer required")
        policy = AffineQualificationPolicy.from_manifest(pointer["policy"])
        policy.validate(package, request, method, code_hash())
        policies = prereg.get("propagation_qualification_policies")
        _require(type(policies) is list and 0 < len(policies) <= 10000
            and policies.count(policy.manifest()) == 1 and len({digest(p) for p in policies}) == len(policies)
            and prereg["primary_metrics"] == [METRIC], "policy/metric not frozen in preregistration")
        attempt = store.attempts()[pointer["source_attempt_id"]]
        _require(attempt["state"] == "SUCCEEDED" and attempt["artifact_id"] == pointer["source_artifact_id"],
            "source attempt/artifact is not completed")
        run = store.manifest("run-"+attempt["run_id"])
        source_spec = store.manifest("study-"+run["study_id"])["spec"]
        source_cell = run["cell"]
        _require(source_spec["runtime_binding"] == spec["runtime_binding"]
            and source_spec["code_hash"] == spec["code_hash"]
            and source_cell["plugin_id"] == "affine-propagation-qualification"
            and source_cell["execution_role"] == "qualification"
            and source_cell["frozen_dynamics"] == cell["frozen_dynamics"]
            and source_cell["propagation_request"] == cell["propagation_request"]
            and source_cell["affine_qualification_policy"] == policy.manifest()
            and source_cell["arm_id"] == cell["arm_id"], "source model/request/arm/policy/root differs")
        # Same family and same sole cumulative arm, not a new qualification arm.
        _require({a["arm_id"]: a for a in source_spec["arms"]}[cell["arm_id"]]
            == {a["arm_id"]: a for a in spec["arms"]}[cell["arm_id"]], "source arm family differs")
        grant = store.authorization(pointer["source_authorization_id"], version=pointer["source_authorization_version"])
        _require(grant["study_id"] == source_spec["study_id"] and grant["protocol_hash"] == source_spec["protocol_hash"]
            and (grant["study_id"] == spec["study_id"] or spec["study_id"] in grant.get("consumer_study_ids", []))
            and "evaluate" in grant["purposes"]
            and datetime.fromisoformat(grant["expires_at"]) > datetime.now(timezone.utc), "source grant does not authorize consumer")
        metadata = store.manifest("artifact-"+attempt["artifact_id"])
        _require(metadata["size_bytes"] <= 512*1024, "source result exceeds its fixed quota")
        content = store.read_artifact(attempt["artifact_id"], purpose="evaluate", authorization=grant)
        result = json.loads(content)
        _require(content == encode(result), "source result is not canonical")
        # Revalidate actual owner artifact, admission, execution and settlement.
        # This does not launch or charge another worker.
        verified_reuse(store, attempt, source_spec, source_cell, qualification_plugin())
        receipt = store.manifest("admission-"+result["admission_hash"])
        blocks = [b for b in receipt["documents"]["protocol"]["blocks"] if b["block_id"] == source_cell["block_id"]]
        _require(receipt["mode"] == "pilot" and len(blocks) == 1 and blocks[0]["split_role"] in {"train", "validation"},
            "source qualification consumed held-out input")
        analysis = result["forecast"]["qualification_analysis"]
        _analysis_values(analysis, policy, package, request, method, result["forecast"]["functional"]["estimate"])
        events = store.events()
        def one(kind):
            matches = [e for e in events if e["event_kind"] == kind and e["payload"].get("attempt_id") == attempt["attempt_id"]]
            _require(len(matches) == 1, "source lacks unique "+kind)
            return matches[0]
        reservation, worker, stop, settlement = (one(k) for k in ("RESERVE", "WORKER_STARTED", "WORKER_TREE_STOPPED", "SETTLE"))
        cost = settlement["payload"]
        _require(type(cost["charged_ms"]) is int and 0 < cost["charged_ms"] <= cost["reserved_ms"]
            <= Fraction(policy.maximum_job_seconds)*1000
            and stop["payload"]["reservation_id"] == cost["reservation_id"]
            and stop["payload"]["observed_elapsed_ms"] == cost["charged_ms"]
            and worker["sequence"] < stop["sequence"] < settlement["sequence"], "source measured cost/stop violates policy")
        evidence = {"schema_version": "managed-affine-analytic-admission-evidence-v1",
            "policy": policy.manifest(), "source_attempt": attempt, "source_run": run,
            "source_artifact": metadata, "source_result": result, "source_admission": receipt,
            "authorization": grant, "reservation_event": reservation, "worker_event": worker,
            "stop_event": stop, "settlement_event": settlement,
            "target_request_hash": request.request_hash, "target_cell_hash": digest(cell),
            "target_spec_hash": digest(spec), "preregistration_hash": digest(prereg)}
        evidence["evidence_hash"] = digest(evidence)
        _require(len(encode(evidence)) <= MAX_EVIDENCE_BYTES, "source evidence exceeds fixed byte quota")
        store.verify_artifact_read(attempt["artifact_id"], purpose="evaluate", authorization=grant)
        return evidence
    except (KeyError, TypeError, ValueError, DataValidationError) as exc:
        if isinstance(exc, ResearchError):
            raise
        raise ResearchError("UNQUALIFIED", "managed analytic qualification evidence is missing or malformed") from exc


def load_worker_admission(store, reference, spec, cell, output):
    receipt = store.manifest("admission-"+reference)
    _require(receipt.get("admission_hash") == reference
        and digest({k: v for k, v in receipt.items() if k != "admission_hash"}) == reference
        and receipt["spec"] == spec and receipt["cell"] == cell
        and receipt["spec_hash"] == digest(spec) and receipt["cell_hash"] == digest(cell)
        and receipt["mode"] == "formal" and receipt["qualification"] == "qualified"
        and receipt["documents"].get("propagation_qualification"), "worker admission source differs")
    attempt = store.attempts()[receipt["attempt_id"]]
    expected_output = store.path/"artifacts"/(".attempt-"+receipt["attempt_id"])/"result.json"
    _require(attempt["state"] == "RUNNING" and attempt["run_id"] == receipt["run_id"]
        and Path(output).resolve() == expected_output.resolve()
        and any(e["event_kind"] == "ADMISSION" and e["payload"] == {"attempt_id": receipt["attempt_id"],
            "run_id": receipt["run_id"], "admission_hash": reference} for e in store.events()),
        "worker admission is not the current owner attempt")
    return receipt


def qualified_analytic_forecast(spec, cell, package, request, method, result, receipt):
    """Current scalar output is checked again; no interval-matrix work here."""
    _require(receipt["spec_hash"] == digest(spec) and receipt["cell_hash"] == digest(cell)
        and receipt["mode"] == "formal" and receipt["qualification"] == "qualified",
        "formal result requires its owner receipt")
    evidence = receipt["documents"]["propagation_qualification"]
    _require(evidence["evidence_hash"] == digest({k: v for k, v in evidence.items() if k != "evidence_hash"})
        and evidence["target_spec_hash"] == digest(spec) and evidence["target_cell_hash"] == digest(cell),
        "formal result qualification evidence differs")
    policy = AffineQualificationPolicy.from_manifest(evidence["policy"])
    from experiments.pirc25.affine import code_hash
    policy.validate(package, request, method, code_hash())
    analysis = evidence["source_result"]["forecast"]["qualification_analysis"]
    original = evidence["source_result"]["forecast"]["functional"]["estimate"]
    continuous, target, bias = _analysis_values(analysis, policy, package, request, method, original)
    current_roundoff = max(abs(Fraction(result.estimate)-target.lo), abs(Fraction(result.estimate)-target.hi))
    _require(current_roundoff <= Fraction(policy.maximum_functional_roundoff), "current analytic output exceeds frozen roundoff tolerance")
    error_upper = _outward_float(max(abs(Fraction(result.estimate)-continuous.lo), abs(Fraction(result.estimate)-continuous.hi)))
    components = {"reference_width_upper": _outward_float(continuous.hi-continuous.lo),
        "implementation_roundoff_upper": _outward_float(current_roundoff),
        "time_bias_absolute_upper": _outward_float(max(abs(bias.lo), abs(bias.hi))),
        "signed_time_bias_bounds": analysis["signed_time_bias_bounds"],
        "model_error": {"value": None, "status": "NOT_IDENTIFIABLE"},
        "scope": "declared-affine-law-only", "qualification_evidence_hash": evidence["evidence_hash"],
        "qualification_source_attempt_id": evidence["source_attempt"]["attempt_id"], "policy_hash": policy.policy_hash}
    return error_upper, components


def validate_formal_analytic_result(receipt, spec, cell, result):
    from application.propagation_execution import validate_propagation_cell
    package, request, config, _ = validate_propagation_cell(spec, cell)
    functional = result["forecast"]["functional"]
    estimate = functional["estimate"]
    _require(type(estimate) in (float, int) and math.isfinite(estimate)
        and functional["kind"] == "analytic" and functional["request_hash"] == request.request_hash
        and functional["estimator_id"] == ("affine-exact-v1" if config["method"] == "exact" else "gaussian-discrete-v1")
        and functional["sample_count"] == 0 and functional["standard_error"] == 0
        and result["forecast"]["model_package_hash"] == package.package_hash
        and result["forecast"]["request_hash"] == request.request_hash, "formal analytic output schema differs")
    error, components = qualified_analytic_forecast(spec, cell, package, request,
        config["method"], SimpleNamespace(estimate=estimate), receipt)
    _require(result["metrics"] == {METRIC: error}
        and result["forecast"].get("qualified_error_components") == components,
        "formal scalar error/qualification provenance differs")
    budget = functional["error_budget"]
    _require(budget["reference"]["value"] == components["reference_width_upper"]
        and budget["reference"]["status"] == "BOUNDED"
        and budget["time_discretization"]["value"] == components["time_bias_absolute_upper"]
        and budget["time_discretization"]["status"] == "BOUNDED"
        and budget["model"]["value"] is None and budget["model"]["status"] == "NOT_IDENTIFIABLE",
        "formal separated error components differ")
