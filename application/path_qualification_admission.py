"""Settled own-path source admission and honest independent target evidence.

Saved bounded arithmetic only in the owner gate. A passed source does not
certify the target's realized sampling; failed target checks remain artifacts.
No new budget/grant authority, numerical replay or Gaussian sampler approval.
"""

from datetime import datetime, timezone
from fractions import Fraction
import json
from types import SimpleNamespace

from domain.errors import DataValidationError
from domain.mixture import finite
from domain.path_qualification import PathQualificationPolicy
from inference.affine_reference import AffineReferenceCertificate, _parse_interval, algorithm_manifest
from inference.path_qualification import saved_path_analysis
from infrastructure.research_store import ResearchError, digest, encode


PAYLOAD_KEY = "managed_path_qualification"
METRIC = "observed_absolute_error_upper_vs_declared_affine_law"
MAX_EVIDENCE_BYTES = 2*1024*1024


def require(condition, detail):
    if not condition:
        raise ResearchError("UNQUALIFIED", "managed path qualification: "+detail)


def functional_object(value, request):
    keys = {"request_hash", "estimator_id", "kind", "estimate", "standard_error", "interval", "interval_kind",
        "sample_count", "error_budget", "status", "diagnostics"}
    require(type(value) is dict and set(value) == keys and type(value["estimate"]) is float
        and finite(value["estimate"]) and type(value["sample_count"]) is int
        and value["sample_count"] == request.samples and value["request_hash"] == request.request_hash,
        "actual functional request/count/finite scalar differs")
    se, interval, diagnostics = value["standard_error"], value["interval"], value["diagnostics"]
    require((se is None or finite(se) and se >= 0)
        and (interval is None or type(interval) in (list, tuple) and len(interval) == 2
            and all(finite(v) for v in interval) and interval[0] <= interval[1])
        and type(diagnostics) in (list, tuple) and 0 < len(diagnostics) <= 8,
        "bounded actual sampling fields required")
    for pair in diagnostics:
        require(type(pair) in (list, tuple) and len(pair) == 2
            and type(pair[0]) is str and 0 < len(pair[0]) <= 64, "bounded unique diagnostic keys required")
        item = pair[1]
        require(item is None or type(item) is bool or finite(item)
            or type(item) is str and len(item) <= 256
            or type(item) in (list, tuple) and len(item) == 2 and all(finite(v) for v in item),
            "bounded primitive diagnostic value required")
    require(len(dict(diagnostics)) == len(diagnostics), "duplicate diagnostic keys")
    unit = "m" if request.functional == "endpoint-x" else "1"
    budget = value["error_budget"]
    require(type(budget) is dict and set(budget) == {"reference", "time_discretization",
        "propagation_approximation", "sampling", "model"}, "complete kernel error budget required")
    for component in budget.values():
        require(type(component) is dict and set(component) == {"value", "units", "estimated_by", "status"}
            and (component["value"] is None or finite(component["value"])) and component["units"] == unit
            and type(component["estimated_by"]) is str and len(component["estimated_by"]) <= 256
            and type(component["status"]) is str and len(component["status"]) <= 64,
            "kernel component primitive/unit schema differs")
    expected = {
        "reference": {"value": None, "units": unit, "estimated_by": "float64 analytic reference; no certified roundoff bound", "status": "NOT_IDENTIFIABLE"},
        "propagation_approximation": {"value": 0., "units": unit, "estimated_by": "constant affine Gaussian model; no nonlinear closure", "status": "IDENTIFIED"},
        "sampling": {"value": se, "units": unit, "estimated_by": "estimator standard error", "status": "ESTIMATED" if se is not None else "NOT_IDENTIFIABLE"},
        "model": {"value": None, "units": unit, "estimated_by": "no observed real dynamics in synthetic recipe", "status": "NOT_IDENTIFIABLE"}}
    require(all(encode(budget[k]) == encode(v) for k, v in expected.items())
        and finite(budget["time_discretization"]["value"])
        and budget["time_discretization"]["status"] == "IDENTIFIED"
        and budget["time_discretization"]["estimated_by"] == "discrete affine Gaussian expectation minus continuous reference",
        "kernel unknowns/sampling/time definitions differ")
    result = SimpleNamespace(**value)
    result.interval = None if interval is None else tuple(interval)
    result.manifest = lambda: value
    return result


def verify_certificate(document, policy, package, request, *, discrete):
    AffineReferenceCertificate(encode(document)).manifest()
    require(document["schema_version"] == ("affine-discrete-certificate-v1" if discrete else "affine-reference-certificate-v1")
        and document["scope"] == ("declared-affine-finite-grid-Gaussian-law" if discrete else "declared-affine-Gaussian-endpoint-law")
        and document["scientific_qualification"] is False and document["algorithm"] == algorithm_manifest()
        and document["code_hash"] == policy.code_hash and document["request_hash"] == request.request_hash
        and document["model_package_hash"] == package.package_hash
        and type(document["operations"]) is int and 0 < document["operations"] <= 200_000
        and document["units"] == ("m" if request.functional == "endpoint-x" else "1"),
        "saved certificate law/source/units differ")
    dimension = 8 if discrete and policy.method == "reversible-heun" else 4
    def vector(values, size):
        require(type(values) is list and len(values) == size, "reference vector dimension differs")
        for value in values:
            _parse_interval(value)
    def matrix(values, size):
        require(type(values) is list and len(values) == size, "reference matrix dimension differs")
        for row in values:
            vector(row, size)
    transition = document["transition_bounds"]
    require(type(transition) is dict and set(transition) == {"F", "offset", "covariance"}, "reference transition fields differ")
    matrix(transition["F"], dimension)
    matrix(transition["covariance"], dimension)
    vector(transition["offset"], dimension)
    vector(document["mean_bounds"], 4)
    matrix(document["covariance_bounds"], 4)
    _parse_interval(document["functional_bounds"])
    if discrete:
        solver = "euler" if policy.method == "importance" else policy.method
        require(document["grid"] == {"solver": solver, "steps": request.steps,
            "time_step": "exact-horizon-rational/steps", "noise": "independent-centered-Gaussian-increments",
            "roundoff_scope": "mathematical-recurrence-only"}
            and type(document["physical_dimension"]) is int and document["physical_dimension"] == 4
            and type(document["auxiliary_dimension"]) is int and document["auxiliary_dimension"] == dimension-4
            and type(document["compositions"]) is int and 0 < document["compositions"] <= 26,
            "saved finite grid/physical-versus-auxiliary dimension differs")


def analysis_values(analysis, policy, package, request, functional):
    """Recalculate every saved scalar/statistic meaning, not numerical engines."""
    policy.validate(package, request, policy.code_hash)
    result = functional_object(functional, request)
    cm, tm = analysis["continuous_certificate"], analysis["target_certificate"]
    verify_certificate(cm, policy, package, request, discrete=False)
    verify_certificate(tm, policy, package, request, discrete=True)
    expected = saved_path_analysis(package, request, policy, result, analysis["completed_statistics"], cm, tm)
    require(encode(analysis) == encode(expected) and analysis["status"] == "PASSED",
        "actual source numerical/statistical checks or saved meanings differ")
    return cm, tm


def prepare_managed_paths(store, spec, cell, execution_package, prereg):
    from .propagation_execution import validate_propagation_cell, request_from_manifest
    from .research_reuse import verified_reuse
    from experiments.pirc25.affine import code_hash
    from experiments.pirc27.path_production_plugin import production_policies
    from experiments.pirc27.path_qualification_plugin import path_qualification_plugin
    try:
        package, request, config, _ = validate_propagation_cell(spec, cell)
        qualification, policy = production_policies(spec, cell, package, request, config)
        pointer = execution_package["payload"][PAYLOAD_KEY]
        require(type(pointer) is dict and set(pointer) == {"schema_version", "policy", "qualification_policy",
            "source_attempt_id", "source_artifact_id", "source_authorization_id", "source_authorization_version"}
            and pointer["schema_version"] == "managed-affine-path-qualification-v1"
            and pointer["policy"] == policy.manifest() and pointer["qualification_policy"] == qualification.manifest(),
            "explicit bound independent target/source pointer required")
        for key, frozen in (("path_production_policies", policy.manifest()),
                            ("path_qualification_policies", qualification.manifest())):
            policies = prereg.get(key)
            require(type(policies) is list and 0 < len(policies) <= 10000
                and policies.count(frozen) == 1 and len({digest(p) for p in policies}) == len(policies),
                "source/independent target policies not uniquely frozen")
        require(prereg["primary_metrics"] == [METRIC], "observed metric not frozen before held-out input")
        attempt = store.attempts()[pointer["source_attempt_id"]]
        require(attempt["state"] == "SUCCEEDED" and attempt["artifact_id"] == pointer["source_artifact_id"], "source not completed")
        run = store.manifest("run-"+attempt["run_id"])
        source_spec = store.manifest("study-"+run["study_id"])["spec"]
        sc = run["cell"]
        source_request = request_from_manifest(sc["propagation_request"])
        require(source_spec["runtime_binding"] == spec["runtime_binding"]
            and source_spec["code_hash"] == spec["code_hash"] == code_hash()
            and sc["plugin_id"] == "affine-path-qualification" and sc["execution_role"] == "qualification"
            and sc["frozen_dynamics"] == cell["frozen_dynamics"]
            and sc["path_qualification_policy"] == qualification.manifest()
            and sc["arm_id"] == cell["arm_id"], "actual own source producer/model/policy/root/arm differs")
        policy.validate_source_request(package, source_request, request, qualification, code_hash())
        require({a["arm_id"]: a for a in source_spec["arms"]}[cell["arm_id"]]
            == {a["arm_id"]: a for a in spec["arms"]}[cell["arm_id"]], "original cumulative family arm differs")
        grant = store.authorization(pointer["source_authorization_id"], version=pointer["source_authorization_version"])
        require(grant["study_id"] == source_spec["study_id"] and grant["protocol_hash"] == source_spec["protocol_hash"]
            and (grant["study_id"] == spec["study_id"] or spec["study_id"] in grant.get("consumer_study_ids", []))
            and "evaluate" in grant["purposes"] and datetime.fromisoformat(grant["expires_at"]) > datetime.now(timezone.utc),
            "source grant does not authorize consumer")
        metadata = store.manifest("artifact-"+attempt["artifact_id"])
        require(metadata["size_bytes"] <= 512*1024, "source result byte quota")
        result = json.loads(store.read_artifact(attempt["artifact_id"], purpose="evaluate", authorization=grant))
        verified_reuse(store, attempt, source_spec, sc, path_qualification_plugin())
        receipt = store.manifest("admission-"+result["admission_hash"])
        blocks = [b for b in receipt["documents"]["protocol"]["blocks"] if b["block_id"] == sc["block_id"]]
        require(receipt["mode"] == "pilot" and len(blocks) == 1 and blocks[0]["split_role"] in {"train", "validation"},
            "source consumed held-out input")
        analysis_values(result["forecast"]["path_qualification_analysis"], qualification, package, source_request,
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
            and type(cost["charged_ms"]) is int and 0 < cost["charged_ms"] <= cost["reserved_ms"] <= Fraction(qualification.maximum_job_seconds)*1000
            and stop["payload"]["reservation_id"] == cost["reservation_id"]
            and stop["payload"]["observed_elapsed_ms"] == cost["charged_ms"]
            and stop["payload"]["confirmation"] == "native-job-or-process-group-no-running-descendants", "native source stop/cost/order differs")
        evidence = {"schema_version": "managed-affine-independent-path-admission-evidence-v1",
            "policy": policy.manifest(), "qualification_policy": qualification.manifest(),
            "source_attempt": attempt, "source_run": run, "source_artifact": metadata,
            "source_result": result, "source_admission": receipt, "authorization": grant,
            "reservation_event": reservation, "worker_event": worker, "stop_event": stop, "settlement_event": settlement,
            "admission_event": admission_event, "completion_event": completions[0], "target_request_hash": request.request_hash,
            "target_cell_hash": digest(cell), "target_spec_hash": digest(spec), "preregistration_hash": digest(prereg)}
        evidence["evidence_hash"] = digest(evidence)
        require(len(encode(evidence)) <= MAX_EVIDENCE_BYTES, "saved owner evidence byte quota")
        store.verify_artifact_read(attempt["artifact_id"], purpose="evaluate", authorization=grant)
        return evidence
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError, RecursionError, DataValidationError) as exc:
        if isinstance(exc, ResearchError):
            raise
        raise ResearchError("UNQUALIFIED", "managed path evidence is missing or malformed") from exc


def target_output_analysis(spec, cell, package, request, functional, last, receipt):
    from .propagation_execution import request_from_manifest
    from experiments.pirc27.path_production_plugin import production_policies
    require(receipt["spec_hash"] == digest(spec) and receipt["cell_hash"] == digest(cell)
        and receipt["mode"] == "formal" and receipt["qualification"] == "qualified", "current formal owner receipt required")
    evidence = receipt["documents"]["propagation_qualification"]
    require(evidence["schema_version"] == "managed-affine-independent-path-admission-evidence-v1"
        and evidence["evidence_hash"] == digest({k: v for k, v in evidence.items() if k != "evidence_hash"})
        and evidence["target_spec_hash"] == digest(spec) and evidence["target_cell_hash"] == digest(cell)
        and evidence["target_request_hash"] == request.request_hash, "current target evidence differs")
    qualification, policy = production_policies(spec, cell, package, request, cell["execution"]["config"])
    require(evidence["policy"] == policy.manifest() and evidence["qualification_policy"] == qualification.manifest(), "current policies differ")
    source_request = request_from_manifest(evidence["source_run"]["cell"]["propagation_request"])
    policy.validate_source_request(package, source_request, request, qualification, policy.code_hash)
    source_forecast = evidence["source_result"]["forecast"]
    cm, tm = analysis_values(source_forecast["path_qualification_analysis"], qualification, package, source_request,
        source_forecast["functional"])
    current = saved_path_analysis(package, request, qualification, functional_object(functional, request), last, cm, tm)
    current.update(schema_version="affine-independent-path-output-analysis-v1",
        scope="one-independent-realized-target-with-owner-admitted-saved-affine-law-reference",
        production_policy_hash=policy.policy_hash, reference_request_hash=source_request.request_hash,
        reference_recomputation="NONE; bounded saved same-law proof reused",
        reference_operations_scope="settled-source-only", maximum_job_seconds=policy.maximum_job_seconds,
        qualification_evidence_hash=evidence["evidence_hash"],
        qualification_source_attempt_id=evidence["source_attempt"]["attempt_id"],
        kernel_error_budget=functional["error_budget"])
    current["analysis_hash"] = digest({k: v for k, v in current.items() if k != "analysis_hash"})
    return current


def validate_formal_path_result(receipt, spec, cell, result):
    from .propagation_execution import validate_propagation_cell
    package, request, _, _ = validate_propagation_cell(spec, cell)
    forecast = result["forecast"]
    current = forecast["path_output_analysis"]
    functional = {**forecast["functional"], "error_budget": current["kernel_error_budget"]}
    expected = target_output_analysis(spec, cell, package, request, functional, current["completed_statistics"], receipt)
    require(encode(current) == encode(expected)
        and forecast["current_output_qualification"] == expected["status"]
        and forecast["model_package_hash"] == package.package_hash and forecast["request_hash"] == request.request_hash,
        "current target statistics/full output/checks/classification differ")
    unit = "m" if request.functional == "endpoint-x" else "1"
    require(result["metrics"] == {METRIC: expected["total_observed_functional_error_upper"]}
        and result["metric_units"] == {METRIC: unit}, "actual observed target metric/unit differs")
    budget = forecast["functional"]["error_budget"]
    reference = {"value": expected["reference_width_upper"], "units": unit,
        "estimated_by": "outward max of saved continuous and method-specific grid functional interval widths", "status": "BOUNDED"}
    time = {"value": expected["absolute_time_bias_upper"], "units": unit,
        "estimated_by": "outward absolute signed grid-minus-continuous expectation bound", "status": "BOUNDED"}
    require(encode(budget) == encode({**current["kernel_error_budget"], "reference": reference, "time_discretization": time}),
        "qualified separated component values/units/definitions differ")
