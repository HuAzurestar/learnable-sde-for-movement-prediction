"""Settled affine geometry preparation, never method or formal admission.

The owner checks bounded saved scalars/intervals and original native provenance.
Matrix exponentials, CDFs, square roots and inverse-CDF searches stay entirely
in the charged worker. A source read grant does not grant a target data read.
"""

from dataclasses import replace
from datetime import datetime, timezone
from fractions import Fraction
import json
import math
from pathlib import Path

from domain.errors import DataValidationError
from domain.probability_calibration import AffineHalfspaceCalibrationPolicy, halfspace_geometry
from inference.affine_probability_calibration import QUANTILE_BISECTIONS, _outward_float
from inference.affine_reference import (AffineReferenceCertificate, GRID, MAX_BYTES,
    MAX_DOUBLINGS, _checked, _parse_interval, algorithm_manifest)
from infrastructure.research_store import ResearchError, digest, encode, identifier


def _require(condition, detail):
    if not condition:
        raise ResearchError("UNQUALIFIED", "probability calibration owner: "+detail)


def _bounded(value, *, nodes=6000, depth_limit=12, string_limit=1235):
    """Bound caller structure before hashing/copying, including cycles."""
    def visit(item, depth):
        nonlocal nodes
        nodes -= 1
        _require(nodes >= 0 and depth <= depth_limit, "saved structure exceeds fixed quota")
        if type(item) is dict:
            _require(len(item) <= nodes and all(type(k) is str and len(k) <= 128 for k in item), "bounded object keys required")
            for child in item.values():
                visit(child, depth+1)
        elif type(item) is list:
            _require(len(item) <= nodes, "bounded array required")
            for child in item:
                visit(child, depth+1)
        elif type(item) is str:
            _require(len(item) <= string_limit, "bounded saved string required")
        elif type(item) is int:
            _require(item.bit_length() <= 4096, "bounded saved integer required")
        elif type(item) is float:
            _require(math.isfinite(item), "finite saved scalar required")
        else:
            _require(item is None or type(item) is bool, "JSON scalar required")
    visit(value, 0)


def _certificate(value, package, request, code_hash, units):
    document = AffineReferenceCertificate(encode(value)).manifest()
    _require(set(document) == {"schema_version", "algorithm", "code_hash", "model_package_hash", "request_hash",
        "scope", "scientific_qualification", "status", "doublings", "operations", "units",
        "functional_bounds", "mean_bounds", "covariance_bounds", "transition_bounds"}
        and document["schema_version"] == "affine-reference-certificate-v1"
        and encode(document["algorithm"]) == encode(algorithm_manifest())
        and document["scope"] == "declared-affine-Gaussian-endpoint-law"
        and document["scientific_qualification"] is False and document["status"] == "BOUNDED"
        and document["code_hash"] == code_hash and document["model_package_hash"] == package.package_hash
        and document["request_hash"] == request.request_hash and document["units"] == units
        and type(document["operations"]) is int and 0 < document["operations"] <= 200000
        and type(document["doublings"]) is int and 0 <= document["doublings"] <= MAX_DOUBLINGS,
        "saved reference identity/quota differs")
    def vector(values):
        _require(type(values) is list and len(values) == 4, "four-state reference vector required")
        for interval in values:
            _parse_interval(interval)
    def matrix(values):
        _require(type(values) is list and len(values) == 4, "four-state reference matrix required")
        for row in values:
            vector(row)
    vector(document["mean_bounds"])
    matrix(document["covariance_bounds"])
    _require(set(document["transition_bounds"]) == {"F", "offset", "covariance"}, "complete reference transition required")
    matrix(document["transition_bounds"]["F"])
    vector(document["transition_bounds"]["offset"])
    matrix(document["transition_bounds"]["covariance"])
    _parse_interval(document["functional_bounds"])
    return document


def saved_calibration_request(analysis, policy, package, template):
    """Validate saved native analysis; never replay its numerical reference."""
    from experiments.pirc25.affine import code_hash
    try:
        _require(type(policy) is AffineHalfspaceCalibrationPolicy, "explicit frozen calibration policy required")
        policy.validate(package, template, code_hash())
        _bounded(analysis)
        _require(type(analysis) is dict and len(encode(analysis)) <= MAX_BYTES
            and analysis["schema_version"] == "affine-halfspace-calibration-analysis-v2"
            and analysis["scope"] == "declared-affine-Gaussian-spatial-endpoint-law-for-one-template"
            and analysis["analysis_hash"] == digest({k: v for k, v in analysis.items() if k != "analysis_hash"})
            and analysis["policy_hash"] == policy.policy_hash and encode(analysis["policy"]) == encode(policy.manifest())
            and analysis["code_hash"] == policy.code_hash and analysis["model_package_hash"] == package.package_hash
            and analysis["source_request_hash"] == template.request_hash
            and encode(analysis["normal"]) == encode(list(template.normal))
            and type(analysis["target_probability"]) in (int, float)
            and analysis["target_probability"] == policy.target_probability
            and analysis["threshold_units"] == "m" and analysis["scientific_qualification"] is False
            and analysis["method_qualification"] is False and analysis["admission_status"] == "NOT_ADMITTED"
            and analysis["cost_status"] == "OWNER_SETTLEMENT_REQUIRED"
            and analysis["maximum_job_seconds"] == policy.maximum_job_seconds,
            "saved analysis identity/scope/flags differ")
        _require(analysis["status"] == "PASSED", "saved numerical checks did not pass")
        _require(set(analysis) == {"schema_version", "scope", "policy_hash", "policy", "code_hash", "model_package_hash",
            "source_request_hash", "target_probability", "normal", "threshold_units", "threshold", "calibrated_request_hash",
            "scientific_qualification", "method_qualification", "cost_status", "admission_status",
            "maximum_quantile_bisections", "executed_quantile_bisections", "quantile_bounds", "quantile_endpoint_tail_bounds",
            "ideal_threshold_bounds", "normalized_ideal_threshold_bounds", "threshold_scale", "maximum_job_seconds",
            "moment_certificate", "moment_certificate_hash", "projected_mean_bounds", "normalized_projected_variance_bounds",
            "probability_certificate", "probability_certificate_hash", "relative_probability_error_upper",
            "relative_probability_width_upper", "checks", "operation_counts", "reference_arithmetic_operations", "status", "analysis_hash"},
            "complete versioned calibration analysis required")
        _require(type(analysis["threshold"]) is float and math.isfinite(analysis["threshold"]), "finite actual float threshold required")
        actual = replace(template, threshold=analysis["threshold"])
        _require(analysis["calibrated_request_hash"] == actual.request_hash, "actual threshold request differs")
        cm = _certificate(analysis["moment_certificate"], package, replace(template, functional="endpoint-x"), policy.code_hash, "m")
        pm = _certificate(analysis["probability_certificate"], package, actual, policy.code_hash, "1")
        _require(analysis["moment_certificate_hash"] == digest(cm)
            and analysis["probability_certificate_hash"] == digest(pm)
            and cm["functional_bounds"] == cm["mean_bounds"][0]
            and all(cm[k] == pm[k] for k in ("mean_bounds", "covariance_bounds", "transition_bounds", "doublings")),
            "component hash/moment law differs")
        counts = analysis["operation_counts"]
        _require(type(counts) is dict and set(counts) == {"moments", "projection_and_quantile", "actual_threshold_reference"}
            and all(type(v) is int and v > 0 for v in counts.values())
            and counts["moments"] == cm["operations"] and counts["actual_threshold_reference"] == pm["operations"]
            and type(analysis["reference_arithmetic_operations"]) is int
            and sum(counts.values()) == analysis["reference_arithmetic_operations"] <= policy.maximum_operations,
            "original shared reference operation counts differ")
        _require(type(analysis["maximum_quantile_bisections"]) is int
            and type(analysis["executed_quantile_bisections"]) is int
            and analysis["maximum_quantile_bisections"] == analysis["executed_quantile_bisections"] == QUANTILE_BISECTIONS,
            "fixed quantile search differs")
        quantile = _parse_interval(analysis["quantile_bounds"])
        tails = analysis["quantile_endpoint_tail_bounds"]
        _require(type(tails) is list and len(tails) == 2, "two saved quantile endpoints required")
        left, right = map(_parse_interval, tails)
        target = Fraction(policy.target_probability)
        _require(0 <= quantile.lo < quantile.hi <= 8 and quantile.hi-quantile.lo == Fraction(8, 2**QUANTILE_BISECTIONS)
            and 0 <= left.lo <= left.hi <= 1 and 0 <= right.lo <= right.hi <= 1
            and left.lo >= target >= right.hi, "saved quantile bracket differs")
        normalized = _parse_interval(analysis["normalized_ideal_threshold_bounds"])
        ideal = _parse_interval(analysis["ideal_threshold_bounds"])
        scale = max(abs(Fraction(v)) for v in template.normal)
        _require(type(analysis["threshold_scale"]) in (int, float) and Fraction(analysis["threshold_scale"]) == scale,
            "exact normal scale differs")
        # The saved physical envelope uses the original dyadic enclosure of
        # the scale, whereas candidate selection scales the normalized midpoint
        # exactly. Preserve both for tiny/large normals, without recomputing CDF.
        scale_lo = Fraction((scale.numerator*GRID)//scale.denominator, GRID)
        scale_hi = Fraction(-((-scale.numerator*GRID)//scale.denominator), GRID)
        products = [_checked(x*y) for x in (normalized.lo, normalized.hi) for y in (scale_lo, scale_hi)]
        lo, hi = min(products), max(products)
        outward = (Fraction((lo.numerator*GRID)//lo.denominator, GRID),
            Fraction(-((-hi.numerator*GRID)//hi.denominator), GRID))
        _require((ideal.lo, ideal.hi) == outward
            and float(_checked((normalized.lo+normalized.hi)/2*scale)) == analysis["threshold"],
            "actual rounded threshold differs from saved normalized candidate")
        _parse_interval(analysis["projected_mean_bounds"])
        variance = _parse_interval(analysis["normalized_projected_variance_bounds"])
        bounds = _parse_interval(pm["functional_bounds"])
        _require(0 <= bounds.lo <= bounds.hi <= 1, "probability bounds outside unit interval")
        error = max(abs(bounds.lo-target), abs(bounds.hi-target))/target
        width = (bounds.hi-bounds.lo)/target
        checks = {"positive_projected_variance": variance.lo > 0, "resolved_probability": pm["status"] == "BOUNDED",
            "relative_probability_error": error <= Fraction(policy.maximum_relative_probability_error),
            "relative_probability_width": width <= Fraction(policy.maximum_relative_probability_width)}
        _require(type(analysis["checks"]) is dict and analysis["checks"] == checks
            and all(type(v) is bool for v in analysis["checks"].values()) and all(checks.values())
            and type(analysis["relative_probability_error_upper"]) is float
            and type(analysis["relative_probability_width_upper"]) is float
            and analysis["relative_probability_error_upper"] == _outward_float(error)
            and analysis["relative_probability_width_upper"] == _outward_float(width),
            "saved numerical checks/diagnostics did not pass")
        return actual
    except (KeyError, TypeError, ValueError, DataValidationError, OverflowError) as exc:
        if isinstance(exc, ResearchError):
            raise
        raise ResearchError("UNQUALIFIED", "probability calibration saved analysis malformed") from exc


def prepare_probability_calibration(store, pointer, *, consumer_study_id):
    """Consume an original settled source for preparation, not target admission.

    A consumer grant is checked before artifact disclosure and physically again
    after all extra source I/O. Returns only bounded proof for that consumer.
    Geometry ownership does not qualify a method, freeze a study or expose test.
    """
    from application.propagation_execution import validate_propagation_cell
    from application.research_reuse import verified_reuse
    from experiments.pirc25.affine import code_hash
    from experiments.pirc27.calibration_plugin import calibration_plugin, calibration_policy
    try:
        identifier(consumer_study_id)
        _require(type(pointer) is dict and set(pointer) == {"schema_version", "policy", "source_attempt_id",
            "source_artifact_id", "source_authorization_id", "source_authorization_version"}
            and pointer["schema_version"] == "managed-affine-halfspace-calibration-v1", "explicit source pointer required")
        for key in ("source_attempt_id", "source_artifact_id", "source_authorization_id"):
            identifier(pointer[key])
        if pointer["source_authorization_version"] is not None:
            identifier(pointer["source_authorization_version"])
        policy = AffineHalfspaceCalibrationPolicy.from_manifest(pointer["policy"])
        attempt = store.attempts()[pointer["source_attempt_id"]]
        _require(attempt["state"] == "SUCCEEDED" and attempt["artifact_id"] == pointer["source_artifact_id"], "source is not settled success")
        run = store.manifest("run-"+attempt["run_id"])
        spec = store.manifest("study-"+run["study_id"])["spec"]
        cell = run["cell"]
        _require(spec["runtime_binding"]["store_id"] == store.store_id
            and Path(spec["runtime_binding"]["root"]).is_absolute()
            and Path(spec["runtime_binding"]["root"]).resolve() == store.path.parent
            and spec["code_hash"] == code_hash() and cell["plugin_id"] == "affine-halfspace-calibration"
            and cell["visibility"] == "synthetic", "source code/root/explicit synthetic adapter differs")
        package, template, config, _ = validate_propagation_cell(spec, cell)
        _require(calibration_policy(spec, cell, package, template, config) == policy, "source frozen policy differs")
        events = store.events()
        def one(kind):
            matches = [e for e in events if e["event_kind"] == kind and e["payload"].get("attempt_id") == attempt["attempt_id"]]
            _require(len(matches) == 1, "unique source "+kind+" required")
            return matches[0]
        admitted = one("ADMISSION")
        receipt = store.manifest("admission-"+admitted["payload"]["admission_hash"])
        blocks = [b for b in receipt["documents"]["protocol"]["blocks"] if b["block_id"] == cell["block_id"]]
        _require(receipt["mode"] == "pilot" and receipt["spec"] == spec and receipt["cell"] == cell
            and len(blocks) == 1 and blocks[0]["split_role"] in {"train", "validation"}, "source held-out/receipt identity differs")
        binding = cell.get("input_binding")
        if binding is not None:
            from experiments.pirc27.input_cases import validate_cell_input_binding
            _require(binding["case"]["source_kind"] == "synthetic-recipe", "causal real-prefix preparation not implemented")
            validate_cell_input_binding(cell, request=template, protocol_block=blocks[0], selection_hash=spec["selection_hash"])
            causal_input_hash = binding["binding_hash"]
        else:
            causal_input_hash = digest({"schema_version": "declared-affine-initial-law-v1",
                "model_package_hash": package.package_hash, "initial_mean": list(template.initial_mean),
                "initial_covariance": [list(row) for row in template.initial_covariance],
                "origin": template.origin, "history_cutoff": template.history_cutoff,
                "provenance": "fixed-synthetic-law-not-source-independence"})
        grant = store.authorization(pointer["source_authorization_id"], version=pointer["source_authorization_version"])
        _require(grant["authorization_id"] == pointer["source_authorization_id"]
            and grant.get("version") == pointer["source_authorization_version"]
            and grant["study_id"] == spec["study_id"] and grant["protocol_hash"] == spec["protocol_hash"]
            and (consumer_study_id == spec["study_id"] or consumer_study_id in grant.get("consumer_study_ids", []))
            and "evaluate" in grant["purposes"]
            and datetime.fromisoformat(grant["expires_at"]) > datetime.now(timezone.utc), "current source consumer permission differs")
        metadata = store.manifest("artifact-"+attempt["artifact_id"])
        _require(type(metadata["size_bytes"]) is int and 0 < metadata["size_bytes"] <= 512*1024, "source result byte quota")
        content = store.read_artifact(attempt["artifact_id"], purpose="evaluate", authorization=grant)
        _require(len(content) <= 512*1024, "source read byte quota")
        result = json.loads(content)
        _bounded(result)
        _require(content == encode(result) and result["admission_hash"] == receipt["admission_hash"]
            and result["qualification"] == "fixture" and result["source_schema"] == "affine-halfspace-calibration-source-v1",
            "canonical fixture source result required")
        verified_reuse(store, attempt, spec, cell, calibration_plugin())
        forecast = result["forecast"]
        _require(forecast["kind"] == "probability-calibration" and "functional" not in forecast
            and forecast["model_package_hash"] == package.package_hash
            and forecast["source_request_hash"] == template.request_hash and forecast["horizons"] == list(template.horizons),
            "source forecast differs")
        actual = saved_calibration_request(forecast["probability_calibration_analysis"], policy, package, template)
        _require(result["metrics"] == {"reference_arithmetic_operations": forecast["probability_calibration_analysis"]["reference_arithmetic_operations"]}
            and result["metric_units"] == {"reference_arithmetic_operations": "operations"}, "source work metric differs")
        # Re-read the original owner events after reuse verification; no derived
        # local cache, synthetic stop, free reference replay or new budget arm.
        events = store.events()
        reservation, worker, stop, settlement = [one(k) for k in ("RESERVE", "WORKER_STARTED", "WORKER_TREE_STOPPED", "SETTLE")]
        completions = [e for e in events if e["event_kind"] == "ATTEMPT" and e["payload"] == attempt]
        cost = settlement["payload"]
        _require(len(completions) == 1 and admitted["sequence"] < worker["sequence"]
            and reservation["sequence"] < worker["sequence"] < stop["sequence"] < settlement["sequence"] < completions[0]["sequence"]
            and type(cost["charged_ms"]) is int and type(cost["reserved_ms"]) is int
            and 0 < cost["charged_ms"] <= cost["reserved_ms"] <= Fraction(policy.maximum_job_seconds)*1000
            and cost["settled"] is True and cost["outcome"] == "SUCCEEDED"
            and stop["payload"]["reservation_id"] == cost["reservation_id"] == digest([store.store_id, attempt["attempt_id"]])
            and stop["payload"]["confirmation"] == "native-job-or-process-group-no-running-descendants"
            and type(stop["payload"]["observed_elapsed_ms"]) is int and stop["payload"]["observed_elapsed_ms"] == cost["charged_ms"],
            "original native settlement/order/caps differ")
        geometry = halfspace_geometry(package, actual, causal_input_hash=causal_input_hash)
        evidence = {"schema_version": "managed-affine-halfspace-calibration-evidence-v1",
            "consumer_study_id": consumer_study_id, "policy": policy.manifest(), "geometry": geometry,
            "geometry_hash": digest(geometry), "source_attempt": attempt, "source_run": run,
            "source_artifact": metadata, "source_result": result, "source_admission": receipt, "authorization": grant,
            "reservation_event": reservation, "worker_event": worker, "stop_event": stop,
            "settlement_event": settlement, "admission_event": admitted, "completion_event": completions[0],
            "source_cost": {"source_id": cost["reservation_id"], "store_id": store.store_id,
                "attempt_id": attempt["attempt_id"], "charged_ms": cost["charged_ms"]},
            "scientific_qualification": False, "method_qualification": False,
            "admission_status": "NOT_ADMITTED", "cost_status": "OWNER_SETTLED"}
        _bounded(evidence, nodes=25000, depth_limit=30, string_limit=16384)
        evidence["evidence_hash"] = digest(evidence)
        _require(len(encode(evidence)) <= 2*1024*1024, "owner evidence byte quota")
        _require(code_hash() == policy.code_hash, "source changed during owner preparation")
        store.verify_artifact_read(attempt["artifact_id"], purpose="evaluate", authorization=grant)
        return evidence
    except (KeyError, TypeError, ValueError, DataValidationError, OverflowError) as exc:
        if isinstance(exc, ResearchError):
            raise
        raise ResearchError("UNQUALIFIED", "probability calibration owner evidence malformed") from exc
