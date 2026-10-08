"""Complete paid preparation table disclosure through the original store.

No calibration search, numerical worker or budget authority is created here.
Missing slots and failed attempts remain explicit. Source permissions precede
any source bytes; original all-attempt costs are deduplicated globally.
"""

from datetime import datetime, timezone
import json

from infrastructure.research_store import ResearchError, digest, encode
from infrastructure.research_visibility import admission_visibility, study_visibility, combine_visibility

from .probability_calibration_admission import _bounded, prepare_probability_calibration
from .research_cost import frozen_cost


TERMINAL_FAILURES = {"FAILED", "INTERRUPTED", "TIMEOUT", "BUDGET_EXHAUSTED", "PREFLIGHT_FAILED", "CANCELLED"}
SOURCE_KINDS = {"RESERVE", "WORKER_STARTED", "WORKER_TREE_STOPPED", "WORKER_STOP_CONFIRMED", "SETTLE", "ADMISSION", "ATTEMPT"}


def _require(condition, detail):
    if not condition:
        raise ResearchError("UNQUALIFIED", "calibration export: " + detail)


def table_for_export(spec):
    from .probability_calibration_consumption import require_calibration_export_support
    require_calibration_export_support(spec)
    return spec.get("propagation_design", {}).get("axis_manifest", {}).get("calibrations", [])


def _source_scope(store, target, entry, target_authorization):
    """Bound and authenticate metadata before disclosing any result bytes."""
    from .propagation_execution import validate_propagation_cell
    from experiments.pirc27.calibration_plugin import calibration_policy
    from .research_evidence import authorize_study
    pointer = entry["source_pointer"]
    attempt = store.attempts()[pointer["source_attempt_id"]]
    run = store.manifest("run-"+attempt["run_id"])
    spec, cell = store.manifest("study-"+run["study_id"])["spec"], run["cell"]
    _require(run["run_id"] == digest({k: v for k, v in run.items() if k != "run_id"})
        and run["spec_hash"] == digest(spec) and run["cell_hash"] == digest(cell)
        and sum(digest(c) == digest(cell) for c in spec["cells"]) == 1
        and attempt["artifact_id"] == pointer["source_artifact_id"]
        and digest(spec["runtime_binding"]) == digest(target["runtime_binding"])
        and spec["code_hash"] == target["code_hash"] and cell["plugin_id"] == "affine-halfspace-calibration"
        and cell["visibility"] == "synthetic", "original synthetic source/run/root identity differs")
    package, request, config, _ = validate_propagation_cell(spec, cell)
    _require(digest(calibration_policy(spec, cell, package, request, config).manifest()) == digest(pointer["policy"])
        and package.package_hash == entry["model_package_hash"] and request.horizons[0] == entry["horizon"],
        "original source policy/model/horizon differs")
    if "input_binding" in cell:
        _require(cell["input_binding"]["case"]["source_kind"] == "synthetic-recipe",
            "private-prefix preparation export adapter is not implemented")
    grant = store.authorization(pointer["source_authorization_id"], version=pointer["source_authorization_version"])
    _require(grant["authorization_id"] == pointer["source_authorization_id"]
        and grant.get("version") == pointer["source_authorization_version"]
        and grant["study_id"] == spec["study_id"] and grant["protocol_hash"] == spec["protocol_hash"]
        and (target["study_id"] == spec["study_id"] or target["study_id"] in grant.get("consumer_study_ids", []))
        and {"evaluate", "export"} <= set(grant["purposes"])
        and {c["block_id"] for c in spec["cells"]} <= set(grant["block_ids"])
        and datetime.fromisoformat(grant["expires_at"]) > datetime.now(timezone.utc),
        "current selected source evaluate/export consumer permission differs")
    events = store.events()
    admissions = [e for e in events if e["event_kind"] == "ADMISSION"
        and e["payload"].get("attempt_id") == attempt["attempt_id"]]
    _require(len(admissions) <= 1, "source has ambiguous admission")
    receipt = store.manifest("admission-"+admissions[0]["payload"]["admission_hash"]) if admissions else None
    lineage = {}
    def manifest(object_id):
        value = store.manifest(object_id)
        _require(len(lineage) < 10000 or object_id in lineage, "bounded source metadata lineage")
        lineage[object_id] = value
        return value
    labels = [study_visibility(manifest, spec)]
    if receipt is not None:
        _require(receipt["admission_hash"] == digest({k: v for k, v in receipt.items() if k != "admission_hash"})
            == admissions[0]["payload"]["admission_hash"]
            and receipt["spec_hash"] == digest(spec) and receipt["cell_hash"] == digest(cell)
            and digest(receipt["spec"]) == digest(spec) and digest(receipt["cell"]) == digest(cell)
            and receipt["attempt_id"] == attempt["attempt_id"] and receipt["run_id"] == run["run_id"]
            and receipt["mode"] == "pilot", "source admission identity differs")
        labels.append(admission_visibility(manifest, receipt))
    metadata = store.manifest("artifact-"+attempt["artifact_id"]) if attempt["artifact_id"] is not None else None
    if metadata is not None:
        _require(metadata["artifact_id"] == attempt["artifact_id"]
            and digest(metadata) == attempt["artifact_manifest_hash"]
            and metadata["study_id"] == spec["study_id"] and metadata["block_ids"] == [cell["block_id"]]
            and metadata["role"] == "result" and metadata["media_type"] == "application/json"
            and type(metadata["size_bytes"]) is int and 0 < metadata["size_bytes"] <= 512*1024,
            "bounded original result artifact differs")
        labels.append(metadata["visibility"])
    visibility = combine_visibility(labels)
    _require(visibility in grant["visibilities"] and visibility in target_authorization["visibilities"],
        "source admission/package/lineage visibility is not covered")
    attachments, extra_grants = [], {}
    documents = receipt.get("documents", {}) if receipt is not None else {}
    groups = [(documents.get("qualification_evidence", []), grant)]
    if documents.get("model_qualification_evidence"):
        groups.append((documents["model_qualification_evidence"], documents["model_authorization"]))
    if documents.get("propagation_qualification"):
        evidence = documents["propagation_qualification"]
        groups.append(([{"artifact": evidence["source_artifact"], "content": evidence["source_result"]}],
            evidence["authorization"]))
    for items, saved_grant in groups:
        if not items:
            continue
        current = store.authorization(saved_grant["authorization_id"], version=saved_grant.get("version"))
        owner = manifest("study-"+current["study_id"])["spec"]
        _require(digest(current) == digest(saved_grant) and current["protocol_hash"] == owner["protocol_hash"]
            and (target["study_id"] == current["study_id"] or target["study_id"] in current.get("consumer_study_ids", []))
            and {"evaluate", "export"} <= set(current["purposes"])
            and visibility in current["visibilities"]
            and datetime.fromisoformat(current["expires_at"]) > datetime.now(timezone.utc),
            "current secondary attachment export consumer permission differs")
        authorize_study(store, current["study_id"], current, "export")
        extra_grants[digest(current)] = current
        for item in items:
            actual = manifest("artifact-"+item["artifact"]["artifact_id"])
            _require(digest(actual) == digest(item["artifact"]) and actual["study_id"] == current["study_id"]
                and actual["visibility"] in current["visibilities"]
                and set(actual["block_ids"]) <= set(current["block_ids"])
                and actual["sha256"] == digest(item["content"]) and actual["size_bytes"] == len(encode(item["content"])),
                "secondary attachment metadata/content export permission differs")
            attachments.append((item, current))
    authorize_study(store, spec["study_id"], grant, "export")
    return spec, cell, run, attempt, receipt, metadata, grant, visibility, lineage, list(extra_grants.values()), attachments


def _record(store, target, entry, authorization):
    from .research_reuse import verified_reuse
    from experiments.pirc27.calibration_plugin import calibration_plugin
    spec, cell, run, attempt, receipt, metadata, grant, visibility, lineage, extra_grants, attachments = _source_scope(store, target, entry, authorization)
    for attachment, attachment_grant in attachments:
        content = store.read_artifact(attachment["artifact"]["artifact_id"], purpose="export", authorization=attachment_grant)
        _require(content == encode(attachment["content"]), "secondary attachment export bytes differ")
    result, proof = None, None
    if metadata is not None:
        content = store.read_artifact(metadata["artifact_id"], purpose="export", authorization=grant)
        _require(len(content) <= 512*1024, "result byte quota")
        result = json.loads(content)
        _bounded(result, nodes=25000, depth_limit=30, string_limit=16384)
        _require(content == encode(result) and result["source_schema"] == "affine-halfspace-calibration-source-v1"
            and result["spec_hash"] == digest(spec) and result["cell_hash"] == digest(cell),
            "canonical original calibration result required")
    if entry["status"] == "CALIBRATED":
        proof = prepare_probability_calibration(store, entry["source_pointer"], consumer_study_id=target["study_id"])
        _require(proof["evidence_hash"] == entry["source_evidence_hash"]
            and proof["geometry_hash"] == entry["geometry_hash"]
            and digest(proof["geometry"]) == digest(entry["geometry"]), "frozen successful owner differs")
        outcome = "calibrated-preparation-not-method-qualification"
    else:
        _require(entry["status"] == "FAILED", "only an explicit failed source can retain failure evidence")
        if attempt["state"] == "SUCCEEDED":
            _require(result is not None and receipt is not None, "completed numerical failure lacks result/admission")
            verified_reuse(store, attempt, spec, cell, calibration_plugin())
            analysis = result["forecast"]["probability_calibration_analysis"]
            _require(analysis["status"] == "FAILED" and analysis["scientific_qualification"] is False
                and analysis["method_qualification"] is False and analysis["admission_status"] == "NOT_ADMITTED",
                "failed slot cannot conceal successful calibration or claim qualification")
            outcome = "recorded-numerical-analysis-failure"
        else:
            _require(attempt["state"] in TERMINAL_FAILURES, "failed slot points to an unfinished source")
            outcome = "recorded-execution-failure-"+attempt["state"]
    history = [a for a in store.attempts().values() if a["run_id"] == run["run_id"]]
    _require(0 < len(history) <= 10000 and sum(a["attempt_id"] == attempt["attempt_id"] for a in history) == 1,
        "complete bounded source history required")
    attempt_ids = {a["attempt_id"] for a in history}
    events = [e for e in store.events() if e["event_kind"] in SOURCE_KINDS
        and e["payload"].get("attempt_id") in attempt_ids]
    cost = frozen_cost(history, events)
    for source in cost["sources"]:
        value = source["payload"]
        _require(value["run_id"] == run["run_id"] and value["study_id"] == spec["study_id"]
            and value["arm_id"] == cell["arm_id"]
            and value["reservation_id"] == digest([store.store_id, value["attempt_id"]])
            and type(value["reserved_ms"]) is int and value["reserved_ms"] > 0
            and type(value["charged_ms"]) is int and value["charged_ms"] >= 0
            and type(value["settled"]) is bool and value["settled"] == (source["event_kind"] == "SETTLE")
            and (value["monotonic_elapsed_ms"] is None or type(value["monotonic_elapsed_ms"]) is int
                and value["monotonic_elapsed_ms"] >= 0)
            and value["charged_ms"] == ((value["reserved_ms"] if value["monotonic_elapsed_ms"] is None
                else value["monotonic_elapsed_ms"]) if value["settled"] else 0), "original all-attempt cost policy differs")
    body = {"schema_version": "calibration-source-export-record-v1", "pointer": entry["source_pointer"],
        "pointer_hash": digest(entry["source_pointer"]), "source_spec": spec, "source_cell": cell,
        "source_run": run, "selected_attempt": attempt, "history": history, "events": events, "cost": cost,
        "source_admission": receipt, "source_artifact": metadata, "source_result": result, "authorization": grant,
        "proof": proof, "outcome": outcome, "visibility": visibility,
        "lineage": lineage, "extra_authorizations": extra_grants,
        "scientific_qualification": False, "method_qualification": False}
    _bounded(body, nodes=100000, depth_limit=32, string_limit=16384)
    _require(len(encode(body)) <= 8*1024*1024, "source record exceeds fixed byte quota")
    if metadata is not None:
        store.verify_artifact_read(metadata["artifact_id"], purpose="export", authorization=grant)
    for attachment, attachment_grant in attachments:
        store.verify_artifact_read(attachment["artifact"]["artifact_id"], purpose="export", authorization=attachment_grant)
    return {**body, "record_hash": digest(body)}


def calibration_cost(records):
    """Summarize actual owner sources, not another budget ledger or per-cell fee."""
    sources = {}
    for record in records:
        store_id = record["source_spec"]["runtime_binding"]["store_id"]
        by_attempt = {e["payload"]["attempt_id"]: e for e in record["cost"]["sources"]}
        for attempt in record["history"]:
            key = (store_id, attempt["attempt_id"])
            event = by_attempt.get(attempt["attempt_id"])
            value = event["payload"] if event is not None else None
            source = {"source_id": digest(list(key)), "store_id": store_id, "attempt_id": attempt["attempt_id"],
                "state": attempt["state"], "event_hash": event["hash"] if event is not None else None,
                "charged_ms": value["charged_ms"] if value is not None and value["settled"] else None,
                "reserved_ms": (value["reserved_ms"] if not value["settled"] else 0) if value is not None else None,
                "measured_ms": value["monotonic_elapsed_ms"] if value is not None and value["settled"] else None,
                "status": ("MEASURED" if value["monotonic_elapsed_ms"] is not None else "UPPER_BOUND")
                    if value is not None and value["settled"] else "RESERVED" if value is not None else "UNAVAILABLE"}
            _require(key not in sources or digest(sources[key]) == digest(source), "conflicting original source costs")
            sources[key] = source
    values = list(sources.values())
    def total(field):
        return sum(v[field] for v in values) if values and all(v[field] is not None for v in values) else None
    return {"unit": "slot-ms", "scope": "unique-calibration-source-attempts-including-failures-not-per-target-cell",
        "unique_attempts": len(values), "charged_ms": total("charged_ms"), "reserved_ms": total("reserved_ms"),
        "measured_ms": total("measured_ms"), "sources": values}


def prepare_calibration_export(store, spec, authorization):
    table = table_for_export(spec)
    if not table:
        return None
    sources, slots, seen, source_bytes = [], [], {}, 0
    # Preflight ALL source grants before any source bytes, not just the first
    # successful slot. A later denied/failed source cannot be dropped.
    for entry in table:
        if entry["source_pointer"] is not None:
            _source_scope(store, spec, entry, authorization)
    for entry in table:
        pointer = entry["source_pointer"]
        reference = digest(pointer) if pointer is not None else None
        if reference is not None and reference not in seen:
            record = _record(store, spec, entry, authorization)
            seen[reference] = record
            sources.append(record)
            source_bytes += len(encode(record))
            _require(source_bytes <= 48*1024*1024, "all-source byte quota")
        if reference is not None:
            record = seen[reference]
            _require((record["proof"] is not None) == (entry["status"] == "CALIBRATED"), "contradictory source dispositions")
        slots.append({"binding_hash": entry["binding_hash"], "status": entry["status"], "reason": entry["reason"],
            "pointer_hash": reference})
    body = {"schema_version": "propagation-calibration-table-evidence-v1", "table_hash": digest(table),
        "slots": slots, "sources": sources, "cost": calibration_cost(sources)}
    _bounded(body, nodes=1000000, depth_limit=35, string_limit=16384)
    _require(len(encode(body)) <= 48*1024*1024, "complete calibration disclosure byte quota")
    return {**body, "evidence_hash": digest(body)}


def verify_calibration_export(store, spec, header, authorization):
    fresh = prepare_calibration_export(store, spec, authorization)
    _require(digest(header) == digest(fresh), "complete source table/history/cost changed before disclosure")
    return [] if fresh is None else [g for r in fresh["sources"]
        for g in [r["authorization"], *r["extra_authorizations"]]]


def calibration_recorded_at(store, spec, cells, header, authorization):
    """Stable evidence-as-of time, NOT a reusable export permission clock.

    Only immutable selected study/grant publications and original attempt
    events contribute. Audit/reuse/disclosure I/O and unrelated studies do not
    mutate an unchanged bundle; every disclosure still checks the live clock.
    """
    from infrastructure.research_store import authorization_key_for_grant
    grants = [authorization, *[g for r in header["sources"] for g in [r["authorization"], *r["extra_authorizations"]]]]
    objects = {"study-"+spec["study_id"], *[authorization_key_for_grant(g) for g in grants]}
    attempts = {a["attempt_id"] for row in cells for a in row["history"]}
    attempts.update(a["attempt_id"] for record in header["sources"] for a in record["history"])
    selected = [e for e in store.events() if e["event_kind"] == "MANIFEST" and e["payload"].get("object_id") in objects
        or e["event_kind"] == "ATTEMPT" and e["payload"].get("attempt_id") in attempts]
    _require({e["payload"].get("object_id") for e in selected} >= objects, "original study/grant publications required")
    return max(datetime.fromisoformat(e["created_at"]) for e in selected).isoformat()
