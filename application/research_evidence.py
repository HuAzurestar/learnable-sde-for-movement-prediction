"""Export an authorized, complete matrix for the paper-side aggregator."""

from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
import io
import json

from infrastructure.research_store import ResearchError, ResearchStore, digest, encode
from application.research_dimensions import comparison_dimensions
from application.research_cost import frozen_cost


def evidence_visibility(spec, cells):
    """Attachments cannot be declassified by a synthetic parent cell."""
    if any(cell.get("visibility") != "synthetic" for cell in spec["cells"]):
        return "restricted"
    for cell in cells:
        documents = cell.get("admission", {}).get("documents", {})
        for key in ("package", "frozen_model"):
            if key in documents and documents[key].get("visibility", "restricted") != "synthetic":
                return "restricted"
        for key in ("qualification_evidence", "model_qualification_evidence"):
            if any(item["artifact"].get("visibility") != "synthetic" for item in documents.get(key, [])):
                return "restricted"
    return "synthetic"


def authorize_study(store, study_id, authorization, purpose):
    try:
        grant = store.manifest("authorization-" + authorization["authorization_id"])
        allowed = (grant == authorization and grant["study_id"] == study_id
                   and purpose in grant["purposes"]
                   and datetime.fromisoformat(grant["expires_at"]) > datetime.now(timezone.utc))
    except (ResearchError, KeyError, ValueError, TypeError):
        allowed = False
    store.append("DISCLOSURE_ALLOWED" if allowed else "DISCLOSURE_DENIED", {
        "study_id": study_id, "purpose": purpose, "authorization_hash": digest(authorization)})
    if not allowed:
        raise ResearchError("UNAUTHORIZED_DATA", "study disclosure is not authorized")


def require_export_visibility(store, spec, cells, authorization):
    visibility = evidence_visibility(spec, cells)
    if visibility not in authorization["visibilities"]:
        store.append("DISCLOSURE_DENIED", {"study_id": spec["study_id"], "purpose": "export",
                   "authorization_hash": digest(authorization), "required_visibility": visibility})
        raise ResearchError("UNAUTHORIZED_DATA", "export grant cannot disclose this evidence visibility")
    return visibility


def export_evidence(store: ResearchStore, study_id: str, authorization: dict):
    authorize_study(store, study_id, authorization, "export")
    spec = store.manifest("study-" + study_id)["spec"]
    if not {c["block_id"] for c in spec["cells"]} <= set(authorization["block_ids"]):
        raise ResearchError("UNAUTHORIZED_DATA", "export does not cover the complete study matrix")
    require_export_visibility(store, spec, [], authorization)
    with store.lock():
        attempts = store._attempts()
        cost_events = store._events()
    runs = {}
    for attempt in attempts.values():
        run = store.manifest("run-" + attempt["run_id"])
        if run["study_id"] == study_id:
            runs.setdefault(run["cell_hash"], []).append(attempt)
    cells = []
    for cell in spec["cells"]:
        history = runs.get(digest(cell), [])
        successful = [a for a in history if a["state"] == "SUCCEEDED"]
        latest = successful[0] if successful else history[-1] if history else None
        row = {"cell_hash": digest(cell), "arm_id": cell["arm_id"], "block_id": cell["block_id"],
               "comparison_dimensions": comparison_dimensions(cell), "registered_cell": cell,
               "seed": cell["seed"], "status": latest["state"] if latest else "MISSING",
               "attempt_id": latest["attempt_id"] if latest else None,
               "run_id": latest["run_id"] if latest else None,
               "artifact_id": latest.get("artifact_id") if latest else None,
               "history": [{"attempt_id": a["attempt_id"], "state": a["state"], "error_code": a["error_code"]} for a in history],
               "metrics": None, "metric_units": None, "qualification": None,
               "cost": frozen_cost(history, cost_events)}
        if successful:
            result = json.loads(store.read_artifact(latest["artifact_id"], purpose="export", authorization=authorization))
            if result["spec_hash"] != digest(spec) or result["cell_hash"] != digest(cell):
                raise ResearchError("CORRUPT_ARTIFACT", "evidence result/spec binding mismatch")
            row.update(metrics=result["metrics"], metric_units=result["metric_units"], qualification=result["qualification"],
                       state_order=result["state_order"], units=result["units"], protocol_hash=result["protocol_hash"])
            if result.get("admission_hash"):
                admission = store.manifest("admission-" + result["admission_hash"])
                body = {key: value for key, value in admission.items() if key != "admission_hash"}
                if (digest(body) != result["admission_hash"] or admission.get("admission_hash") != digest(body)
                        or admission["spec_hash"] != digest(spec) or admission["cell_hash"] != digest(cell)
                        or admission["attempt_id"] != latest["attempt_id"] or admission["run_id"] != latest["run_id"]
                        or admission["qualification"] != result["qualification"]):
                    raise ResearchError("UNQUALIFIED", "result admission identity differs from authoritative attempt")
                require_export_visibility(store, spec, [{"admission": admission}], authorization)
                # Qualification attachments are additional disclosures, not
                # automatically public because execution was authorized.
                documents = admission.get("documents", {})
                from infrastructure.research_store import encode
                for attachment in documents.get("qualification_evidence", []):
                    content = store.read_artifact(attachment["artifact"]["artifact_id"], purpose="export", authorization=authorization)
                    if content != encode(attachment["content"]):
                        raise ResearchError("CORRUPT_ARTIFACT", "qualification export binding changed")
                if documents.get("model_qualification_evidence"):
                    model_grant = documents["model_authorization"]
                    if (model_grant["study_id"] != study_id and
                            study_id not in model_grant.get("consumer_study_ids", [])):
                        raise ResearchError("UNAUTHORIZED_DATA", "model evidence export consumer differs")
                    for attachment in documents["model_qualification_evidence"]:
                        content = store.read_artifact(attachment["artifact"]["artifact_id"], purpose="export", authorization=model_grant)
                        if content != encode(attachment["content"]):
                            raise ResearchError("CORRUPT_ARTIFACT", "model qualification export binding changed")
                row.update(admission=admission, admission_hash=result["admission_hash"])
            elif result["qualification"] == "qualified":
                raise ResearchError("UNQUALIFIED", "qualified result lacks execution admission evidence")
        cells.append(row)
    visibility = require_export_visibility(store, spec, cells, authorization)
    payload = {"schema_version": "pirc25-evidence-bundle-v1", "study_id": study_id,
               "spec_hash": digest(spec), "protocol_hash": spec["protocol_hash"], "data_hash": spec["data_hash"],
               "code_hash": spec["code_hash"], "feature_hash": spec["feature_hash"], "selection_hash": spec["selection_hash"],
               "comparison_family": spec["comparison_family"], "independent_unit": "block_id",
               "comparison_plan": spec.get("comparison_plan"),
               "visibility": visibility,
               "expected_cells": [{"cell_hash": digest(c), "arm_id": c["arm_id"], "block_id": c["block_id"], "seed": c["seed"],
                                   "comparison_dimensions": comparison_dimensions(c)} for c in spec["cells"]],
               "cells": cells, "disclosure_scope": "authorized-local-export"}
    bundle = {**payload, "bundle_hash": digest(payload)}
    store.publish("bundle-" + bundle["bundle_hash"], bundle)
    return bundle


def accept_aggregate(store, aggregate, study_id):
    """Freeze a TSDE-produced aggregate; the runtime/UI never re-compute it."""
    spec = store.manifest("study-" + study_id)["spec"]
    body = {key: value for key, value in aggregate.items() if key != "aggregate_hash"}
    if (aggregate.get("schema_version") != "pirc25-aggregate-v1" or aggregate.get("aggregate_hash") != digest(body)
            or aggregate.get("study_id") != study_id or aggregate.get("spec_hash") != digest(spec)
            or aggregate.get("protocol_hash") != spec["protocol_hash"]):
        raise ResearchError("CONTRACT_MISMATCH", "aggregate identity differs from the registered study")
    bundle = store.manifest("bundle-" + aggregate.get("source_bundle_hash", ""))
    if (bundle["spec_hash"] != digest(spec) or aggregate.get("expected_cell_count") != len(spec["cells"])
            or aggregate.get("cell_dispositions") != bundle["cells"]):
        raise ResearchError("CONTRACT_MISMATCH", "aggregate is not bound to the frozen complete matrix")
    from infrastructure.research_store import encode
    visibility = evidence_visibility(spec, bundle["cells"])
    return store.artifact(encode(aggregate), role="aggregate", visibility=visibility,
                          block_ids=[c["block_id"] for c in spec["cells"]], study_id=study_id)


def expected_metrics_csv(aggregate):
    """Validate the frozen v1 table serialization without recalculating scores."""
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["aggregate_hash", "arm_id", "metric", "value", "unit", "independent_n", "expected_cells", "successful_cells", "status", "stratum_id", "comparison_dimensions", "charged_ms", "reserved_ms", "measured_ms", "cost_unit", "cost_scope", "status_rates"])
    for arm in aggregate["arms"]:
        costs = [arm["cost"][key] for key in ("charged_ms", "reserved_ms", "measured_ms", "unit", "scope")]
        rates = encode(arm.get("status_rates")).decode()
        dimensions = encode(arm["comparison_dimensions"]).decode()
        if not arm["metrics"]:
            writer.writerow([aggregate["aggregate_hash"], arm["arm_id"], "", "", "", arm["independent_n"],
                             arm["expected_cells"], arm["successful_cells"], arm["status"], arm["stratum_id"],
                             dimensions, *costs, rates])
        for metric, value in sorted(arm["metrics"].items()):
            writer.writerow([aggregate["aggregate_hash"], arm["arm_id"], metric, value, arm["metric_units"][metric],
                             arm["independent_n"], arm["expected_cells"], arm["successful_cells"], arm["status"],
                             arm["stratum_id"], dimensions, *costs, rates])
    return stream.getvalue().encode()


def expected_paper_index(aggregate, table):
    claims = []
    for arm in aggregate["arms"]:
        for metric, number in sorted(arm["metrics"].items()):
            claims.append({"claim_id": digest([arm["arm_id"], arm["stratum_id"], metric]),
                "metric": metric, "value": number, "arm_id": arm["arm_id"], "stratum_id": arm["stratum_id"],
                "comparison_dimensions": arm["comparison_dimensions"], "evidence_status": arm["status"],
                "independent_n": arm["independent_n"], "cost": arm["cost"],
                "unit": arm["metric_units"][metric], "aggregate_hash": aggregate["aggregate_hash"],
                "attempt_ids": [cell["attempt_id"] for cell in aggregate["cell_dispositions"]
                                if cell["arm_id"] == arm["arm_id"] and cell["status"] == "SUCCEEDED"
                                and cell["block_id"] in arm["complete_block_ids"]
                                and encode(comparison_dimensions(cell)) == encode(arm["comparison_dimensions"])]})
    return {"schema_version": "pirc25-paper-evidence-v1", "study_id": aggregate["study_id"],
            "aggregate_hash": aggregate["aggregate_hash"], "table_sha256": hashlib.sha256(table).hexdigest(),
            "code_hash": aggregate["code_hash"], "evidence_status": "active", "relation": None,
            "disclosure_scope": aggregate["disclosure_scope"], "claims": claims}


def accept_evidence_package(store, directory, expected_hash):
    """Import the same frozen JSON/CSV/evidence files that TSDE produced."""
    from pathlib import Path

    root = Path(directory).resolve()
    contents = {}
    for name in ("manifest.json", "aggregate.json", "metrics.csv", "PaperEvidenceIndex.json"):
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ResearchError("UNAUTHORIZED_DATA", "evidence package path escapes root")
        contents[name] = path.read_bytes()
    manifest = json.loads(contents["manifest.json"])
    if manifest.get("schema_version") != "pirc25-evidence-package-v1" or manifest.get("aggregate_hash") != expected_hash:
        raise ResearchError("CONTRACT_MISMATCH", "evidence package version/hash mismatch")
    for name in ("aggregate.json", "metrics.csv", "PaperEvidenceIndex.json"):
        if hashlib.sha256(contents[name]).hexdigest() != manifest["files"].get(name):
            raise ResearchError("CORRUPT_ARTIFACT", "evidence package content differs from manifest")
    aggregate = json.loads(contents["aggregate.json"])
    if aggregate.get("aggregate_hash") != expected_hash:
        raise ResearchError("CORRUPT_ARTIFACT", "aggregate does not match package")
    index = json.loads(contents["PaperEvidenceIndex.json"])
    if index.get("aggregate_hash") != expected_hash or index.get("table_sha256") != manifest["files"]["metrics.csv"]:
        raise ResearchError("CONTRACT_MISMATCH", "paper evidence/table version differs")
    try:
        if contents["metrics.csv"] != expected_metrics_csv(aggregate):
            raise ResearchError("CONTRACT_MISMATCH", "frozen CSV differs from aggregate values")
        if contents["PaperEvidenceIndex.json"] != encode(expected_paper_index(aggregate, contents["metrics.csv"])):
            raise ResearchError("CONTRACT_MISMATCH", "paper evidence index differs from aggregate values")
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ResearchError):
            raise
        raise ResearchError("CONTRACT_MISMATCH", "incomplete frozen evidence package") from exc
    artifact = accept_aggregate(store, aggregate, aggregate["study_id"])
    attachments = {}
    for name, role in (("metrics.csv", "table"), ("PaperEvidenceIndex.json", "evidence-index")):
        attachments[role] = store.artifact(contents[name], role=role, visibility=artifact["visibility"],
                                          block_ids=artifact["block_ids"], study_id=artifact["study_id"],
                                          media_type="text/csv" if role == "table" else "application/json")["artifact_id"]
    package = {"schema_version": "pirc25-imported-evidence-v1", "aggregate_hash": expected_hash,
               "aggregate_id": artifact["artifact_id"], "study_id": artifact["study_id"], **attachments}
    store.publish("comparison-" + expected_hash, package)
    return package
