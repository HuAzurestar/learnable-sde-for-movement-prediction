"""Export an authorized, complete matrix for the paper-side aggregator."""

from __future__ import annotations

from datetime import datetime, timezone
import csv
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import stat

from infrastructure.research_store import ResearchError, ResearchStore, digest, encode
from application.research_dimensions import comparison_dimensions
from application.research_cost import frozen_cost


def evidence_visibility(store, spec, cells):
    """Source artifact metadata cannot be declassified by a synthetic cell."""
    from infrastructure.research_visibility import study_visibility, admission_visibility, combine_visibility
    labels = [study_visibility(store.manifest, spec)]
    for cell in cells:
        if cell.get("artifact_id"):
            labels.append(store.manifest("artifact-" + cell["artifact_id"])["visibility"])
        if cell.get("admission"):
            labels.append(admission_visibility(store.manifest, cell["admission"]))
    return combine_visibility(labels)


def _study_disclosure_allowed(store, study_id, authorization, purpose):
    try:
        grant = store.authorization(authorization["authorization_id"], version=authorization.get("version"))
        return (grant == authorization and grant["study_id"] == study_id
                and purpose in grant["purposes"]
                and datetime.fromisoformat(grant["expires_at"]) > datetime.now(timezone.utc))
    except (ResearchError, KeyError, ValueError, TypeError):
        return False


def authorize_study(store, study_id, authorization, purpose):
    allowed = _study_disclosure_allowed(store, study_id, authorization, purpose)
    store.append("DISCLOSURE_ALLOWED" if allowed else "DISCLOSURE_DENIED", {
        "study_id": study_id, "purpose": purpose, "authorization_hash": digest(authorization)})
    if not allowed:
        raise ResearchError("UNAUTHORIZED_DATA", "study disclosure is not authorized")
    # Flushing the permission journal can itself cross the expiry boundary.
    # A successful pre-I/O check is not permission to read after that write.
    require_unexpired_disclosure(store, study_id, authorization, purpose)
    verify_study_disclosure(store, study_id, authorization, purpose)
    if store._read_snapshot() is not None:
        # Metadata-only queries/exports have no raw-byte completion guard.
        # Rehash their own grant after the outer physical verification too.
        store._read_completion(
            lambda: verify_study_disclosure(store, study_id, authorization, purpose),
            lambda: require_unexpired_disclosure(store, study_id, authorization, purpose))


def verify_study_disclosure(store, study_id, authorization, purpose):
    """Fresh post-journal authority without reopening an ALLOWED-write loop."""
    if not _study_disclosure_allowed(store, study_id, authorization, purpose):
        store.append("DISCLOSURE_DENIED", {"study_id": study_id, "purpose": purpose,
            "authorization_hash": digest(authorization)})
        raise ResearchError("UNAUTHORIZED_DATA", "grant changed during disclosure verification")


def require_unexpired_disclosure(store, study_id, authorization, purpose):
    """Last clock guard for an already authenticated immutable study grant.

    This is not a replacement for authorize_study: that function must verify
    the actual manifest, study and purpose first. The valid path performs no
    I/O so final journal/physical-integrity work cannot follow its clock check.
    An expired grant is durably denied before any value can be disclosed.
    """
    try:
        current = datetime.fromisoformat(authorization['expires_at']) > datetime.now(timezone.utc)
    except (KeyError, ValueError, TypeError):
        current = False
    if not current:
        store.append('DISCLOSURE_DENIED', {'study_id': study_id, 'purpose': purpose,
            'authorization_hash': digest(authorization)})
        raise ResearchError('UNAUTHORIZED_DATA', 'study disclosure authorization expired')


def require_export_visibility(store, spec, cells, authorization):
    visibility = evidence_visibility(store, spec, cells)
    if visibility not in authorization["visibilities"]:
        store.append("DISCLOSURE_DENIED", {"study_id": spec["study_id"], "purpose": "export",
                   "authorization_hash": digest(authorization), "required_visibility": visibility})
        raise ResearchError("UNAUTHORIZED_DATA", "export grant cannot disclose this evidence visibility")
    return visibility


def export_evidence(store: ResearchStore, study_id: str, authorization: dict):
    # Retain one current-thread/PID verified scope and its actual OS lock for
    # the whole source/cost assembly. This is not a persistent grant cache.
    with store._read_transaction():
        authorize_study(store, study_id, authorization, "export")
        snapshot = _export_data_snapshot(store)
        bundle = _assemble_evidence(store, study_id, authorization)
        grants = _fresh_export_authority(store, bundle, authorization)
        if _export_data_snapshot(store) != snapshot:
            raise ResearchError("INDEX_STALE", "authority changed while assembling evidence; retry export")
        object_id = "bundle-" + bundle["bundle_hash"]
        store.publish(object_id, bundle)
        grants = _fresh_export_authority(store, bundle, authorization)
        # The only permitted new data event is this exact private bundle's
        # immutable publication. Disclosure/read journals do not change data.
        current = _export_data_snapshot(store, after=snapshot,
            publication={"object_id": object_id, "sha256": digest(bundle)})
        if current != snapshot:
            raise ResearchError("INDEX_STALE", "authority changed while publishing evidence; retry export")
    # The final uncached physical validation can itself cross expiry. No
    # valid-path I/O may follow this last clock check before returning bytes.
    _require_current_export_grants(store, grants)
    return bundle


def _export_data_snapshot(store, *, after=(), publication=None):
    kinds = {"MANIFEST", "ATTEMPT", "RESERVE", "SETTLE", "ARM_CLOSED", "RECOVERY_HOLD"}
    result, publications = [], 0
    last_sequence = after[-1][0] if after else 0
    for event in store.events():
        if event["event_kind"] not in kinds:
            continue
        if (publication is not None and event["sequence"] > last_sequence
                and event["event_kind"] == "MANIFEST" and event["payload"] == publication):
            publications += 1
            if publications > 1:
                raise ResearchError("INDEX_STALE", "evidence publication changed more than once")
            continue
        # Include hashes, not just the largest sequence, so forensic recovery
        # cannot replace a data prefix with a different equal-sized chain.
        result.append((event["sequence"], event["hash"]))
    return tuple(result)


def _export_grants(bundle, authorization):
    grants = {digest(authorization): authorization}
    for cell in bundle["cells"]:
        documents = cell.get("admission", {}).get("documents", {})
        if documents.get("model_qualification_evidence"):
            grant = documents["model_authorization"]
            if (grant["study_id"] != bundle["study_id"]
                    and bundle["study_id"] not in grant.get("consumer_study_ids", [])):
                raise ResearchError("UNAUTHORIZED_DATA", "model evidence export consumer differs")
            grants[digest(grant)] = grant
        if documents.get("propagation_qualification"):
            grant = documents["propagation_qualification"]["authorization"]
            if grant["study_id"] != bundle["study_id"] and bundle["study_id"] not in grant.get("consumer_study_ids", []):
                raise ResearchError("UNAUTHORIZED_DATA", "analytic qualification export consumer differs")
            grants[digest(grant)] = grant
    return list(grants.values())


def _require_current_export_grants(store, grants):
    # All manifests/studies/purposes have just been freshly authenticated.
    # Checking the earliest expiry last protects every consumed grant at the
    # same disclosure instant, including a model grant shorter than the main.
    earliest = min(grants, key=lambda grant: datetime.fromisoformat(grant["expires_at"]))
    require_unexpired_disclosure(store, earliest["study_id"], earliest, "export")


def _fresh_export_authority(store, bundle, authorization):
    spec = store.manifest("study-" + bundle["study_id"])["spec"]
    if spec != bundle["registered_spec"] or digest(spec) != bundle["spec_hash"]:
        raise ResearchError("CORRUPT_ARTIFACT", "export study source binding changed")
    if not {cell["block_id"] for cell in spec["cells"]} <= set(authorization["block_ids"]):
        raise ResearchError("UNAUTHORIZED_DATA", "export does not cover the complete study matrix")
    require_export_visibility(store, spec, bundle["cells"], authorization)
    grants = _export_grants(bundle, authorization)
    for index, grant in enumerate(grants):
        study_id = bundle["study_id"] if index == 0 else grant["study_id"]
        authorize_study(store, study_id, grant, "export")
    # All successful permission journals must finish before the last rehash.
    # Do not add another ALLOWED journal here: its I/O would reopen the same
    # integrity window. Denial must still be durable, and failure propagates.
    for index, grant in enumerate(grants):
        study_id = bundle["study_id"] if index == 0 else grant["study_id"]
        verify_study_disclosure(store, study_id, grant, "export")
    _require_current_export_grants(store, grants)
    return grants


def authorize_evidence_publication(store, bundle, authorization):
    """Reauthenticate a frozen export after target fsync, before its rename.

    The CLI cannot reuse export_evidence's earlier permission after directory
    creation, existing-file checks, encoding and flushing the target bytes.
    This guard does not recalculate metrics, rerun cells or alter old costs.
    """
    with store._read_transaction():
        authorize_study(store, bundle["study_id"], authorization, "export")
        body = {key: value for key, value in bundle.items() if key != "bundle_hash"}
        if (bundle["bundle_hash"] != digest(body)
                or store.manifest("bundle-" + bundle["bundle_hash"]) != bundle):
            raise ResearchError("CORRUPT_ARTIFACT", "target export differs from its immutable bundle")
        grants = _fresh_export_authority(store, bundle, authorization)
    _require_current_export_grants(store, grants)


def _assemble_evidence(store, study_id, authorization):
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
            for field in ("metric_definitions", "comparison_diagnostics"):
                if field in result:
                    row[field] = result[field]
            if result.get("admission_hash"):
                admission = store.manifest("admission-" + result["admission_hash"])
                body = {key: value for key, value in admission.items() if key != "admission_hash"}
                if (digest(body) != result["admission_hash"] or admission.get("admission_hash") != digest(body)
                        or admission["spec_hash"] != digest(spec) or admission["cell_hash"] != digest(cell)
                        or admission["attempt_id"] != latest["attempt_id"] or admission["run_id"] != latest["run_id"]
                        or admission["qualification"] != result["qualification"]):
                    raise ResearchError("UNQUALIFIED", "result admission identity differs from authoritative attempt")
                if "cell_packages" in (spec.get("admission") or {}) or "admission_selection" in admission:
                    from infrastructure.research_admission_selection import verify_admission_selection
                    verify_admission_selection(spec, cell, admission)
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
                if documents.get("propagation_qualification"):
                    evidence = documents["propagation_qualification"]
                    # The independent reader needs the actual target bytes,
                    # not a row of copied metric/pass fields. Both attachments
                    # are already covered by the target artifact export grant.
                    row.update(result=result, result_artifact=store.manifest("artifact-"+latest["artifact_id"]))
                    source_grant = evidence["authorization"]
                    if source_grant["study_id"] != study_id and study_id not in source_grant.get("consumer_study_ids", []):
                        raise ResearchError("UNAUTHORIZED_DATA", "analytic qualification export consumer differs")
                    content = store.read_artifact(evidence["source_artifact"]["artifact_id"],
                        purpose="export", authorization=source_grant)
                    if content != encode(evidence["source_result"]):
                        raise ResearchError("CORRUPT_ARTIFACT", "analytic qualification export binding changed")
                row.update(admission=admission, admission_hash=result["admission_hash"])
            elif result["qualification"] == "qualified" or "cell_packages" in (spec.get("admission") or {}):
                raise ResearchError("UNQUALIFIED", "qualified or per-cell-bound result lacks execution admission evidence")
        cells.append(row)
    visibility = require_export_visibility(store, spec, cells, authorization)
    payload = {"schema_version": "pirc25-evidence-bundle-v1", "study_id": study_id,
               "spec_hash": digest(spec), "protocol_hash": spec["protocol_hash"], "data_hash": spec["data_hash"],
               "code_hash": spec["code_hash"], "feature_hash": spec["feature_hash"], "selection_hash": spec["selection_hash"],
               "comparison_family": spec["comparison_family"], "independent_unit": "block_id",
               "comparison_plan": spec.get("comparison_plan"),
               "registered_spec": spec,
               "visibility": visibility,
               "expected_cells": [{"cell_hash": digest(c), "arm_id": c["arm_id"], "block_id": c["block_id"], "seed": c["seed"],
                                   "comparison_dimensions": comparison_dimensions(c)} for c in spec["cells"]],
               "cells": cells, "disclosure_scope": "authorized-local-export"}
    bundle = {**payload, "bundle_hash": digest(payload)}
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
    # Four-file descriptive inputs are not proof of a supervised statistical
    # result. Removing both proof fields must never promote a fixture or an
    # operator-rehashed summary to formal evidence.
    managed = "adjudication" in aggregate or "computation_ref" in aggregate
    qualification = aggregate.get("qualification")
    if qualification == "formal" and not managed:
        raise ResearchError("UNQUALIFIED", "formal aggregate requires its managed computation proof")
    if not managed and qualification is not None and (
            not isinstance(qualification, str) or qualification not in {"descriptive", "engineering-fixture"}):
        raise ResearchError("UNQUALIFIED", "unsupported descriptive aggregate qualification")
    if managed:
        if "adjudication" not in aggregate or "computation_ref" not in aggregate:
            raise ResearchError("CONTRACT_MISMATCH", "managed adjudication requires its computation reference")
        from application.research_computation import verified_computation
        verified_computation(store, aggregate["computation_ref"], aggregate=aggregate)
    from infrastructure.research_store import encode
    visibility = evidence_visibility(store, spec, bundle["cells"])
    return store.artifact(encode(aggregate), role="aggregate", visibility=visibility,
                          block_ids=[c["block_id"] for c in spec["cells"]], study_id=study_id)


def expected_metrics_csv(aggregate):
    """Validate the frozen v1 table serialization without recalculating scores."""
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    extra = [name for name in ("adjudication", "computation_ref") if name in aggregate]
    writer.writerow(["aggregate_hash", "arm_id", "metric", "value", "unit", "independent_n", "expected_cells", "successful_cells", "status", "stratum_id", "comparison_dimensions", "charged_ms", "reserved_ms", "measured_ms", "cost_unit", "cost_scope", "status_rates", *extra])
    frozen = [encode(aggregate[name]).decode() for name in extra]
    for arm in aggregate["arms"]:
        costs = [arm["cost"][key] for key in ("charged_ms", "reserved_ms", "measured_ms", "unit", "scope")]
        rates = encode(arm.get("status_rates")).decode()
        dimensions = encode(arm["comparison_dimensions"]).decode()
        if not arm["metrics"]:
            writer.writerow([aggregate["aggregate_hash"], arm["arm_id"], "", "", "", arm["independent_n"],
                             arm["expected_cells"], arm["successful_cells"], arm["status"], arm["stratum_id"],
                             dimensions, *costs, rates, *frozen])
        for metric, value in sorted(arm["metrics"].items()):
            writer.writerow([aggregate["aggregate_hash"], arm["arm_id"], metric, value, arm["metric_units"][metric],
                             arm["independent_n"], arm["expected_cells"], arm["successful_cells"], arm["status"],
                             arm["stratum_id"], dimensions, *costs, rates, *frozen])
    return stream.getvalue().encode()


def expected_paper_index(aggregate, table, figure_index=None):
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
            "disclosure_scope": aggregate["disclosure_scope"], "claims": claims,
            **({"figure_index_hash": digest(figure_index)} if figure_index is not None else {}),
            **{name: aggregate[name] for name in ("adjudication", "computation_ref") if name in aggregate}}


def _package_file_bytes(root, name, limit, *, root_identity=None):
    """Read one stable regular package file without a stat/read allocation race."""
    from infrastructure.research_files import opened_regular_file
    root = Path(os.path.abspath(root))
    path = root / name
    if type(limit) is not int or limit < 0:
        raise ResearchError("RESOURCE_PLAN_REJECTED", "invalid evidence package byte quota")
    with opened_regular_file(root, path, maximum_bytes=limit, root_identity=root_identity) as (stream, size, _):
        # Bound allocation to the actually admitted size, not just the larger
        # quota. One extra byte detects growth even after the handle check.
        content = stream.read(size + 1)
        # The shared context verifies native/ctime identity and quota BEFORE
        # classifying short/grown returned content, including after real I/O.
    if len(content) > limit:
        raise ResearchError("RESOURCE_PLAN_REJECTED", "evidence package file exceeds byte quota")
    if len(content) != size:
        raise ResearchError("CORRUPT_ARTIFACT", "evidence package file changed during read")
    return content


def accept_evidence_package(store, directory, expected_hash):
    """Import the same frozen JSON/CSV/evidence files that TSDE produced."""
    from infrastructure.research_publication import opened_directory
    root = Path(os.path.abspath(directory))
    with opened_directory(root) as binding:
        # One actual import phase, not persistent provenance/permission facts.
        # Retain every original member read, source check and durable write,
        # then freshly verify the complete final chain before returning IDs.
        with store._read_transaction():
            package = _accept_evidence_package(store, root, expected_hash, binding)
        # Final physical authority I/O may itself replace the external root.
        # Keep its original handle alive and verify AFTER that I/O as well.
        binding[2]()
        return package


def _accept_evidence_package(store, root, expected_hash, binding):
    original_root = os.fstat(binding[1])
    root_identity = original_root.st_dev, original_root.st_ino

    def read_member(name, limit):
        binding[2]()
        content = _package_file_bytes(root, name, limit, root_identity=root_identity)
        binding[2]()
        return content

    contents = {}
    for name in ("manifest.json", "aggregate.json", "metrics.csv", "PaperEvidenceIndex.json"):
        contents[name] = read_member(name, 64 * 1024 * 1024)
    manifest = json.loads(contents["manifest.json"])
    if manifest.get("schema_version") != "pirc25-evidence-package-v1" or manifest.get("aggregate_hash") != expected_hash:
        raise ResearchError("CONTRACT_MISMATCH", "evidence package version/hash mismatch")
    for name in ("aggregate.json", "metrics.csv", "PaperEvidenceIndex.json"):
        if hashlib.sha256(contents[name]).hexdigest() != manifest["files"].get(name):
            raise ResearchError("CORRUPT_ARTIFACT", "evidence package content differs from manifest")
    aggregate = json.loads(contents["aggregate.json"])
    figure_index, figure_contents = None, {}
    if aggregate.get("aggregate_hash") != expected_hash:
        raise ResearchError("CORRUPT_ARTIFACT", "aggregate does not match package")
    if "adjudication" in aggregate or "computation_ref" in aggregate:
        from application.research_computation import verified_computation
        path = root / "ComputationReceipt.json"
        if (path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file()
                or path.stat().st_size > 1024 * 1024):
            raise ResearchError("CONTRACT_MISMATCH", "managed comparison requires a bounded computation receipt")
        contents["ComputationReceipt.json"] = read_member("ComputationReceipt.json", 1024 * 1024)
        if hashlib.sha256(contents["ComputationReceipt.json"]).hexdigest() != manifest["files"].get("ComputationReceipt.json"):
            raise ResearchError("CORRUPT_ARTIFACT", "computation receipt differs from package manifest")
        receipt = json.loads(contents["ComputationReceipt.json"])
        _, worker_result = verified_computation(store, aggregate.get("computation_ref", {}), receipt=receipt, aggregate=aggregate)
        if (contents["metrics.csv"] != worker_result["metrics_csv"].encode() or
                contents["PaperEvidenceIndex.json"] != encode(worker_result["paper_index"])):
            raise ResearchError("CORRUPT_ARTIFACT", "managed evidence is not the exact worker serialization")
        if "figure_index" in worker_result or "figures" in worker_result:
            from application.research_figures import validate_figure_package
            validate_figure_package(aggregate, worker_result.get("figure_index"), worker_result.get("figures"))
            figure_index = worker_result["figure_index"]
            files = {"FigureIndex.json": encode(figure_index),
                     **{name: value.encode("utf-8") for name, value in worker_result["figures"].items()}}
            for name, expected_content in files.items():
                path = root / name
                if (path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file() or
                        path.stat().st_size > 2 * 1024 * 1024):
                    raise ResearchError("CONTRACT_MISMATCH", "frozen figure file is missing or outside quota")
                content = read_member(name, 2 * 1024 * 1024)
                if content != expected_content or hashlib.sha256(content).hexdigest() != manifest["files"].get(name):
                    raise ResearchError("CORRUPT_ARTIFACT", "figure is not the exact managed worker output")
                contents[name] = content
            figure_contents = {name: contents[name] for name in worker_result["figures"]}
    if set(manifest.get("files", {})) != set(contents) - {"manifest.json"}:
        raise ResearchError("CONTRACT_MISMATCH", "evidence manifest contains missing or extra files")
    index = json.loads(contents["PaperEvidenceIndex.json"])
    if index.get("aggregate_hash") != expected_hash or index.get("table_sha256") != manifest["files"]["metrics.csv"]:
        raise ResearchError("CONTRACT_MISMATCH", "paper evidence/table version differs")
    try:
        if contents["metrics.csv"] != expected_metrics_csv(aggregate):
            raise ResearchError("CONTRACT_MISMATCH", "frozen CSV differs from aggregate values")
        if contents["PaperEvidenceIndex.json"] != encode(expected_paper_index(aggregate, contents["metrics.csv"], figure_index)):
            raise ResearchError("CONTRACT_MISMATCH", "paper evidence index differs from aggregate values")
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ResearchError):
            raise
        raise ResearchError("CONTRACT_MISMATCH", "incomplete frozen evidence package") from exc
    binding[2]()
    artifact = accept_aggregate(store, aggregate, aggregate["study_id"])
    attachments = {}
    for name, role in (("metrics.csv", "table"), ("PaperEvidenceIndex.json", "evidence-index")):
        attachments[role] = store.artifact(contents[name], role=role, visibility=artifact["visibility"],
                                          block_ids=artifact["block_ids"], study_id=artifact["study_id"],
                                          media_type="text/csv" if role == "table" else "application/json")["artifact_id"]
    if "ComputationReceipt.json" in contents:
        attachments["computation-receipt"] = store.artifact(contents["ComputationReceipt.json"],
            role="computation-receipt", visibility=artifact["visibility"], block_ids=artifact["block_ids"],
            study_id=artifact["study_id"])["artifact_id"]
    if figure_index is not None:
        attachments["figure-index"] = store.artifact(contents["FigureIndex.json"], role="figure-index",
            visibility=artifact["visibility"], block_ids=artifact["block_ids"], study_id=artifact["study_id"])["artifact_id"]
        attachments["figures"] = []
        for entry in figure_index["figures"]:
            figure = store.artifact(figure_contents[entry["filename"]], role="comparison-figure",
                visibility=artifact["visibility"], block_ids=artifact["block_ids"], study_id=artifact["study_id"],
                media_type="image/svg+xml")
            attachments["figures"].append({**entry, "artifact_id": figure["artifact_id"]})
    package = {"schema_version": "pirc25-imported-evidence-v1", "aggregate_hash": expected_hash,
               "aggregate_id": artifact["artifact_id"], "study_id": artifact["study_id"], **attachments}
    store.publish("comparison-" + expected_hash, package)
    return package
