"""Bindings and owner-side integrity checks for shared statistical jobs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from infrastructure.research_store import ResearchError, digest, encode

MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_OPERATIONS = 20_000_000


def paper_identity(root):
    root = Path(root).resolve()
    directory = root / "scripts" / "pirc25"
    paths = sorted(directory.glob("*.py"))
    if not paths or len(paths) > 64:
        raise ResearchError("CONTRACT_MISMATCH", "bounded paper implementation is required")
    files = {}
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(root) or path.stat().st_size > MAX_INPUT_BYTES:
            raise ResearchError("UNAUTHORIZED_DATA", "paper source is outside the declared root/quota")
        files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True, timeout=10).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "scripts/pirc25"],
                               check=True, capture_output=True, text=True, timeout=10).stdout.strip())
    return {"schema_version": "pirc25-paper-code-v1", "git_sha": head,
            "source_tree_hash": digest(files), "files": files, "source_tree_dirty": dirty}


def comparison_plan(bundle, max_operations):
    """A conservative registered-size bound; no metric computation here."""
    if type(max_operations) is not int or not 0 < max_operations <= MAX_OPERATIONS:
        raise ResearchError("RESOURCE_PLAN_REJECTED", "invalid statistical operation quota")
    rows = bundle.get("cells")
    if not isinstance(rows, list) or not rows or len(rows) > 100_000:
        raise ResearchError("RESOURCE_PLAN_REJECTED", "registered cell quota exceeded")
    frozen_plan = bundle.get("comparison_plan") or {}
    if not isinstance(frozen_plan, dict):
        raise ResearchError("RESOURCE_PLAN_REJECTED", "comparison plan must be a mapping")
    policy = frozen_plan.get("adjudication_spec")
    if policy is not None and not isinstance(policy, dict):
        raise ResearchError("RESOURCE_PLAN_REJECTED", "statistical policy must be a mapping")
    contrasts = policy.get("contrasts", []) if isinstance(policy, dict) else []
    if not isinstance(contrasts, list) or len(contrasts) > 256:
        raise ResearchError("RESOURCE_PLAN_REJECTED", "fixed comparison family quota exceeded")
    # Preparation, descriptive means/serialization and bootstrap allocations
    # all have bounds. Failed/incomplete blocks never lower the preflight cap.
    if any(not isinstance(row, dict) or not isinstance(row.get("block_id"), str) or
           (row.get("metrics") is not None and not isinstance(row["metrics"], dict)) for row in rows):
        raise ResearchError("RESOURCE_PLAN_REJECTED", "registered computation rows are malformed")
    metric_count = max((len(row.get("metrics") or {}) for row in rows), default=0)
    if metric_count > 256:
        raise ResearchError("RESOURCE_PLAN_REJECTED", "metric quota exceeded")
    preparation = len(rows) * (4 + 4 * metric_count)
    interval = (policy.get("interval") or {}) if isinstance(policy, dict) else {}
    if not isinstance(interval, dict):
        raise ResearchError("RESOURCE_PLAN_REJECTED", "interval policy must be a mapping")
    repetitions = interval.get("replicates", 0)
    if type(repetitions) is not int or not 0 <= repetitions <= 10000:
        raise ResearchError("RESOURCE_PLAN_REJECTED", "bootstrap repetition quota exceeded")
    for contrast in contrasts:
        if not isinstance(contrast, dict):
            raise ResearchError("RESOURCE_PLAN_REJECTED", "contrast policy must be a mapping")
        weights = contrast.get("stratum_weights", [])
        if not isinstance(weights, list) or len(weights) > 256:
            raise ResearchError("RESOURCE_PLAN_REJECTED", "weighted stratum quota exceeded")
        preparation += len(rows) * (1 + 2 * len(weights))
    blocks = len({row["block_id"] for row in rows})
    resampling = repetitions * blocks * len(contrasts)
    planned = preparation + resampling
    if planned > max_operations:
        raise ResearchError("RESOURCE_PLAN_REJECTED", "registered statistical plan exceeds operation quota")
    return {"schema_version": "pirc25-computation-plan-v1", "maximum_input_bytes": MAX_INPUT_BYTES,
            "maximum_output_bytes": MAX_OUTPUT_BYTES, "maximum_operations": max_operations,
            "registered_cells": len(rows), "registered_blocks": blocks,
            "preparation_operation_bound": preparation, "bootstrap_operation_bound": resampling,
            "planned_operation_bound": planned, "allocation": "shared-job-on-frozen-reference-arm"}


def _verified_computation(store, reference, *, receipt=None, aggregate=None):
    """No caller-supplied cost/verdict can replace the authoritative worker.

    Owner integrity read only. It must not return private worker bytes to a
    browser; the query layer separately authorizes original-study artifacts.
    """
    with store.lock():
        proof = store._manifest(reference["manifest_id"])
        request = store._manifest(reference["request_manifest_id"])
        if (proof.get("schema_version") != "pirc25-computation-receipt-v1" or
                proof.get("computation_ref") != reference or request.get("computation_ref") != reference or
                proof.get("request_hash") != digest(request) or (receipt is not None and receipt != proof)):
            raise ResearchError("CORRUPT_ARTIFACT", "statistical computation reference/receipt changed")
        attempt = store._attempts().get(reference["attempt_id"])
        if not attempt or attempt["state"] != "SUCCEEDED" or attempt["run_id"] != reference["run_id"]:
            raise ResearchError("CORRUPT_ARTIFACT", "computation has no successful authoritative attempt")
        run = store._manifest("run-" + attempt["run_id"])
        origin = store._manifest("study-" + run["study_id"])["spec"]["computation_origin"]
        if (run["spec_hash"] != reference["computation_spec_hash"] or run["arm_id"] != proof["cost"]["arm_id"] or
                run["study_id"] != request["computation_study_id"] or
                origin["input_aggregate_hash"] != reference["input_aggregate_hash"] or
                origin["source_bundle_hash"] != reference["source_bundle_hash"] or
                origin["paper_identity"] != request["paper_identity"] or origin["formal"] != request["formal"] or
                attempt["artifact_id"] != proof["result_artifact_id"]):
            raise ResearchError("CORRUPT_ARTIFACT", "computation run or charged arm identity changed")
        metadata = store._manifest("artifact-" + attempt["artifact_id"])
        if (metadata["role"] != "result" or metadata["study_id"] != run["study_id"] or
                digest(metadata) != attempt["artifact_manifest_hash"]):
            raise ResearchError("CORRUPT_ARTIFACT", "computation result metadata changed")
        result = json.loads(store._verified_artifact_content(metadata))
        if (result.get("schema_version") != "pirc25-computation-result-v1" or
                result.get("request_hash") != digest(request) or result.get("computation_ref") != reference or
                result["aggregate"]["computation_ref"] != reference or
                result["aggregate"]["aggregate_hash"] != proof["aggregate_hash"] or
                result["aggregate"]["source_bundle_hash"] != reference["source_bundle_hash"] or
                (aggregate is not None and aggregate != result["aggregate"])):
            raise ResearchError("CORRUPT_ARTIFACT", "imported adjudication is not the managed worker output")
        events = store._events()
        reserves = [e for e in events if e["event_kind"] == "RESERVE" and
                    e["payload"].get("reservation_id") == reference["reservation_id"]]
        starts = [e for e in events if e["event_kind"] == "WORKER_STARTED" and
                  e["payload"].get("reservation_id") == reference["reservation_id"]]
        settles = [e for e in events if e["event_kind"] == "SETTLE" and
                   e["payload"].get("reservation_id") == reference["reservation_id"]]
        if len(reserves) != 1 or len(starts) != 1 or len(settles) != 1:
            raise ResearchError("CORRUPT_ARTIFACT", "computation reserve/worker/settle chain missing")
        reserved, started, settled = reserves[0], starts[0], settles[0]
        charge = settled["payload"]
        if (not reserved["sequence"] < started["sequence"] < settled["sequence"] or
                any(e["payload"].get("attempt_id") != reference["attempt_id"] for e in (reserved, started, settled)) or
                charge.get("outcome") != "SUCCEEDED" or not charge.get("settled") or
                type(charge.get("monotonic_elapsed_ms")) is not int or
                charge["monotonic_elapsed_ms"] != charge["charged_ms"] or
                charge["charged_ms"] <= 0 or charge["arm_id"] != run["arm_id"] or
                charge["charged_ms"] > reserved["payload"]["reserved_ms"] or
                any(e["payload"].get("run_id") != run["run_id"] or
                    e["payload"].get("study_id") != run["study_id"] or
                    e["payload"].get("arm_id") != run["arm_id"] for e in (reserved, settled)) or
                proof["settlement_event_hash"] != settled["hash"] or
                proof["cost"] != {"arm_id": run["arm_id"], "charged_ms": charge["charged_ms"],
                    "unit": "slot-ms", "scope": "whole-shared-computation-job", "basis": "measured-monotonic"}):
            raise ResearchError("CORRUPT_ARTIFACT", "computation measured cost/settlement changed")
        return proof, result


def verified_computation(store, reference, *, receipt=None, aggregate=None):
    if not isinstance(reference, dict) or not all(isinstance(reference.get(key), str) for key in (
            "manifest_id", "request_manifest_id", "attempt_id", "run_id", "computation_spec_hash",
            "reservation_id", "input_aggregate_hash", "source_bundle_hash", "allocation")):
        raise ResearchError("CONTRACT_MISMATCH", "managed comparison reference is incomplete")
    try:
        return _verified_computation(store, reference, receipt=receipt, aggregate=aggregate)
    except (KeyError, TypeError, ValueError) as exc:
        raise ResearchError("CORRUPT_ARTIFACT", "managed computation provenance is malformed") from exc
