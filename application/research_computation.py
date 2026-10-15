"""Bindings and owner-side integrity checks for shared statistical jobs."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess

from infrastructure.research_store import ResearchError, digest, encode
from infrastructure.research_files import source_file_hash, opened_regular_file

MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_OPERATIONS = 20_000_000


def computation_input_key(kind):
    if type(kind) is not str or kind not in {"aggregate", "bundle"}:
        raise ResearchError("CONTRACT_MISMATCH", "comparison input kind must be aggregate or bundle")
    return "input_bundle_hash" if kind == "bundle" else "input_aggregate_hash"


def computation_input(reference):
    if type(reference) is not dict:
        raise ResearchError("CONTRACT_MISMATCH", "comparison input reference must be an object")
    kind = reference.get("input_kind", "aggregate")
    key = computation_input_key(kind)
    other = "input_aggregate_hash" if kind == "bundle" else "input_bundle_hash"
    if (type(reference.get(key)) is not str or not reference[key] or other in reference
            or kind == "bundle" and reference[key] != reference.get("source_bundle_hash")):
        raise ResearchError("CONTRACT_MISMATCH", "comparison input reference is mixed or incomplete")
    return key, kind


def paper_identity(root):
    root = Path(os.path.abspath(root))
    directory = root / "scripts" / "pirc25"
    paths = sorted(directory.glob("*.py"))
    if not paths or len(paths) > 64:
        raise ResearchError("CONTRACT_MISMATCH", "bounded paper implementation is required")
    parent = root / "scripts" / "__init__.py"
    # Python executes a present parent initializer before scripts.pirc25. Its
    # absence is also a frozen namespace-package decision, not an omission.
    files = {"scripts/__init__.py": None}
    if parent.exists() or parent.is_symlink():
        paths.append(parent)
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ResearchError("UNAUTHORIZED_DATA", "paper source is outside the declared root/quota")
        try:
            source_hash = source_file_hash(root, path, maximum_bytes=MAX_INPUT_BYTES)
        except (OSError, UnicodeError) as exc:
            raise ResearchError("CONTRACT_MISMATCH", "paper source is not readable UTF-8") from exc
        files[path.relative_to(root).as_posix()] = source_hash
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True, timeout=10).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "scripts/pirc25", "scripts/__init__.py"],
                               check=True, capture_output=True, text=True, timeout=10).stdout.strip())
    return {"schema_version": "pirc25-paper-code-v1", "git_sha": head,
            "source_tree_hash": digest(files), "files": files, "source_tree_dirty": dirty}


def require_resource_summary_support(root):
    """Check the opt-in API without importing or executing Paper source."""
    root = Path(root)
    try:
        for filename, function in (("comparison.py", "compare_package"),
                                   ("computation_plan.py", "comparison_package_plan")):
            path = root/"scripts"/"pirc25"/filename
            with opened_regular_file(root, path, maximum_bytes=MAX_INPUT_BYTES) as (stream, size, _):
                body = stream.read(size+1)
            if len(body) != size:
                raise ValueError("source moved")
            definitions = [n for n in ast.parse(body.decode("utf-8")).body
                           if isinstance(n, ast.FunctionDef) and n.name == function]
            if len(definitions) != 1:
                raise ValueError("missing API")
            defaults = {a.arg: d for a, d in zip(definitions[0].args.kwonlyargs, definitions[0].args.kw_defaults)}
            option = defaults.get("include_resource_summaries")
            if not isinstance(option, ast.Constant) or option.value is not False:
                raise ValueError("missing default-off resource API")
    except (OSError, ValueError, SyntaxError, UnicodeError) as exc:
        raise ResearchError("CONTRACT_MISMATCH", "paired Paper lacks the default-off resource summary API") from exc


def comparison_plan(bundle, max_operations, *, include_resource_summaries=False):
    """A conservative registered-size bound; no metric computation here."""
    if type(include_resource_summaries) is not bool:
        raise ResearchError("CONTRACT_MISMATCH", "resource opt-in must be an explicit boolean")
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
    from application.propagation_resource_summary import resource_work
    summary_work, graph_sources = resource_work(rows) if include_resource_summaries else (0, 0)
    preparation = len(rows) * (4 + 4 * metric_count) + summary_work
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
    from application.research_dimensions import comparison_dimensions
    horizons = {str(comparison_dimensions(row).get("horizon")) for row in rows
                if "horizon" in comparison_dimensions(row)}
    if len(horizons) > 256:
        raise ResearchError("RESOURCE_PLAN_REJECTED", "comparison figure horizon quota exceeded")
    figure_count = 1 + len(horizons)
    graph_operations = figure_count * (256 + len(rows) * (8 + 8 * metric_count) + graph_sources)
    planned = preparation + resampling + graph_operations
    if planned > max_operations:
        raise ResearchError("RESOURCE_PLAN_REJECTED", "registered statistical plan exceeds operation quota")
    return {"schema_version": "pirc25-computation-plan-v1", "maximum_input_bytes": MAX_INPUT_BYTES,
            "maximum_output_bytes": MAX_OUTPUT_BYTES, "maximum_operations": max_operations,
            "registered_cells": len(rows), "registered_blocks": blocks,
            "preparation_operation_bound": preparation, "bootstrap_operation_bound": resampling,
            "comparison_figure_count_bound": figure_count, "figure_byte_bound": 2 * 1024 * 1024,
            "graph_operation_bound": graph_operations,
            "planned_operation_bound": planned, "allocation": "shared-job-on-frozen-reference-arm",
            **({"include_resource_summaries": True} if include_resource_summaries else {})}


def _verified_computation(store, reference, *, receipt=None, aggregate=None):
    """No caller-supplied cost/verdict can replace the authoritative worker.

    Owner integrity read only. It must not return private worker bytes to a
    browser; the query layer separately authorizes original-study artifacts.
    """
    with store._read_transaction():
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
        input_key, input_kind = computation_input(reference)
        other_input = "input_aggregate_hash" if input_kind == "bundle" else "input_bundle_hash"
        if (origin.get("input_kind", "aggregate") != input_kind or request.get("input_kind", "aggregate") != input_kind
                or other_input in origin):
            raise ResearchError("CORRUPT_ARTIFACT", "computation input kind changed")
        enabled = request.get("include_resource_summaries", False)
        if (type(enabled) is not bool or origin.get("include_resource_summaries", False) is not enabled
                or request["resource_plan"].get("include_resource_summaries", False) is not enabled
                or origin["resource_plan"] != request["resource_plan"]):
            raise ResearchError("CORRUPT_ARTIFACT", "computation resource opt-in changed")
        if (run["spec_hash"] != reference["computation_spec_hash"] or run["arm_id"] != proof["cost"]["arm_id"] or
                run["study_id"] != request["computation_study_id"] or
                origin[input_key] != reference[input_key] or
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
        from application.propagation_resource_summary import validate_resource_summaries
        validate_resource_summaries(result["aggregate"], enabled=enabled)
        _verify_job_cost(store, reference, run, proof)
        return proof, result


def _verify_job_cost(store, reference, run, proof):
    """Shared comparison/case job proof. Caller holds the authority lock."""
    events = store._events()
    chains = [[e for e in events if e['event_kind'] == kind and
               e['payload'].get('reservation_id') == reference['reservation_id']]
              for kind in ('RESERVE', 'WORKER_STARTED', 'SETTLE')]
    if any(len(chain) != 1 for chain in chains):
        raise ResearchError('CORRUPT_ARTIFACT', 'computation reserve/worker/settle chain missing')
    reserved, started, settled = [chain[0] for chain in chains]
    charge = settled['payload']
    if (not reserved['sequence'] < started['sequence'] < settled['sequence'] or
            any(e['payload'].get('attempt_id') != reference['attempt_id'] for e in (reserved, started, settled)) or
            charge.get('outcome') != 'SUCCEEDED' or not charge.get('settled') or
            type(charge.get('monotonic_elapsed_ms')) is not int or
            charge['monotonic_elapsed_ms'] != charge['charged_ms'] or
            charge['charged_ms'] <= 0 or charge['arm_id'] != run['arm_id'] or
            charge['charged_ms'] > reserved['payload']['reserved_ms'] or
            any(e['payload'].get('run_id') != run['run_id'] or e['payload'].get('study_id') != run['study_id'] or
                e['payload'].get('arm_id') != run['arm_id'] for e in (reserved, settled)) or
            proof['settlement_event_hash'] != settled['hash'] or
            proof['cost'] != {'arm_id': run['arm_id'], 'charged_ms': charge['charged_ms'],
                             'unit': 'slot-ms', 'scope': 'whole-shared-computation-job', 'basis': 'measured-monotonic'}):
        raise ResearchError('CORRUPT_ARTIFACT', 'computation measured cost/settlement changed')


def verified_computation(store, reference, *, receipt=None, aggregate=None):
    computation_input(reference)
    if not isinstance(reference, dict) or not all(isinstance(reference.get(key), str) for key in (
            "manifest_id", "request_manifest_id", "attempt_id", "run_id", "computation_spec_hash",
            "reservation_id", "source_bundle_hash", "allocation")):
        raise ResearchError("CONTRACT_MISMATCH", "managed comparison reference is incomplete")
    try:
        return _verified_computation(store, reference, receipt=receipt, aggregate=aggregate)
    except (KeyError, TypeError, ValueError) as exc:
        raise ResearchError("CORRUPT_ARTIFACT", "managed computation provenance is malformed") from exc
