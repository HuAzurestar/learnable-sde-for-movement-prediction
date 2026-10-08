"""Audited development query demand and explicitly conditional cost scenarios.

No forecasts, model changes or numerical acceptance. Feature rows include the
baseline's history encoder; only non-base rows reach the raw map provider. Map
rows are not SQL calls, unique geometries, cache misses or measured map latency.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from itertools import product
import json
import math
from pathlib import Path

from . import candidate_power as power
from .brownian import integration_grid
from .direct_linear_evidence import add_evidence_arguments, read_bound
from .direct_linear_rollout import resolve_map_backend
from .inference import PRIMARY_FAMILY, SEEDS
from .precision_check import KEYS, load_particle_evidence
from .qualification import _hash
from .resource_replay import audited, numerical_summary

VERSION = "pirc17-candidate-query-demand-v1"
BROWNIAN_VALUE_LIMIT_BYTES = 512 * 1024**2
FAMILY = tuple(sorted({name for pair in PRIMARY_FAMILY.values() for name in pair}))
WALL_FIELD = "rollout_wall_seconds_including_lazy_map_initialization"


def source_hashes():
    return {**power.source_hashes(), "query_demand.py": _hash(Path(__file__))}


def _axis(values, minimum, label):
    if (not values or any(type(v) is not int or v < minimum for v in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"unique integer {label} scenarios required")


def _intervals(horizons, step, history):
    if (not horizons or any(isinstance(t, bool) or not isinstance(t, (int, float))
            or not math.isfinite(t) for t in horizons)
            or isinstance(step, bool) or not isinstance(step, (int, float))
            or not math.isfinite(step) or step <= 0
            or isinstance(history, bool) or not isinstance(history, (int, float))
            or not math.isfinite(history) or history <= 0):
        raise ValueError("finite actual horizons and positive clock steps required")
    return _grid_intervals(horizons, step, history)


@lru_cache(maxsize=256)
def _grid_intervals(horizons, step, history):
    return len(integration_grid(horizons, step, history)) - 1


def project(sources, *, step_seconds, source_particles, particle_counts, block_counts, seed_counts):
    """Arithmetic on complete-audited sources; run() supplies the evidence gate.

Scenario blocks each have ONE hypothetical origin with a measured time-grid
template. Template extrema are not bounds on unseen origins. Wall projection
uses measured per-configuration run means, linear particle scaling and the
specified run count; it excludes all non-rollout work and is not a guarantee.
    """
    _axis([source_particles], 3, "source particle")
    _axis(particle_counts, 3, "particle")
    _axis(block_counts, 1, "block")
    _axis(seed_counts, 1, "seed count")
    if any(n > len(SEEDS) for n in seed_counts):
        raise ValueError("seed scenarios cannot exceed the five registered labels")
    whole, selected, totals, seen = [], [], [], set()
    identity = None
    for rows in sources:
        init, header, end = rows[0], rows[1], rows[-1]
        current_identity = (
            power._without(init, power.WORKLOAD_FIELDS | {"started_at", "wall_seconds", "limit_origins"}),
            power._without(header, power.WORKLOAD_FIELDS | {
                "sample_ids", "selected_independent_block_count", "input_validation_seconds"}))
        if identity is not None and identity != current_identity:
            raise ValueError("query-demand sources mix runtime, model, map or selection identities")
        identity = current_identity
        if not set(FAMILY) <= set(header["configurations"]):
            raise ValueError("each source must retain the complete primary configuration family")
        if step_seconds not in header["max_steps_seconds"] or source_particles not in header["particle_counts"]:
            raise ValueError("selected numerical setting must exist in every complete source")
        runs = rows[2:-1]
        if (end.get("status") != "complete" or len(runs) != init["expected_run_count"]
                or any(r.get("type") != "run" or r.get("status") != "success" for r in runs)):
            raise ValueError("complete sources required; no partial or failed-row salvage")
        feature_rows = map_rows = invalid_rows = 0
        for row in runs:
            key = tuple(row[k] for k in KEYS)
            if key in seen:
                raise ValueError("duplicate workload across query-demand sources")
            seen.add(key)
            count = row["particles"]
            if type(count) is not int or count < 3:
                raise ValueError("invalid source particle count")
            intervals = _intervals(tuple(row["actual_horizons_seconds"]), row["max_step_seconds"],
                                   header["physical_history_step_seconds"])
            expected = intervals * count
            if type(row.get("feature_query_rows")) is not int or row["feature_query_rows"] != expected:
                raise ValueError("recorded feature query rows differ from the actual integration grid")
            missing, wall = row["invalid_feature_rows"], row[WALL_FIELD]
            if type(missing) is not int or not 0 <= missing <= expected:
                raise ValueError("invalid-feature rows must retain their full denominator")
            if isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall < 0:
                raise ValueError("finite nonnegative measured rollout wall time required")
            feature_rows += expected
            map_rows += 0 if row["configuration"] == "base" else expected
            invalid_rows += missing
            whole.append(row)
            if row["max_step_seconds"] == step_seconds and count == source_particles and row["configuration"] in FAMILY:
                selected.append((row, intervals))
        totals.append({"runs": len(runs), "feature_rows": feature_rows, "raw_map_query_rows": map_rows,
                       "invalid_feature_rows": invalid_rows, "measured_rollout_seconds": math.fsum(r[WALL_FIELD] for r in runs)})
    if not selected:
        raise ValueError("no complete primary setting available")
    origins, blocks, actual = {}, set(), set()
    for row, _ in selected:
        sample, block = row["sample_id"], row["independent_block_id"]
        origin_identity = (block, tuple(row["actual_horizons_seconds"]))
        if origins.setdefault(sample, origin_identity) != origin_identity:
            raise ValueError("origin changes block or observed time grid")
        blocks.add(block)
        actual.add((sample, row["configuration"], row["seed"]))
    if actual != set(product(origins, FAMILY, SEEDS)) or len(actual) != len(selected):
        raise ValueError("complete origin-by-family-by-five-seed grid required")
    timings = {}
    for name in FAMILY:
        values = [row[WALL_FIELD] for row, _ in selected if row["configuration"] == name]
        timings[name] = {"measured_runs": len(values), "total_seconds": math.fsum(values),
                         "mean_seconds": math.fsum(values)/len(values), "min_seconds": min(values), "max_seconds": max(values)}
    lo, hi = min(n for _, n in selected), max(n for _, n in selected)
    family_mean = math.fsum(t["mean_seconds"] for t in timings.values())
    scenarios = []
    for count, n_blocks, n_seeds in product(particle_counts, block_counts, seed_counts):
        workload = n_blocks*n_seeds
        # The scenario driver contains this selected step only. Extension with
        # other steps must recheck the UNION grid; do not equate the two guards.
        value_bytes = (hi+1)*count*2*8
        wall = family_mean*workload*(count/source_particles)
        if not math.isfinite(wall):
            raise ValueError("wall-time scenario overflow")
        scenarios.append({"particles": count, "blocks": n_blocks, "origins_per_block": 1, "seed_count": n_seeds,
            "primary_runs": workload*len(FAMILY),
            "feature_rows_template_min": workload*len(FAMILY)*count*lo,
            "feature_rows_template_max": workload*len(FAMILY)*count*hi,
            "raw_map_query_rows_template_min": workload*(len(FAMILY)-1)*count*lo,
            "raw_map_query_rows_template_max": workload*(len(FAMILY)-1)*count*hi,
            "brownian_values_bytes_one_stream_template_max": value_bytes,
            "within_brownian_values_guard_for_templates": value_bytes <= BROWNIAN_VALUE_LIMIT_BYTES,
            "linear_rollout_only_hours": wall/3600, "launch_authorized": False})
    return {"whole_source_counts": totals, "whole_runs": len(whole), "selected_runs": len(selected),
        "source_origins": len(origins), "source_independent_blocks": len(blocks), "registered_seeds": list(SEEDS),
        "selected_configurations": list(FAMILY), "source_particles": source_particles, "step_seconds": step_seconds,
        "observed_integration_intervals_min": lo, "observed_integration_intervals_max": hi,
        "selected_feature_rows": sum(r["feature_query_rows"] for r, _ in selected),
        "selected_raw_map_query_rows": sum(r["feature_query_rows"] for r, _ in selected if r["configuration"] != "base"),
        "selected_invalid_feature_rows": sum(r["invalid_feature_rows"] for r, _ in selected),
        "measured_rollout_seconds_by_configuration": timings, "scenarios": scenarios,
        "brownian_values_limit_bytes": BROWNIAN_VALUE_LIMIT_BYTES,
        "limits": ["template counts only; not actual unseen blocks, representative coverage or a final sample decision",
            "feature rows include baseline history; raw-map rows are not SQL calls, unique geometries or cache misses",
            "linear rollout-only cost scenarios use all selected observed runs, including overlapping system/test load",
            "no controlled cold/hot p50/p95 or peak process-memory measurement; guard success is not memory qualification",
            "Brownian values only, one stream and selected step; allocation temporaries, maps, history and scoring are additional",
            "input validation, training, O(N^2) offline scoring/audits, I/O and supervisor overhead excluded from wall scenarios",
            "four LIO configurations and 28 required NEX326 slots remain unmeasured; this is not a full-matrix budget"]}


def run(*, planning, planning_sha256, ledgers, audits, fit, fit_sha256, fit_ledger,
        fit_ledger_sha256, training_policy_sha256, particle_counts, block_counts, seed_counts, output):
    output = Path(output).resolve()
    if output.exists() or any(output.is_relative_to(Path(p).resolve().with_suffix(".particles")) for p in ledgers):
        raise FileExistsError("refusing to overwrite evidence or write inside an immutable particle directory")
    original = read_bound(planning, planning_sha256)
    if (original.get("schema_version") != power.VERSION or original.get("status") != "complete"
            or original.get("source_sha256") != power.source_hashes()
            or original.get("certified") is not False or original.get("formal_training_accepted") is not False
            or original.get("final_eval_label_prediction_metric_reads") != 0):
        raise ValueError("bound complete candidate planning with unchanged sources required")
    ledger_hashes, audit_hashes = original["input_ledger_sha256"], original["input_audit_sha256"]
    if not ledgers or not len(ledgers) == len(audits) == len(ledger_hashes) == len(audit_hashes):
        raise ValueError("all original whole ledgers and audits required in their bound order")
    evidence = dict(fit=fit, fit_sha256=fit_sha256, fit_ledger=fit_ledger,
        fit_ledger_sha256=fit_ledger_sha256, training_policy_sha256=training_policy_sha256)
    binding = {k: v for k, v in evidence.items() if k.endswith("sha256")}
    if original["candidate_binding"] != binding:
        raise ValueError("candidate evidence differs from the planning artifact")
    sources = source_hashes()
    bound = {Path(planning): planning_sha256, Path(fit): fit_sha256, Path(fit_ledger): fit_ledger_sha256}
    rows_by_source, reports = [], []
    for ledger, audit, sha, audit_sha in zip(ledgers, audits, ledger_hashes, audit_hashes):
        rows, report = read_bound(ledger, sha, jsonl=True), read_bound(audit, audit_sha)
        tolerance = report["numerical_audit"]["tolerance_m"]
        if (report["status"] != "complete" or tolerance != original["parameters"]["delta_m"]/4
                or audited(rows, Path(ledger).parent, sha, tolerance, evidence) != report):
            raise ValueError("whole source audit no longer reproduces at its original tolerance")
        _, modules = resolve_map_backend(rows[1]["map_backend"])
        bound.update({Path(m.__file__): rows[1]["map_source_sha256"][m.__name__] for m in modules})
        bound.update({Path(ledger): sha, Path(audit): audit_sha})
        rows_by_source.append(rows)
        reports.append(report)
    if [r["counts"] for r in reports] != original["source_counts"]:
        raise ValueError("source counts differ from bound planning")
    parameters = original["parameters"]
    projection = project(rows_by_source, step_seconds=parameters["step_seconds"], source_particles=parameters["particles"],
        particle_counts=particle_counts, block_counts=block_counts, seed_counts=seed_counts)
    result = {"schema_version": VERSION, "status": "complete", "source_sha256": sources,
        "planning_sha256": planning_sha256, "input_ledger_sha256": ledger_hashes, "input_audit_sha256": audit_hashes,
        "candidate_binding": binding, "source_numerical": [numerical_summary(r) for r in reports],
        "projection": projection, "certified": False, "numerically_qualified": False, "full_budget_qualified": False,
        "formal_training_accepted": False, "final_eval_label_prediction_metric_reads": 0, "new_forecasts": 0}
    for rows, path in zip(rows_by_source, ledgers):
        for row in rows[2:-1]:
            load_particle_evidence(row, Path(path).parent)
    if source_hashes() != sources or any(_hash(p) != sha for p, sha in bound.items()):
        raise ValueError("bound source or evidence changed during query-demand analysis")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_evidence_arguments(parser)
    parser.add_argument("--planning", type=Path, required=True)
    parser.add_argument("--planning-sha256", required=True)
    parser.add_argument("--ledger", dest="ledgers", type=Path, action="append", required=True)
    parser.add_argument("--audit", dest="audits", type=Path, action="append", required=True)
    for name in ("particle", "block", "seed"):
        parser.add_argument("--"+name+"-counts", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = run(**vars(parser.parse_args()))
    print(json.dumps({k: result[k] for k in ("status", "new_forecasts", "full_budget_qualified", "source_numerical")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
