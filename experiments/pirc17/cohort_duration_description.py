"""Describe recorded time spans for every original final window; no experiments.

Only retained identity/clock columns are decoded. Saved eligibility and selection
remain unchanged, including invalid-clock windows. Recorded spans are not a
certification of physical GPX time or new forecast horizons.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .checkpoint_resume import load
from .cohort_geography_description import (COHORT_SHA, COVERAGE, ELIGIBILITY_FILE_SHA,
    ELIGIBILITY_SHA, METADATA, POPULATION_SHA, TEMPORAL)
from .protocol_core import canonical, decode, digest, file_hash, read_json, under, unpack

COLUMNS = ["dataset_version", "file_id", "segment_id", "split", "independent_block_id",
           "point_index", "absolute_epoch_ns"]
NS = 1_000_000_000


def window_spans(sample, rows, dataset_id):
    """Sample bounds are segment offsets, not original-file point indexes."""
    bounds = [sample[k] for k in ("history_start", "history_end", "target_start", "target_end")]
    if (sample["split"] != "final_eval" or any(type(x) is not int for x in bounds)
            or not 0 <= bounds[0] <= bounds[1] < bounds[2] <= bounds[3]
            or bounds[2] != bounds[1] + 1 or len(rows) != bounds[3] + 1):
        raise ValueError("complete original final segment with unchanged bounds required")
    if (any(set(r) != set(COLUMNS) or type(r["point_index"]) is not int
            or type(r["absolute_epoch_ns"]) is not int for r in rows)
            or len({r["point_index"] for r in rows}) != len(rows)):
        raise ValueError("unique identity and integer-clock columns only required")
    ordered = sorted(rows, key=lambda r: r["point_index"])
    for row in ordered:
        if any(row[k] != v for k, v in (("dataset_version", dataset_id),
                ("file_id", sample["file_id"]), ("segment_id", sample["segment_id"]),
                ("split", sample["split"]), ("independent_block_id", sample["independent_block_id"]))):
            raise ValueError("original clock identity differs")
    first, origin, last = (ordered[i]["absolute_epoch_ns"] for i in (bounds[0], bounds[1], bounds[3]))
    return {"visible_history_ns": origin - first, "followup_ns": last - origin,
            "whole_window_ns": last - first}


def describe(values):
    if any(type(v) is not int for v in values):
        raise ValueError("recorded integer-nanosecond differences required")
    if not values:
        return {"windows": 0, "minimum_seconds": None, "q25_seconds": None,
                "median_seconds": None, "q75_seconds": None, "maximum_seconds": None,
                "nonpositive_count": 0, "at_least_1800s_count": 0}
    seconds = np.asarray(values, dtype=float) / NS
    quartiles = np.quantile(seconds, [.25, .5, .75], method="linear")
    return {"windows": len(values), "minimum_seconds": float(seconds.min()),
            "q25_seconds": float(quartiles[0]), "median_seconds": float(quartiles[1]),
            "q75_seconds": float(quartiles[2]), "maximum_seconds": float(seconds.max()),
            "nonpositive_count": sum(v <= 0 for v in values),
            "at_least_1800s_count": sum(v >= 1800 * NS for v in values)}


def summarize(rows, clocks, selected_ids):
    groups = {k: [] for k in ("released", "metadata_complete", "temporal_support",
        "joint_feature_validity", "selected_primary", "metadata_excluded", "temporal_excluded",
        "coverage_excluded_after_temporal", "eligible_not_selected")}
    seen = set()
    for row in rows:
        sid, reasons = row["sample_id"], row["reasons"]
        if (row["split"] != "final_eval" or sid in seen or sid not in clocks
                or not isinstance(reasons, list) or any(not isinstance(r, str) for r in reasons)
                or len(set(reasons)) != len(reasons) or set(reasons) - METADATA - TEMPORAL - COVERAGE
                or type(row["eligible"]) is not bool or row["eligible"] != (not reasons)):
            raise ValueError("complete unchanged eligibility population required")
        seen.add(sid)
        groups["released"].append(row)
        if METADATA.intersection(reasons):
            groups["metadata_excluded"].append(row)
            continue
        groups["metadata_complete"].append(row)
        if TEMPORAL.intersection(reasons):
            groups["temporal_excluded"].append(row)
            continue
        groups["temporal_support"].append(row)
        if COVERAGE.intersection(reasons):
            groups["coverage_excluded_after_temporal"].append(row)
            continue
        groups["joint_feature_validity"].append(row)
        groups["selected_primary" if sid in selected_ids else "eligible_not_selected"].append(row)
    if (seen != set(clocks) or not selected_ids
            or {r["sample_id"] for r in groups["selected_primary"]} != selected_ids
            or len({r["independent_block_id"] for r in groups["selected_primary"]}) != len(selected_ids)):
        raise ValueError("complete clock support and unchanged eligible block selection required")
    return {name: {"windows": len(group),
            "recording_hash_blocks": len({r["independent_block_id"] for r in group}),
            **{span: describe([clocks[r["sample_id"]][span + "_ns"] for r in group])
               for span in ("visible_history", "followup", "whole_window")}}
            for name, group in groups.items()}


def project(runtime, eligibility_path, cohort_path):
    settings, _ = load(runtime)
    context = unpack(read_json(settings["context"]), expected_sha256=settings["context_sha256"])
    bundle = unpack(read_json(settings["bundle"]), expected_sha256=settings["bundle_sha256"])
    protocol, execution = unpack(bundle["protocol"]), unpack(bundle["execution"])
    population = unpack(context["population"], expected_sha256=POPULATION_SHA)
    eligibility = unpack(read_json(eligibility_path, expected_file_sha256=ELIGIBILITY_FILE_SHA), expected_sha256=ELIGIBILITY_SHA)
    if (population["eligibility_sha256"] != ELIGIBILITY_SHA
            or any(value["protocol_sha256"] != bundle["protocol"]["sha256"]
                   or value["execution_sha256"] != bundle["execution"]["sha256"]
                   or value["prior_performance_reads"] != 0 for value in (population, eligibility))
            or context["protocol_sha256"] != bundle["protocol"]["sha256"]
            or context["execution_sha256"] != bundle["execution"]["sha256"]
            or execution["protocol_sha256"] != bundle["protocol"]["sha256"]):
        raise ValueError("original saved input/population chain differs")
    cohort = read_json(cohort_path, expected_file_sha256=COHORT_SHA)
    binding = protocol["dataset_inputs"]
    release, snapshot = (Path(settings["input_paths"][k]) for k in ("release", "snapshot"))
    sample_path = release / "samples.jsonl"
    sample_sha = binding["release_artifact_sha256"]["samples.jsonl"]
    if file_hash(sample_path) != sample_sha:
        raise ValueError("original complete sample metadata differs")
    samples = {}
    with sample_path.open("rb") as stream:
        while raw := stream.readline(16385):
            if len(raw) > 16384:
                raise ValueError("bounded sample metadata row required")
            sample = decode(raw)
            if sample["split"] == "final_eval":
                if sample["sample_id"] in samples or sample["data_version"] != binding["dataset_id"]:
                    raise ValueError("unique final release identity required")
                samples[sample["sample_id"]] = sample
    rows = eligibility["rows"]
    names = ("sample_id", "segment_id", "independent_block_id")
    if (len(samples) != len(rows)
            or any(tuple(samples[r["sample_id"]][k] for k in names) != tuple(r[k] for k in names) for r in rows)
            or digest({k: sorted({r[k] for r in rows}) for k in names}) != population["full_population_identity_sha256"]
            or digest(sorted([r[k] for k in names] for r in rows)) != population["full_population_row_identity_sha256"]):
        raise ValueError("original full final population required; no selected-only clock sample")
    manifest_sha = binding["snapshot"]["manifest_sha256"]
    manifest = read_json(snapshot / "manifest.json", expected_file_sha256=manifest_sha)
    entries = {}
    for entry in manifest["files"]:
        if entry["split"] == "final_eval":
            if entry["file_id"] in entries:
                raise ValueError("unique final feature source required")
            entries[entry["file_id"]] = entry
    groups = defaultdict(list)
    for sample in samples.values():
        groups[sample["file_id"]].append(sample)
    clocks, sources = {}, {}
    for file_id, group in sorted(groups.items()):
        entry = entries[file_id]
        path = under(snapshot, entry["path"])
        if file_hash(path) != entry["sha256"] or pq.ParquetFile(path).metadata.num_rows != entry["row_count"]:
            raise ValueError("bound final clock file differs")
        table = pq.read_table(path, columns=COLUMNS, filters=[("segment_id", "in", [s["segment_id"] for s in group])])
        segments = defaultdict(list)
        for row in table.to_pylist():
            segments[row["segment_id"]].append(row)
        for sample in group:
            clocks[sample["sample_id"]] = window_spans(sample, segments[sample["segment_id"]], binding["dataset_id"])
        sources[file_id] = entry["sha256"]
    selected = population["selection"]["selected"]
    if len(selected) != 46 or digest(selected) != population["selection"]["selection_sha256"]:
        raise ValueError("unchanged original selection required")
    stages = summarize(rows, clocks, {r["sample_id"] for r in selected})
    for stage in ("released", "metadata_complete", "temporal_support", "joint_feature_validity", "selected_primary"):
        if any(stages[stage][k] != cohort["stages"][stage][k] for k in ("windows", "recording_hash_blocks")):
            raise ValueError("original saved stage denominators differ")
    return {"schema_version": "pirc17-original-final-cohort-duration-description-v1", "stages": stages,
        "bindings": {"protocol_sha256": bundle["protocol"]["sha256"], "execution_sha256": bundle["execution"]["sha256"],
                     "population_sha256": POPULATION_SHA, "eligibility_sha256": ELIGIBILITY_SHA,
                     "eligibility_file_sha256": ELIGIBILITY_FILE_SHA, "samples_file_sha256": sample_sha,
                     "snapshot_manifest_sha256": manifest_sha, "base_cohort_description_sha256": COHORT_SHA,
                     "used_clock_files": len(sources), "used_clock_sources_sha256": digest(sources)},
        "definition": "All released final midpoint windows, one value per original window; visible=origin-first, followup=last-origin, whole=last-first; segment-offset bounds; linear descriptive quartiles, not forecast interpolation",
        "scope": {"decoded_columns": COLUMNS, "original_eligibility_or_selection_recomputed": False,
                  "positions_speeds_feature_values_or_prediction_arrays_decoded": False,
                  "invalid_clock_windows_dropped_or_time_differences_clipped": False,
                  "physical_source_clock_provenance_certified": False,
                  "new_fits_forecasts_scores_resampling_or_map_queries": 0,
                  "whole_snapshot_revalidation_or_forecast_restart_gate": False,
                  "independent_saved_output_audit_completed": False,
                  "private_identities_epochs_paths_or_coordinates_exported": False,
                  "all_review_items_or_paper_complete": False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "eligibility", "cohort", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = project(args.runtime, args.eligibility, args.cohort)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print("Described all12370 original final-window clocks; no experiments or eligibility changes.")


if __name__ == "__main__":
    main()
