"""Describe retained terrain-development prefix clocks, without new experiments.

Read only identity/clock columns of the original selected development segments.
No coordinates, velocity estimates, map pixels, fits or forecasts are decoded.
The first-tick span is buffer arithmetic, not a simulated sensitivity experiment.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .checkpoint_resume import load
from .feature_design_description import ORIGINAL_CONTEXT_FILE_SHA, ROLES
from .protocol_core import canonical, digest, file_hash, read_json, under, unpack
from .terrain_fit_scope_description import METADATA_COLUMNS, first_interval_ns

NS = 1_000_000_000
SOURCES = ("experiments/pirc17/development.py", "experiments/pirc17/origins.py",
           "experiments/pirc17/rollout.py")


def prefix_clock_record(sample, rows, dataset_id):
    """Use segment offsets, not original-file point identities or nominal dt."""
    first_interval_ns(sample, rows, dataset_id)
    start, end = sample["history_start"], sample["history_end"]
    if type(start) is not int or not 0 <= start < end:
        raise ValueError("at least two visible development history points required")
    ordered = sorted(rows, key=lambda row: row["point_index"])
    prefix = ordered[max(start, end - 2):end + 1]
    ticks = []
    for row in prefix:
        for key, expected in (("dataset_version", dataset_id), ("split", sample["split"]),
                              ("file_id", sample["file_id"]), ("segment_id", sample["segment_id"]),
                              ("independent_block_id", sample["independent_block_id"])):
            if row[key] != expected:
                raise ValueError("original prefix clock identity mismatch")
        if type(row["absolute_epoch_ns"]) is not int:
            raise ValueError("integer recorded prefix clock ticks required")
        ticks.append(row["absolute_epoch_ns"])
    if len(ticks) not in (2, 3) or any(not 0 < b - a <= 60 * NS for a, b in zip(ticks, ticks[1:])):
        raise ValueError("prefix gaps outside original positive 60-second support")
    return {"points": len(ticks), "origin_span_ns": ticks[-1] - ticks[0],
            "last_gap_ns": ticks[-1] - ticks[-2],
            "first_tick_span_ns": ticks[-1] - ticks[-2] + 5 * NS}


def describe_spans(values_ns, *, maximum_ns):
    if (not values_ns or any(type(x) is not int or not 0 < x <= maximum_ns for x in values_ns)):
        raise ValueError("positive bounded integer-nanosecond prefix spans required")
    seconds = np.asarray(values_ns, dtype=float) / NS
    quantiles = np.quantile(seconds, [.25, .5, .75], method="linear")
    edges = [0, 5, 10, 30, 60, 120]
    bins = {f"({a},{b}]": sum(a * NS < x <= b * NS for x in values_ns)
            for a, b in zip(edges, edges[1:])}
    if sum(bins.values()) != len(values_ns):
        raise ValueError("span outside descriptive bins")
    return {"count": len(values_ns), "minimum_seconds": float(seconds.min()),
            "q25_seconds": float(quantiles[0]), "median_seconds": float(quantiles[1]),
            "q75_seconds": float(quantiles[2]), "maximum_seconds": float(seconds.max()),
            "mean_seconds": float(seconds.mean()), "span_bin_counts_seconds": bins,
            "compared_with_stable_10s": {
                "shorter": sum(x < 10 * NS for x in values_ns),
                "equal": sum(x == 10 * NS for x in values_ns),
                "longer": sum(x > 10 * NS for x in values_ns)}}


def describe_role(records):
    if not records or any(r["points"] not in (2, 3) for r in records):
        raise ValueError("retained two/three-point prefix records required")
    return {"windows": len(records),
            "retained_point_counts": {str(n): sum(r["points"] == n for r in records) for n in (2, 3)},
            "origin_span": describe_spans([r["origin_span_ns"] for r in records], maximum_ns=120 * NS),
            "last_gap": describe_spans([r["last_gap_ns"] for r in records], maximum_ns=60 * NS),
            "first_tick_span_from_buffer_rule": describe_spans(
                [r["first_tick_span_ns"] for r in records], maximum_ns=65 * NS)}


def project(runtime, original_context, feature_path, feature_sha, history_path, history_sha):
    settings, _ = load(runtime)
    original = unpack(read_json(original_context, expected_file_sha256=ORIGINAL_CONTEXT_FILE_SHA))
    inputs = unpack(original["input_identity"])
    current = unpack(read_json(settings["context"]), expected_sha256=settings["context_sha256"])
    if inputs["terrain_development_identity"] != unpack(current["input_identity"])["terrain_development_identity"]:
        raise ValueError("original and resumed terrain development populations differ")
    bundle = unpack(read_json(settings["bundle"]))
    protocol, execution = unpack(bundle["protocol"]), unpack(bundle["execution"])
    if original["protocol_sha256"] != bundle["protocol"]["sha256"]:
        raise ValueError("original development protocol differs")
    feature = read_json(feature_path, expected_file_sha256=feature_sha)
    history = read_json(history_path, expected_file_sha256=history_sha)
    if (feature["original_input_identity_sha256"] != original["input_identity"]["sha256"]
            or feature["original_context_file_sha256"] != ORIGINAL_CONTEXT_FILE_SHA
            or history["protocol_sha256"] != bundle["protocol"]["sha256"]
            or history["execution_sha256"] != bundle["execution"]["sha256"]
            or history["terrain_history_tick_seconds"] != 5
            or history["stable_terrain_secant_span_seconds"] != 10):
        raise ValueError("original feature/history projection bindings differ")
    repository = Path(__file__).resolve().parents[2]
    sources = {}
    for name in SOURCES:
        expected = protocol["source_sha256"]["PSDE-SDE/" + name]
        if (file_hash(repository / name) != expected
                or execution["source_sha256"]["PSDE-SDE/" + name] != expected):
            raise ValueError("original prefix/buffer source differs")
        sources[name] = expected
    binding = protocol["dataset_inputs"]
    release, snapshot = (Path(settings["input_paths"][name]) for name in ("release", "snapshot"))
    manifest = read_json(snapshot / "manifest.json", expected_file_sha256=binding["snapshot"]["manifest_sha256"])
    samples_path = release / "samples.jsonl"
    if file_hash(samples_path) != binding["release_artifact_sha256"]["samples.jsonl"]:
        raise ValueError("original sample index differs")
    development = inputs["terrain_development_identity"]
    wanted = {sid for role in ROLES for sid in development["sample_ids"][role]}
    if len(wanted) != sum(ROLES.values()):
        raise ValueError("complete disjoint original development windows required")
    samples = {}
    with samples_path.open(encoding="utf-8") as stream:
        for line in stream:
            sample = json.loads(line)
            if sample["sample_id"] in wanted:
                if sample["sample_id"] in samples:
                    raise ValueError("duplicate original development sample")
                samples[sample["sample_id"]] = sample
    if set(samples) != wanted:
        raise ValueError("original development sample missing")
    groups = defaultdict(list)
    for role, count in ROLES.items():
        if len(development["sample_ids"][role]) != count:
            raise ValueError("original development role count differs")
        for sid in development["sample_ids"][role]:
            sample = samples[sid]
            if sample["split"] != role:
                raise ValueError("no final-evaluation numeric rows allowed")
            groups[role, sample["file_id"]].append(sample)
    entries = {(e["split"], e["file_id"]): e for e in manifest["files"]}
    records, selected_sources = {}, {}
    for (role, file_id), group in sorted(groups.items()):
        entry = entries[role, file_id]
        bound = development["sources"][role + ":" + file_id]
        if entry["sha256"] != bound["feature_sha256"] or entry["condition_sha256"] != bound["condition_sha256"]:
            raise ValueError("original development clock binding differs")
        path = under(snapshot, entry["path"])
        if file_hash(path) != entry["sha256"] or pq.ParquetFile(path).metadata.num_rows != entry["row_count"]:
            raise ValueError("selected development clock file differs")
        table = pq.read_table(path, columns=METADATA_COLUMNS,
                              filters=[("split", "=", role),
                                       ("segment_id", "in", sorted({s["segment_id"] for s in group}))])
        segments = defaultdict(list)
        for row in table.to_pylist():
            segments[row["segment_id"]].append(row)
        for sample in group:
            records[sample["sample_id"]] = prefix_clock_record(
                sample, segments[sample["segment_id"]], binding["dataset_id"])
        selected_sources[role + ":" + file_id] = entry["sha256"]
    if (len(selected_sources) != feature["selected_feature_files"]
            or digest(selected_sources) != feature["selected_sources_sha256"] or set(records) != wanted):
        raise ValueError("original selected design/clock population differs")
    return {"schema_version": "pirc17-original-development-history-description-v1",
            "original_context_file_sha256": ORIGINAL_CONTEXT_FILE_SHA,
            "original_input_identity_sha256": original["input_identity"]["sha256"],
            "protocol_sha256": bundle["protocol"]["sha256"],
            "execution_sha256": bundle["execution"]["sha256"],
            "base_feature_projection_sha256": feature_sha, "base_history_projection_sha256": history_sha,
            "selected_feature_files": len(selected_sources), "selected_sources_sha256": digest(selected_sources),
            "metadata_columns_decoded": METADATA_COLUMNS, "source_sha256": sources,
            "terrain_history_tick_seconds": 5, "stable_terrain_secant_span_seconds": 10,
            "roles": {role: describe_role([records[sid] for sid in development["sample_ids"][role]]) for role in ROLES},
            "scope": {"population": "404 terrain training and 81 validation origins, not ordinary-method resampled history",
                      "first_tick_spans_are_buffer_arithmetic_not_simulated_outputs": True,
                      "positions_velocities_drift_residuals_maps_or_final_eval_numeric_rows_decoded": False,
                      "clock_ticks_sample_identifiers_or_private_source_paths_exported": False,
                      "new_fits_forecasts_scores_resampling_or_map_queries": 0,
                      "training_rollout_distribution_equivalence_established": False,
                      "history_sensitivity_or_out_of_bounds_rates_established": False,
                      "physical_source_clock_provenance_certified": False,
                      "whole_snapshot_revalidated": False,
                      "independent_saved_forecast_audit_completed": False,
                      "all_review_items_or_paper_complete": False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "original-context", "feature-projection", "history-projection", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("feature-projection-sha256", "history-projection-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = project(args.runtime, args.original_context, args.feature_projection,
                     args.feature_projection_sha256, args.history_projection, args.history_projection_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print("Described original 404/81 terrain development prefix clocks; no fits or forecasts.")


if __name__ == "__main__":
    main()
