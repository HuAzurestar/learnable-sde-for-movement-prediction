"""Describe saved interval-speed scalars for the unchanged final population.

Read only original release identities and source time/speed columns. The source
ordinal join matches the frozen adapter, but does not construct model inputs,
deduplicate observations, select new windows, fit, score or predict. Unusable
joins/scalars remain explicit statistical failures in the original denominator.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import duckdb
import numpy as np

from .checkpoint_resume import load
from .cohort_geography_description import (COHORT_SHA, COVERAGE, ELIGIBILITY_FILE_SHA,
    ELIGIBILITY_SHA, METADATA, POPULATION_SHA, TEMPORAL)
from .protocol_core import canonical, decode, digest, file_hash, read_json, unpack

ALIGNMENT_COLUMNS = {"file_id": "VARCHAR", "segment_id": "VARCHAR",
    "source_segment_id": "VARCHAR", "segment_point_index": "BIGINT", "source_point_index": "BIGINT"}
SOURCE_COLUMNS = ["file_id", "segment_id", "t", "speed"]


def window_statistic(sample, row):
    """No finite-value subset, clipping, repair or silent loss of windows."""
    bounds = [sample[k] for k in ("history_start", "history_end", "target_start", "target_end")]
    if (sample["split"] != "final_eval" or any(type(x) is not int for x in bounds)
            or not 0 == bounds[0] <= bounds[1] < bounds[2] <= bounds[3]
            or bounds[2] != bounds[1] + 1):
        raise ValueError("unchanged complete original midpoint window required")
    if row is None:
        return {"median_saved_speed": None, "reason": "missing-alignment-window"}
    checks = [
        (row["n"] == bounds[3] + 1 and row["unique_indexes"] == row["n"]
         and row["first_index"] == 0 and row["last_index"] == bounds[3], "incomplete-or-duplicate-window-indexes"),
        (row["file_count"] == 1 and row["file_id"] == sample["file_id"] and row["parent_count"] == 1,
         "alignment-window-identity-mismatch"),
        (row["missing_source"] == 0, "missing-source-point"),
        (row["wrong_source_file"] == 0, "source-file-identity-mismatch"),
        (row["parent_count_mismatch"] == 0, "source-parent-count-mismatch"),
        (row["parent_order_invalid"] == 0, "ambiguous-source-parent-order"),
        (row["invalid_speed"] == 0, "nonfinite-or-negative-saved-speed"),
    ]
    for ok, reason in checks:
        if not ok:
            return {"median_saved_speed": None, "reason": reason}
    value = row["median_speed"]
    if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
        raise ValueError("complete valid source speed aggregate required")
    return {"median_saved_speed": float(value), "reason": None}


def joined_statistics(alignment, trajectory, samples, connection):
    """Two bounded-column alignment scans; only needed parent rows are ranked.

    Ranking BEFORE filtering refined windows is important: refinement does not
    reset the parent source ordinal. Count/uniqueness guards forbid an ordinal
    join when the retained alignment omits parent points or has ambiguous order.
    """
    connection.execute("CREATE TEMP TABLE wanted(segment_id VARCHAR)")
    connection.executemany("INSERT INTO wanted VALUES (?)", [(s["segment_id"],) for s in samples.values()])
    definition = ",".join("'" + key + "':'" + value + "'" for key, value in ALIGNMENT_COLUMNS.items())
    scan = "read_json(?, format='newline_delimited', columns={" + definition + "})"
    connection.execute("CREATE TEMP TABLE source_ids AS SELECT DISTINCT a.source_segment_id FROM "
                       + scan + " a JOIN wanted w USING(segment_id)", [str(alignment)])
    connection.execute("""CREATE TEMP TABLE alignment_ranked AS SELECT a.*,
        row_number() OVER(PARTITION BY source_segment_id ORDER BY source_point_index)-1 AS source_ordinal,
        count(*) OVER(PARTITION BY source_segment_id) AS alignment_count,
        count(DISTINCT source_point_index) OVER(PARTITION BY source_segment_id) AS alignment_unique
        FROM """ + scan + " a JOIN source_ids p USING(source_segment_id)", [str(alignment)])
    query = """WITH trajectory_ranked AS (
        SELECT cast(t.file_id AS VARCHAR) AS trajectory_file_id,
          cast(t.segment_id AS VARCHAR) AS source_segment_id,
          cast(t.speed AS DOUBLE) AS speed,
          row_number() OVER(PARTITION BY t.segment_id ORDER BY t.t)-1 AS source_ordinal,
          count(*) OVER(PARTITION BY t.segment_id) AS trajectory_count,
          count(DISTINCT t.t) OVER(PARTITION BY t.segment_id) AS trajectory_unique,
          count(*) FILTER(WHERE NOT isfinite(t.t) OR t.t IS NULL)
            OVER(PARTITION BY t.segment_id) AS invalid_source_time
        FROM read_parquet(?) t JOIN source_ids p ON cast(t.segment_id AS VARCHAR)=p.source_segment_id
      ), joined AS (
        SELECT a.*, t.trajectory_file_id, t.speed, t.trajectory_count,
               t.trajectory_unique, t.invalid_source_time
        FROM alignment_ranked a JOIN wanted w USING(segment_id)
        LEFT JOIN trajectory_ranked t USING(source_segment_id,source_ordinal)
      ) SELECT segment_id, count(*) AS n, count(DISTINCT segment_point_index) AS unique_indexes,
        min(segment_point_index) AS first_index, max(segment_point_index) AS last_index,
        count(DISTINCT file_id) AS file_count, min(file_id) AS file_id,
        count(DISTINCT source_segment_id) AS parent_count,
        count(*) FILTER(WHERE trajectory_count IS NULL) AS missing_source,
        count(*) FILTER(WHERE trajectory_file_id IS NULL OR trajectory_file_id != file_id) AS wrong_source_file,
        count(*) FILTER(WHERE trajectory_count != alignment_count) AS parent_count_mismatch,
        count(*) FILTER(WHERE alignment_unique != alignment_count OR trajectory_unique != trajectory_count
                        OR invalid_source_time > 0 OR source_point_index < 0) AS parent_order_invalid,
        count(*) FILTER(WHERE speed IS NULL OR NOT isfinite(speed) OR speed < 0) AS invalid_speed,
        median(speed) AS median_speed FROM joined GROUP BY segment_id"""
    cursor = connection.execute(query, [str(trajectory)])
    names = [item[0] for item in cursor.description]
    grouped = {row["segment_id"]: row for row in (dict(zip(names, values)) for values in cursor.fetchall())}
    if set(grouped) - {s["segment_id"] for s in samples.values()}:
        raise ValueError("only original final windows may be described")
    return {sid: window_statistic(sample, grouped.get(sample["segment_id"])) for sid, sample in samples.items()}


def describe(records):
    valid = [r["median_saved_speed"] for r in records if r["reason"] is None]
    failed = Counter(r["reason"] for r in records if r["reason"] is not None)
    if any(not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0 for v in valid):
        raise ValueError("finite nonnegative complete-window medians required")
    if any(r["median_saved_speed"] is not None for r in records if r["reason"] is not None):
        raise ValueError("failed statistics must not retain a successful-subset estimate")
    values = [None] * 5 if failed or not valid else [float(x) for x in np.quantile(valid, [0, .25, .5, .75, 1], method="linear")]
    return {"windows": len(records), "statistic_successes": len(valid), "statistic_failures": sum(failed.values()),
            "failure_reasons": dict(sorted(failed.items())),
            **dict(zip(("minimum", "q25", "median", "q75", "maximum"), values)),
            "full_stage_distribution_available": not failed and bool(records)}


def summarize(rows, statistics, selected_ids):
    groups = {k: [] for k in ("released", "metadata_complete", "temporal_support", "joint_feature_validity",
        "selected_primary", "metadata_excluded", "temporal_excluded", "coverage_excluded_after_temporal", "eligible_not_selected")}
    seen = set()
    for row in rows:
        sid, reasons = row["sample_id"], row["reasons"]
        if (row["split"] != "final_eval" or sid in seen or sid not in statistics
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
    if (seen != set(statistics) or not selected_ids
            or {r["sample_id"] for r in groups["selected_primary"]} != selected_ids
            or len({r["independent_block_id"] for r in groups["selected_primary"]}) != len(selected_ids)):
        raise ValueError("complete population and unchanged one-origin-per-block selection required")
    return {name: {"windows": len(group), "recording_hash_blocks": len({r["independent_block_id"] for r in group}),
                  "window_median_saved_speed": describe([statistics[r["sample_id"]] for r in group])}
            for name, group in groups.items()}


def project(runtime, eligibility_path, cohort_path, construction_path, builder_path, scratch):
    settings, _ = load(runtime)
    context = unpack(read_json(settings["context"]), expected_sha256=settings["context_sha256"])
    bundle = unpack(read_json(settings["bundle"]), expected_sha256=settings["bundle_sha256"])
    protocol, execution = unpack(bundle["protocol"]), unpack(bundle["execution"])
    population = unpack(context["population"], expected_sha256=POPULATION_SHA)
    eligibility = unpack(read_json(eligibility_path, expected_file_sha256=ELIGIBILITY_FILE_SHA), expected_sha256=ELIGIBILITY_SHA)
    if (population["eligibility_sha256"] != ELIGIBILITY_SHA
            or any(v["protocol_sha256"] != bundle["protocol"]["sha256"]
                   or v["execution_sha256"] != bundle["execution"]["sha256"]
                   or v["prior_performance_reads"] != 0 for v in (population, eligibility))
            or context["protocol_sha256"] != bundle["protocol"]["sha256"]
            or context["execution_sha256"] != bundle["execution"]["sha256"]
            or execution["protocol_sha256"] != bundle["protocol"]["sha256"]):
        raise ValueError("original input/population bindings required")
    cohort = read_json(cohort_path, expected_file_sha256=COHORT_SHA)
    binding = protocol["dataset_inputs"]
    release = Path(settings["input_paths"]["release"])
    sample_path, alignment = release / "samples.jsonl", release / "alignment.jsonl"
    trajectory = Path(settings["input_paths"]["trajectory_path"])
    for path, expected in ((sample_path, binding["release_artifact_sha256"]["samples.jsonl"]),
            (alignment, binding["release_artifact_sha256"]["alignment.jsonl"]),
            (trajectory, binding["source_trajectory_sha256"])):
        if file_hash(path) != expected:
            raise ValueError("original metadata/source file differs")
    samples = {}
    with sample_path.open("rb") as stream:
        while raw := stream.readline(16385):
            if len(raw) > 16384:
                raise ValueError("bounded sample metadata row required")
            sample = decode(raw)
            if sample["split"] == "final_eval":
                if sample["sample_id"] in samples or sample["data_version"] != binding["dataset_id"]:
                    raise ValueError("unique original final release identity required")
                samples[sample["sample_id"]] = sample
    rows, names = eligibility["rows"], ("sample_id", "segment_id", "independent_block_id")
    if (len(samples) != len(rows) or len({s["segment_id"] for s in samples.values()}) != len(samples)
            or any(tuple(samples[r["sample_id"]][k] for k in names) != tuple(r[k] for k in names) for r in rows)
            or digest({k: sorted({r[k] for r in rows}) for k in names}) != population["full_population_identity_sha256"]
            or digest(sorted([r[k] for k in names] for r in rows)) != population["full_population_row_identity_sha256"]):
        raise ValueError("unchanged full final population required")
    selected = population["selection"]["selected"]
    if len(selected) != 46 or digest(selected) != population["selection"]["selection_sha256"]:
        raise ValueError("original sealed selection required")
    scratch.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="pirc17-speed-stat-", dir=scratch) as temp:
        connection = duckdb.connect(config={"threads": 1, "memory_limit": "512MB",
            "max_temp_directory_size": "2GB", "temp_directory": temp})
        try:
            statistics = joined_statistics(alignment, trajectory, samples, connection)
        finally:
            connection.close()
    stages = summarize(rows, statistics, {r["sample_id"] for r in selected})
    for stage in ("released", "metadata_complete", "temporal_support", "joint_feature_validity", "selected_primary"):
        if any(stages[stage][k] != cohort["stages"][stage][k] for k in ("windows", "recording_hash_blocks")):
            raise ValueError("original stage denominator differs")
    return {"schema_version": "pirc17-original-final-cohort-speed-description-v1", "stages": stages,
        "bindings": {"protocol_sha256": bundle["protocol"]["sha256"], "execution_sha256": bundle["execution"]["sha256"],
            "population_sha256": POPULATION_SHA, "eligibility_sha256": ELIGIBILITY_SHA,
            "eligibility_file_sha256": ELIGIBILITY_FILE_SHA, "base_cohort_description_sha256": COHORT_SHA,
            "samples_file_sha256": binding["release_artifact_sha256"]["samples.jsonl"],
            "alignment_file_sha256": binding["release_artifact_sha256"]["alignment.jsonl"],
            "trajectory_file_sha256": binding["source_trajectory_sha256"],
            "retained_construction_script_sha256": file_hash(construction_path),
            "retained_builder_script_sha256": file_hash(builder_path)},
        "definition": "Median of every saved source speed scalar over the complete original released window, then equal-window linear descriptive quartiles; not pooled-point or duration-weighted speed",
        "declared_unit": "m/s according to the retained construction script; historical source-unit provenance is not independently certified",
        "scope": {"alignment_columns_decoded": list(ALIGNMENT_COLUMNS), "trajectory_columns_decoded": SOURCE_COLUMNS,
            "source_speed_is_saved_interval_scalar_not_norm_of_centered_vx_vy": True,
            "source_endpoint_padding_retained": True, "source_construction_scripts_bound_by_original_release": False,
            "statistical_failures_retained_in_original_denominators": True,
            "source_order_count_guards_not_physical_provenance_certification": True,
            "equal_window_summary_not_independent_participant_distribution": True,
            "future_observation_used_only_for_posthoc_population_description": True,
            "velocity_outlier_clipping_or_clock_speed_reconstruction": False,
            "original_eligibility_or_selection_recomputed": False,
            "model_inputs_or_centered_velocity_components_reconstructed": False,
            "positions_maps_predictions_or_scientific_scores_decoded": False,
            "new_fits_forecasts_scores_resampling_or_map_queries": 0,
            "whole_snapshot_revalidation_or_forecast_restart_gate": False,
            "independent_saved_output_audit_completed": False,
            "private_identities_epochs_paths_or_coordinates_exported": False,
            "all_review_items_or_paper_complete": False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "eligibility", "cohort", "construction", "builder", "scratch", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    start = perf_counter()
    result = project(args.runtime, args.eligibility, args.cohort, args.construction, args.builder, args.scratch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print(canonical({"event": "original_population_speed_description", "elapsed_seconds": perf_counter()-start,
        "windows": result["stages"]["released"]["windows"],
        "statistic_failures": result["stages"]["released"]["window_median_saved_speed"]["statistic_failures"],
        "new_experiments": 0}).decode())


if __name__ == "__main__":
    main()
