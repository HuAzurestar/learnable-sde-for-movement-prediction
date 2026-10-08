"""Describe geography of the original final cohort, without new experiments.

Join saved eligibility and sample identities to source harvesting labels only.
No raw coordinate/speed strings, feature pixels or prediction arrays are decoded.
Source labels are not certified point-level locations or participant identities.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from .checkpoint_resume import load
from .protocol_core import canonical, decode, digest, file_hash, read_json, unpack

ELIGIBILITY_SHA = "d9600487e5e7b2ad7fcca5c955b59b0c255fc51d4a9e729508b987452da0ffc5"
ELIGIBILITY_FILE_SHA = "7ce338a1a2a67ef0642b484497960916004d872686d036ec25facd6a9df44718"
POPULATION_SHA = "7061f44ef8647f7bee55822ca7e494e53d3c058e5973d6198e997893c70bb020"
COHORT_SHA = "bb48e9fa6b09807f6ad8ebfb4036d286e2506d9b1abccf08398438df6422bf05"
GEOGRAPHY_SHA = "90acfb5e2eb57778a25fcb79b3066c7e8c7815beb462a1748b914c9d2f4ebe3e"
CLUSTER_SHA = "6816971e6cdb5c615cbd627ea7ef0c819f941005f0c083c5f4f445c9c1adbf38"
METADATA = {
    "invalid-integer-point-metadata", "incomplete-or-duplicate-original-indexes",
    "alignment-identity-mismatch", "invalid-or-duplicate-source-indexes", "timestamp-outside-int64",
}
TEMPORAL = {
    "insufficient-visible-prefix", "nonpositive-observation-gap", "observation-gap-exceeds-60s",
    "followup-shorter-than-1800s", "missing-distinct-original-score-targets",
}
COVERAGE = {f"invalid-{group}-coverage-origin-through-target-end"
            for group in ("surface", "road", "river", "worldcover", "history")}


def summarize(rows, sample_labels, selected_ids):
    """Partition windows, but count each recording block once within each stage.

    The same block can have both rejected and accepted windows; rejection block
    counts are therefore NOT additive to admitted block counts.
    """
    stages = {name: [] for name in ("released", "metadata_complete", "temporal_support",
              "joint_feature_validity", "selected_primary", "metadata_excluded",
              "temporal_excluded", "coverage_excluded_after_temporal", "eligible_not_selected")}
    labels_by_block, seen = defaultdict(set), set()
    if not isinstance(selected_ids, set) or not selected_ids:
        raise ValueError("explicit unchanged selected sample set required")
    for row in rows:
        sid = row["sample_id"]
        label = sample_labels.get(sid)
        if (row["split"] != "final_eval" or sid in seen or label is None
                or len(label) != 3 or any(not isinstance(v, str) for v in label)
                or len(label[0]) != 2 or not label[0].isascii() or not label[0].isupper()):
            raise ValueError("unique final sample with explicit source country/region/area required")
        seen.add(sid)
        labels_by_block[row["independent_block_id"]].add(tuple(label))
        reasons = row["reasons"]
        if (not isinstance(reasons, list) or any(not isinstance(r, str) for r in reasons)
                or len(set(reasons)) != len(reasons)
                or set(reasons) - METADATA - TEMPORAL - COVERAGE
                or type(row["eligible"]) is not bool or row["eligible"] != (not reasons)):
            raise ValueError("unchanged explicit saved eligibility dispositions required")
        record = {"sample_id": sid, "block": row["independent_block_id"], "country": label[0],
                  "region_missing": not label[1], "harvest_area_missing": not label[2]}
        stages["released"].append(record)
        if METADATA.intersection(reasons):
            stages["metadata_excluded"].append(record)
            continue
        stages["metadata_complete"].append(record)
        if TEMPORAL.intersection(reasons):
            stages["temporal_excluded"].append(record)
            continue
        stages["temporal_support"].append(record)
        if COVERAGE.intersection(reasons):
            stages["coverage_excluded_after_temporal"].append(record)
            continue
        stages["joint_feature_validity"].append(record)
        stages["selected_primary" if sid in selected_ids else "eligible_not_selected"].append(record)
    if (seen != set(sample_labels) or not selected_ids <= seen
            or {r["sample_id"] for r in stages["selected_primary"]} != selected_ids
            or any(len(v) != 1 for v in labels_by_block.values())):
        raise ValueError("complete population, eligible selection and unambiguous block labels required")
    if len({r["block"] for r in stages["selected_primary"]}) != len(selected_ids):
        raise ValueError("one unchanged selected origin per recording block required")
    def counts(records):
        countries = sorted({r["country"] for r in records})
        return {"windows": len(records), "recording_hash_blocks": len({r["block"] for r in records}),
                "source_country_count": len(countries),
                "source_region_missing_windows": sum(r["region_missing"] for r in records),
                "source_harvest_area_missing_windows": sum(r["harvest_area_missing"] for r in records),
                "countries": {
                    c: {"windows": sum(r["country"] == c for r in records),
                        "recording_hash_blocks": len({r["block"] for r in records if r["country"] == c})}
                    for c in countries}}
    result = {name: counts(records) for name, records in stages.items()}
    selected = set(result["selected_primary"]["countries"])
    # Two largest released-window countries absent from the fixed selection.
    others = sorted(set(result["released"]["countries"]) - selected,
                    key=lambda c: (-result["released"]["countries"][c]["windows"], c))[:2]
    display = sorted(selected | set(others))
    columns = ("released", "temporal_support", "joint_feature_validity", "selected_primary")
    display_rows = [{"source_country": c, **{stage: result[stage]["countries"].get(c,
                    {"windows": 0, "recording_hash_blocks": 0}) for stage in columns}} for c in display]
    remainder = {stage: {key: sum(value[key] for c, value in result[stage]["countries"].items()
                         if c not in display) for key in ("windows", "recording_hash_blocks")}
                 for stage in columns}
    display_rows.append({"source_country": "Other source countries", **remainder})
    display_rows.append({"source_country": "Total", **{stage: {key: result[stage][key]
                        for key in ("windows", "recording_hash_blocks")} for stage in columns}})
    return {"stages": result, "display_rows": display_rows,
            "display_rule": "All selected countries plus the two largest released-window countries absent from selection; remainder pooled; no outcomes inspected",
            "countries_absent_from_selected": sorted(set(result["released"]["countries"]) - selected),
            "rejection_block_counts_not_additive_to_admitted_block_counts": True}


def project(runtime, eligibility_path, raw_path, metadata_path, cohort_path, geography_path):
    settings, _ = load(runtime)
    context = unpack(read_json(settings["context"]), expected_sha256=settings["context_sha256"])
    bundle = unpack(read_json(settings["bundle"]), expected_sha256=settings["bundle_sha256"])
    protocol, execution = unpack(bundle["protocol"]), unpack(bundle["execution"])
    if (context["protocol_sha256"] != bundle["protocol"]["sha256"]
            or context["execution_sha256"] != bundle["execution"]["sha256"]
            or execution["protocol_sha256"] != bundle["protocol"]["sha256"]):
        raise ValueError("original context/protocol/execution identity differs")
    population = unpack(context["population"], expected_sha256=POPULATION_SHA)
    eligibility = unpack(read_json(eligibility_path, expected_file_sha256=ELIGIBILITY_FILE_SHA),
                         expected_sha256=ELIGIBILITY_SHA)
    if (population["eligibility_sha256"] != ELIGIBILITY_SHA or
            any(value["protocol_sha256"] != bundle["protocol"]["sha256"]
                or value["execution_sha256"] != bundle["execution"]["sha256"]
                or value["prior_performance_reads"] != 0 for value in (population, eligibility))):
        raise ValueError("original saved pre-performance denominator required")
    cohort = read_json(cohort_path, expected_file_sha256=COHORT_SHA)
    geography = read_json(geography_path, expected_file_sha256=GEOGRAPHY_SHA)
    release = Path(settings["input_paths"]["release"])
    binding = protocol["dataset_inputs"]
    dataset = read_json(release / "dataset.json", expected_file_sha256=binding["dataset_file_sha256"])
    raw_sha = dataset["source"]["raw_harvest"]["sha256"]
    if file_hash(raw_path) != raw_sha:
        raise ValueError("original recording-to-cluster source differs")
    metadata = read_json(metadata_path, expected_file_sha256=CLUSTER_SHA)
    mapping = pq.read_table(raw_path, columns=["file_id", "cluster_A"]).to_pylist()
    clusters = {}
    for row in mapping:
        if row["file_id"] in clusters:
            raise ValueError("duplicate raw file identity")
        clusters[row["file_id"]] = row["cluster_A"]
    samples_path = release / "samples.jsonl"
    sample_sha = binding["release_artifact_sha256"]["samples.jsonl"]
    if file_hash(samples_path) != sample_sha:
        raise ValueError("original complete release sample metadata differs")
    samples, sample_labels = {}, {}
    with samples_path.open("rb") as stream:
        while line := stream.readline(16385):
            if len(line) > 16384:
                raise ValueError("bounded sample metadata row required")
            row = decode(line)
            if row["split"] != "final_eval":
                continue
            if row["sample_id"] in samples or row["data_version"] != binding["dataset_id"]:
                raise ValueError("unique final release identity required")
            samples[row["sample_id"]] = row
            sample_labels[row["sample_id"]] = metadata[str(clusters[row["file_id"]])]
    rows = eligibility["rows"]
    names = ("sample_id", "segment_id", "independent_block_id")
    if (len(samples) != len(rows) or
            any(tuple(samples[r["sample_id"]][k] for k in names) != tuple(r[k] for k in names)
                for r in rows)
            or digest({k: sorted({r[k] for r in rows}) for k in names}) != population["full_population_identity_sha256"]
            or digest(sorted([r[k] for k in names] for r in rows)) != population["full_population_row_identity_sha256"]):
        raise ValueError("complete original denominator join differs; no successful subset")
    selected = population["selection"]["selected"]
    if (population["selection"]["selection_sha256"] != digest(selected)
            or len(selected) != 46 or len({r["sample_id"] for r in selected}) != 46):
        raise ValueError("unchanged sealed selection required")
    result = summarize(rows, sample_labels, {r["sample_id"] for r in selected})
    for stage in ("released", "metadata_complete", "temporal_support", "joint_feature_validity", "selected_primary"):
        if any(result["stages"][stage][k] != cohort["stages"][stage][k]
               for k in ("windows", "recording_hash_blocks")):
            raise ValueError("original saved cohort denominator differs")
    selected_countries = {k: v["recording_hash_blocks"] for k, v in result["stages"]["selected_primary"]["countries"].items()}
    if selected_countries != geography["countries"]:
        raise ValueError("same original selected geography required")
    result.update({"schema_version": "pirc17-original-final-cohort-geography-description-v1",
        "bindings": {"protocol_sha256": bundle["protocol"]["sha256"], "execution_sha256": bundle["execution"]["sha256"],
                     "context_sha256": settings["context_sha256"], "population_sha256": POPULATION_SHA,
                     "eligibility_sha256": ELIGIBILITY_SHA, "eligibility_file_sha256": ELIGIBILITY_FILE_SHA,
                     "samples_file_sha256": sample_sha, "raw_harvest_file_sha256": raw_sha,
                     "cluster_metadata_sha256": CLUSTER_SHA, "base_cohort_description_sha256": COHORT_SHA,
                     "base_selected_geography_sha256": GEOGRAPHY_SHA},
        "scope": {"raw_columns_decoded": ["file_id", "cluster_A"],
                  "source_country_labels_not_verified_point_locations": True,
                  "cluster_metadata_in_original_release_binding": False,
                  "source_country_not_participant_or_spatial_independence_unit": True,
                  "positions_speeds_durations_features_predictions_or_scores_decoded": False,
                  "private_sample_block_file_ids_or_coordinates_exported": False,
                  "new_fits_forecasts_scores_resampling_or_map_queries": 0,
                  "original_eligibility_recomputed_or_selection_changed": False,
                  "geographical_performance_or_causal_effect_estimated": False,
                  "independent_saved_output_audit_completed": False,
                  "all_review_items_or_paper_complete": False}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "eligibility", "raw", "metadata", "cohort", "geography", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = project(args.runtime, args.eligibility, args.raw, args.metadata, args.cohort, args.geography)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print("Described all12370 original final windows and46 unchanged selected origins; no experiments.")


if __name__ == "__main__":
    main()
