"""Describe exact numeric source-sequence agreement; never fit or forecast.

Uses whole bound condition records serving the original selected windows.
Exact agreement is a descriptive screen, not near-route/participant/UTC proof.
Private identities, clocks, coordinates and per-record fingerprints stay local.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow.parquet as pq

from .checkpoint_resume import load
from .method_algorithm_description import INVENTORY_SHA
from .mode_partition_description import role_counts
from .protocol_core import canonical, decode, digest, file_hash, read_json, under, unpack

ROLES = ("train", "adapt", "validation", "final_eval")
EXPECTED_WINDOWS = dict(zip(ROLES, (328, 76, 81, 46)))
EXPECTED_FILES = dict(zip(ROLES, (266, 65, 66, 46)))


def fingerprints(lonlat, ticks):
    """Preserve row order/count; no rounding, interpolation or point deletion."""
    positions = np.array(lonlat, dtype="<f8", copy=True)
    epochs = np.asarray(ticks)
    if (positions.ndim != 2 or positions.shape[1] != 2 or not len(positions)
            or epochs.ndim != 1 or len(epochs) != len(positions)
            or epochs.dtype.kind != "i"):
        raise ValueError("nonempty two-coordinate rows and matching integer clocks required")
    positions[positions == 0] = 0.0  # Normalize signed zero only.
    positions[np.isnan(positions)] = np.nan  # One explicit NaN representation.
    epochs = epochs.astype("<i8", copy=False)
    from hashlib import sha256
    route = sha256(b"pirc17-exact-lonlat-v1\0" + len(positions).to_bytes(8, "big")
                   + positions.tobytes()).hexdigest()
    timed = sha256(b"pirc17-exact-lonlat-clock-v1\0" + bytes.fromhex(route)
                   + epochs.tobytes()).hexdigest()
    return {"coordinate_fingerprint": route, "coordinate_clock_fingerprint": timed,
            "points": len(positions),
            "nonfinite_coordinate_rows": int((~np.isfinite(positions).all(axis=1)).sum()),
            "missing_clock_rows": int((epochs == np.iinfo(np.int64).min).sum())}


def summarize(records):
    """Count source-file pairs, not windows, particles or independent people."""
    if not records or any(r["role"] not in ROLES for r in records):
        raise ValueError("all four named roles required")
    if len({r["file_id"] for r in records}) != len(records):
        raise ValueError("unique selected file identity required across roles")
    groups = {role: [r for r in records if r["role"] == role] for role in ROLES}
    if any(not group for group in groups.values()):
        raise ValueError("no missing role or successful subset")
    keys = ("coordinate_fingerprint", "coordinate_clock_fingerprint")
    def within(group, key):
        counts = Counter(r[key] for r in group)
        blocks = defaultdict(Counter)
        for row in group:
            blocks[row[key]][row["recording_hash_block"]] += 1
        total_pairs = sum(n * (n - 1) // 2 for n in counts.values())
        same_block_pairs = sum(n * (n - 1) // 2 for v in blocks.values() for n in v.values())
        return {"equal_sequence_file_pairs": total_pairs,
                "equal_sequence_groups": sum(n > 1 for n in counts.values()),
                "files_in_equal_sequence_groups": sum(n for n in counts.values() if n > 1),
                "equal_sequence_pairs_outside_same_recording_hash_block": total_pairs - same_block_pairs}
    result = {"roles": {role: {"source_files": len(group),
        "source_rows": sum(r["points"] for r in group),
        "nonfinite_coordinate_rows": sum(r["nonfinite_coordinate_rows"] for r in group),
        "missing_clock_rows": sum(r["missing_clock_rows"] for r in group),
        **{key: within(group, key) for key in keys}} for role, group in groups.items()}}
    result["cross_role"] = {}
    for a, b in combinations(ROLES, 2):
        values = {"possible_source_file_pairs": len(groups[a]) * len(groups[b])}
        for key in keys:
            ca, cb = (Counter(r[key] for r in groups[role]) for role in (a, b))
            common = ca.keys() & cb.keys()
            values[key] = {"equal_sequence_file_pairs": sum(ca[k] * cb[k] for k in common),
                "shared_sequence_groups": len(common),
                "matching_files_left": sum(ca[k] for k in common),
                "matching_files_right": sum(cb[k] for k in common)}
        result["cross_role"][a + ":" + b] = values
    return result


def project(runtime, inventory_path):
    settings, _ = load(runtime)
    bundle = unpack(read_json(settings["bundle"]), expected_sha256=settings["bundle_sha256"])
    protocol = unpack(bundle["protocol"])
    context = unpack(read_json(settings["context"]), expected_sha256=settings["context_sha256"])
    if context["protocol_sha256"] != bundle["protocol"]["sha256"]:
        raise ValueError("original context/protocol differs")
    inventory = read_json(inventory_path, expected_file_sha256=INVENTORY_SHA)
    candidates = [m for m in inventory["models"] if m["matrix"] == "NEX326-methods"
                  and "arm-14/reptile" in m["prediction_configs"]]
    if len(candidates) != 1:
        raise ValueError("unique original prepared role accounting required")
    binding = candidates[0]
    record = unpack(read_json(binding["original_model_record_path"],
        expected_file_sha256=binding["original_model_record_file_sha256"]),
        expected_sha256=binding["original_model_record_content_sha256"])
    training = record["artifact"]["training"]
    role_counts(training)
    roles_by_segment = {r["segment_id"]: r["role"] for r in training["per_segment"]}
    if len(roles_by_segment) != 485:
        raise ValueError("complete unique original development accounting required")
    population = unpack(context["population"])
    selected = population["selection"]["selected"]
    if len(selected) != 46 or digest(selected) != population["selection"]["selection_sha256"]:
        raise ValueError("original selected final cohort differs")
    final_ids = {r["sample_id"] for r in selected}
    dataset_binding = protocol["dataset_inputs"]
    release = Path(settings["input_paths"]["release"])
    samples_path = release / "samples.jsonl"
    samples_sha = dataset_binding["release_artifact_sha256"]["samples.jsonl"]
    if file_hash(samples_path) != samples_sha:
        raise ValueError("original release sample identities differ")
    samples = {role: [] for role in ROLES}
    with samples_path.open("rb") as stream:
        while raw := stream.readline(16385):
            if len(raw) > 16384:
                raise ValueError("bounded sample metadata required")
            row = decode(raw)
            role = "final_eval" if row["sample_id"] in final_ids else roles_by_segment.get(row["segment_id"])
            if role is None:
                continue
            expected_split = "train" if role == "adapt" else role
            if row["split"] != expected_split or row["data_version"] != dataset_binding["dataset_id"]:
                raise ValueError("original selected sample role differs")
            samples[role].append(row)
    if {role: len(v) for role, v in samples.items()} != EXPECTED_WINDOWS:
        raise ValueError("all original 531 selected windows required")
    development = unpack(context["input_identity"])["terrain_development_identity"]
    if ({r["sample_id"] for role in ("train", "adapt") for r in samples[role]}
            != set(development["sample_ids"]["train"])
            or {r["sample_id"] for r in samples["validation"]}
            != set(development["sample_ids"]["validation"])
            or {r["sample_id"] for r in samples["final_eval"]} != final_ids):
        raise ValueError("original terrain/method population correspondence differs")
    dataset = read_json(release / "dataset.json", expected_file_sha256=dataset_binding["dataset_file_sha256"])
    manifest_path = release / "condition_file_manifest.jsonl"
    condition_manifest_sha = dataset["artifacts"][manifest_path.name]["sha256"]
    if file_hash(manifest_path) != condition_manifest_sha:
        raise ValueError("release condition manifest differs")
    entries = {}
    for raw in manifest_path.read_bytes().splitlines():
        row = decode(raw)
        if row["file_id"] in entries:
            raise ValueError("duplicate condition file identity")
        entries[row["file_id"]] = row
    snapshot = Path(settings["input_paths"]["snapshot"])
    feature_manifest = read_json(snapshot / "manifest.json",
        expected_file_sha256=dataset_binding["snapshot"]["manifest_sha256"])
    features = {(r["split"], r["file_id"]): r for r in feature_manifest["files"]}
    root = Path(settings["input_paths"]["data_root"]) / "cond_slices"
    records, sources = [], {}
    for role in ROLES:
        files = sorted({r["file_id"] for r in samples[role]})
        if len(files) != EXPECTED_FILES[role]:
            raise ValueError("original selected file denominators differ")
        for file_id in files:
            file_blocks = {r["independent_block_id"] for r in samples[role] if r["file_id"] == file_id}
            if len(file_blocks) != 1:
                raise ValueError("source file must bind one original recording hash block")
            entry = entries[file_id]
            split = "train" if role == "adapt" else role
            if features[split, file_id]["condition_sha256"] != entry["sha256"]:
                raise ValueError("feature/condition original binding differs")
            path = under(root, entry["relative_path"])
            if file_hash(path) != entry["sha256"]:
                raise ValueError("selected original condition bytes differ")
            table = pq.read_table(path, columns=["file_id", "lon", "lat", "t"])
            if set(table["file_id"].to_pylist()) != {file_id}:
                raise ValueError("source condition file identity differs")
            xy = np.column_stack((table["lon"].to_numpy(), table["lat"].to_numpy()))
            ticks = table["t"].to_numpy().astype("datetime64[ns]").astype(np.int64)
            records.append({"role": role, "file_id": file_id,
                            "recording_hash_block": next(iter(file_blocks)), **fingerprints(xy, ticks)})
            sources[file_id] = entry["sha256"]
    result = summarize(records)
    for role in ROLES:
        result["roles"][role]["selected_windows"] = len(samples[role])
        result["roles"][role]["selected_recording_hash_blocks"] = len({r["independent_block_id"] for r in samples[role]})
    result.update({"schema_version": "pirc17-selected-source-sequence-description-v1",
        "bindings": {"protocol_sha256": bundle["protocol"]["sha256"],
            "producer_source_sha256": file_hash(Path(__file__)),
            "context_sha256": settings["context_sha256"], "inventory_sha256": INVENTORY_SHA,
            "original_model_record_file_sha256": binding["original_model_record_file_sha256"],
            "samples_file_sha256": samples_sha, "condition_manifest_sha256": condition_manifest_sha,
            "snapshot_manifest_sha256": dataset_binding["snapshot"]["manifest_sha256"],
            "selected_condition_files": len(sources), "selected_condition_sources_sha256": digest(sources)},
        "definition": "Exact entire selected source-recording row sequence, float64 lon/lat in original order; row count retained, signed zero and NaN representation normalized, no rounding/resampling/reversal/translation; second fingerprint appends retained integer-nanosecond condition clocks including NaT sentinel",
        "scope": {"decoded_condition_columns": ["file_id", "lon", "lat", "t"],
            "whole_source_recordings_not_only_selected_window_or_fitted_transitions": True,
            "whole_source_rows_are_not_independent_observations": True,
            "near_route_partial_overlap_or_repeated_visit_search_performed": False,
            "participant_or_physical_clock_independence_established": False,
            "physical_utc_or_coordinate_provenance_certified": False,
            "original_eligibility_selection_or_model_parameters_changed": False,
            "new_fits_forecasts_scores_resampling_or_map_queries": 0,
            "independent_saved_forecast_output_audit_completed": False,
            "private_ids_coordinates_epochs_paths_or_record_fingerprints_exported": False,
            "all_review_items_or_paper_complete": False}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "inventory", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    start = perf_counter()
    result = project(args.runtime, args.inventory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print(canonical({"event": "saved_source_sequence_description", "elapsed_seconds": perf_counter()-start,
        "source_files": result["bindings"]["selected_condition_files"],
        "cross_role": result["cross_role"], "new_experiments": 0}).decode())


if __name__ == "__main__":
    main()
