"""Count original Reptile tasks using saved accounting and source labels only.

No fitting, direction labels, residuals, resampling, model inputs or predictions
are reconstructed. Region task keys reproduce the bound adapter's city-first
fallback literally; they are labels, not certified spatial/participant units.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import duckdb

from .checkpoint_resume import load
from .method_algorithm_description import INVENTORY_SHA
from .mode_partition_description import role_counts
from .protocol_core import canonical, decode, digest, file_hash, read_json, unpack

SOURCES = ("experiments/nex326/pirc20_adapter.py", "experiments/nex326/model.py", "experiments/pirc17/method_training.py")


def task_key(city, region):
    if any(v is not None and not isinstance(v, str) for v in (city, region)):
        raise ValueError("source VARCHAR label or explicit null required")
    city = str(city or "").strip()
    if city and city.lower() != "nan":
        return city, "city"
    return (str(region), "region") if region else ("unknown", "unknown")


def summarize(rows, labels, model):
    """Task count threshold, repeated exposure and membership, not mode counts."""
    seen, train, role_transitions, groups = set(), {}, Counter(), defaultdict(list)
    for row in rows:
        sid, role, count = row["segment_id"], row["role"], row["transition_count"]
        if (sid in seen or role not in {"train", "adapt", "validation"}
                or type(count) is not int or count < 3):
            raise ValueError("unique positive original role accounting required")
        seen.add(sid)
        role_transitions[role] += count
        if role == "train": train[sid] = count
    if (not train or set(labels) != set(train)
            or set(role_transitions) != {"train", "adapt", "validation"}):
        raise ValueError("all and only original training-segment labels required")
    sources = Counter()
    for sid, count in train.items():
        label, kind = labels[sid]
        if not isinstance(label, str) or not label or kind not in {"city", "region", "unknown"}:
            raise ValueError("exact nonempty adapter label with source kind required")
        groups[label].append((sid, count))
        sources[kind] += 1
    eligible = {k: v for k, v in groups.items() if len(v) >= 3}
    excluded = {k: v for k, v in groups.items() if len(v) < 3}
    if len(eligible) < 2:
        raise ValueError("original Reptile requires at least two eligible tasks")
    task_transitions = sum(n for members in eligible.values() for _, n in members)
    for field in ("meta_task_count", "meta_outer_epochs", "meta_inner_steps", "training_sample_count"):
        if type(model[field]) is not int:
            raise ValueError("saved integer model accounting required")
    exposure = role_transitions["train"] + 4*task_transitions + role_transitions["adapt"]
    if (model["meta_task_count"] != len(eligible) or model["meta_outer_epochs"] != 4
            or model["meta_inner_steps"] != 5 or model["training_sample_count"] != exposure):
        raise ValueError("original saved task/exposure accounting differs")
    def counts(current):
        return {"label_groups": len(current), "segments": sum(len(v) for v in current.values()),
                "transitions_per_pass": sum(n for members in current.values() for _, n in members)}
    tasks = [{"task": f"T{i+1:02d}", "segments": len(eligible[k]),
              "transitions_per_pass": sum(n for _, n in eligible[k]),
              "segment_rank_cardinalities": [sum(min(2, 3*r//len(eligible[k])) == m for r in range(len(eligible[k])))
                                              for m in range(3)]}
             for i, k in enumerate(sorted(eligible))]
    histogram = Counter(len(v) for v in eligible.values())
    return {"all_training_groups": counts(groups), "eligible_tasks": counts(eligible),
        "ineligible_small_groups": counts(excluded), "training_label_source_segments": dict(sorted(sources.items())),
        "tasks": tasks, "eligible_task_size_histogram": {str(k): v for k, v in sorted(histogram.items())},
        "training_role_transitions": dict(sorted(role_transitions.items())), "outer_epochs": 4, "inner_steps": 5,
        "task_visits": 4*len(eligible), "repeated_task_transition_exposures": 4*task_transitions,
        "saved_training_sample_count": model["training_sample_count"],
        "exposure_reconciliation": {"global_initialization": role_transitions["train"],
            "repeated_meta_tasks": 4*task_transitions, "target_adaptation": role_transitions["adapt"],
            "total": exposure},
        "membership_identity_sha256": digest(sorted((sid, *labels[sid], n) for sid, n in train.items()))}


def project(runtime, inventory_path):
    settings, _ = load(runtime)
    bundle = unpack(read_json(settings["bundle"]), expected_sha256=settings["bundle_sha256"])
    protocol, execution = unpack(bundle["protocol"]), unpack(bundle["execution"])
    if execution["protocol_sha256"] != bundle["protocol"]["sha256"]:
        raise ValueError("original protocol/execution binding required")
    inventory = read_json(inventory_path, expected_file_sha256=INVENTORY_SHA)
    if (inventory["model_count"], inventory["method_fit_count"], inventory["terrain_fit_count"]) != (26, 16, 10):
        raise ValueError("complete original model inventory required")
    bindings = [b for b in inventory["models"] if b["matrix"] == "NEX326-methods" and "arm-14/reptile" in b["prediction_configs"]]
    if len(bindings) != 1:
        raise ValueError("unique original Reptile fit required")
    binding = bindings[0]
    record = unpack(read_json(binding["original_model_record_path"],
        expected_file_sha256=binding["original_model_record_file_sha256"]), expected_sha256=binding["original_model_record_content_sha256"])
    if (record["fit_identity"] != binding["fit_identity"] or record["parameter_identity"] != binding["parameter_identity"]
            or record["matrix_sha256"] != inventory["matrix_sha256"] or record["protocol_sha256"] != bundle["protocol"]["sha256"]):
        raise ValueError("original model record binding differs")
    training, model = (record["artifact"][k] for k in ("training", "model"))
    role_counts(training)
    if (training["training_identity_sha256"] != digest({k: v for k, v in training.items() if k != "training_identity_sha256"})
            or training["reference_interval_seconds"] != 60 or model["transfer_method"] != "meta_reptile"):
        raise ValueError("original complete prepared accounting required")
    repository, source_hashes = Path(__file__).resolve().parents[2], {}
    for name in SOURCES:
        expected = execution["source_sha256"]["PSDE-SDE/" + name]
        if file_hash(repository / name) != expected or protocol["source_sha256"]["PSDE-SDE/" + name] != expected:
            raise ValueError("original adapter/task source differs")
        source_hashes[name] = expected
    for name, expected in training["source_sha256"].items():
        if file_hash(repository / name) != expected:
            raise ValueError("original preparation source differs")
    dataset = protocol["dataset_inputs"]
    samples_path = Path(settings["input_paths"]["release"]) / "samples.jsonl"
    sample_sha = dataset["release_artifact_sha256"]["samples.jsonl"]
    if file_hash(samples_path) != sample_sha:
        raise ValueError("original sample-to-recording mapping differs")
    roles = {r["segment_id"]: r["role"] for r in training["per_segment"]}
    samples = {}
    with samples_path.open("rb") as stream:
        while line := stream.readline(16385):
            if len(line) > 16384: raise ValueError("bounded release metadata row required")
            row = decode(line)
            sid = row["segment_id"]
            if sid not in roles: continue
            if (sid in samples or row["data_version"] != dataset["dataset_id"]
                    or row["split"] != ("validation" if roles[sid] == "validation" else "train")):
                raise ValueError("unique original role-to-release identity required")
            samples[sid] = row
    if set(samples) != set(roles):
        raise ValueError("complete original485-segment mapping required")
    trajectory = Path(settings["input_paths"]["trajectory_path"])
    if file_hash(trajectory) != dataset["source_trajectory_sha256"]:
        raise ValueError("original source metadata file differs")
    wanted = sorted({samples[sid]["file_id"] for sid, role in roles.items() if role == "train"})
    connection = duckdb.connect(config={"threads": 1, "memory_limit": "256MB"})
    try:
        connection.execute("CREATE TEMP TABLE wanted(file_id VARCHAR)")
        connection.executemany("INSERT INTO wanted VALUES (?)", [(name,) for name in wanted])
        source_rows = connection.execute("""SELECT DISTINCT cast(s.file_id AS VARCHAR),
            cast(s.city AS VARCHAR), cast(s.region AS VARCHAR)
            FROM read_parquet(?) s JOIN wanted w ON cast(s.file_id AS VARCHAR)=w.file_id""", [str(trajectory)]).fetchall()
    finally:
        connection.close()
    by_file = {}
    for file_id, city, region in source_rows:
        if file_id in by_file:
            raise ValueError("ambiguous source-file labels; cannot infer first-point group")
        by_file[file_id] = task_key(city, region)
    if set(by_file) != set(wanted):
        raise ValueError("all original training recordings need source labels")
    labels = {sid: by_file[samples[sid]["file_id"]] for sid, role in roles.items() if role == "train"}
    result = summarize(training["per_segment"], labels, model)
    result.update({"schema_version": "pirc17-original-reptile-task-description-v1",
        "bindings": {"inventory_sha256": INVENTORY_SHA, "original_record_file_sha256": binding["original_model_record_file_sha256"],
            "original_record_content_sha256": binding["original_model_record_content_sha256"],
            "training_identity_sha256": training["training_identity_sha256"], "protocol_sha256": bundle["protocol"]["sha256"],
            "execution_sha256": bundle["execution"]["sha256"], "source_sha256": source_hashes,
            "samples_file_sha256": sample_sha, "trajectory_file_sha256": dataset["source_trajectory_sha256"],
            "training_source_recordings": len(wanted)},
        "group_definition": "Exact adapter region: strip city; use it unless empty or case-insensitive nan, else unstripped region or unknown; group exact strings, no country key or case normalization",
        "task_display_rule": "Anonymous T01..T30 in original Python lexicographic label order; no labels or member identities exported",
        "scope": {"source_columns_decoded": ["file_id", "city", "region"], "training_segments": 328,
            "original_source_labels_constant_within_used_recordings_verified": True,
            "task_groups_are_not_certified_cities_regions_participants_or_independent_samples": True,
            "small_groups_remain_in_global_initialization": True,
            "label_membership_reconciled_from_original_records_not_new_training": True,
            "intermediate_parameters_or_gmm_residual_labels_recovered": False,
            "final_mixture_probabilities_inverted_to_transition_frequencies": False,
            "coordinates_clocks_speeds_predictions_or_scientific_scores_decoded_from_source": False,
            "new_fits_forecasts_scores_resampling_or_map_queries": 0,
            "new_experimental_qualification_or_final_acceptance": False,
            "independent_saved_output_audit_completed": False,
            "raw_labels_segment_file_ids_private_paths_or_coefficients_exported": False,
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
    with args.output.open("xb") as stream: stream.write(canonical(result) + b"\n")
    print(canonical({"event": "saved_reptile_task_description", "elapsed_seconds": perf_counter()-start,
        "training_segments": 328, "eligible_tasks": result["eligible_tasks"]["label_groups"], "new_experiments": 0}).decode())


if __name__ == "__main__":
    main()
