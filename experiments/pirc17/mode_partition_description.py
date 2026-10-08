"""Describe retained role counts and deterministic segment-rank cardinalities.

No fitting, residual reconstruction, labels, coordinates or resampling. Equal
segment counts are not equal transition counts, empirical probabilities or
behavioural-state correspondence across independently ranked batches.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .method_algorithm_description import INVENTORY_SHA
from .protocol_core import canonical, digest, file_hash, read_json, unpack

ROLES = {"train": 328, "adapt": 76, "validation": 81}
SOURCES = ("experiments/nex326/model.py", "experiments/pirc17/method_training.py")


def segment_rank_counts(count):
    if type(count) is not int or count < 1:
        raise ValueError("positive integer segment count required")
    return [sum(min(2, 3 * rank // count) == mode for rank in range(count)) for mode in range(3)]


def role_counts(training):
    rows = training["per_segment"]
    if not rows or len({r["segment_id"] for r in rows}) != len(rows):
        raise ValueError("unique original prepared segment identities required")
    actual = Counter()
    transitions = Counter()
    for row in rows:
        role, n = row["role"], row["transition_count"]
        if (role not in ROLES or type(n) is not int or n < 3
                or row["reference_interval_seconds"] != training["reference_interval_seconds"]
                or row["scoring_observations_changed"] is not False):
            raise ValueError("original complete uniform role accounting required")
        actual[role] += 1
        transitions[role] += n
    if (dict(actual) != ROLES or training["sample_counts"] != ROLES
            or dict(transitions) != training["transitions_by_role"]):
        raise ValueError("saved per-segment and aggregate role counts differ")
    return {role: {"segments": actual[role], "all_transitions": transitions[role],
                   "whole_role_heading_rank_segment_counts": segment_rank_counts(actual[role]),
                   "multimode_transition_frequencies": None} for role in ROLES}


def project(inventory_path, bundle_path):
    inventory = read_json(inventory_path, expected_file_sha256=INVENTORY_SHA)
    bundle = unpack(read_json(bundle_path))
    protocol, execution = unpack(bundle["protocol"]), unpack(bundle["execution"])
    if execution["protocol_sha256"] != bundle["protocol"]["sha256"]:
        raise ValueError("execution/protocol identity differs")
    repository = Path(__file__).resolve().parents[2]
    source_hashes = {}
    for relative in SOURCES:
        expected = execution["source_sha256"]["PSDE-SDE/" + relative]
        if (file_hash(repository / relative) != expected
                or protocol["source_sha256"]["PSDE-SDE/" + relative] != expected):
            raise ValueError("original ranking/training-accounting source differs")
        source_hashes[relative] = expected
    rows, seen, slots, population = [], set(), set(), None
    for binding in inventory["models"]:
        if binding["matrix"] != "NEX326-methods":
            continue
        record = unpack(read_json(binding["original_model_record_path"],
                                  expected_file_sha256=binding["original_model_record_file_sha256"]),
                        expected_sha256=binding["original_model_record_content_sha256"])
        if (record["fit_identity"] != binding["fit_identity"]
                or record["parameter_identity"] != binding["parameter_identity"]
                or record["matrix_sha256"] != inventory["matrix_sha256"]
                or record["protocol_sha256"] != bundle["protocol"]["sha256"]
                or record["fit_identity"] in seen):
            raise ValueError("original saved method binding differs or repeats")
        seen.add(record["fit_identity"])
        training, model = (record["artifact"][name] for name in ("training", "model"))
        if (training["training_identity_sha256"] != digest({k: v for k, v in training.items()
                                                          if k != "training_identity_sha256"})
                or any(training["source_sha256"][name] != sha for name, sha in source_hashes.items())):
            raise ValueError("original prepared training identity differs")
        counts = role_counts(training)
        identity = digest(sorted((r["role"], r["segment_id"]) for r in training["per_segment"]))
        if population is None:
            population = identity
        if identity != population or slots.intersection(binding["prediction_configs"]):
            raise ValueError("prepared role population differs or prediction slots repeat")
        slots.update(binding["prediction_configs"])
        rows.append({"representative_slot": binding["representative_slot"],
                     "prediction_slots": sorted(binding["prediction_configs"]),
                     "original_record_sha256": binding["original_model_record_file_sha256"],
                     "model_kind": model["model_kind"],
                     "reference_interval_seconds": training["reference_interval_seconds"],
                     "roles": counts})
    if len(rows) != 16 or len(slots) != 28:
        raise ValueError("complete original sixteen-fit/twenty-eight-slot scope required")
    return {"schema_version": "pirc17-original-mode-partition-description-v1",
            "inventory_sha256": INVENTORY_SHA, "protocol_sha256": bundle["protocol"]["sha256"],
            "execution_sha256": bundle["execution"]["sha256"], "source_sha256": source_hashes,
            "prepared_role_population_sha256": population,
            "whole_role_rank_segment_counts": {role: segment_rank_counts(n) for role, n in ROLES.items()},
            "methods": sorted(rows, key=lambda r: r["representative_slot"]),
            "rank_rule": "Sort by atan2(net north displacement, net east displacement), then segment identity; label=min(2,floor(3*zero_based_rank/K))",
            "scope": {"counts_describe_complete_role_batches_not_all_Reptile_region_tasks": True,
                      "segment_balance_implies_transition_balance": False,
                      "actual_segment_labels_or_multimode_transition_frequencies_recovered": False,
                      "final_blended_probabilities_inverted_to_frequencies": False,
                      "gmm_residual_partitions_reconstructed": False,
                      "raw_coordinates_times_labels_residuals_or_private_paths_exported": False,
                      "new_fits_forecasts_scores_resampling_or_map_queries": 0,
                      "independent_saved_output_audit_completed": False,
                      "all_review_items_or_paper_complete": False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inventory", "bundle", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = project(args.inventory, args.bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print("Described sixteen saved role-accounting records and rank cardinalities; no fitting or labels reconstructed.")


if __name__ == "__main__":
    main()
