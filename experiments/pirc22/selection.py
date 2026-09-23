"""Reproducible 1-SE and paired-block selection for PIRC-22 RunRecords."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence
from uuid import uuid4

import numpy as np

from .conditioners import CONDITIONER_BY_ID
from .representations import RepresentationMatrix


SELECTION_SCHEMA_VERSION = "pirc22-benchmark-selection-v1"
DEFAULT_MINIMUM_FOLDS = 5
DEFAULT_MINIMUM_BLOCKS_PER_FOLD = 1
DEFAULT_MINIMUM_BOOTSTRAP_BLOCKS = 8
DEFAULT_BOOTSTRAP_REPLICATES = 2000
DEFAULT_BOOTSTRAP_SEED = 20260922
DEFAULT_MAXIMUM_FOLD_IMBALANCE = 2.0


class BenchmarkSelectionError(ValueError):
    """Benchmark evidence cannot produce a trustworthy frozen selection."""


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkSelectionError(f"cannot read benchmark evidence: {path}") from error
    if not isinstance(value, dict):
        raise BenchmarkSelectionError(f"benchmark evidence is not an object: {path}")
    return value


def load_benchmark_evidence(
    output_root: str | Path,
) -> tuple[dict[str, object], dict[str, object], tuple[dict[str, object], ...]]:
    root = Path(output_root).resolve()
    manifest = _read_object(root / "manifest.json")
    summary = _read_object(root / "summary.json")
    manifest_protocol = dict(manifest)
    protocol_identity = manifest_protocol.pop("protocol_identity_sha256", None)
    provider_identity = manifest_protocol.pop("provider_identity", None)
    provider_identity_sha256 = manifest_protocol.pop(
        "provider_identity_sha256", None
    )
    if protocol_identity != _canonical_hash(manifest_protocol):
        raise BenchmarkSelectionError("benchmark manifest protocol identity mismatch")
    if (
        not isinstance(provider_identity, Mapping)
        or provider_identity_sha256 != _canonical_hash(provider_identity)
        or provider_identity.get("final_eval_read_count") != 0
    ):
        raise BenchmarkSelectionError("benchmark provider identity is invalid")
    summary_copy = dict(summary)
    summary_identity = summary_copy.pop("summary_identity_sha256", None)
    if summary_identity != _canonical_hash(summary_copy):
        raise BenchmarkSelectionError("benchmark summary identity mismatch")
    if (
        summary.get("protocol_identity_sha256") != protocol_identity
        or not summary.get("complete")
        or summary.get("terminal_cell_count") != manifest.get("expected_cell_count")
    ):
        raise BenchmarkSelectionError("benchmark cube is not complete")
    records: list[dict[str, object]] = []
    for record_path in sorted((root / "cells").glob("*/record.json")):
        record = _read_object(record_path)
        identity_payload = dict(record)
        identity = identity_payload.pop("record_identity_sha256", None)
        if identity != _canonical_hash(identity_payload):
            raise BenchmarkSelectionError(
                f"benchmark RunRecord identity mismatch: {record_path.parent.name}"
            )
        if record.get("protocol_identity_sha256") != protocol_identity:
            raise BenchmarkSelectionError("benchmark RunRecord protocol mismatch")
        if record.get("final_eval_read_count") != 0:
            raise BenchmarkSelectionError("benchmark evidence read final evaluation")
        artifacts = record.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise BenchmarkSelectionError("benchmark RunRecord artifact map is invalid")
        for name, expected in artifacts.items():
            artifact = record_path.parent / str(name)
            if not artifact.is_file() or _sha256(artifact) != expected:
                raise BenchmarkSelectionError("benchmark artifact identity mismatch")
        records.append(record)
    if len(records) != manifest.get("expected_cell_count"):
        raise BenchmarkSelectionError("benchmark RunRecord count is incomplete")
    axes = {
        (
            str(record.get("candidate_id")),
            str(record.get("conditioner_id")),
            str(record.get("fold_id")),
            int(record.get("seed", -1)),
        )
        for record in records
    }
    expected_axes = {
        (candidate, conditioner, str(fold["fold_id"]), int(seed))
        for candidate in manifest.get("candidate_ids", [])
        for conditioner in manifest.get("conditioner_ids", [])
        for fold in manifest.get("folds", [])
        for seed in manifest.get("seeds", [])
    }
    if axes != expected_axes or len(axes) != len(records):
        raise BenchmarkSelectionError("benchmark RunRecord cube axes are invalid")
    return manifest, summary, tuple(records)


def _candidate_key(record: Mapping[str, object]) -> str:
    return f"{record['candidate_id']}::{record['conditioner_id']}"


def _eligible(mean: float, threshold: float, direction: str) -> bool:
    return mean <= threshold if direction == "lower_is_better" else mean >= threshold


def _best_key(values: Mapping[str, float], direction: str) -> str:
    if direction == "lower_is_better":
        return min(values, key=lambda key: (values[key], key))
    return min(values, key=lambda key: (-values[key], key))


def build_benchmark_selection(
    manifest: Mapping[str, object],
    summary: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    *,
    matrix: RepresentationMatrix,
    selection_version: str,
    minimum_folds: int = DEFAULT_MINIMUM_FOLDS,
    minimum_blocks_per_fold: int = DEFAULT_MINIMUM_BLOCKS_PER_FOLD,
    minimum_bootstrap_blocks: int = DEFAULT_MINIMUM_BOOTSTRAP_BLOCKS,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    maximum_fold_imbalance: float = DEFAULT_MAXIMUM_FOLD_IMBALANCE,
) -> dict[str, object]:
    if not selection_version:
        raise BenchmarkSelectionError("selection_version is required")
    for name, value in (
        ("minimum_folds", minimum_folds),
        ("minimum_blocks_per_fold", minimum_blocks_per_fold),
        ("minimum_bootstrap_blocks", minimum_bootstrap_blocks),
        ("bootstrap_replicates", bootstrap_replicates),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise BenchmarkSelectionError(f"{name} must be a positive integer")
    if not np.isfinite(maximum_fold_imbalance) or maximum_fold_imbalance < 1:
        raise BenchmarkSelectionError(
            "maximum_fold_imbalance must be finite and at least one"
        )
    if manifest.get("matrix_identity_sha256") != matrix.matrix_identity_sha256:
        raise BenchmarkSelectionError("benchmark matrix identity mismatch")
    shared_protocol = manifest.get("shared_protocol")
    if not isinstance(shared_protocol, Mapping):
        raise BenchmarkSelectionError("shared benchmark protocol is invalid")
    direction = str(shared_protocol.get("primary_metric_direction", ""))
    metric = str(shared_protocol.get("primary_metric", ""))
    if direction not in {"lower_is_better", "higher_is_better"} or not metric:
        raise BenchmarkSelectionError("primary metric contract is invalid")
    if any(record.get("final_eval_read_count") != 0 for record in records):
        raise BenchmarkSelectionError("selection evidence read final evaluation")
    fold_ids = tuple(str(item["fold_id"]) for item in manifest.get("folds", []))
    seeds = tuple(int(value) for value in manifest.get("seeds", []))
    expected_per_candidate = len(fold_ids) * len(seeds)
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for record in records:
        grouped.setdefault(_candidate_key(record), []).append(record)

    summaries: dict[str, dict[str, object]] = {}
    complete_means: dict[str, float] = {}
    block_scores: dict[str, dict[str, list[float]]] = {}
    complexity: dict[str, tuple[float, ...]] = {}
    for key, values in sorted(grouped.items()):
        successes = [value for value in values if value.get("status") == "success"]
        failures = [value for value in values if value.get("status") == "failed"]
        fold_scores: list[dict[str, object]] = []
        complete = len(values) == expected_per_candidate and not failures
        for fold_id in fold_ids:
            fold_values = [
                value
                for value in successes
                if value.get("fold_id") == fold_id
            ]
            fold_seed_values = {
                int(value["seed"]): float(value["validation_metric"])
                for value in fold_values
            }
            if set(fold_seed_values) != set(seeds):
                complete = False
            fold_scores.append(
                {
                    "fold_id": fold_id,
                    "seed_scores": [
                        {"seed": seed, "score": fold_seed_values.get(seed)}
                        for seed in seeds
                    ],
                    "s_c_k": (
                        float(np.mean(list(fold_seed_values.values())))
                        if len(fold_seed_values) == len(seeds)
                        else None
                    ),
                }
            )
        valid_fold_scores = [
            float(item["s_c_k"])
            for item in fold_scores
            if item["s_c_k"] is not None
        ]
        mean_score = (
            float(np.mean(valid_fold_scores))
            if len(valid_fold_scores) == len(fold_ids)
            else None
        )
        seed_summaries = []
        for seed in seeds:
            seed_values = [
                float(value["validation_metric"])
                for value in successes
                if int(value["seed"]) == seed
            ]
            seed_summaries.append(
                {
                    "seed": seed,
                    "mean_score_across_folds": (
                        float(np.mean(seed_values))
                        if len(seed_values) == len(fold_ids)
                        else None
                    ),
                }
            )
        complete_seed_means = [
            float(value["mean_score_across_folds"])
            for value in seed_summaries
            if value["mean_score_across_folds"] is not None
        ]
        candidate_id, conditioner_id = key.split("::", 1)
        try:
            candidate = matrix.candidate(candidate_id)
            conditioner = CONDITIONER_BY_ID[conditioner_id]
        except (KeyError, ValueError) as error:
            raise BenchmarkSelectionError(
                f"unknown candidate configuration: {key}"
            ) from error
        representative = successes[0] if successes else None
        dimension_fields = (
            "input_dimension",
            "trainable_parameter_count",
            "layer_count",
        )
        if representative is not None and any(
            value.get(field) != representative.get(field)
            for value in successes
            for field in dimension_fields
        ):
            raise BenchmarkSelectionError(
                f"candidate dimensions changed across RunRecords: {key}"
            )
        summaries[key] = {
            "candidate_key": key,
            "candidate_id": candidate_id,
            "conditioner_id": conditioner_id,
            "complete": complete,
            "success_count": len(successes),
            "failure_count": len(failures),
            "failure_reasons": sorted(
                {
                    str(value.get("failure", {}).get("message", "unknown"))
                    for value in failures
                }
            ),
            "fold_scores": fold_scores,
            "seed_summaries": seed_summaries,
            "across_seed_standard_deviation": (
                float(np.std(complete_seed_means, ddof=1))
                if len(complete_seed_means) > 1
                else (0.0 if len(complete_seed_means) == 1 else None)
            ),
            "mean_score": mean_score,
            "mean_train_metric": (
                float(np.mean([float(value["train_metric"]) for value in successes]))
                if successes
                else None
            ),
            "mean_train_validation_gap": (
                float(
                    np.mean(
                        [float(value["train_validation_gap"]) for value in successes]
                    )
                )
                if successes
                else None
            ),
            "mean_runtime_seconds": (
                float(np.mean([float(value["runtime_seconds"]) for value in successes]))
                if successes
                else None
            ),
            "runtime_standard_deviation_seconds": (
                float(
                    np.std(
                        [float(value["runtime_seconds"]) for value in successes],
                        ddof=1,
                    )
                )
                if len(successes) > 1
                else (0.0 if successes else None)
            ),
            "minimum_train_independent_block_count": (
                min(int(value["train_independent_block_count"]) for value in successes)
                if successes
                and all("train_independent_block_count" in value for value in successes)
                else None
            ),
            "minimum_validation_independent_block_count": (
                min(
                    int(value["validation_independent_block_count"])
                    for value in successes
                )
                if successes
                and all(
                    "validation_independent_block_count" in value
                    for value in successes
                )
                else None
            ),
            "input_dimension": (
                int(representative["input_dimension"]) if representative else None
            ),
            "trainable_parameter_count": (
                int(representative["trainable_parameter_count"])
                if representative
                else None
            ),
            "layer_count": (
                int(representative["layer_count"]) if representative else None
            ),
            "hidden_widths": list(conditioner.hidden_widths),
            "representation_order": candidate.order,
            "representation_family": candidate.family,
            "variant_ids": list(candidate.variant_ids),
            "composition_ids": list(candidate.composition_ids),
            "interaction_ids": list(candidate.interaction_ids),
        }
        if complete and mean_score is not None and representative is not None:
            complete_means[key] = mean_score
            complexity[key] = (
                float(representative["trainable_parameter_count"]),
                float(representative["input_dimension"]),
                float(representative["layer_count"]),
                float(summaries[key]["mean_runtime_seconds"]),
                float(representative["candidate_order"]),
                float(list(CONDITIONER_BY_ID).index(conditioner_id)),
            )
            for value in successes:
                metrics = value.get("validation_block_metrics")
                if not isinstance(metrics, Mapping) or not metrics:
                    continue
                for block_id, score in metrics.items():
                    block_scores.setdefault(key, {}).setdefault(
                        str(block_id), []
                    ).append(float(score))

    if not complete_means:
        raw_best = None
        selected = None
        selection_status = "inconclusive_no_complete_candidate"
        selection_mode = "inconclusive"
        standard_error = threshold = None
        eligible_keys: list[str] = []
    else:
        raw_best = _best_key(complete_means, direction)
        fold_counts = [
            int(item.get("validation_independent_block_count", 0))
            for item in manifest.get("folds", [])
        ]
        positive_counts = [value for value in fold_counts if value > 0]
        imbalance = (
            max(positive_counts) / min(positive_counts)
            if len(positive_counts) == len(fold_counts) and positive_counts
            else float("inf")
        )
        raw_fold_scores = [
            float(item["s_c_k"])
            for item in summaries[raw_best]["fold_scores"]
        ]
        if (
            len(fold_ids) >= minimum_folds
            and positive_counts
            and min(positive_counts) >= minimum_blocks_per_fold
            and imbalance <= maximum_fold_imbalance
        ):
            selection_mode = "one_standard_error"
            standard_error = (
                float(np.std(raw_fold_scores, ddof=1) / np.sqrt(len(raw_fold_scores)))
                if len(raw_fold_scores) > 1
                else 0.0
            )
            comparison_means = complete_means
        else:
            common_blocks = set(block_scores.get(raw_best, {}))
            for key in complete_means:
                common_blocks &= set(block_scores.get(key, {}))
            if len(common_blocks) >= minimum_bootstrap_blocks:
                selection_mode = "paired_block_bootstrap"
                ordered_blocks = sorted(common_blocks)
                block_means = {
                    key: np.asarray(
                        [np.mean(block_scores[key][block]) for block in ordered_blocks],
                        dtype=float,
                    )
                    for key in complete_means
                }
                comparison_means = {
                    key: float(np.mean(values)) for key, values in block_means.items()
                }
                raw_best = _best_key(comparison_means, direction)
                rng = np.random.default_rng(bootstrap_seed)
                indexes = rng.integers(
                    0,
                    len(ordered_blocks),
                    size=(bootstrap_replicates, len(ordered_blocks)),
                )
                bootstrapped_best = np.mean(block_means[raw_best][indexes], axis=1)
                standard_error = float(np.std(bootstrapped_best, ddof=1))
            else:
                selection_mode = "inconclusive"
                selection_status = "inconclusive_insufficient_folds_and_blocks"
                comparison_means = complete_means
                standard_error = threshold = None
                eligible_keys = []
                selected = None
        if selection_mode != "inconclusive":
            raw_mean = comparison_means[raw_best]
            threshold = (
                raw_mean + standard_error
                if direction == "lower_is_better"
                else raw_mean - standard_error
            )
            eligible_keys = sorted(
                key
                for key, value in comparison_means.items()
                if _eligible(value, threshold, direction)
            )
            selected = min(eligible_keys, key=lambda key: (*complexity[key], key))
            selection_status = "selected"

    comparison_scores = (
        comparison_means if complete_means else {}
    )
    for key, candidate_summary in summaries.items():
        candidate_summary["selection_score"] = comparison_scores.get(key)
        candidate_summary["eligible_under_threshold"] = key in eligible_keys

    complexity_order = sorted(
        complete_means,
        key=lambda key: (*complexity[key], key),
    )
    selected_configuration = None
    if selected is not None:
        candidate_id, conditioner_id = selected.split("::", 1)
        candidate = matrix.candidate(candidate_id)
        conditioner = CONDITIONER_BY_ID[conditioner_id]
        selected_configuration = {
            "candidate_id": candidate_id,
            "conditioner_id": conditioner_id,
            "variant_ids": list(candidate.variant_ids),
            "composition_ids": list(candidate.composition_ids),
            "interaction_ids": list(candidate.interaction_ids),
            "derived_transforms": [
                {
                    "transform": value.transform,
                    "source_variant_ids": list(value.source_variant_ids),
                    "output_dim": value.output_dim,
                    "parameters": dict(value.parameters),
                }
                for value in candidate.derived_transforms
            ],
            "categorical_embedding": (
                {
                    "source_variant_id": candidate.categorical_embedding.source_variant_id,
                    "vocabulary": candidate.categorical_embedding.vocabulary,
                    "embedding_dim": candidate.categorical_embedding.embedding_dim,
                }
                if candidate.categorical_embedding
                else None
            ),
            "model_input_dim": candidate.model_input_dim,
            "hidden_widths": list(conditioner.hidden_widths),
            "layer_count": conditioner.layer_count,
            "training_config": dict(manifest.get("training_config", {})),
        }
    selection: dict[str, object] = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "selection_version": selection_version,
        "status": selection_status,
        "source_protocol_identity_sha256": manifest.get(
            "protocol_identity_sha256"
        ),
        "source_summary_identity_sha256": summary.get("summary_identity_sha256"),
        "matrix_id": matrix.matrix_id,
        "matrix_identity_sha256": matrix.matrix_identity_sha256,
        "metric": metric,
        "metric_direction": direction,
        "fold_count": len(fold_ids),
        "fold_definitions": [dict(value) for value in manifest.get("folds", [])],
        "seed_count": len(seeds),
        "minimum_folds": minimum_folds,
        "minimum_blocks_per_fold": minimum_blocks_per_fold,
        "minimum_bootstrap_blocks": minimum_bootstrap_blocks,
        "maximum_fold_imbalance": maximum_fold_imbalance,
        "selection_mode": selection_mode,
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "unit": "independent_block_id",
            "paired_across_candidates": True,
        },
        "raw_best_candidate_key": raw_best,
        "selected_candidate_key": selected,
        "standard_error_best": standard_error,
        "admission_threshold": threshold,
        "eligible_candidate_keys": eligible_keys,
        "complexity_order": complexity_order,
        "candidate_summaries": [summaries[key] for key in sorted(summaries)],
        "selected_configuration": selected_configuration,
        "selection_rationale": (
            "Selected the least complex eligible configuration under the "
            "registered parameter/input/layer/runtime ordering."
            if selected is not None
            else "Capacity selection remained inconclusive under the registered fallback."
        ),
        "final_eval_read_count": 0,
        "immutability_policy": "new_selection_version_required",
    }
    selection["selection_identity_sha256"] = _canonical_hash(selection)
    return selection


def select_benchmark(
    output_root: str | Path,
    *,
    matrix: RepresentationMatrix,
    selection_version: str,
    **kwargs: object,
) -> dict[str, object]:
    manifest, summary, records = load_benchmark_evidence(output_root)
    return build_benchmark_selection(
        manifest,
        summary,
        records,
        matrix=matrix,
        selection_version=selection_version,
        **kwargs,
    )


def write_benchmark_selection(
    path: str | Path, selection: Mapping[str, object]
) -> None:
    destination = Path(path).resolve()
    payload = dict(selection)
    identity = payload.pop("selection_identity_sha256", None)
    if payload.get("schema_version") != SELECTION_SCHEMA_VERSION or (
        identity != _canonical_hash(payload)
    ):
        raise BenchmarkSelectionError("BenchmarkSelection identity is invalid")
    payload["selection_identity_sha256"] = identity
    if destination.exists():
        existing = _read_object(destination)
        if existing == payload:
            return
        raise BenchmarkSelectionError(
            "BenchmarkSelection is immutable; publish a new selection_version"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}-{uuid4().hex}.tmp"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, destination)


__all__ = [
    "BenchmarkSelectionError",
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_MAXIMUM_FOLD_IMBALANCE",
    "DEFAULT_MINIMUM_BLOCKS_PER_FOLD",
    "DEFAULT_MINIMUM_BOOTSTRAP_BLOCKS",
    "DEFAULT_MINIMUM_FOLDS",
    "SELECTION_SCHEMA_VERSION",
    "build_benchmark_selection",
    "load_benchmark_evidence",
    "select_benchmark",
    "write_benchmark_selection",
]
