from __future__ import annotations

import copy

import pytest

from experiments.pirc22.representations import load_representation_matrix
from experiments.pirc22.selection import (
    BenchmarkSelectionError,
    build_benchmark_selection,
    write_benchmark_selection,
)


def _evidence(
    direction: str,
    scores: dict[str, list[float]],
    *,
    blocks_per_fold: int = 2,
    seeds: tuple[int, ...] = (11, 12),
):
    matrix = load_representation_matrix()
    fold_count = len(next(iter(scores.values())))
    folds = [
        {
            "fold_id": f"fold-{index:02d}",
            "validation_independent_block_count": blocks_per_fold,
            "validation_segment_count": blocks_per_fold,
        }
        for index in range(fold_count)
    ]
    manifest = {
        "matrix_identity_sha256": matrix.matrix_identity_sha256,
        "folds": folds,
        "seeds": list(seeds),
        "training_config": {"maximum_epochs": 20, "patience": 3},
        "shared_protocol": {
            "primary_metric": "fixture_score",
            "primary_metric_direction": direction,
        },
    }
    records = []
    for candidate_order, (candidate_key, fold_scores) in enumerate(scores.items()):
        candidate_id, conditioner_id = candidate_key.split("::")
        complex_candidate = conditioner_id != "linear"
        for fold_index, fold_score in enumerate(fold_scores):
            for seed_index, seed in enumerate(seeds):
                centered_score = fold_score + (-0.01 if seed_index == 0 else 0.01)
                records.append(
                    {
                        "candidate_id": candidate_id,
                        "candidate_order": candidate_order,
                        "conditioner_id": conditioner_id,
                        "fold_id": f"fold-{fold_index:02d}",
                        "seed": seed,
                        "status": "success",
                        "validation_metric": centered_score,
                        "train_metric": centered_score - 0.05,
                        "train_validation_gap": 0.05,
                        "validation_block_metrics": {
                            f"block-{fold_index:02d}-{block_index:02d}": (
                                centered_score
                                + (0.8 if block_index % 2 else -0.8)
                            )
                            for block_index in range(blocks_per_fold)
                        },
                        "runtime_seconds": 2.0 if complex_candidate else 1.0,
                        "input_dimension": 8 if complex_candidate else 0,
                        "trainable_parameter_count": 82 if complex_candidate else 2,
                        "layer_count": 1 if complex_candidate else 0,
                        "train_independent_block_count": 12,
                        "validation_independent_block_count": blocks_per_fold,
                        "final_eval_read_count": 0,
                    }
                )
    return matrix, manifest, {"summary_identity_sha256": "a" * 64}, records


@pytest.mark.parametrize(
    ("direction", "complex_scores", "simple_score"),
    [
        ("lower_is_better", [0.8, 1.2, 0.9, 1.1, 1.0], 1.05),
        ("higher_is_better", [1.2, 0.8, 1.1, 0.9, 1.0], 0.95),
    ],
)
def test_one_se_supports_both_directions_and_selects_simpler_candidate(
    direction, complex_scores, simple_score
):
    simple = "R00-no-terrain::linear"
    complex_key = "R01-legacy-scalars::mlp-small-16"
    matrix, manifest, summary, records = _evidence(
        direction,
        {
            simple: [simple_score] * 5,
            complex_key: complex_scores,
        },
    )

    selection = build_benchmark_selection(
        manifest,
        summary,
        records,
        matrix=matrix,
        selection_version="fixture-v1",
    )

    assert selection["selection_mode"] == "one_standard_error"
    assert selection["raw_best_candidate_key"] == complex_key
    assert selection["selected_candidate_key"] == simple
    assert selection["eligible_candidate_keys"] == [simple, complex_key]
    assert selection["complexity_order"] == [simple, complex_key]
    assert selection["final_eval_read_count"] == 0
    candidate = next(
        item
        for item in selection["candidate_summaries"]
        if item["candidate_key"] == complex_key
    )
    assert [item["s_c_k"] for item in candidate["fold_scores"]] == pytest.approx(
        complex_scores
    )


def test_paired_block_bootstrap_is_deterministic_when_folds_are_few():
    simple = "R00-no-terrain::linear"
    complex_key = "R01-legacy-scalars::mlp-small-16"
    matrix, manifest, summary, records = _evidence(
        "lower_is_better",
        {simple: [1.05, 1.05], complex_key: [1.0, 1.0]},
        blocks_per_fold=4,
    )

    first = build_benchmark_selection(
        manifest,
        summary,
        records,
        matrix=matrix,
        selection_version="fixture-v1",
        bootstrap_replicates=250,
    )
    second = build_benchmark_selection(
        manifest,
        summary,
        records,
        matrix=matrix,
        selection_version="fixture-v1",
        bootstrap_replicates=250,
    )

    assert first == second
    assert first["selection_mode"] == "paired_block_bootstrap"
    assert first["raw_best_candidate_key"] == complex_key
    assert first["selected_candidate_key"] == simple
    assert first["standard_error_best"] > 0


def test_insufficient_folds_and_blocks_is_inconclusive():
    matrix, manifest, summary, records = _evidence(
        "lower_is_better",
        {
            "R00-no-terrain::linear": [1.1, 1.1],
            "R01-legacy-scalars::mlp-small-16": [1.0, 1.0],
        },
        blocks_per_fold=2,
    )

    selection = build_benchmark_selection(
        manifest,
        summary,
        records,
        matrix=matrix,
        selection_version="fixture-v1",
    )

    assert selection["status"] == "inconclusive_insufficient_folds_and_blocks"
    assert selection["selection_mode"] == "inconclusive"
    assert selection["raw_best_candidate_key"].endswith("mlp-small-16")
    assert selection["selected_candidate_key"] is None
    assert selection["admission_threshold"] is None


def test_failed_candidate_is_retained_but_ineligible():
    failed = "R02-local-statistics-standardized::linear"
    matrix, manifest, summary, records = _evidence(
        "lower_is_better",
        {
            "R00-no-terrain::linear": [1.1] * 5,
            "R01-legacy-scalars::mlp-small-16": [1.0] * 5,
        },
    )
    for fold in manifest["folds"]:
        for seed in manifest["seeds"]:
            records.append(
                {
                    "candidate_id": failed.split("::")[0],
                    "conditioner_id": "linear",
                    "fold_id": fold["fold_id"],
                    "seed": seed,
                    "status": "failed",
                    "failure": {"message": "declared failure"},
                    "final_eval_read_count": 0,
                }
            )

    selection = build_benchmark_selection(
        manifest,
        summary,
        records,
        matrix=matrix,
        selection_version="fixture-v1",
    )

    failed_summary = next(
        item
        for item in selection["candidate_summaries"]
        if item["candidate_key"] == failed
    )
    assert not failed_summary["complete"]
    assert failed_summary["failure_reasons"] == ["declared failure"]
    assert failed not in selection["eligible_candidate_keys"]


def test_benchmark_selection_is_content_addressed_and_immutable(tmp_path):
    matrix, manifest, summary, records = _evidence(
        "lower_is_better",
        {
            "R00-no-terrain::linear": [1.1] * 5,
            "R01-legacy-scalars::mlp-small-16": [1.0] * 5,
        },
    )
    first = build_benchmark_selection(
        manifest,
        summary,
        records,
        matrix=matrix,
        selection_version="fixture-v1",
    )
    changed = build_benchmark_selection(
        manifest,
        summary,
        records,
        matrix=matrix,
        selection_version="fixture-v2",
    )
    path = tmp_path / "BenchmarkSelection.json"

    write_benchmark_selection(path, first)
    write_benchmark_selection(path, copy.deepcopy(first))
    with pytest.raises(BenchmarkSelectionError, match="immutable"):
        write_benchmark_selection(path, changed)
