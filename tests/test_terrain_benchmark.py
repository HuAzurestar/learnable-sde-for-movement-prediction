from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from experiments.nex326.cohort import Cohort, Segment
from experiments.nex326.pirc20_adapter import PIRC20Segment
from experiments.terrain_benchmark import (
    BenchmarkDataBoundary,
    BenchmarkIsolationError,
)


def _segment(segment_id: str, block_id: str) -> Segment:
    return PIRC20Segment(
        segment_id=segment_id,
        source_domain="human",
        region="fixture",
        time=np.arange(4, dtype=float),
        state=np.column_stack((np.arange(4, dtype=float), np.zeros(4))),
        conditions={},
        has_terrain=True,
        independent_block_id=block_id,
    )


def _cohort() -> Cohort:
    cohort = Cohort(
        schema_version="nex326-cohort-v1",
        dataset_id="pirc20-fixture-v1",
        data_version="pirc20-fixture-v1",
        purpose="pirc22-fixture",
        splits={
            "train": (_segment("train-a", "train-block-a"),),
            "adapt": (_segment("adapt-a", "adapt-block-a"),),
            "validation": (
                _segment("validation-a1", "validation-block-a"),
                _segment("validation-a2", "validation-block-a"),
                _segment("validation-b", "validation-block-b"),
                _segment("validation-c", "validation-block-c"),
            ),
            "evaluation": (),
            "animal_pretrain": (),
        },
        unavailable_reasons={
            "evaluation": "final eval remains sealed",
            "animal_pretrain": "not used by benchmark",
        },
        fingerprint="fixture-cohort-fingerprint",
    )
    cohort.validate()
    return cohort


def test_grouped_folds_are_deterministic_balanced_and_identity_bound():
    first = BenchmarkDataBoundary(_cohort(), fold_count=3, fold_seed=17)
    second = BenchmarkDataBoundary(_cohort(), fold_count=3, fold_seed=17)

    assert first.identity_record == second.identity_record
    assert first.benchmark_data_identity_sha256 == second.benchmark_data_identity_sha256
    assert [fold.segment_count for fold in first.folds] == [2, 1, 1]
    assigned_blocks = [
        block_id for fold in first.folds for block_id in fold.validation_block_ids
    ]
    assert sorted(assigned_blocks) == [
        "validation-block-a",
        "validation-block-b",
        "validation-block-c",
    ]
    for fold in first.folds:
        segments = first.segments("validation", fold_id=fold.fold_id)
        assert {segment.independent_block_id for segment in segments} == set(
            fold.validation_block_ids
        )


def test_fit_population_is_train_only_and_preserves_block_identity():
    boundary = BenchmarkDataBoundary(_cohort(), fold_count=2)

    fit = boundary.fit_segments(fold_id=boundary.folds[0].fold_id)
    adapt = boundary.segments("adapt")

    assert [segment.segment_id for segment in fit] == ["train-a"]
    assert fit[0].independent_block_id == "train-block-a"
    assert {segment.segment_id for segment in fit}.isdisjoint(
        segment.segment_id for segment in adapt
    )
    assert boundary.identity_record["fit_role"] == "train"


def test_final_eval_access_fails_closed_and_is_audited_without_a_read():
    boundary = BenchmarkDataBoundary(_cohort(), fold_count=2)

    with pytest.raises(BenchmarkIsolationError, match="not readable"):
        boundary.segments("final_eval")
    with pytest.raises(BenchmarkIsolationError, match="explicit fold_id"):
        boundary.segments("validation")

    audit = boundary.audit_record
    assert audit["final_eval_materialized"] is False
    assert audit["final_eval_read_count"] == 0
    assert [attempt["role"] for attempt in audit["denied_attempts"]] == [
        "final_eval",
        "validation",
    ]


def test_boundary_rejects_materialized_eval_cross_role_blocks_and_missing_identity():
    cohort = _cohort()
    with_eval = replace(
        cohort,
        splits={**cohort.splits, "evaluation": (_segment("eval-a", "eval-block"),)},
    )
    with pytest.raises(BenchmarkIsolationError, match="materialized final"):
        BenchmarkDataBoundary(with_eval)

    crossed = replace(
        cohort,
        splits={
            **cohort.splits,
            "adapt": (_segment("adapt-a", "train-block-a"),),
        },
    )
    with pytest.raises(BenchmarkIsolationError, match="crosses benchmark roles"):
        BenchmarkDataBoundary(crossed)

    missing = replace(
        cohort,
        splits={
            **cohort.splits,
            "train": (replace(cohort.splits["train"][0], independent_block_id=None),),
        },
    )
    with pytest.raises(BenchmarkIsolationError, match="lacks independent_block_id"):
        BenchmarkDataBoundary(missing)
