from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.pirc22.conditioners import TrainingConfig
from experiments.pirc22.representations import load_representation_matrix
from experiments.pirc22.runner import (
    BenchmarkRunner,
    BenchmarkRunnerError,
    PreparedBenchmark,
)
from experiments.pirc22.selection import select_benchmark
from experiments.terrain_benchmark import BenchmarkFold


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _fold(index: int) -> BenchmarkFold:
    fold_id = f"fold-{index:02d}"
    blocks = (f"validation-block-{index}",)
    segments = (f"validation-segment-{index}",)
    return BenchmarkFold(
        fold_id=fold_id,
        validation_block_ids=blocks,
        validation_segment_ids=segments,
        identity_sha256=_hash(
            {"fold_id": fold_id, "blocks": blocks, "segments": segments}
        ),
    )


class _Provider:
    def __init__(self, *, fail_candidate: str | None = None, drift: bool = False):
        self.fail_candidate = fail_candidate
        self.drift = drift

    @property
    def identity_record(self):
        return {"provider": "fixture-v1", "final_eval_read_count": 0}

    def prepare(self, candidate, fold):
        if candidate.candidate_id == self.fail_candidate:
            raise RuntimeError("declared fixture failure")
        fold_index = int(fold.fold_id.rsplit("-", 1)[1])
        train_rows = 18
        validation_rows = 8
        target_train_x = np.linspace(-1.0, 1.0, train_rows)
        target_validation_x = np.linspace(-0.8, 0.8, validation_rows)
        train_targets = np.column_stack(
            (0.4 * target_train_x + 0.1, -0.2 * target_train_x)
        )
        validation_targets = np.column_stack(
            (0.4 * target_validation_x + 0.1, -0.2 * target_validation_x)
        )
        train_targets += fold_index * 0.001
        validation_targets += fold_index * 0.001
        embedding = candidate.categorical_embedding
        continuous_dim = candidate.model_input_dim - (
            embedding.embedding_dim if embedding else 0
        )
        rng = np.random.default_rng(1000 + candidate.order * 10 + fold_index)
        train_features = rng.normal(size=(train_rows, continuous_dim))
        validation_features = rng.normal(size=(validation_rows, continuous_dim))
        if continuous_dim:
            train_features[:, 0] = target_train_x
            validation_features[:, 0] = target_validation_x
        train_tokens = validation_tokens = None
        cardinality = embedding_dim = None
        if embedding:
            cardinality = 12
            embedding_dim = embedding.embedding_dim
            train_tokens = np.arange(train_rows, dtype=np.int64) % cardinality
            validation_tokens = np.arange(validation_rows, dtype=np.int64) % cardinality
        base_train = np.zeros_like(train_targets)
        base_validation = np.zeros_like(validation_targets)
        control = {
            "fold_id": fold.fold_id,
            "train_targets": hashlib.sha256(train_targets.tobytes()).hexdigest(),
            "validation_targets": hashlib.sha256(
                validation_targets.tobytes()
            ).hexdigest(),
            "base_model": "fixed-base-v1",
        }
        if self.drift and candidate.order:
            control["base_model"] = "changed-base"
        return PreparedBenchmark(
            train_features=train_features,
            train_targets=train_targets,
            train_base_predictions=base_train,
            validation_features=validation_features,
            validation_targets=validation_targets,
            validation_base_predictions=base_validation,
            train_block_ids=("train-block-a", "train-block-b"),
            validation_block_ids=fold.validation_block_ids,
            train_row_block_ids=tuple(
                "train-block-a" if index < train_rows // 2 else "train-block-b"
                for index in range(train_rows)
            ),
            validation_row_block_ids=(fold.validation_block_ids[0],)
            * validation_rows,
            control_identity_sha256=_hash(control),
            problem_identity_sha256=_hash(
                {"candidate": candidate.candidate_id, "fold": fold.fold_id}
            ),
            train_tokens=train_tokens,
            validation_tokens=validation_tokens,
            categorical_cardinality=cardinality,
            embedding_dim=embedding_dim,
        )


def _runner(path: Path, *, seeds=(11, 12), candidates=None):
    return BenchmarkRunner(
        path,
        matrix=load_representation_matrix(),
        data_identity={"benchmark_data_identity_sha256": "a" * 64},
        folds=(_fold(1), _fold(2)),
        seeds=seeds,
        candidate_ids=candidates or ("R00-no-terrain", "R01-legacy-scalars"),
        conditioner_ids=("linear", "mlp-small-16"),
        training_config=TrainingConfig(
            learning_rate=0.02,
            weight_decay=0.0,
            batch_size=18,
            maximum_epochs=8,
            patience=3,
        ),
    )


def _records(root: Path) -> list[dict[str, object]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "cells").glob("*/record.json"))
    ]


def test_fixture_cube_is_complete_auditable_and_resumable(tmp_path):
    runner = _runner(tmp_path / "run")
    first = runner.run_all(_Provider())
    record_paths = sorted((tmp_path / "run" / "cells").glob("*/record.json"))
    mtimes = {path: path.stat().st_mtime_ns for path in record_paths}
    second = runner.run_all(_Provider())

    assert first == second
    assert first["expected_cell_count"] == 16
    assert first["terminal_cell_count"] == 16
    assert first["status_counts"] == {"success": 16, "failed": 0}
    assert {path: path.stat().st_mtime_ns for path in record_paths} == mtimes
    for record in _records(tmp_path / "run"):
        assert record["final_eval_read_count"] == 0
        assert record["primary_metric_direction"] == "lower_is_better"
        assert record["train_independent_block_count"] == 2
        assert record["validation_independent_block_count"] == 1
        assert set(record["validation_block_metrics"]) == {
            f"validation-block-{int(record['fold_id'].rsplit('-', 1)[1])}"
        }
        assert record["epochs_completed"] <= 8
        assert set(record["artifacts"]) == {
            "checkpoint.json",
            "learning_curve.json",
        }
    selection = select_benchmark(
        tmp_path / "run",
        matrix=load_representation_matrix(),
        selection_version="fixture-v1",
    )
    assert selection["status"] == "inconclusive_insufficient_folds_and_blocks"
    assert selection["source_summary_identity_sha256"] == first[
        "summary_identity_sha256"
    ]


def test_failures_are_terminal_rows_and_control_drift_cannot_succeed(tmp_path):
    failed_root = tmp_path / "failed"
    summary = _runner(
        failed_root,
        seeds=(11,),
        candidates=("R00-no-terrain", "R01-legacy-scalars"),
    ).run_all(_Provider(fail_candidate="R01-legacy-scalars"))
    assert summary["status_counts"] == {"success": 4, "failed": 4}
    failures = [record for record in _records(failed_root) if record["status"] == "failed"]
    assert all(record["failure"]["type"] == "RuntimeError" for record in failures)

    drift_root = tmp_path / "drift"
    drift_summary = _runner(
        drift_root,
        seeds=(11,),
        candidates=("R00-no-terrain", "R01-legacy-scalars"),
    ).run_all(_Provider(drift=True))
    assert drift_summary["status_counts"]["failed"] == 4
    assert any(
        "control identity changed" in record["failure"]["message"]
        for record in _records(drift_root)
        if record["status"] == "failed"
    )


def test_resume_rejects_protocol_or_artifact_identity_drift(tmp_path):
    root = tmp_path / "run"
    _runner(root, seeds=(11,)).run_all(_Provider())
    with pytest.raises(BenchmarkRunnerError, match="resume identity mismatch"):
        _runner(root, seeds=(99,)).run_all(_Provider())

    checkpoint = next((root / "cells").glob("*/checkpoint.json"))
    checkpoint.write_text("{}\n", encoding="utf-8")
    with pytest.raises(BenchmarkRunnerError, match="artifact identity mismatch"):
        _runner(root, seeds=(11,)).run_all(_Provider())
