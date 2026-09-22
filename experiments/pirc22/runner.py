"""Atomic, resumable execution of the PIRC-22 benchmark candidate cube."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from time import perf_counter
from typing import Mapping, Protocol, Sequence
from uuid import uuid4

import numpy as np

from experiments.terrain_benchmark import BenchmarkFold

from .conditioners import (
    CONDITIONER_BY_ID,
    ConditionerError,
    TrainingConfig,
    fit_conditioner,
    predict_conditioner,
)
from .representations import RepresentationCandidate, RepresentationMatrix


RUNNER_SCHEMA_VERSION = "pirc22-benchmark-runner-v1"
RUN_RECORD_SCHEMA_VERSION = "pirc22-benchmark-run-record-v1"
SUMMARY_SCHEMA_VERSION = "pirc22-benchmark-summary-v1"
PRIMARY_METRIC = "validation_mean_squared_error"
PRIMARY_METRIC_DIRECTION = "lower_is_better"


class BenchmarkRunnerError(ValueError):
    """A benchmark run cannot safely start, resume, or commit."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


@dataclass(frozen=True)
class PreparedBenchmark:
    train_features: np.ndarray
    train_targets: np.ndarray
    train_base_predictions: np.ndarray
    validation_features: np.ndarray
    validation_targets: np.ndarray
    validation_base_predictions: np.ndarray
    train_block_ids: tuple[str, ...]
    validation_block_ids: tuple[str, ...]
    control_identity_sha256: str
    problem_identity_sha256: str
    train_tokens: np.ndarray | None = None
    validation_tokens: np.ndarray | None = None
    categorical_cardinality: int | None = None
    embedding_dim: int | None = None


class BenchmarkProblemProvider(Protocol):
    @property
    def identity_record(self) -> Mapping[str, object]: ...

    def prepare(
        self,
        candidate: RepresentationCandidate,
        fold: BenchmarkFold,
    ) -> PreparedBenchmark: ...


def _validate_problem(
    problem: PreparedBenchmark, candidate: RepresentationCandidate, fold: BenchmarkFold
) -> None:
    arrays = (
        problem.train_features,
        problem.train_targets,
        problem.train_base_predictions,
        problem.validation_features,
        problem.validation_targets,
        problem.validation_base_predictions,
    )
    if any(np.asarray(value).ndim != 2 for value in arrays):
        raise BenchmarkRunnerError("prepared benchmark arrays must be matrices")
    if not len(problem.train_features) or not len(problem.validation_features):
        raise BenchmarkRunnerError("prepared benchmark train/validation rows are empty")
    if not (
        len(problem.train_features)
        == len(problem.train_targets)
        == len(problem.train_base_predictions)
    ):
        raise BenchmarkRunnerError("prepared train rows disagree")
    if not (
        len(problem.validation_features)
        == len(problem.validation_targets)
        == len(problem.validation_base_predictions)
    ):
        raise BenchmarkRunnerError("prepared validation rows disagree")
    if problem.train_targets.shape[1] != 2 or problem.validation_targets.shape[1] != 2:
        raise BenchmarkRunnerError("benchmark targets must be two-dimensional drift")
    if problem.train_base_predictions.shape != problem.train_targets.shape or (
        problem.validation_base_predictions.shape != problem.validation_targets.shape
    ):
        raise BenchmarkRunnerError("base prediction shape disagrees with targets")
    if any(not np.isfinite(np.asarray(value, dtype=float)).all() for value in arrays):
        raise BenchmarkRunnerError("prepared benchmark contains non-finite values")
    if not problem.train_block_ids or not problem.validation_block_ids:
        raise BenchmarkRunnerError("prepared benchmark block identities are empty")
    if set(problem.train_block_ids) & set(problem.validation_block_ids):
        raise BenchmarkRunnerError("independent blocks cross train and validation")
    if tuple(sorted(problem.validation_block_ids)) != tuple(
        sorted(fold.validation_block_ids)
    ):
        raise BenchmarkRunnerError("prepared validation blocks do not match the fold")
    if len(problem.control_identity_sha256) != 64 or len(problem.problem_identity_sha256) != 64:
        raise BenchmarkRunnerError("prepared benchmark identity is invalid")

    embedding = candidate.categorical_embedding
    if embedding is None:
        expected_continuous_dim = candidate.model_input_dim
        if problem.train_tokens is not None or problem.validation_tokens is not None:
            raise BenchmarkRunnerError("unexpected categorical tokens")
        if problem.categorical_cardinality is not None or problem.embedding_dim is not None:
            raise BenchmarkRunnerError("unexpected categorical embedding metadata")
    else:
        expected_continuous_dim = candidate.model_input_dim - embedding.embedding_dim
        if problem.train_tokens is None or problem.validation_tokens is None:
            raise BenchmarkRunnerError("embedding candidate is missing categorical tokens")
        if problem.embedding_dim != embedding.embedding_dim:
            raise BenchmarkRunnerError("embedding dimension disagrees with matrix")
        if problem.categorical_cardinality is None:
            raise BenchmarkRunnerError("embedding vocabulary size is absent")
    if problem.train_features.shape[1] != expected_continuous_dim or (
        problem.validation_features.shape[1] != expected_continuous_dim
    ):
        raise BenchmarkRunnerError("prepared feature dimension disagrees with matrix")


class BenchmarkRunner:
    def __init__(
        self,
        output_root: str | Path,
        *,
        matrix: RepresentationMatrix,
        data_identity: Mapping[str, object],
        folds: Sequence[BenchmarkFold],
        seeds: Sequence[int],
        candidate_ids: Sequence[str] | None = None,
        conditioner_ids: Sequence[str] | None = None,
        training_config: TrainingConfig = TrainingConfig(),
    ) -> None:
        training_config.validate()
        self.output_root = Path(output_root).resolve()
        self.matrix = matrix
        self.data_identity = dict(data_identity)
        self.folds = tuple(folds)
        self.seeds = tuple(seeds)
        selected_candidates = tuple(
            candidate_ids
            if candidate_ids is not None
            else (candidate.candidate_id for candidate in matrix.candidates)
        )
        selected_conditioners = tuple(
            conditioner_ids
            if conditioner_ids is not None
            else CONDITIONER_BY_ID.keys()
        )
        if (
            not self.folds
            or not self.seeds
            or not selected_candidates
            or not selected_conditioners
            or len(self.seeds) != len(set(self.seeds))
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in self.seeds)
        ):
            raise BenchmarkRunnerError("benchmark cube axes must be nonempty and unique")
        if len({fold.fold_id for fold in self.folds}) != len(self.folds):
            raise BenchmarkRunnerError("benchmark fold IDs must be unique")
        if len(selected_candidates) != len(set(selected_candidates)) or len(
            selected_conditioners
        ) != len(set(selected_conditioners)):
            raise BenchmarkRunnerError("benchmark candidate/model IDs must be unique")
        unknown_models = set(selected_conditioners) - set(CONDITIONER_BY_ID)
        if unknown_models:
            raise BenchmarkRunnerError(f"unknown conditioners: {sorted(unknown_models)}")
        self.candidates = tuple(matrix.candidate(value) for value in selected_candidates)
        self.conditioner_ids = selected_conditioners
        self.training_config = training_config
        self.cells_root = self.output_root / "cells"
        self.protocol = self._protocol()
        self.protocol_identity_sha256 = _canonical_hash(self.protocol)

    def _protocol(self) -> dict[str, object]:
        implementation_path = Path(__file__)
        expected_cells = (
            len(self.candidates)
            * len(self.conditioner_ids)
            * len(self.folds)
            * len(self.seeds)
        )
        return {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "matrix_id": self.matrix.matrix_id,
            "matrix_identity_sha256": self.matrix.matrix_identity_sha256,
            "data_identity": self.data_identity,
            "data_identity_sha256": _canonical_hash(self.data_identity),
            "candidate_ids": [value.candidate_id for value in self.candidates],
            "conditioner_ids": list(self.conditioner_ids),
            "folds": [
                {
                    "fold_id": fold.fold_id,
                    "identity_sha256": fold.identity_sha256,
                    "validation_block_ids": list(fold.validation_block_ids),
                }
                for fold in self.folds
            ],
            "seeds": list(self.seeds),
            "training_config": asdict(self.training_config),
            "shared_protocol": {
                "base_sde": "provider_control_identity_fixed_per_fold",
                "optimizer_family": "Adam",
                "primary_metric": PRIMARY_METRIC,
                "primary_metric_direction": PRIMARY_METRIC_DIRECTION,
                "final_eval_access": "forbidden",
            },
            "implementation": {
                "path": implementation_path.name,
                "sha256": _sha256(implementation_path),
            },
            "expected_cell_count": expected_cells,
        }

    def _initialize(self, provider: BenchmarkProblemProvider) -> None:
        provider_identity = dict(provider.identity_record)
        if provider_identity.get("final_eval_read_count") != 0:
            raise BenchmarkRunnerError(
                "benchmark provider must attest zero final-eval reads"
            )
        manifest = {
            **self.protocol,
            "protocol_identity_sha256": self.protocol_identity_sha256,
            "provider_identity": provider_identity,
            "provider_identity_sha256": _canonical_hash(provider_identity),
        }
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.cells_root.mkdir(exist_ok=True)
        manifest_path = self.output_root / "manifest.json"
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise BenchmarkRunnerError("benchmark manifest is unreadable") from error
            if existing != manifest:
                raise BenchmarkRunnerError("benchmark resume identity mismatch")
        else:
            temporary = self.output_root / f".manifest-{uuid4().hex}.tmp"
            _write_json(temporary, manifest)
            os.replace(temporary, manifest_path)

    def _run_id(
        self,
        candidate: RepresentationCandidate,
        conditioner_id: str,
        fold: BenchmarkFold,
        seed: int,
    ) -> str:
        digest = _canonical_hash(
            {
                "protocol_identity_sha256": self.protocol_identity_sha256,
                "candidate_id": candidate.candidate_id,
                "conditioner_id": conditioner_id,
                "fold_id": fold.fold_id,
                "seed": seed,
            }
        )
        return f"pirc22-{digest[:24]}"

    def _existing(self, run_id: str) -> dict[str, object] | None:
        cell = self.cells_root / run_id
        if not cell.exists():
            return None
        record_path = cell / "record.json"
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BenchmarkRunnerError(f"resume cell is unreadable: {run_id}") from error
        identity = record.pop("record_identity_sha256", None)
        if identity != _canonical_hash(record):
            raise BenchmarkRunnerError(f"resume record identity mismatch: {run_id}")
        record["record_identity_sha256"] = identity
        if record.get("run_id") != run_id or (
            record.get("protocol_identity_sha256") != self.protocol_identity_sha256
        ):
            raise BenchmarkRunnerError(f"resume record protocol mismatch: {run_id}")
        artifacts = record.get("artifacts", {})
        if not isinstance(artifacts, Mapping):
            raise BenchmarkRunnerError(f"resume record artifacts are invalid: {run_id}")
        for name, expected in artifacts.items():
            artifact = cell / str(name)
            if not artifact.is_file() or _sha256(artifact) != expected:
                raise BenchmarkRunnerError(f"resume artifact identity mismatch: {run_id}")
        return record

    def _commit(
        self,
        run_id: str,
        record: dict[str, object],
        artifacts: Mapping[str, object],
    ) -> dict[str, object]:
        cell = self.cells_root / run_id
        if cell.exists():
            raise BenchmarkRunnerError(f"benchmark cell already exists: {run_id}")
        temporary = self.output_root / f".cell-{run_id}-{uuid4().hex}.tmp"
        temporary.mkdir()
        try:
            artifact_hashes: dict[str, str] = {}
            for name, payload in artifacts.items():
                path = temporary / name
                _write_json(path, payload)
                artifact_hashes[name] = _sha256(path)
            record["artifacts"] = artifact_hashes
            record["record_identity_sha256"] = _canonical_hash(record)
            _write_json(temporary / "record.json", record)
            os.replace(temporary, cell)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        return record

    def _run_cell(
        self,
        provider: BenchmarkProblemProvider,
        candidate: RepresentationCandidate,
        conditioner_id: str,
        fold: BenchmarkFold,
        seed: int,
        control_identities: dict[str, str],
        problem_cache: dict[tuple[str, str], PreparedBenchmark],
    ) -> dict[str, object]:
        run_id = self._run_id(candidate, conditioner_id, fold, seed)
        existing = self._existing(run_id)
        if existing is not None:
            if existing.get("status") == "success":
                existing_control = str(existing.get("control_identity_sha256", ""))
                previous_control = control_identities.setdefault(
                    fold.fold_id, existing_control
                )
                if not existing_control or previous_control != existing_control:
                    raise BenchmarkRunnerError(
                        "resumed control identity changed across candidates"
                    )
            return existing
        started = perf_counter()
        common: dict[str, object] = {
            "schema_version": RUN_RECORD_SCHEMA_VERSION,
            "run_id": run_id,
            "protocol_identity_sha256": self.protocol_identity_sha256,
            "matrix_id": self.matrix.matrix_id,
            "candidate_id": candidate.candidate_id,
            "candidate_order": candidate.order,
            "conditioner_id": conditioner_id,
            "fold_id": fold.fold_id,
            "fold_identity_sha256": fold.identity_sha256,
            "seed": seed,
            "primary_metric": PRIMARY_METRIC,
            "primary_metric_direction": PRIMARY_METRIC_DIRECTION,
            "final_eval_read_count": 0,
        }
        artifacts: dict[str, object] = {}
        try:
            problem_key = (candidate.candidate_id, fold.fold_id)
            problem = problem_cache.get(problem_key)
            if problem is None:
                problem = provider.prepare(candidate, fold)
                _validate_problem(problem, candidate, fold)
                problem_cache[problem_key] = problem
            previous_control = control_identities.setdefault(
                fold.fold_id, problem.control_identity_sha256
            )
            if previous_control != problem.control_identity_sha256:
                raise BenchmarkRunnerError(
                    "base SDE/data/target control identity changed across candidates"
                )
            train_residual = problem.train_targets - problem.train_base_predictions
            validation_residual = (
                problem.validation_targets - problem.validation_base_predictions
            )
            result = fit_conditioner(
                conditioner_id,
                problem.train_features,
                train_residual,
                problem.validation_features,
                validation_residual,
                seed=seed,
                config=self.training_config,
                train_tokens=problem.train_tokens,
                validation_tokens=problem.validation_tokens,
                categorical_cardinality=problem.categorical_cardinality,
                embedding_dim=problem.embedding_dim,
            )
            train_predictions = problem.train_base_predictions + predict_conditioner(
                result.model,
                problem.train_features,
                categorical_tokens=problem.train_tokens,
            )
            validation_predictions = (
                problem.validation_base_predictions
                + predict_conditioner(
                    result.model,
                    problem.validation_features,
                    categorical_tokens=problem.validation_tokens,
                )
            )
            train_loss = float(np.mean((train_predictions - problem.train_targets) ** 2))
            validation_loss = float(
                np.mean((validation_predictions - problem.validation_targets) ** 2)
            )
            record = {
                **common,
                "status": "success",
                "problem_identity_sha256": problem.problem_identity_sha256,
                "control_identity_sha256": problem.control_identity_sha256,
                "input_dimension": candidate.model_input_dim,
                "layer_count": CONDITIONER_BY_ID[conditioner_id].layer_count,
                "hidden_widths": list(
                    CONDITIONER_BY_ID[conditioner_id].hidden_widths
                ),
                "trainable_parameter_count": result.model.trainable_parameter_count,
                "train_row_count": len(problem.train_features),
                "validation_row_count": len(problem.validation_features),
                "train_independent_block_count": len(set(problem.train_block_ids)),
                "validation_independent_block_count": len(
                    set(problem.validation_block_ids)
                ),
                "train_metric": train_loss,
                "validation_metric": validation_loss,
                "train_validation_gap": validation_loss - train_loss,
                "best_epoch": result.best_epoch,
                "epochs_completed": len(result.learning_curve),
                "checkpoint_identity_sha256": result.checkpoint[
                    "checkpoint_identity_sha256"
                ],
                "validation_prediction_sha256": _array_hash(
                    validation_predictions
                ),
                "runtime_seconds": perf_counter() - started,
                "failure": None,
            }
            artifacts = {
                "checkpoint.json": dict(result.checkpoint),
                "learning_curve.json": {
                    "points": [asdict(point) for point in result.learning_curve]
                },
            }
        except Exception as error:
            record = {
                **common,
                "status": "failed",
                "runtime_seconds": perf_counter() - started,
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error)[:1000],
                },
            }
            artifacts = {}
        return self._commit(run_id, record, artifacts)

    def run_all(self, provider: BenchmarkProblemProvider) -> dict[str, object]:
        self._initialize(provider)
        records: list[dict[str, object]] = []
        control_identities: dict[str, str] = {}
        problem_cache: dict[tuple[str, str], PreparedBenchmark] = {}
        for candidate in self.candidates:
            for conditioner_id in self.conditioner_ids:
                for fold in self.folds:
                    for seed in self.seeds:
                        records.append(
                            self._run_cell(
                                provider,
                                candidate,
                                conditioner_id,
                                fold,
                                seed,
                                control_identities,
                                problem_cache,
                            )
                        )
        counts = {
            status: sum(record["status"] == status for record in records)
            for status in ("success", "failed")
        }
        expected = int(self.protocol["expected_cell_count"])
        summary: dict[str, object] = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "protocol_identity_sha256": self.protocol_identity_sha256,
            "expected_cell_count": expected,
            "terminal_cell_count": len(records),
            "status_counts": counts,
            "complete": len(records) == expected,
            "run_record_identities": sorted(
                str(record["record_identity_sha256"]) for record in records
            ),
        }
        summary["summary_identity_sha256"] = _canonical_hash(summary)
        temporary = self.output_root / f".summary-{uuid4().hex}.tmp"
        _write_json(temporary, summary)
        os.replace(temporary, self.output_root / "summary.json")
        return summary


__all__ = [
    "PRIMARY_METRIC",
    "PRIMARY_METRIC_DIRECTION",
    "RUNNER_SCHEMA_VERSION",
    "BenchmarkProblemProvider",
    "BenchmarkRunner",
    "BenchmarkRunnerError",
    "PreparedBenchmark",
]
