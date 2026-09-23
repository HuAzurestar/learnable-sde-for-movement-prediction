"""Execute the frozen PIRC-22 matrix on a bounded production cohort.

This operational harness lives beside private outputs rather than in Git.  Its own
content hash is embedded in the provider identity and the runner manifest.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from experiments.nex326.cohort import Cohort, Segment
from experiments.nex326.pirc21_adapter import FeatureSelection, FeatureSnapshotAdapter
from experiments.nex326.pirc21_interactions import interaction_output_columns
from experiments.nex326.pirc21_runtime import PIRC21FeatureRuntime
from experiments.pirc22.conditioners import TrainingConfig
from experiments.pirc22.representations import (
    RepresentationCandidate,
    RepresentationMatrix,
    load_representation_matrix,
)
from experiments.pirc22.runner import BenchmarkRunner, PreparedBenchmark
from experiments.pirc22.selection import load_benchmark_evidence
from experiments.terrain_benchmark import BenchmarkDataBoundary, BenchmarkFold


PROVIDER_VERSION = "pirc22-production-provider-v1"
BASE_MODEL_VERSION = "pirc22-shared-ridge-drift-v1"
DEFAULT_SEEDS = (20260814, 20260815, 20260816)


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def pirc20_file_id(segment_id: str) -> str:
    parts = segment_id.split(":", 3)
    if len(parts) != 4 or parts[0] != "r1t" or not parts[1]:
        raise ValueError(f"unexpected PIRC-20 segment identity: {segment_id}")
    return parts[1]


def ordered_union(values: Sequence[Sequence[str]]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for group in values for item in group))


def union_selection(matrix: RepresentationMatrix) -> FeatureSelection:
    return FeatureSelection(
        variant_ids=ordered_union(
            [candidate.variant_ids for candidate in matrix.candidates]
        ),
        composition_ids=ordered_union(
            [candidate.composition_ids for candidate in matrix.candidates]
        ),
        interaction_ids=ordered_union(
            [candidate.interaction_ids for candidate in matrix.candidates]
        ),
        include_validity_indicators=True,
    )


def scoped_feature_runtime(
    snapshot_root: Path,
    selection: FeatureSelection,
    train_segments: Sequence[Segment],
) -> PIRC21FeatureRuntime:
    """Build the production runtime while limiting train-fit I/O to selected files."""

    segment_ids = tuple(segment.segment_id for segment in train_segments)
    file_ids = tuple(sorted({pirc20_file_id(value) for value in segment_ids}))
    adapter = FeatureSnapshotAdapter(snapshot_root, selection).fit(
        file_ids=file_ids,
        segment_ids=segment_ids,
    )
    runtime = object.__new__(PIRC21FeatureRuntime)
    runtime.adapter = adapter
    output_columns = adapter.output_columns
    model_columns = [f"pirc21:{name}" for name in output_columns]
    model_columns.extend(f"pirc21:{name}__valid" for name in output_columns)
    interactions = interaction_output_columns(selection.interaction_ids)
    model_columns.extend(f"pirc21:{name}" for name in interactions)
    model_columns.extend(f"pirc21:{name}__valid" for name in interactions)
    runtime.interaction_columns = interactions
    runtime.condition_names = tuple(model_columns)
    return runtime


def attach_production_features(
    runtime: PIRC21FeatureRuntime,
    boundary: BenchmarkDataBoundary,
) -> tuple[tuple[Segment, ...], dict[str, Segment]]:
    train = boundary.fit_segments()
    validation = tuple(
        segment
        for fold in boundary.folds
        for segment in boundary.segments("validation", fold_id=fold.fold_id)
    )
    cohort = Cohort(
        schema_version="nex326-cohort-v1",
        dataset_id=str(runtime.adapter.dataset_id),
        data_version=str(runtime.adapter.dataset_id),
        purpose="pirc22_train_validation_benchmark",
        splits={
            "train": train,
            "validation": validation,
            "adapt": (),
            "evaluation": (),
            "animal_pretrain": (),
        },
        unavailable_reasons={
            "adapt": "not consumed by the PIRC-22 conditioner fit",
            "evaluation": "sealed; unavailable to the PIRC-22 benchmark",
            "animal_pretrain": "outside the PIRC-22 benchmark scope",
        },
        fingerprint=canonical_hash(
            {
                "benchmark_data_identity_sha256": (
                    boundary.benchmark_data_identity_sha256
                ),
                "purpose": "pirc22_train_validation_benchmark",
            }
        ),
    )
    attached = runtime.attach(cohort)
    return attached.splits["train"], {
        segment.segment_id: segment for segment in attached.splits["validation"]
    }


def transition_targets(
    segments: Sequence[Segment],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    states: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    block_ids: list[str] = []
    for segment in segments:
        dt = np.diff(segment.time)
        states.append(segment.state[:-1])
        targets.append(np.diff(segment.state, axis=0) / dt[:, None])
        block_id = str(getattr(segment, "independent_block_id", ""))
        if not block_id:
            raise ValueError(f"segment lacks independent block: {segment.segment_id}")
        block_ids.extend([block_id] * (len(segment.time) - 1))
    return np.concatenate(states), np.concatenate(targets), tuple(block_ids)


@dataclass(frozen=True)
class FeatureRows:
    numeric: np.ndarray
    valid: np.ndarray


class ProductionProvider:
    def __init__(
        self,
        *,
        boundary: BenchmarkDataBoundary,
        matrix: RepresentationMatrix,
        snapshot_root: Path,
        harness_path: Path,
    ) -> None:
        self.boundary = boundary
        self.matrix = matrix
        self.runtime = scoped_feature_runtime(
            snapshot_root,
            union_selection(matrix),
            boundary.fit_segments(),
        )
        self.train_segments, self.validation_by_id = attach_production_features(
            self.runtime, boundary
        )
        self.train_states, self.train_targets, self.train_row_blocks = (
            transition_targets(self.train_segments)
        )
        base_features = np.column_stack(
            (np.ones(len(self.train_states)), self.train_states)
        )
        ridge = 1e-6 * np.eye(base_features.shape[1])
        self.base_weights = np.linalg.solve(
            base_features.T @ base_features + ridge,
            base_features.T @ self.train_targets,
        )
        self.base_identity = canonical_hash(
            {
                "version": BASE_MODEL_VERSION,
                "ridge": 1e-6,
                "weights": self.base_weights.tolist(),
                "train_targets_sha256": array_hash(self.train_targets),
                "train_segment_ids": [
                    segment.segment_id for segment in self.train_segments
                ],
            }
        )
        self._variant_columns = {
            str(item["variant_id"]): tuple(
                str(column["name"]) for column in item["output_columns"]
            )
            for item in self.runtime.adapter.spec["variants"]
        }
        self._composition_columns = {
            str(item["composition_id"]): tuple(
                str(column["name"]) for column in item["output_columns"]
            )
            for item in self.runtime.adapter.spec["compositions"]
        }
        snapshot_manifest = self.runtime.adapter.manifest
        self._identity = {
            "provider_version": PROVIDER_VERSION,
            "provider_source_sha256": file_hash(harness_path),
            "benchmark_data_identity_sha256": (
                boundary.benchmark_data_identity_sha256
            ),
            "snapshot_id": snapshot_manifest["snapshot_id"],
            "snapshot_content_inventory_sha256": snapshot_manifest[
                "content_inventory_sha256"
            ],
            "feature_runtime_identity": self.runtime.identity_record,
            "base_model_version": BASE_MODEL_VERSION,
            "base_model_identity_sha256": self.base_identity,
            "semantic_roles_read": ["train", "validation"],
            "final_eval_identity_sealed": True,
            "final_eval_content_materialized": False,
            "final_eval_read_count": 0,
        }

    @property
    def identity_record(self) -> Mapping[str, object]:
        return self._identity

    def _point_rows(
        self, segments: Sequence[Segment], names: Sequence[str]
    ) -> FeatureRows:
        numeric = np.concatenate(
            [
                np.column_stack(
                    [segment.conditions[f"pirc21:{name}"][:-1] for name in names]
                )
                for segment in segments
            ],
            axis=0,
        )
        valid = np.concatenate(
            [
                np.column_stack(
                    [
                        segment.conditions[f"pirc21:{name}__valid"][:-1]
                        for name in names
                    ]
                )
                for segment in segments
            ],
            axis=0,
        ).astype(bool)
        return FeatureRows(numeric, valid)

    def _legacy_rows(self, segments: Sequence[Segment]) -> np.ndarray:
        return np.concatenate(
            [
                np.column_stack(
                    [
                        segment.conditions.get(
                            "terrain_elevation", np.zeros(len(segment.time))
                        )[:-1],
                        segment.conditions.get(
                            "terrain_slope", np.zeros(len(segment.time))
                        )[:-1],
                    ]
                )
                for segment in segments
            ],
            axis=0,
        )

    def _base_columns(self, candidate: RepresentationCandidate) -> tuple[str, ...]:
        return tuple(
            name
            for variant_id in candidate.variant_ids
            for name in self._variant_columns[variant_id]
        ) + tuple(
            name
            for composition_id in candidate.composition_ids
            for name in self._composition_columns[composition_id]
        )

    def _candidate_features(
        self,
        candidate: RepresentationCandidate,
        train: Sequence[Segment],
        validation: Sequence[Segment],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        if candidate.candidate_id == "R00-no-terrain":
            return (
                np.empty((len(self.train_targets), 0)),
                np.empty((sum(len(value.time) - 1 for value in validation), 0)),
                None,
                None,
            )
        if candidate.legacy_condition_names:
            return self._legacy_rows(train), self._legacy_rows(validation), None, None

        base_names = self._base_columns(candidate)
        if candidate.categorical_embedding is not None:
            if len(base_names) != 1:
                raise ValueError("embedding representation must expose one token")
            train_rows = self._point_rows(train, base_names)
            validation_rows = self._point_rows(validation, base_names)
            train_tokens = np.where(
                train_rows.valid[:, 0], train_rows.numeric[:, 0], 0
            ).astype(np.int64)
            validation_tokens = np.where(
                validation_rows.valid[:, 0], validation_rows.numeric[:, 0], 0
            ).astype(np.int64)
            return (
                train_rows.valid.astype(float),
                validation_rows.valid.astype(float),
                train_tokens,
                validation_tokens,
            )

        if candidate.derived_transforms:
            if len(candidate.derived_transforms) != 1:
                raise ValueError("production provider expects one registered transform")
            transform = candidate.derived_transforms[0]
            source_names = tuple(
                name
                for variant_id in transform.source_variant_ids
                for name in self._variant_columns[variant_id]
            )
            train_rows = self._point_rows(train, source_names)
            validation_rows = self._point_rows(validation, source_names)
            if transform.transform == "clipped_standardize":
                clip = float(dict(transform.parameters)["clip_sigma"])
                train_numeric = np.zeros_like(train_rows.numeric)
                validation_numeric = np.zeros_like(validation_rows.numeric)
                for index in range(len(source_names)):
                    observed = train_rows.numeric[train_rows.valid[:, index], index]
                    mean = float(np.mean(observed))
                    scale = float(np.std(observed))
                    if not np.isfinite(scale) or scale < 1e-12:
                        scale = 1.0
                    train_numeric[:, index] = np.clip(
                        (train_rows.numeric[:, index] - mean) / scale, -clip, clip
                    )
                    validation_numeric[:, index] = np.clip(
                        (validation_rows.numeric[:, index] - mean) / scale,
                        -clip,
                        clip,
                    )
                train_numeric[~train_rows.valid] = 0.0
                validation_numeric[~validation_rows.valid] = 0.0
                return (
                    np.concatenate((train_numeric, train_rows.valid), axis=1),
                    np.concatenate(
                        (validation_numeric, validation_rows.valid), axis=1
                    ),
                    None,
                    None,
                )
            if transform.transform == "fixed_rbf":
                parameters = dict(transform.parameters)
                knots = np.asarray(parameters["knots_m"], dtype=float)
                width = float(parameters["width_m"])

                def rbf(rows: FeatureRows) -> np.ndarray:
                    source = rows.numeric.copy()
                    for index, variant_id in enumerate(transform.source_variant_ids):
                        if variant_id.endswith("distance_log1p"):
                            source[:, index] = np.expm1(source[:, index])
                    values = np.exp(
                        -0.5
                        * ((source[:, :, None] - knots[None, None, :]) / width) ** 2
                    ).reshape(len(source), -1)
                    valid = np.repeat(rows.valid, len(knots), axis=1)
                    values[~valid] = 0.0
                    return np.concatenate((values, valid), axis=1)

                return rbf(train_rows), rbf(validation_rows), None, None
            raise ValueError(f"unsupported registered transform: {transform.transform}")

        output_names = (
            base_names if candidate.include_base_outputs else ()
        ) + interaction_output_columns(candidate.interaction_ids)
        train_rows = self._point_rows(train, output_names)
        validation_rows = self._point_rows(validation, output_names)
        return (
            np.concatenate((train_rows.numeric, train_rows.valid), axis=1),
            np.concatenate(
                (validation_rows.numeric, validation_rows.valid), axis=1
            ),
            None,
            None,
        )

    def _base_predict(self, states: np.ndarray) -> np.ndarray:
        return np.column_stack((np.ones(len(states)), states)) @ self.base_weights

    def prepare(
        self, candidate: RepresentationCandidate, fold: BenchmarkFold
    ) -> PreparedBenchmark:
        validation = tuple(
            self.validation_by_id[segment_id]
            for segment_id in fold.validation_segment_ids
        )
        validation_states, validation_targets, validation_row_blocks = (
            transition_targets(validation)
        )
        train_features, validation_features, train_tokens, validation_tokens = (
            self._candidate_features(candidate, self.train_segments, validation)
        )
        embedding = candidate.categorical_embedding
        expected_continuous = candidate.model_input_dim - (
            embedding.embedding_dim if embedding else 0
        )
        if train_features.shape[1] != expected_continuous:
            raise ValueError(
                f"candidate feature dimension mismatch: {candidate.candidate_id}: "
                f"{train_features.shape[1]} != {expected_continuous}"
            )
        control = {
            "benchmark_data_identity_sha256": (
                self.boundary.benchmark_data_identity_sha256
            ),
            "fold_identity_sha256": fold.identity_sha256,
            "base_model_identity_sha256": self.base_identity,
            "train_targets_sha256": array_hash(self.train_targets),
            "validation_targets_sha256": array_hash(validation_targets),
            "train_row_blocks_sha256": canonical_hash(self.train_row_blocks),
            "validation_row_blocks_sha256": canonical_hash(validation_row_blocks),
        }
        problem = {
            **control,
            "candidate_id": candidate.candidate_id,
            "train_features_sha256": array_hash(train_features),
            "validation_features_sha256": array_hash(validation_features),
        }
        return PreparedBenchmark(
            train_features=train_features,
            train_targets=self.train_targets,
            train_base_predictions=self._base_predict(self.train_states),
            validation_features=validation_features,
            validation_targets=validation_targets,
            validation_base_predictions=self._base_predict(validation_states),
            train_block_ids=tuple(sorted(set(self.train_row_blocks))),
            validation_block_ids=fold.validation_block_ids,
            train_row_block_ids=self.train_row_blocks,
            validation_row_block_ids=validation_row_blocks,
            control_identity_sha256=canonical_hash(control),
            problem_identity_sha256=canonical_hash(problem),
            train_tokens=train_tokens,
            validation_tokens=validation_tokens,
            categorical_cardinality=(12 if embedding else None),
            embedding_dim=(embedding.embedding_dim if embedding else None),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--condition-root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-segments", type=int, default=16)
    parser.add_argument("--adapt-segments", type=int, default=8)
    parser.add_argument("--validation-segments", type=int, default=40)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    limits = {
        "train": args.train_segments,
        "adapt": args.adapt_segments,
        "validation": args.validation_segments,
    }
    boundary = BenchmarkDataBoundary.from_pirc20_release(
        args.cohort,
        args.trajectory,
        args.condition_root,
        fold_count=args.folds,
        maximum_segments_per_role=limits,
    )
    matrix = load_representation_matrix()
    provider = ProductionProvider(
        boundary=boundary,
        matrix=matrix,
        snapshot_root=args.snapshot,
        harness_path=Path(__file__).resolve(),
    )
    runner = BenchmarkRunner(
        args.output,
        matrix=matrix,
        data_identity=boundary.identity_record,
        folds=boundary.folds,
        seeds=DEFAULT_SEEDS,
        training_config=TrainingConfig(
            learning_rate=1e-3,
            weight_decay=1e-4,
            batch_size=256,
            maximum_epochs=40,
            patience=6,
        ),
    )
    summary = runner.run_all(provider)
    manifest, verified_summary, records = load_benchmark_evidence(args.output)
    receipt: dict[str, object] = {
        "schema_version": "pirc22-production-run-receipt-v1",
        "scientific_role": "bounded_production_train_validation_selection",
        "implementation_sha": "eba0b9dfe6d61babf379d3897e1a424d5890f3ee",
        "provider_source_sha256": file_hash(Path(__file__).resolve()),
        "limits": limits,
        "fold_count": len(boundary.folds),
        "seeds": list(DEFAULT_SEEDS),
        "candidate_count": len(matrix.candidates),
        "conditioner_count": 5,
        "expected_cell_count": manifest["expected_cell_count"],
        "terminal_cell_count": len(records),
        "status_counts": summary["status_counts"],
        "protocol_identity_sha256": manifest["protocol_identity_sha256"],
        "summary_identity_sha256": verified_summary["summary_identity_sha256"],
        "benchmark_data_identity_sha256": (
            boundary.benchmark_data_identity_sha256
        ),
        "snapshot_id": provider.identity_record["snapshot_id"],
        "snapshot_content_inventory_sha256": provider.identity_record[
            "snapshot_content_inventory_sha256"
        ],
        "access_audit": boundary.audit_record,
        "final_eval_read_count": 0,
    }
    receipt["receipt_identity_sha256"] = canonical_hash(receipt)
    write_json(args.output / "run-receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
