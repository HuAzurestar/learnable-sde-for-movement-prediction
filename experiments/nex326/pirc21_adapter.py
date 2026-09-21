"""Versioned PIRC-21 FeatureRow adapter for PSDE model inputs.

The adapter consumes one immutable DSDE snapshot, validates every registered
artifact, applies only transformations declared by its feature spec, and keeps
missing values separate from numeric values through an explicit validity mask.
Train-fitted transformations are fit from the frozen ``train`` split only.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .pirc21_interactions import (
    INTERACTION_REGISTRY_VERSION,
    PIRC21InteractionError,
    interaction_registry_fingerprint,
    validate_interaction_selection,
)


FEATURE_SPEC_SCHEMA_VERSION = "pirc21-feature-spec-v1"
FEATURE_SNAPSHOT_SCHEMA_VERSION = "pirc21-feature-snapshot-v1"
ADAPTER_VERSION = "pirc21-psde-adapter-v1"
_FINAL_EVAL_SPLIT = "final_eval"
_AGGREGATIONS = {"mean", "std", "min", "max", "last", "valid_fraction"}
_SUPPORTED_TRANSFORMS = {
    "identity",
    "log1p",
    "clipped_standardize",
    "threshold_bins",
    "fixed_rbf",
    "visible_time_difference",
    "unit_vector",
    "relative_heading_sin_cos",
    "tangent_normal_projection",
    "one_hot",
    "grouped_one_hot",
    "embedding_identity",
}
_SUPPORTED_COMPOSITIONS = {
    "concatenate",
    "elementwise_product",
    "scalar_vector_product",
    "pairwise_product",
}
_IDENTITY_COLUMNS = (
    "dataset_version",
    "point_id",
    "file_id",
    "point_index",
    "absolute_epoch_ns",
    "segment_id",
    "split",
    "independent_block_id",
)
_WORLDCOVER_CODES = (10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)
_WORLDCOVER_GROUPS = {
    "built": {50},
    "vegetated": {10, 20, 30, 40, 95, 100},
    "water_wetland": {80, 90},
    "bare_snow_unknown": {60, 70},
}


class PIRC21AdapterError(ValueError):
    """A PIRC-21 snapshot or requested model layout is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PIRC21AdapterError(f"cannot read JSON artifact: {path.name}") from error
    if not isinstance(value, dict):
        raise PIRC21AdapterError(f"JSON artifact is not an object: {path.name}")
    return value


def _safe_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise PIRC21AdapterError(
            f"snapshot path escapes its root: {relative}"
        ) from error
    if not path.is_file() or path.is_symlink():
        raise PIRC21AdapterError(
            f"snapshot artifact is missing or symlinked: {relative}"
        )
    return path


def _numeric(table: pa.Table, name: str) -> np.ndarray:
    if name not in table.column_names:
        raise PIRC21AdapterError(f"FeatureRow column is absent: {name}")
    values = table[name].to_pylist()
    result = np.empty(len(values), dtype=float)
    for index, value in enumerate(values):
        try:
            result[index] = float(value) if value is not None else math.nan
        except (TypeError, ValueError):
            result[index] = math.nan
    return result


def _int64(table: pa.Table, name: str) -> np.ndarray:
    if name not in table.column_names:
        raise PIRC21AdapterError(f"FeatureRow column is absent: {name}")
    values = table[name].to_pylist()
    if any(value is None for value in values):
        raise PIRC21AdapterError(f"integer identity column contains nulls: {name}")
    try:
        return np.asarray(values, dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as error:
        raise PIRC21AdapterError(
            f"integer identity column is invalid: {name}"
        ) from error


def _strings(table: pa.Table, name: str) -> np.ndarray:
    if name not in table.column_names:
        raise PIRC21AdapterError(f"FeatureRow column is absent: {name}")
    values = table[name].to_pylist()
    if any(value is None for value in values):
        raise PIRC21AdapterError(f"identity/status column contains nulls: {name}")
    return np.asarray([str(value) for value in values], dtype=object)


@dataclass(frozen=True)
class FeatureSelection:
    """One ordered, independently cacheable set of model inputs."""

    variant_ids: tuple[str, ...] = ()
    composition_ids: tuple[str, ...] = ()
    interaction_ids: tuple[str, ...] = ()
    segment_aggregations: tuple[str, ...] = ()
    include_validity_indicators: bool = True

    def validate(self) -> None:
        for name, values in (
            ("variant_ids", self.variant_ids),
            ("composition_ids", self.composition_ids),
            ("interaction_ids", self.interaction_ids),
            ("segment_aggregations", self.segment_aggregations),
        ):
            if not isinstance(values, tuple) or len(values) != len(set(values)):
                raise PIRC21AdapterError(f"{name} must be an ordered unique tuple")
        unknown = set(self.segment_aggregations) - _AGGREGATIONS
        if unknown:
            raise PIRC21AdapterError(
                f"unsupported segment aggregations: {sorted(unknown)}"
            )
        if not isinstance(self.include_validity_indicators, bool):
            raise PIRC21AdapterError("include_validity_indicators must be boolean")


@dataclass(frozen=True)
class SegmentFeatureBatch:
    segment_ids: tuple[str, ...]
    columns: tuple[str, ...]
    values: np.ndarray
    valid: np.ndarray
    cache_identity: str
    include_validity_indicators: bool

    def model_matrix(self, *, include_validity: bool | None = None) -> np.ndarray:
        if include_validity is None:
            include_validity = self.include_validity_indicators
        if not include_validity and not self.valid.all():
            raise PIRC21AdapterError(
                "validity indicators cannot be disabled while segment values are missing"
            )
        numeric = np.where(self.valid, self.values, 0.0)
        if include_validity and numeric.shape[1]:
            return np.concatenate((numeric, self.valid.astype(float)), axis=1)
        return numeric


@dataclass(frozen=True)
class PointFeatureBatch:
    split: str
    point_ids: tuple[str, ...]
    segment_ids: tuple[str, ...]
    absolute_epoch_ns: np.ndarray
    columns: tuple[str, ...]
    values: np.ndarray
    valid: np.ndarray
    cache_identity: str
    include_validity_indicators: bool

    def model_matrix(self, *, include_validity: bool | None = None) -> np.ndarray:
        """Return finite model input plus explicit missingness indicators.

        Zero is used only as the numeric placeholder paired with a false validity
        bit; callers can never mistake it for a valid observed zero.
        """

        if include_validity is None:
            include_validity = self.include_validity_indicators
        if not include_validity and not self.valid.all():
            raise PIRC21AdapterError(
                "validity indicators cannot be disabled while point values are missing"
            )
        numeric = np.where(self.valid, self.values, 0.0)
        if include_validity and numeric.shape[1]:
            return np.concatenate((numeric, self.valid.astype(float)), axis=1)
        return numeric

    def aggregate(self, operations: Sequence[str]) -> SegmentFeatureBatch:
        operations = tuple(operations)
        unknown = set(operations) - _AGGREGATIONS
        if not operations or unknown or len(operations) != len(set(operations)):
            raise PIRC21AdapterError(
                "segment aggregations must be non-empty and unique"
            )
        segment_ids = tuple(sorted(set(self.segment_ids)))
        output_columns = tuple(
            f"{column}__{operation}"
            for operation in operations
            for column in self.columns
        )
        values = np.full((len(segment_ids), len(output_columns)), np.nan, dtype=float)
        valid = np.zeros_like(values, dtype=bool)
        segment_array = np.asarray(self.segment_ids, dtype=object)
        for segment_offset, segment_id in enumerate(segment_ids):
            row_indexes = np.flatnonzero(segment_array == segment_id)
            order = row_indexes[
                np.argsort(self.absolute_epoch_ns[row_indexes], kind="stable")
            ]
            target_offset = 0
            for operation in operations:
                for column_index in range(len(self.columns)):
                    mask = self.valid[order, column_index]
                    observed = self.values[order, column_index][mask]
                    if operation == "valid_fraction":
                        values[segment_offset, target_offset] = float(np.mean(mask))
                        valid[segment_offset, target_offset] = True
                    elif observed.size:
                        if operation == "mean":
                            value = float(np.mean(observed))
                        elif operation == "std":
                            value = float(np.std(observed))
                        elif operation == "min":
                            value = float(np.min(observed))
                        elif operation == "max":
                            value = float(np.max(observed))
                        else:
                            value = float(observed[-1])
                        values[segment_offset, target_offset] = value
                        valid[segment_offset, target_offset] = True
                    target_offset += 1
        identity = _canonical_hash(
            {
                "point_cache_identity": self.cache_identity,
                "segment_aggregations": operations,
                "segment_ids": segment_ids,
            }
        )
        return SegmentFeatureBatch(
            segment_ids,
            output_columns,
            values,
            valid,
            identity,
            self.include_validity_indicators,
        )


@dataclass(frozen=True)
class _VariantResult:
    values: np.ndarray
    valid: np.ndarray
    columns: tuple[str, ...]


class FeatureSnapshotAdapter:
    """Validate one immutable snapshot and expose configured PSDE matrices."""

    def __init__(self, snapshot_root: str | Path, selection: FeatureSelection) -> None:
        selection.validate()
        self.root = Path(snapshot_root).resolve()
        if not self.root.is_dir():
            raise PIRC21AdapterError("feature snapshot root is absent")
        self.manifest = _json(_safe_file(self.root, "manifest.json"))
        self.spec = _json(_safe_file(self.root, "feature_spec.json"))
        self.selection = selection
        self._validate_snapshot()
        self._factor_by_id = {
            str(item["factor_id"]): item for item in self.spec["factors"]
        }
        self._variant_by_id = {
            str(item["variant_id"]): item for item in self.spec["variants"]
        }
        self._composition_by_id = {
            str(item["composition_id"]): item for item in self.spec["compositions"]
        }
        self._column_factor: dict[str, str] = {}
        for factor_id, factor in self._factor_by_id.items():
            for column in factor["value_columns"]:
                self._column_factor[str(column["name"])] = factor_id
        self._validate_selection()
        self._fit_state: dict[str, dict[str, list[float]]] | None = None

    @property
    def dataset_id(self) -> str:
        return str(self.manifest["dataset_id"])

    @property
    def feature_spec_id(self) -> str:
        return str(self.spec["feature_spec_id"])

    @property
    def output_columns(self) -> tuple[str, ...]:
        names: list[str] = []
        for variant_id in self.selection.variant_ids:
            names.extend(
                str(column["name"])
                for column in self._variant_by_id[variant_id]["output_columns"]
            )
        for composition_id in self.selection.composition_ids:
            names.extend(
                str(column["name"])
                for column in self._composition_by_id[composition_id]["output_columns"]
            )
        return tuple(names)

    @property
    def fit_state_fingerprint(self) -> str:
        return _canonical_hash(self._fit_state or {})

    @property
    def identity_record(self) -> dict[str, object]:
        """Return the JSON-safe identity to embed in caches and RunRecords."""

        return {
            "adapter_version": ADAPTER_VERSION,
            "dataset_id": self.dataset_id,
            "snapshot_id": str(self.manifest["snapshot_id"]),
            "content_inventory_sha256": str(self.manifest["content_inventory_sha256"]),
            "feature_spec_id": self.feature_spec_id,
            "feature_spec_sha256": str(self.manifest["feature_spec_sha256"]),
            "processing_version": str(self.spec["processing_version"]),
            "variant_ids": list(self.selection.variant_ids),
            "composition_ids": list(self.selection.composition_ids),
            "interaction_ids": list(self.selection.interaction_ids),
            "interaction_registry_version": INTERACTION_REGISTRY_VERSION,
            "interaction_registry_sha256": interaction_registry_fingerprint(),
            "segment_aggregation_ids": list(self.selection.segment_aggregations),
            "selection_fingerprint": self.selection_fingerprint,
            "fit_state": self._fit_state or {},
            "fit_state_fingerprint": self.fit_state_fingerprint,
        }

    @property
    def selection_fingerprint(self) -> str:
        return _canonical_hash(
            {
                "adapter_version": ADAPTER_VERSION,
                "variant_ids": self.selection.variant_ids,
                "composition_ids": self.selection.composition_ids,
                "interaction_ids": self.selection.interaction_ids,
                "interaction_registry_sha256": interaction_registry_fingerprint(),
                "segment_aggregations": self.selection.segment_aggregations,
                "include_validity_indicators": self.selection.include_validity_indicators,
            }
        )

    def _validate_snapshot(self) -> None:
        if self.manifest.get("schema_version") != FEATURE_SNAPSHOT_SCHEMA_VERSION:
            raise PIRC21AdapterError("unsupported feature snapshot schema")
        if self.manifest.get("status") != "valid":
            raise PIRC21AdapterError("feature snapshot is not valid")
        if self.spec.get("schema_version") != FEATURE_SPEC_SCHEMA_VERSION:
            raise PIRC21AdapterError("unsupported feature spec schema")
        if self.manifest.get("feature_spec_id") != self.spec.get("feature_spec_id"):
            raise PIRC21AdapterError("manifest/spec identity mismatch")
        fingerprint = _canonical_hash(self.spec)
        if self.manifest.get("feature_spec_sha256") != fingerprint:
            raise PIRC21AdapterError("feature spec fingerprint mismatch")
        coverage = _safe_file(self.root, "coverage_report.json")
        if self.manifest.get("coverage_report_sha256") != _sha256(coverage):
            raise PIRC21AdapterError("coverage report hash mismatch")
        files = self.manifest.get("files")
        if not isinstance(files, list) or not files:
            raise PIRC21AdapterError("snapshot file inventory is empty")
        normalized_files: list[Mapping[str, Any]] = []
        for index, item in enumerate(files):
            if not isinstance(item, Mapping):
                raise PIRC21AdapterError(f"invalid snapshot file entry: {index}")
            if not isinstance(item.get("path"), str) or not item["path"]:
                raise PIRC21AdapterError(f"snapshot file path is invalid: {index}")
            normalized_files.append(item)
        seen_paths: set[str] = set()
        inventory = hashlib.sha256()
        for item in sorted(normalized_files, key=lambda value: str(value["path"])):
            relative = str(item.get("path", ""))
            if not relative or relative in seen_paths:
                raise PIRC21AdapterError(f"duplicate/empty snapshot path: {relative!r}")
            seen_paths.add(relative)
            path = _safe_file(self.root, relative)
            digest = _sha256(path)
            row_count = int(item.get("row_count", -1))
            if digest != item.get("sha256"):
                raise PIRC21AdapterError(f"feature file hash mismatch: {relative}")
            if pq.ParquetFile(path).metadata.num_rows != row_count:
                raise PIRC21AdapterError(f"feature file row count mismatch: {relative}")
            inventory.update(f"{relative}\0{digest}\0{row_count}\n".encode("utf-8"))
        if inventory.hexdigest() != self.manifest.get("content_inventory_sha256"):
            raise PIRC21AdapterError("feature content inventory mismatch")

    def _validate_selection(self) -> None:
        unknown_variants = set(self.selection.variant_ids) - set(self._variant_by_id)
        if unknown_variants:
            raise PIRC21AdapterError(f"unknown variants: {sorted(unknown_variants)}")
        unknown_compositions = set(self.selection.composition_ids) - set(
            self._composition_by_id
        )
        if unknown_compositions:
            raise PIRC21AdapterError(
                f"unknown compositions: {sorted(unknown_compositions)}"
            )
        selected = set(self.selection.variant_ids)
        output_names: list[str] = []
        for variant_id in self.selection.variant_ids:
            variant = self._variant_by_id[variant_id]
            transform = str(variant["transform"])
            if transform not in _SUPPORTED_TRANSFORMS:
                raise PIRC21AdapterError(
                    f"unsupported transform for {variant_id}: {transform}"
                )
            columns = tuple(str(item["name"]) for item in variant["output_columns"])
            if len(columns) != int(variant["output_dim"]):
                raise PIRC21AdapterError(
                    f"variant output dimension mismatch: {variant_id}"
                )
            output_names.extend(columns)
        for composition_id in self.selection.composition_ids:
            composition = self._composition_by_id[composition_id]
            dependencies = {str(value) for value in composition["input_variant_ids"]}
            if not dependencies <= selected:
                raise PIRC21AdapterError(
                    f"composition {composition_id} requires unselected variants"
                )
            operator = str(composition["operator"])
            if operator not in _SUPPORTED_COMPOSITIONS:
                raise PIRC21AdapterError(
                    f"unsupported composition for {composition_id}: {operator}"
                )
            columns = tuple(str(item["name"]) for item in composition["output_columns"])
            if len(columns) != int(composition["output_dim"]):
                raise PIRC21AdapterError(
                    f"composition output dimension mismatch: {composition_id}"
                )
            output_names.extend(columns)
        if len(output_names) != len(set(output_names)):
            raise PIRC21AdapterError("selected adapter output columns are not unique")
        try:
            validate_interaction_selection(
                self.selection.interaction_ids, output_names
            )
        except PIRC21InteractionError as error:
            raise PIRC21AdapterError(str(error)) from error
        aggregation_ids = {
            str(item["aggregation_id"])
            for item in self.spec.get("segment_aggregations", ())
        }
        missing_aggregations = (
            set(self.selection.segment_aggregations) - aggregation_ids
        )
        if missing_aggregations:
            raise PIRC21AdapterError(
                "segment aggregations are not registered in the feature spec: "
                f"{sorted(missing_aggregations)}"
            )

    def _entries(
        self, split: str, *, file_ids: Sequence[str] | None = None
    ) -> list[Mapping[str, Any]]:
        wanted = set(file_ids) if file_ids is not None else None
        entries = [
            item
            for item in self.manifest["files"]
            if str(item.get("split")) == split
            and (wanted is None or str(item.get("file_id")) in wanted)
        ]
        if not entries:
            if wanted is not None:
                raise PIRC21AdapterError(
                    f"snapshot file identities are absent: {sorted(wanted)}"
                )
            raise PIRC21AdapterError(f"snapshot split is absent: {split}")
        if wanted is not None:
            found = {str(item.get("file_id")) for item in entries}
            if found != wanted:
                raise PIRC21AdapterError(
                    f"snapshot file identities are absent: {sorted(wanted - found)}"
                )
        return sorted(entries, key=lambda item: str(item["path"]))

    def _required_columns(self) -> list[str]:
        names = list(_IDENTITY_COLUMNS)
        factors: set[str] = set()
        for variant_id in self.selection.variant_ids:
            variant = self._variant_by_id[variant_id]
            parameters = variant["parameters"]
            for name in (
                *parameters.get("input_columns", ()),
                *parameters.get("reference_columns", ()),
            ):
                name = str(name)
                if name not in names:
                    names.append(name)
                if name in self._column_factor:
                    factors.add(self._column_factor[name])
        for factor_id in sorted(factors):
            status = str(self._factor_by_id[factor_id]["status_column"])
            if status not in names:
                names.append(status)
        return names

    def _load(
        self,
        split: str,
        *,
        final_eval_unlock: str | None,
        file_ids: Sequence[str] | None = None,
        segment_ids: Sequence[str] | None = None,
    ) -> pa.Table:
        if split == _FINAL_EVAL_SPLIT and final_eval_unlock != self.dataset_id:
            raise PIRC21AdapterError(
                "final_eval is sealed; provide the exact dataset ID acknowledgement"
            )
        columns = self._required_columns()
        tables: list[pa.Table] = []
        for entry in self._entries(split, file_ids=file_ids):
            path = _safe_file(self.root, str(entry["path"]))
            table = pq.read_table(path, columns=columns).combine_chunks()
            metadata = table.schema.metadata or {}
            expected_spec = str(self.manifest["feature_spec_sha256"]).encode("ascii")
            if metadata.get(b"pirc21.feature_spec_sha256") not in (None, expected_spec):
                raise PIRC21AdapterError("Parquet feature spec metadata mismatch")
            tables.append(table)
        table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        if segment_ids is not None:
            wanted_segments = tuple(dict.fromkeys(segment_ids))
            table = table.filter(
                pc.is_in(table["segment_id"], value_set=pa.array(wanted_segments))
            )
            found_segments = set(_strings(table, "segment_id").tolist())
            if found_segments != set(wanted_segments):
                raise PIRC21AdapterError(
                    "snapshot segment identities are absent: "
                    f"{sorted(set(wanted_segments) - found_segments)}"
                )
        if any(value != split for value in _strings(table, "split")):
            raise PIRC21AdapterError(f"FeatureRow split identity mismatch: {split}")
        if any(
            value != self.dataset_id for value in _strings(table, "dataset_version")
        ):
            raise PIRC21AdapterError("FeatureRow dataset identity mismatch")
        point_ids = _strings(table, "point_id")
        if len(point_ids) != len(set(point_ids.tolist())):
            raise PIRC21AdapterError("FeatureRow point identities are not unique")
        return table

    def _raw_inputs(
        self, table: pa.Table, variant: Mapping[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        parameters = variant["parameters"]
        input_names = [str(name) for name in parameters["input_columns"]]
        reference_names = [
            str(name) for name in parameters.get("reference_columns", ())
        ]
        names = input_names + reference_names
        arrays = np.column_stack([_numeric(table, name) for name in names])
        valid = np.isfinite(arrays).all(axis=1)
        required_factors = {self._column_factor[name] for name in names}
        for factor_id in required_factors:
            status_name = str(self._factor_by_id[factor_id]["status_column"])
            valid &= _strings(table, status_name) == "valid"
        return arrays, valid

    def fit(self) -> "FeatureSnapshotAdapter":
        """Fit registered train-only transforms from the train split and nothing else."""

        train_fitted = [
            variant_id
            for variant_id in self.selection.variant_ids
            if self._variant_by_id[variant_id]["fit_scope"] == "train_only"
        ]
        state: dict[str, dict[str, list[float]]] = {}
        if train_fitted:
            table = self._load("train", final_eval_unlock=None)
            for variant_id in train_fitted:
                variant = self._variant_by_id[variant_id]
                values, valid = self._raw_inputs(table, variant)
                observed = values[valid]
                if not len(observed):
                    raise PIRC21AdapterError(
                        f"train split has no valid rows for fitted variant: {variant_id}"
                    )
                center = np.mean(observed, axis=0)
                scale = np.std(observed, axis=0)
                scale = np.where(scale > 1e-12, scale, 1.0)
                state[variant_id] = {
                    "center": center.tolist(),
                    "scale": scale.tolist(),
                }
        self._fit_state = state
        return self

    def _variant(self, table: pa.Table, variant_id: str) -> _VariantResult:
        variant = self._variant_by_id[variant_id]
        transform = str(variant["transform"])
        parameters = variant["parameters"]
        raw, row_valid = self._raw_inputs(table, variant)
        output_columns = tuple(
            str(column["name"]) for column in variant["output_columns"]
        )
        valid: np.ndarray
        if transform == "identity":
            values = raw
            valid = np.repeat(row_valid[:, None], raw.shape[1], axis=1)
        elif transform == "log1p":
            scale = float(parameters.get("scale_m", 1.0))
            if not math.isfinite(scale) or scale <= 0:
                raise PIRC21AdapterError(f"invalid log1p scale: {variant_id}")
            domain = raw[:, :1] >= 0
            values = np.log1p(np.maximum(raw[:, :1], 0.0) / scale)
            valid = row_valid[:, None] & domain
        elif transform == "clipped_standardize":
            if self._fit_state is None or variant_id not in self._fit_state:
                raise PIRC21AdapterError(
                    f"fit() is required before using train-fitted variant: {variant_id}"
                )
            state = self._fit_state[variant_id]
            center = np.asarray(state["center"], dtype=float)
            scale = np.asarray(state["scale"], dtype=float)
            clip = float(parameters["clip_sigma"])
            values = np.clip((raw - center) / scale, -clip, clip)
            valid = np.repeat(row_valid[:, None], raw.shape[1], axis=1)
        elif transform == "unit_vector":
            vector = raw[:, :2]
            norm = np.linalg.norm(vector, axis=1)
            usable = row_valid & (norm > 1e-12)
            values = np.full_like(vector, np.nan)
            values[usable] = vector[usable] / norm[usable, None]
            valid = np.repeat(usable[:, None], 2, axis=1)
        elif transform == "relative_heading_sin_cos":
            if raw.shape[1] != 4:
                raise PIRC21AdapterError(
                    f"relative heading needs two input and two reference columns: {variant_id}"
                )
            heading = raw[:, :2]
            reference = raw[:, 2:]
            heading_norm = np.linalg.norm(heading, axis=1)
            reference_norm = np.linalg.norm(reference, axis=1)
            usable = row_valid & (heading_norm > 1e-12) & (reference_norm > 1e-12)
            heading = heading / np.where(heading_norm > 0, heading_norm, 1.0)[:, None]
            reference = (
                reference / np.where(reference_norm > 0, reference_norm, 1.0)[:, None]
            )
            sin_delta = (
                heading[:, 0] * reference[:, 1] - heading[:, 1] * reference[:, 0]
            )
            cos_delta = np.sum(heading * reference, axis=1)
            values = np.column_stack((sin_delta, cos_delta))
            valid = np.repeat(usable[:, None], 2, axis=1)
        elif transform == "visible_time_difference":
            values = np.full((len(raw), raw.shape[1]), np.nan, dtype=float)
            valid = np.zeros_like(values, dtype=bool)
            segments = _strings(table, "segment_id")
            times = _int64(table, str(parameters["time_column"]))
            point_indexes = _int64(table, "point_index")
            for segment_id in sorted(set(segments.tolist())):
                offsets = np.flatnonzero(segments == segment_id)
                order = offsets[np.lexsort((point_indexes[offsets], times[offsets]))]
                for previous, current in zip(order[:-1], order[1:]):
                    dt_seconds = (times[current] - times[previous]) / 1_000_000_000.0
                    if row_valid[previous] and row_valid[current] and dt_seconds > 0:
                        values[current] = (raw[current] - raw[previous]) / dt_seconds
                        valid[current] = True
        elif transform == "grouped_one_hot":
            groups = [str(value) for value in parameters["groups"]]
            categories = raw[:, 0]
            values = np.zeros((len(raw), len(groups)), dtype=float)
            for index, group in enumerate(groups):
                if group not in _WORLDCOVER_GROUPS:
                    raise PIRC21AdapterError(f"unknown WorldCover group: {group}")
                values[:, index] = np.isin(categories, list(_WORLDCOVER_GROUPS[group]))
            unmatched = ~np.isin(
                categories,
                sorted(set().union(*_WORLDCOVER_GROUPS.values())),
            )
            if "bare_snow_unknown" in groups:
                unknown_index = groups.index("bare_snow_unknown")
                values[:, unknown_index] = np.logical_or(
                    values[:, unknown_index].astype(bool), unmatched
                ).astype(float)
            valid = np.repeat(row_valid[:, None], len(groups), axis=1)
        elif transform == "embedding_identity":
            categories = raw[:, 0]
            tokens = np.zeros(len(raw), dtype=float)
            for index, code in enumerate(_WORLDCOVER_CODES, start=1):
                tokens[categories == code] = index
            values = tokens[:, None]
            valid = row_valid[:, None]
        elif transform == "one_hot":
            categories = [float(value) for value in parameters["categories"]]
            values = np.column_stack(
                [raw[:, 0] == value for value in categories]
            ).astype(float)
            valid = np.repeat(row_valid[:, None], len(categories), axis=1)
        elif transform == "threshold_bins":
            edges = np.asarray(parameters["edges"], dtype=float)
            indexes = np.digitize(raw[:, 0], edges)
            values = np.column_stack(
                [indexes == index for index in range(len(edges) + 1)]
            ).astype(float)
            valid = np.repeat(row_valid[:, None], values.shape[1], axis=1)
        elif transform == "fixed_rbf":
            centers = np.asarray(parameters["centers"], dtype=float)
            width = float(parameters["width"])
            if width <= 0:
                raise PIRC21AdapterError(f"invalid RBF width: {variant_id}")
            values = np.exp(-0.5 * ((raw[:, :1] - centers[None, :]) / width) ** 2)
            valid = np.repeat(row_valid[:, None], values.shape[1], axis=1)
        elif transform == "tangent_normal_projection":
            if raw.shape[1] != 4:
                raise PIRC21AdapterError(
                    f"tangent/normal projection requires two vectors: {variant_id}"
                )
            first = raw[:, :2]
            second = raw[:, 2:]
            values = np.column_stack(
                (
                    np.sum(first * second, axis=1),
                    first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0],
                )
            )
            valid = np.repeat(row_valid[:, None], 2, axis=1)
        else:  # guarded by selection validation
            raise PIRC21AdapterError(f"unsupported transform: {transform}")
        if values.shape[1] != len(output_columns):
            raise PIRC21AdapterError(
                f"transform output does not match declared columns: {variant_id}"
            )
        values = np.where(valid, values, np.nan)
        return _VariantResult(values, valid, output_columns)

    def _composition(
        self,
        composition_id: str,
        variants: Mapping[str, _VariantResult],
    ) -> _VariantResult:
        composition = self._composition_by_id[composition_id]
        inputs = [variants[str(value)] for value in composition["input_variant_ids"]]
        operator = str(composition["operator"])
        columns = tuple(str(item["name"]) for item in composition["output_columns"])
        if operator == "concatenate":
            values = np.concatenate([item.values for item in inputs], axis=1)
            valid = np.concatenate([item.valid for item in inputs], axis=1)
        elif operator == "elementwise_product":
            if len({item.values.shape[1] for item in inputs}) != 1:
                raise PIRC21AdapterError(
                    f"elementwise dimensions disagree: {composition_id}"
                )
            values = np.prod(np.stack([item.values for item in inputs]), axis=0)
            valid = np.logical_and.reduce([item.valid for item in inputs])
        elif operator == "scalar_vector_product":
            if len(inputs) != 2:
                raise PIRC21AdapterError(
                    f"scalar/vector composition needs two inputs: {composition_id}"
                )
            scalar, vector = (
                (inputs[0], inputs[1])
                if inputs[0].values.shape[1] == 1
                else (inputs[1], inputs[0])
            )
            if scalar.values.shape[1] != 1:
                raise PIRC21AdapterError(
                    f"scalar/vector composition has no scalar: {composition_id}"
                )
            values = scalar.values * vector.values
            valid = scalar.valid & vector.valid
        else:
            semantics = tuple(composition.get("parameters", {}).get("semantics", ()))
            if semantics == ("uphill_projection", "contour_projection"):
                if (
                    len(inputs) != 2
                    or inputs[0].values.shape[1] < 3
                    or inputs[1].values.shape[1] != 2
                ):
                    raise PIRC21AdapterError(
                        f"terrain projection inputs are invalid: {composition_id}"
                    )
                downslope = inputs[0].values[:, 1:3]
                history = inputs[1].values
                uphill = -downslope
                contour = np.column_stack((-uphill[:, 1], uphill[:, 0]))
                values = np.column_stack(
                    (
                        np.sum(history * uphill, axis=1),
                        np.sum(history * contour, axis=1),
                    )
                )
                usable = inputs[0].valid[:, 1:3].all(axis=1) & inputs[1].valid.all(
                    axis=1
                )
                valid = np.repeat(usable[:, None], 2, axis=1)
            else:
                if len(inputs) != 2 or inputs[0].values.shape != inputs[1].values.shape:
                    raise PIRC21AdapterError(
                        f"pairwise composition dimensions disagree: {composition_id}"
                    )
                values = inputs[0].values * inputs[1].values
                valid = inputs[0].valid & inputs[1].valid
        if values.shape[1] != len(columns):
            raise PIRC21AdapterError(
                f"composition output does not match declared columns: {composition_id}"
            )
        return _VariantResult(np.where(valid, values, np.nan), valid, columns)

    def transform(
        self,
        split: str,
        *,
        final_eval_unlock: str | None = None,
        file_ids: Sequence[str] | None = None,
        segment_ids: Sequence[str] | None = None,
    ) -> PointFeatureBatch:
        fitted_required = any(
            self._variant_by_id[variant_id]["fit_scope"] == "train_only"
            for variant_id in self.selection.variant_ids
        )
        if self._fit_state is None:
            if fitted_required:
                raise PIRC21AdapterError("fit() is required for this selection")
            self._fit_state = {}
        table = self._load(
            split,
            final_eval_unlock=final_eval_unlock,
            file_ids=file_ids,
            segment_ids=segment_ids,
        )
        variants = {
            variant_id: self._variant(table, variant_id)
            for variant_id in self.selection.variant_ids
        }
        results = list(variants.values()) + [
            self._composition(composition_id, variants)
            for composition_id in self.selection.composition_ids
        ]
        if results:
            values = np.concatenate([result.values for result in results], axis=1)
            valid = np.concatenate([result.valid for result in results], axis=1)
            columns = tuple(column for result in results for column in result.columns)
        else:
            values = np.empty((table.num_rows, 0), dtype=float)
            valid = np.empty((table.num_rows, 0), dtype=bool)
            columns = ()
        entries = self._entries(split, file_ids=file_ids)
        cache_identity = _canonical_hash(
            {
                "adapter_version": ADAPTER_VERSION,
                "dataset_id": self.dataset_id,
                "snapshot_id": self.manifest["snapshot_id"],
                "content_inventory_sha256": self.manifest["content_inventory_sha256"],
                "feature_spec_id": self.feature_spec_id,
                "feature_spec_sha256": self.manifest["feature_spec_sha256"],
                "selection_fingerprint": self.selection_fingerprint,
                "fit_state_fingerprint": self.fit_state_fingerprint,
                "split": split,
                "split_files": [str(item["sha256"]) for item in entries],
                "file_ids": sorted(file_ids) if file_ids is not None else None,
                "segment_ids": sorted(segment_ids) if segment_ids is not None else None,
            }
        )
        batch = PointFeatureBatch(
            split=split,
            point_ids=tuple(_strings(table, "point_id").tolist()),
            segment_ids=tuple(_strings(table, "segment_id").tolist()),
            absolute_epoch_ns=_int64(table, "absolute_epoch_ns"),
            columns=columns,
            values=values,
            valid=valid,
            cache_identity=cache_identity,
            include_validity_indicators=self.selection.include_validity_indicators,
        )
        return batch

    def transform_segments(
        self, split: str, *, final_eval_unlock: str | None = None
    ) -> SegmentFeatureBatch:
        if not self.selection.segment_aggregations:
            raise PIRC21AdapterError(
                "segment_aggregations must be selected for segment output"
            )
        return self.transform(split, final_eval_unlock=final_eval_unlock).aggregate(
            self.selection.segment_aggregations
        )


__all__ = [
    "ADAPTER_VERSION",
    "FeatureSelection",
    "FeatureSnapshotAdapter",
    "PIRC21AdapterError",
    "PointFeatureBatch",
    "SegmentFeatureBatch",
]
