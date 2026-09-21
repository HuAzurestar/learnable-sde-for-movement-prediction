"""Attach PIRC-21 feature variants to the actual NEX326 SDE runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .cohort import Cohort, Segment
from .pirc20_runtime import PIRC20NEX326Runner
from .pirc21_adapter import FeatureSelection, FeatureSnapshotAdapter, PointFeatureBatch
from .pirc21_interactions import (
    INTERACTION_REGISTRY_VERSION,
    evaluate_interactions,
    interaction_output_columns,
    interaction_registry_fingerprint,
)


RUNTIME_VERSION = "pirc21-nex326-runtime-v2"
_ROLE_TO_SNAPSHOT_SPLIT = {
    "train": "train",
    "adapt": "train",
    "validation": "validation",
    "evaluation": "final_eval",
}


class PIRC21RuntimeError(ValueError):
    """A feature snapshot cannot be aligned with the SDE cohort."""


def _pirc20_file_id(segment_id: str) -> str | None:
    parts = segment_id.split(":", 3)
    if len(parts) != 4 or parts[0] != "r1t" or not parts[1]:
        return None
    return parts[1]


class PIRC21Segment(Segment):
    """A Segment whose additional conditions are owned by a PIRC-21 identity."""

    def validate(self) -> None:
        legacy_conditions = {
            name: values
            for name, values in self.conditions.items()
            if not name.startswith("pirc21:")
        }
        legacy = Segment(
            segment_id=self.segment_id,
            source_domain=self.source_domain,
            region=self.region,
            time=self.time,
            state=self.state,
            conditions=legacy_conditions,
            has_terrain=self.has_terrain,
            endpoint_prior_mean=self.endpoint_prior_mean,
            endpoint_prior_covariance=self.endpoint_prior_covariance,
            endpoint_prior_source=self.endpoint_prior_source,
            endpoint_prior_derived_from_truth=self.endpoint_prior_derived_from_truth,
        )
        legacy.validate()
        for name, values in self.conditions.items():
            if name.startswith("pirc21:") and (
                values.shape != self.time.shape or not np.isfinite(values).all()
            ):
                raise PIRC21RuntimeError(
                    f"segment {self.segment_id} PIRC-21 condition is invalid: {name}"
                )


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extend_implementation_identity(
    identity: Mapping[str, object],
) -> dict[str, object]:
    extended = dict(identity)
    files = [dict(item) for item in identity["files"]]  # type: ignore[index]
    existing = {str(item["path"]) for item in files}
    for path in (Path(__file__), Path(__file__).with_name("pirc21_adapter.py")):
        if path.name not in existing:
            files.append({"path": path.name, "sha256": _sha256(path)})
    source_bundle = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    runtime = dict(identity["runtime"])  # type: ignore[arg-type]
    execution_identity = hashlib.sha256(
        json.dumps(
            {"source_bundle_sha256": source_bundle, "runtime": runtime},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    extended.update(
        {
            "source_bundle_sha256": source_bundle,
            "execution_identity_sha256": execution_identity,
            "files": files,
            "pirc21_runtime_version": RUNTIME_VERSION,
        }
    )
    return extended


class PIRC21FeatureRuntime:
    """Fit registered transforms and align their matrices to PIRC-20 segments."""

    def __init__(
        self,
        snapshot_root: str | Path,
        selection: FeatureSelection,
    ) -> None:
        self.adapter = FeatureSnapshotAdapter(snapshot_root, selection).fit()
        output_columns = self.adapter.output_columns
        model_columns = [f"pirc21:{name}" for name in output_columns]
        if selection.include_validity_indicators:
            model_columns.extend(f"pirc21:{name}__valid" for name in output_columns)
        interaction_columns = interaction_output_columns(selection.interaction_ids)
        model_columns.extend(f"pirc21:{name}" for name in interaction_columns)
        if selection.include_validity_indicators:
            model_columns.extend(
                f"pirc21:{name}__valid" for name in interaction_columns
            )
        self.interaction_columns = interaction_columns
        self.condition_names = tuple(model_columns)

    @property
    def identity_record(self) -> dict[str, object]:
        return {
            "runtime_version": RUNTIME_VERSION,
            "interaction_registry_version": INTERACTION_REGISTRY_VERSION,
            "interaction_registry_sha256": interaction_registry_fingerprint(),
            **self.adapter.identity_record,
            "model_condition_names": list(self.condition_names),
            "runtime_identity_sha256": _canonical_hash(
                {
                    "adapter": self.adapter.identity_record,
                    "model_condition_names": self.condition_names,
                }
            ),
        }

    def _batch(self, split: str, segments: tuple[Segment, ...]) -> PointFeatureBatch:
        segment_ids = tuple(segment.segment_id for segment in segments)
        parsed_file_ids = tuple(_pirc20_file_id(value) for value in segment_ids)
        file_ids = (
            tuple(sorted(value for value in parsed_file_ids if value is not None))
            if all(value is not None for value in parsed_file_ids)
            else None
        )
        if split == "final_eval":
            return self.adapter.transform(
                split,
                final_eval_unlock=self.adapter.dataset_id,
                file_ids=file_ids,
                segment_ids=segment_ids,
            )
        return self.adapter.transform(
            split,
            file_ids=file_ids,
            segment_ids=segment_ids,
        )

    def attach(self, cohort: Cohort) -> Cohort:
        if cohort.data_version != self.adapter.dataset_id:
            raise PIRC21RuntimeError(
                "feature snapshot dataset does not match the cohort data version"
            )
        scoped_segments: dict[str, list[Segment]] = {}
        for role, segments in cohort.splits.items():
            if not segments:
                continue
            snapshot_split = _ROLE_TO_SNAPSHOT_SPLIT.get(role)
            if snapshot_split is None:
                raise PIRC21RuntimeError(
                    f"PIRC-21 features are unavailable for cohort role: {role}"
                )
            scoped_segments.setdefault(snapshot_split, []).extend(segments)
        batches = {
            split: self._batch(split, tuple(segments))
            for split, segments in scoped_segments.items()
        }
        attached_splits: dict[str, tuple[Segment, ...]] = {}
        for role, segments in cohort.splits.items():
            if not segments:
                attached_splits[role] = ()
                continue
            snapshot_split = _ROLE_TO_SNAPSHOT_SPLIT.get(role)
            if snapshot_split is None:
                raise PIRC21RuntimeError(
                    f"PIRC-21 features are unavailable for cohort role: {role}"
                )
            batch = batches[snapshot_split]
            matrix = batch.model_matrix()
            attached: list[Segment] = []
            for segment in segments:
                segment_indexes = np.flatnonzero(
                    np.asarray(batch.segment_ids, dtype=object) == segment.segment_id
                )
                if not len(segment_indexes):
                    raise PIRC21RuntimeError(
                        f"segment is absent from feature snapshot: {segment.segment_id}"
                    )
                order = segment_indexes[
                    np.argsort(batch.absolute_epoch_ns[segment_indexes], kind="stable")
                ]
                relative_time = (
                    batch.absolute_epoch_ns[order] - batch.absolute_epoch_ns[order[0]]
                ).astype(float) / 1_000_000_000.0
                selected_indexes: list[int] = []
                search_start = 0
                for target_time in segment.time:
                    matches = np.flatnonzero(
                        np.isclose(
                            relative_time[search_start:],
                            target_time,
                            rtol=1e-9,
                            atol=1e-6,
                        )
                    )
                    if not len(matches):
                        raise PIRC21RuntimeError(
                            f"feature timestamp is absent: {segment.segment_id}:{target_time}"
                        )
                    selected = search_start + int(matches[0])
                    selected_indexes.append(int(order[selected]))
                    search_start = selected + 1
                indexes = np.asarray(selected_indexes, dtype=int)
                if len(indexes) != len(segment.time):
                    raise PIRC21RuntimeError(
                        f"feature row count does not match: {segment.segment_id}"
                    )
                conditions = dict(segment.conditions)
                selected = matrix[indexes]
                base_condition_names = tuple(
                    f"pirc21:{name}" for name in batch.columns
                )
                if self.adapter.selection.include_validity_indicators:
                    base_condition_names += tuple(
                        f"pirc21:{name}__valid" for name in batch.columns
                    )
                for column_index, name in enumerate(base_condition_names):
                    conditions[name] = selected[:, column_index].copy()
                source_values = {
                    name: batch.values[indexes, column_index].copy()
                    for column_index, name in enumerate(batch.columns)
                }
                source_valid = {
                    name: batch.valid[indexes, column_index].copy()
                    for column_index, name in enumerate(batch.columns)
                }
                interactions = evaluate_interactions(
                    segment,
                    source_values,
                    source_valid,
                    self.adapter.selection.interaction_ids,
                )
                if (
                    not self.adapter.selection.include_validity_indicators
                    and not interactions.valid.all()
                ):
                    raise PIRC21RuntimeError(
                        "validity indicators cannot be disabled while interaction "
                        "values are missing"
                    )
                interaction_numeric = np.where(
                    interactions.valid, interactions.values, 0.0
                )
                for column_index, name in enumerate(interactions.columns):
                    conditions[f"pirc21:{name}"] = interaction_numeric[
                        :, column_index
                    ].copy()
                if self.adapter.selection.include_validity_indicators:
                    for column_index, name in enumerate(interactions.columns):
                        conditions[f"pirc21:{name}__valid"] = interactions.valid[
                            :, column_index
                        ].astype(float)
                attached.append(
                    PIRC21Segment(
                        segment_id=segment.segment_id,
                        source_domain=segment.source_domain,
                        region=segment.region,
                        time=segment.time,
                        state=segment.state,
                        conditions=conditions,
                        has_terrain=segment.has_terrain,
                        endpoint_prior_mean=segment.endpoint_prior_mean,
                        endpoint_prior_covariance=segment.endpoint_prior_covariance,
                        endpoint_prior_source=segment.endpoint_prior_source,
                        endpoint_prior_derived_from_truth=segment.endpoint_prior_derived_from_truth,
                    )
                )
            attached_splits[role] = tuple(attached)
        result = Cohort(
            schema_version=cohort.schema_version,
            dataset_id=cohort.dataset_id,
            data_version=cohort.data_version,
            purpose=f"{cohort.purpose}+pirc21_features",
            splits=attached_splits,
            unavailable_reasons=cohort.unavailable_reasons,
            fingerprint=_canonical_hash(
                {
                    "cohort_fingerprint": cohort.fingerprint,
                    "feature_identity": self.identity_record,
                }
            ),
        )
        result.validate()
        return result


class PIRC21NEX326Runner(PIRC20NEX326Runner):
    """Run NEX326 with one configuration-selected PIRC-21 feature matrix."""

    def __init__(self, spec, cohort, output_root, *, feature_runtime, **kwargs) -> None:
        self.pirc21_features = feature_runtime
        attached = feature_runtime.attach(cohort)
        super().__init__(
            spec,
            attached,
            output_root,
            **kwargs,
        )
        self.implementation = _extend_implementation_identity(self.implementation)

    def _config(self, subconfig: Mapping[str, object]) -> dict[str, object]:
        config = super()._config(subconfig)
        existing = tuple(str(name) for name in config.get("condition", ()))
        config["condition"] = list(
            dict.fromkeys((*existing, *self.pirc21_features.condition_names))
        )
        config["pirc21_feature_selection"] = self.pirc21_features.identity_record
        return config

    def run_all(self):
        records = super().run_all()
        manifest_path = self.output_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["pirc21_features"] = self.pirc21_features.identity_record
        temporary = self.output_root / ".manifest-pirc21.tmp"
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(manifest_path)
        return records


__all__ = [
    "PIRC21FeatureRuntime",
    "PIRC21NEX326Runner",
    "PIRC21RuntimeError",
    "PIRC21Segment",
    "RUNTIME_VERSION",
]
