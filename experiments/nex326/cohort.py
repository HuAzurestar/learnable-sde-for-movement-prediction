"""Versioned cohort loader used by every NEX326 arm."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


REQUIRED_SPLITS = ("train", "validation", "adapt", "evaluation", "animal_pretrain")
CONDITION_NAMES = (
    "solar_elev",
    "is_day",
    "temperature",
    "precipitation",
    "terrain_elevation",
    "terrain_slope",
)


class CohortError(ValueError):
    """A cohort cannot satisfy the registered experiment contract."""


@dataclass(frozen=True)
class Segment:
    segment_id: str
    source_domain: str
    region: str
    time: np.ndarray
    state: np.ndarray
    conditions: Mapping[str, np.ndarray]
    has_terrain: bool
    endpoint_prior_mean: np.ndarray | None = None
    endpoint_prior_covariance: np.ndarray | None = None
    endpoint_prior_source: str | None = None
    endpoint_prior_derived_from_truth: bool = False

    def validate(self) -> None:
        if not self.segment_id or self.time.ndim != 1:
            raise CohortError("segment identity/time is invalid")
        if self.state.shape != (len(self.time), 2) or len(self.time) < 4:
            raise CohortError(f"segment {self.segment_id} must contain >=4 two-dimensional points")
        if not np.isfinite(self.state).all() or not np.isfinite(self.time).all():
            raise CohortError(f"segment {self.segment_id} contains non-finite values")
        if not np.all(np.diff(self.time) > 0):
            raise CohortError(f"segment {self.segment_id} time is not strictly increasing")
        unknown_conditions = set(self.conditions) - set(CONDITION_NAMES)
        if unknown_conditions:
            raise CohortError(
                f"segment {self.segment_id} has unknown conditions: {sorted(unknown_conditions)}"
            )
        for name, values in self.conditions.items():
            if values.shape != self.time.shape or not np.isfinite(values).all():
                raise CohortError(f"segment {self.segment_id} condition {name} is invalid")
        if self.endpoint_prior_derived_from_truth:
            raise CohortError(f"segment {self.segment_id} endpoint prior is derived from evaluation truth")


@dataclass(frozen=True)
class Cohort:
    schema_version: str
    dataset_id: str
    data_version: str
    purpose: str
    splits: Mapping[str, tuple[Segment, ...]]
    unavailable_reasons: Mapping[str, str]
    fingerprint: str

    def validate(self) -> None:
        if self.schema_version != "nex326-cohort-v1":
            raise CohortError("unsupported cohort schema")
        for split in REQUIRED_SPLITS:
            if split not in self.splits:
                raise CohortError(f"required split is absent: {split}")
            if not self.splits[split] and split not in self.unavailable_reasons:
                raise CohortError(
                    f"empty split {split} requires an explicit unavailable reason"
                )
        seen: dict[str, str] = {}
        for split, segments in self.splits.items():
            for segment in segments:
                segment.validate()
                if segment.segment_id in seen:
                    raise CohortError(
                        f"segment {segment.segment_id} overlaps {seen[segment.segment_id]} and {split}"
                    )
                seen[segment.segment_id] = split
        if any(segment.source_domain != "animal" for segment in self.splits["animal_pretrain"]):
            raise CohortError("animal_pretrain split must be explicitly animal sourced")
        if any(segment.source_domain == "animal" for segment in self.splits["evaluation"]):
            raise CohortError("human evaluation may not contain animal pretraining segments")


def _canonical_fingerprint(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _generated_segment(
    split: str,
    index: int,
    points: int,
    dt: float,
    prior: Mapping[str, object],
) -> Segment:
    split_offset = REQUIRED_SPLITS.index(split) * 100
    global_index = split_offset + index
    time = np.arange(points, dtype=float) * dt
    mode = index % 3
    direction = (-1.0, 0.35, 1.0)[mode]
    region = ("north", "central", "south")[(index // 3) % 3]
    phase = 0.35 * global_index
    solar = 35.0 * np.sin((time / 3600.0) + phase)
    is_day = (solar >= 0).astype(float)
    temperature = 18.0 + 0.12 * solar + (index % 4)
    precipitation = np.maximum(0.0, np.sin(phase + time / 240.0))
    terrain_elevation = 80.0 + 12.0 * (index % 4) + 0.01 * time
    terrain_slope = 0.03 + 0.01 * ((index + np.arange(points)) % 4)
    source_domain = "animal" if split == "animal_pretrain" else "human"
    species_scale = 1.45 if source_domain == "animal" else 1.0
    target_scale = 0.78 if split in {"adapt", "evaluation"} else 1.0
    velocity_x = species_scale * target_scale * (0.018 + 0.006 * direction)
    velocity_y = species_scale * target_scale * (0.012 - 0.004 * direction)
    x = np.zeros(points, dtype=float)
    y = np.zeros(points, dtype=float)
    x[0] = global_index * 0.2
    y[0] = -global_index * 0.1
    for step in range(1, points):
        condition_dx = 0.00002 * solar[step - 1] + 0.00004 * terrain_elevation[step - 1]
        condition_dy = -0.0007 * precipitation[step - 1] + 0.00003 * temperature[step - 1]
        coupling_x = 0.00002 * y[step - 1]
        coupling_y = -0.00001 * x[step - 1]
        x[step] = x[step - 1] + dt * (velocity_x + condition_dx + coupling_x)
        y[step] = y[step - 1] + dt * (velocity_y + condition_dy + coupling_y)
    state = np.column_stack([x, y])
    endpoint_mean = endpoint_cov = None
    endpoint_source = None
    derived = False
    if split == "evaluation":
        # The generator emulates an independent planning feed.  The fixed bias is
        # registered here; the runner never reads the evaluation target to build it.
        endpoint_mean = state[2] + (points - 3) * dt * np.array([velocity_x, velocity_y])
        endpoint_mean = endpoint_mean + np.array([0.6 * direction, -0.4])
        sd = float(prior.get("standard_deviation", 1.25))
        endpoint_cov = np.eye(2) * sd * sd
        endpoint_source = str(prior.get("source", "unknown"))
        derived = bool(prior.get("derived_from_evaluation_truth", False))
    return Segment(
        segment_id=f"{split}-{index:03d}",
        source_domain=source_domain,
        region=region,
        time=time,
        state=state,
        conditions={
            "solar_elev": solar,
            "is_day": is_day,
            "temperature": temperature,
            "precipitation": precipitation,
            "terrain_elevation": terrain_elevation,
            "terrain_slope": terrain_slope,
        },
        has_terrain=(index % 4 != 3),
        endpoint_prior_mean=endpoint_mean,
        endpoint_prior_covariance=endpoint_cov,
        endpoint_prior_source=endpoint_source,
        endpoint_prior_derived_from_truth=derived,
    )


def _parse_segment(payload: Mapping[str, object]) -> Segment:
    prior = payload.get("endpoint_prior") or {}
    conditions = payload.get("conditions", {})
    return Segment(
        segment_id=str(payload["segment_id"]),
        source_domain=str(payload["source_domain"]),
        region=str(payload["region"]),
        time=np.asarray(payload["time"], dtype=float),
        state=np.asarray(payload["state"], dtype=float),
        conditions={
            name: np.asarray(conditions[name], dtype=float)
            for name in CONDITION_NAMES
            if name in conditions
        },
        has_terrain=bool(payload.get("has_terrain", False)),
        endpoint_prior_mean=(np.asarray(prior["mean"], dtype=float) if "mean" in prior else None),
        endpoint_prior_covariance=(np.asarray(prior["covariance"], dtype=float) if "covariance" in prior else None),
        endpoint_prior_source=(str(prior["source"]) if "source" in prior else None),
        endpoint_prior_derived_from_truth=bool(prior.get("derived_from_evaluation_truth", False)),
    )


def load_cohort(path: Path | str) -> Cohort:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    split_status = payload.get("split_status", {})
    unavailable_reasons = {
        split: str(status.get("reason", "unspecified external data unavailability"))
        for split, status in split_status.items()
        if status.get("status") == "unavailable"
    }
    unavailable_reasons.update(
        {
            f"condition:{name}": str(
                status.get("reason", "unspecified condition data unavailability")
            )
            for name, status in payload.get("feature_status", {}).items()
            if status.get("status") == "unavailable"
        }
    )
    endpoint_prior_status = payload.get("endpoint_prior_status", {})
    if endpoint_prior_status.get("status") == "unavailable":
        unavailable_reasons["endpoint_prior"] = str(
            endpoint_prior_status.get(
                "reason", "unspecified endpoint-prior data unavailability"
            )
        )
    if "generator" in payload:
        generator = payload["generator"]
        points = int(generator["points_per_segment"])
        dt = float(generator["dt_seconds"])
        prior = generator.get("endpoint_prior", {})
        splits = {
            split: tuple(
                _generated_segment(split, index, points, dt, prior)
                for index in range(int(generator["split_counts"][split]))
            )
            for split in REQUIRED_SPLITS
        }
    else:
        splits = {
            split: tuple(_parse_segment(item) for item in payload.get("splits", {}).get(split, []))
            for split in REQUIRED_SPLITS
        }
    cohort = Cohort(
        schema_version=str(payload["schema_version"]),
        dataset_id=str(payload["dataset_id"]),
        data_version=str(payload["data_version"]),
        purpose=str(payload.get("purpose", "scientific")),
        splits=splits,
        unavailable_reasons=unavailable_reasons,
        fingerprint=_canonical_fingerprint(payload),
    )
    cohort.validate()
    return cohort


def resample_segment(segment: Segment, dt_seconds: float) -> Segment:
    if dt_seconds <= 0:
        raise CohortError("dt_seconds must be positive")
    grid = np.arange(segment.time[0], segment.time[-1] + 1e-9, dt_seconds)
    if grid[-1] < segment.time[-1]:
        grid = np.append(grid, segment.time[-1])
    state = np.column_stack(
        [np.interp(grid, segment.time, segment.state[:, coordinate]) for coordinate in range(2)]
    )
    conditions = {
        name: np.interp(grid, segment.time, values)
        for name, values in segment.conditions.items()
    }
    return Segment(
        segment_id=segment.segment_id,
        source_domain=segment.source_domain,
        region=segment.region,
        time=grid,
        state=state,
        conditions=conditions,
        has_terrain=segment.has_terrain,
        endpoint_prior_mean=segment.endpoint_prior_mean,
        endpoint_prior_covariance=segment.endpoint_prior_covariance,
        endpoint_prior_source=segment.endpoint_prior_source,
        endpoint_prior_derived_from_truth=segment.endpoint_prior_derived_from_truth,
    )


def select_segments(
    segments: Sequence[Segment],
    *,
    dt_seconds: float,
    coverage_mask: str | None = None,
) -> tuple[Segment, ...]:
    selected = tuple(segment for segment in segments if coverage_mask != "has_terrain" or segment.has_terrain)
    if not selected:
        raise CohortError(f"coverage mask {coverage_mask!r} selected no segments")
    return tuple(resample_segment(segment, dt_seconds) for segment in selected)


__all__ = [
    "Cohort",
    "CohortError",
    "CONDITION_NAMES",
    "Segment",
    "load_cohort",
    "resample_segment",
    "select_segments",
]
