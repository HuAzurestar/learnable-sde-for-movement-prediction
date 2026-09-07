"""Validated values passed between data, models, algorithms, and applications."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, NewType, Optional, Tuple

import torch

from .errors import DataValidationError


def _finite(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise DataValidationError(f"{name} 包含 NaN 或 infinity")


@dataclass(frozen=True)
class ModelContext:
    """Condition and latent regime required for one model call."""

    condition: Optional[torch.Tensor] = None
    regime: Optional[torch.Tensor | int] = None


@dataclass(frozen=True)
class TrajectorySegment:
    """Canonical trajectory segment consumed by loaders and algorithms.

    ``time/state/condition/metadata`` properties provide descriptive domain
    names while ``t/x/cond/meta`` preserve the concise mathematical notation.
    """

    t: torch.Tensor
    x: torch.Tensor
    cond: Optional[torch.Tensor] = None
    dt: float = 60.0
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def segment_id(self) -> str:
        return str(self.meta.get("segment_id", ""))

    @property
    def time(self) -> torch.Tensor:
        return self.t

    @property
    def state(self) -> torch.Tensor:
        return self.x

    @property
    def condition(self) -> Optional[torch.Tensor]:
        return self.cond

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.meta

    @property
    def source(self) -> str:
        return str(self.meta.get("source", "unknown"))

    def validate(self) -> None:
        if not self.segment_id:
            raise DataValidationError("segment_id 不能为空")
        if self.t.ndim != 1:
            raise DataValidationError("time 必须为 (T,)")
        if self.x.ndim != 2:
            raise DataValidationError("state 必须为 (T, state_dim)")
        if len(self.t) != len(self.x):
            raise DataValidationError("time/state 长度不一致")
        if len(self.t) < 2:
            raise DataValidationError("轨迹段至少需要两个时刻")
        _finite(self.t, "time")
        _finite(self.x, "state")
        if not torch.all(torch.diff(self.t) > 0):
            raise DataValidationError("time 必须严格递增")
        if self.dt <= 0:
            raise DataValidationError("dt 必须为正")
        if self.cond is not None:
            if len(self.cond) != len(self.t):
                raise DataValidationError("condition/time 长度不一致")
            _finite(self.cond, "condition")


@dataclass(frozen=True)
class TrajectoryDataset:
    train: Tuple[TrajectorySegment, ...]
    validation: Tuple[TrajectorySegment, ...] = ()
    evaluation: Tuple[TrajectorySegment, ...] = ()

    def validate(self) -> None:
        all_segments = self.train + self.validation + self.evaluation
        for segment in all_segments:
            segment.validate()
        ids = [segment.segment_id for segment in all_segments]
        if len(ids) != len(set(ids)):
            raise DataValidationError("dataset split 间存在重复 segment_id")


@dataclass(frozen=True)
class TransitionBatch:
    x: torch.Tensor
    y: torch.Tensor
    dt: torch.Tensor
    condition: Optional[torch.Tensor] = None
    segment_ids: Tuple[str, ...] = ()

    def validate(self) -> None:
        if self.x.ndim != 2 or self.y.ndim != 2 or self.x.shape != self.y.shape:
            raise DataValidationError("x/y 必须具有相同的 (N, state_dim) 形状")
        if self.dt.ndim != 1 or len(self.dt) != len(self.x):
            raise DataValidationError("dt 必须为 (N,)")
        _finite(self.x, "x")
        _finite(self.y, "y")
        _finite(self.dt, "dt")
        if not torch.all(self.dt > 0):
            raise DataValidationError("dt 必须为正")
        if self.condition is not None and len(self.condition) != len(self.x):
            raise DataValidationError("condition/batch 长度不一致")


@dataclass(frozen=True)
class GaussianTransition:
    mean: torch.Tensor
    covariance: torch.Tensor

    def validate(self) -> None:
        if self.mean.ndim < 1:
            raise DataValidationError("mean 缺少 state 维")
        d = self.mean.shape[-1]
        if self.covariance.shape[-2:] != (d, d):
            raise DataValidationError("covariance 与 mean 维度不匹配")
        _finite(self.mean, "mean")
        _finite(self.covariance, "covariance")
        if not torch.allclose(
            self.covariance,
            self.covariance.transpose(-1, -2),
            rtol=1e-7,
            atol=1e-10,
        ):
            raise DataValidationError("covariance 必须对称")


@dataclass(frozen=True)
class ForecastRequest:
    initial_state: torch.Tensor
    horizons: torch.Tensor
    n_samples: int
    context: ModelContext = field(default_factory=ModelContext)

    def validate(self) -> None:
        if self.initial_state.ndim != 1:
            raise DataValidationError("initial_state 必须为 (state_dim,)")
        if self.horizons.ndim != 1 or self.horizons.numel() == 0:
            raise DataValidationError("horizons 必须为非空一维张量")
        _finite(self.initial_state, "initial_state")
        _finite(self.horizons, "horizons")
        if not torch.all(self.horizons > 0):
            raise DataValidationError("horizons 必须为正")
        if len(self.horizons) > 1 and not torch.all(torch.diff(self.horizons) > 0):
            raise DataValidationError("horizons 必须严格递增")
        if self.n_samples <= 0:
            raise DataValidationError("n_samples 必须为正")


@dataclass(frozen=True)
class Forecast:
    samples: torch.Tensor
    mean: Optional[torch.Tensor] = None
    covariance: Optional[torch.Tensor] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


EvidenceId = NewType("EvidenceId", str)


@dataclass(frozen=True)
class SearchEvidence:
    """Validated evidence context consumed by a conditioning component.

    Coordinates use ``spatial_unit`` and an explicitly declared CRS when one is
    known.  Times are offsets in seconds from the forecast origin.  ``None`` is
    the deliberate missing state for CRS and confidence.
    """

    evidence_id: EvidenceId
    kind: str
    region: Tuple[float, float]
    time_window: Tuple[float, float]
    spatial_unit: str
    crs: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not str(self.evidence_id).strip():
            raise DataValidationError("evidence_id 不能为空")
        if not self.kind.strip():
            raise DataValidationError("evidence kind 不能为空")
        if len(self.region) != 2 or not all(math.isfinite(v) for v in self.region):
            raise DataValidationError("evidence region 必须为两个有限坐标")
        if self.region[0] > self.region[1]:
            raise DataValidationError("evidence region 下界不得大于上界")
        if len(self.time_window) != 2 or not all(
            math.isfinite(v) for v in self.time_window
        ):
            raise DataValidationError("evidence time_window 必须为两个有限秒数")
        if self.time_window[0] > self.time_window[1]:
            raise DataValidationError("evidence time_window 必须按时间递增")
        if not self.spatial_unit.strip():
            raise DataValidationError("evidence spatial_unit 不能为空")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise DataValidationError("evidence confidence 必须在 [0, 1]")


@dataclass(frozen=True)
class ConditionedForecast:
    """A forecast conditioned by one or more immutable evidence facts."""

    forecast: Forecast
    evidence_ids: Tuple[EvidenceId, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObservationSet:
    """Observed states aligned with the horizons of one forecast."""

    values: torch.Tensor
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.values.ndim < 1 or self.values.numel() == 0:
            raise DataValidationError("observation values 必须为非空张量")
        _finite(self.values, "observation values")


@dataclass(frozen=True)
class FitResult:
    converged: bool
    iterations: int
    objective_history: Tuple[float, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def final_objective(self) -> Optional[float]:
        return self.objective_history[-1] if self.objective_history else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "converged": self.converged,
            "iterations": self.iterations,
            "objective_history": list(self.objective_history),
            "final_objective": self.final_objective,
            "diagnostics": dict(self.diagnostics),
        }


# The existing split-aware dataset remains the single training-data fact type.
TrainingData = TrajectoryDataset


@dataclass(frozen=True)
class ArtifactReference:
    """Auditable location and digest for one committed run artifact."""

    path: str
    sha256: str
    size_bytes: int

    def validate(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise DataValidationError("artifact path must not be empty")
        if (
            not isinstance(self.sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None
        ):
            raise DataValidationError("artifact sha256 must be 64 lowercase hex digits")
        if not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool):
            raise DataValidationError("artifact size_bytes must be an integer")
        if self.size_bytes < 0:
            raise DataValidationError("artifact size_bytes must not be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactReference":
        try:
            reference = cls(
                path=payload["path"],
                sha256=payload["sha256"],
                size_bytes=payload["size_bytes"],
            )
        except (KeyError, TypeError) as exc:
            raise DataValidationError("invalid artifact reference") from exc
        reference.validate()
        return reference


@dataclass(frozen=True)
class RunRecord:
    """Minimum audit record committed beside one run's visible artifacts."""

    run_id: str
    issue_id: str
    experiment_id: str
    code_version: str
    data_id: str
    split: str
    config: Mapping[str, Any]
    seed: int
    components: Mapping[str, str]
    reproducibility: str
    missing_reproducibility: Tuple[str, ...]
    artifacts: Mapping[str, ArtifactReference] = field(default_factory=dict)
    status: str = "succeeded"
    failure_stage: Optional[str] = None
    failure_reason: Optional[str] = None

    def validate(self) -> None:
        required_strings = {
            "run_id": self.run_id,
            "issue_id": self.issue_id,
            "experiment_id": self.experiment_id,
            "code_version": self.code_version,
            "data_id": self.data_id,
            "split": self.split,
        }
        for name, value in required_strings.items():
            if not isinstance(value, str) or not value.strip():
                raise DataValidationError(f"RunRecord {name} must not be empty")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise DataValidationError("RunRecord seed must be an integer")
        if not isinstance(self.config, Mapping):
            raise DataValidationError("RunRecord config must be a mapping")
        config = dict(self.config)
        try:
            encoded_config = json.dumps(config, ensure_ascii=False, allow_nan=False)
            decoded_config = json.loads(encoded_config)
        except (TypeError, ValueError) as exc:
            raise DataValidationError("RunRecord config must be JSON serializable") from exc
        if decoded_config != config:
            raise DataValidationError("RunRecord config must preserve JSON round-trip")
        if not isinstance(self.components, Mapping):
            raise DataValidationError("RunRecord components must be a mapping")
        if not self.components:
            raise DataValidationError("RunRecord components must not be empty")
        for name, value in self.components.items():
            if (
                not isinstance(name, str)
                or not name.strip()
                or not isinstance(value, str)
                or not value.strip()
            ):
                raise DataValidationError(
                    "RunRecord component names and values must be non-empty strings"
                )
        if not isinstance(self.status, str):
            raise DataValidationError("RunRecord status must be a string")
        if self.status not in {"succeeded", "failed"}:
            raise DataValidationError("RunRecord status must be succeeded or failed")
        if self.status == "failed":
            if not isinstance(self.failure_stage, str):
                raise DataValidationError("RunRecord failure_stage must be a string")
            if not self.failure_stage.strip():
                raise DataValidationError("failed RunRecord requires failure_stage")
            if not isinstance(self.failure_reason, str):
                raise DataValidationError("RunRecord failure_reason must be a string")
            if not self.failure_reason.strip():
                raise DataValidationError("failed RunRecord requires failure_reason")
        elif self.failure_stage is not None or self.failure_reason is not None:
            raise DataValidationError(
                "successful RunRecord must not contain failure_stage or failure_reason"
            )
        if not isinstance(self.reproducibility, str):
            raise DataValidationError("RunRecord reproducibility must be a string")
        if self.reproducibility not in {"partial", "complete"}:
            raise DataValidationError(
                "RunRecord reproducibility must be partial or complete"
            )
        if not isinstance(self.missing_reproducibility, tuple):
            raise DataValidationError(
                "RunRecord missing_reproducibility must be a tuple"
            )
        if self.reproducibility == "partial" and not self.missing_reproducibility:
            raise DataValidationError(
                "partial reproducibility requires missing_reproducibility fields"
            )
        if self.reproducibility == "complete" and self.missing_reproducibility:
            raise DataValidationError(
                "complete reproducibility cannot list missing_reproducibility fields"
            )
        if any(
            not isinstance(name, str) or not name.strip()
            for name in self.missing_reproducibility
        ):
            raise DataValidationError(
                "missing_reproducibility fields must be non-empty strings"
            )
        if not isinstance(self.artifacts, Mapping):
            raise DataValidationError("RunRecord artifacts must be a mapping")
        for name, reference in self.artifacts.items():
            if not isinstance(name, str) or not name.strip():
                raise DataValidationError("RunRecord artifact name must not be empty")
            if not isinstance(reference, ArtifactReference):
                raise DataValidationError(
                    "RunRecord artifacts must contain ArtifactReference values"
                )
            reference.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "run_id": self.run_id,
            "issue_id": self.issue_id,
            "experiment_id": self.experiment_id,
            "code_version": self.code_version,
            "data_id": self.data_id,
            "split": self.split,
            "config": dict(self.config),
            "seed": self.seed,
            "components": dict(self.components),
            "artifacts": {
                name: reference.to_dict()
                for name, reference in self.artifacts.items()
            },
            "status": self.status,
            "failure_stage": self.failure_stage,
            "failure_reason": self.failure_reason,
            "reproducibility": self.reproducibility,
            "missing_reproducibility": list(self.missing_reproducibility),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunRecord":
        try:
            artifact_payload = payload.get("artifacts", {})
            record = cls(
                run_id=payload["run_id"],
                issue_id=payload["issue_id"],
                experiment_id=payload["experiment_id"],
                code_version=payload["code_version"],
                data_id=payload["data_id"],
                split=payload["split"],
                config=dict(payload["config"]),
                seed=payload["seed"],
                components=dict(payload["components"]),
                artifacts={
                    name: ArtifactReference.from_dict(reference)
                    for name, reference in artifact_payload.items()
                },
                status=payload["status"],
                failure_stage=payload.get("failure_stage"),
                failure_reason=payload.get("failure_reason"),
                reproducibility=payload["reproducibility"],
                missing_reproducibility=tuple(payload["missing_reproducibility"]),
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise DataValidationError("invalid RunRecord payload") from exc
        record.validate()
        return record
