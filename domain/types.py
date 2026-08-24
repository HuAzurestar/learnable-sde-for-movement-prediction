"""Validated values passed between data, models, algorithms, and applications."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

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
