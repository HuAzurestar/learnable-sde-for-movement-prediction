"""EM estimator for the segment-level latent-regime SDE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch

from data.validation import (
    DataValidationError,
    ensure_finite,
    ensure_positive,
    validate_phase_space,
)
from domain import FitResult
from models.base import RegimeParameterUpdate
from models.segment_constant import SegmentConstantSDE

from .base import Estimator, FitContext


@dataclass(frozen=True)
class SegmentEMData:
    """Typed training input: one sampling interval per phase-space segment."""

    segments: tuple[torch.Tensor, ...]
    dts: tuple[float, ...]

    @classmethod
    def uniform(
        cls,
        segments: Sequence[torch.Tensor],
        dt: float,
    ) -> "SegmentEMData":
        return cls(tuple(segments), tuple(float(dt) for _ in segments))

    def validate(self) -> None:
        if not self.segments:
            raise DataValidationError("segments 为空 —— 无数据可拟合，中止")
        if len(self.segments) != len(self.dts):
            raise DataValidationError("segments/dts 长度不一致")
        for index, segment in enumerate(self.segments):
            validate_phase_space(segment, source=f"segments[{index}]")
        dt_tensor = torch.as_tensor(self.dts, dtype=torch.float64)
        ensure_finite(dt_tensor, name="dts")
        ensure_positive(dt_tensor, name="dts")


def _kmeans_warmstart(
    segments: Sequence[torch.Tensor],
    n_modes: int,
    generator: torch.Generator,
) -> list[int]:
    features = []
    for segment in segments:
        velocity = segment[..., 1]
        displacement = torch.diff(segment[..., 0])
        features.append(
            torch.stack(
                [
                    torch.log(velocity.std() + 1e-6),
                    torch.log(displacement.std() + 1e-6),
                ]
            )
        )
    feature_matrix = torch.stack(features)
    indices = torch.randperm(
        feature_matrix.shape[0],
        device=feature_matrix.device,
        generator=generator,
    )[:n_modes]
    centers = feature_matrix[indices].clone()
    assignments = torch.zeros(feature_matrix.shape[0], dtype=torch.long)
    for _ in range(30):
        distances = (
            feature_matrix.unsqueeze(1) - centers.unsqueeze(0)
        ).pow(2).sum(-1)
        assignments = distances.argmin(dim=1)
        for regime in range(n_modes):
            selected = assignments == regime
            if selected.sum() > 0:
                centers[regime] = feature_matrix[selected].mean(dim=0)
    return [int(value) for value in assignments]


class SegmentEM(Estimator[SegmentConstantSDE, SegmentEMData]):
    """E-step segment posterior followed by weighted Gaussian M-step."""

    def __init__(self, max_iter: int = 50, tol: float = 1e-5) -> None:
        self.max_iter = max_iter
        self.tol = tol

    def fit(
        self,
        model: SegmentConstantSDE,
        data: SegmentEMData,
        context: FitContext,
    ) -> FitResult:
        data.validate()
        segments = data.segments
        dts = data.dts
        median_dt = float(torch.median(torch.tensor(dts, dtype=torch.float64)))
        n_modes = model.n_regimes

        labels = _kmeans_warmstart(segments, n_modes, context.generator)
        initial_weights = []
        for label in labels:
            weight = torch.zeros(n_modes, dtype=model.dtype, device=model.device)
            weight[label] = 1.0
            initial_weights.append(weight)
        for regime in range(n_modes):
            F, c, covariance = self._weighted_ls(segments, initial_weights, regime)
            A, b, B = model.discrete_to_continuous(F, c, covariance, median_dt)
            model.apply_em_update(RegimeParameterUpdate(regime, A, b, B))

        previous_nll = float("inf")
        history: list[float] = []
        for iteration in range(self.max_iter):
            posteriors = []
            segment_nll_sum = 0.0
            for segment, dt in zip(segments, dts):
                posterior = model.segment_posterior(segment, dt)
                posteriors.append(posterior)
                segment_nll_sum += (
                    posterior * model.segment_nll(segment, dt)
                ).sum().detach().item()
            history.append(segment_nll_sum)
            for regime in range(n_modes):
                F, c, covariance = self._weighted_ls(segments, posteriors, regime)
                A, b, B = model.discrete_to_continuous(F, c, covariance, median_dt)
                model.apply_em_update(RegimeParameterUpdate(regime, A, b, B))
            model.update_regime_prior(torch.stack(posteriors).mean(dim=0))
            if abs(previous_nll - segment_nll_sum) < self.tol:
                return FitResult(True, iteration + 1, tuple(history))
            previous_nll = segment_nll_sum
        return FitResult(False, self.max_iter, tuple(history))

    @staticmethod
    def _weighted_ls(
        segments: Sequence[torch.Tensor],
        weights: Sequence[torch.Tensor],
        regime: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xs, ys, ws = [], [], []
        for segment, posterior in zip(segments, weights):
            x = segment[:-1]
            y = segment[1:]
            xs.append(x)
            ys.append(y)
            ws.append(posterior[regime].expand_as(x[..., 0]))
        X = torch.cat(xs, dim=0)
        Y = torch.cat(ys, dim=0)
        W = torch.cat(ws, dim=0)
        augmented = torch.cat([X, torch.ones_like(X[..., :1])], dim=-1)
        root_weight = W.sqrt().clamp_min(1e-12).unsqueeze(-1)
        theta = torch.linalg.lstsq(
            augmented * root_weight,
            Y * root_weight,
        ).solution
        F = theta[:2].T
        c = theta[2]
        residual = Y - augmented @ theta
        covariance = (
            (W.unsqueeze(-1) * residual).T @ residual / (W.sum() + 1e-12)
        )
        covariance = 0.5 * (covariance + covariance.T)
        return F, c, covariance


__all__ = ["SegmentEM", "SegmentEMData"]
