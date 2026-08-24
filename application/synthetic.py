"""Deterministic synthetic data fixture used by the public smoke workflow."""

from __future__ import annotations

import torch

from domain import ModelContext
from estimation.em import SegmentEMData
from inference.base import ExactGaussianEngine
from models.segment_constant import SegmentConstantSDE


def make_synthetic_em_data(
    n_segments: int = 12,
    length: int = 120,
    dt: float = 60.0,
    seed: int = 20260814,
) -> tuple[SegmentEMData, tuple[int, ...]]:
    parameter_sets = (
        {"gamma": 0.05, "linear_drift": -0.02, "diffusion": 0.12},
        {"gamma": 0.01, "linear_drift": 0.0, "diffusion": 0.35},
    )
    segments, true_regimes = [], []
    dts = torch.full((length - 1,), dt, dtype=torch.float64)
    engine = ExactGaussianEngine()
    for index in range(n_segments):
        regime = index % len(parameter_sets)
        model = SegmentConstantSDE(n_modes=1, dt_ref=dt)
        model.set_regime_parameters(0, **parameter_sets[regime], constant_drift=0.0)
        segment = engine.rollout(
            model,
            torch.zeros(2, dtype=torch.float64),
            dts,
            1,
            ModelContext(regime=0),
            torch.Generator().manual_seed(seed + index),
        ).squeeze(0)
        segments.append(segment)
        true_regimes.append(regime)
    return SegmentEMData.uniform(segments, dt), tuple(true_regimes)
