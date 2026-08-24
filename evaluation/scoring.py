"""Canonical proper scoring rules used only for evaluation."""

from __future__ import annotations

import math

import torch

from domain import Forecast
from .base import ScoringRule


class EnergyScore(ScoringRule):
    """Canonical ``E||X-y|| - 0.5 E||X-X'||`` sample score."""

    @property
    def name(self) -> str:
        return "energy_score"

    def score(self, forecast: Forecast, observation: torch.Tensor) -> torch.Tensor:
        samples = forecast.samples
        if samples.ndim < 2:
            raise ValueError("forecast samples need sample and state axes")
        term1 = torch.linalg.vector_norm(samples - observation, dim=-1).mean(dim=0)
        pairwise = torch.linalg.vector_norm(
            samples.unsqueeze(1) - samples.unsqueeze(0), dim=-1
        )
        term2 = 0.5 * pairwise.mean(dim=(0, 1))
        return (term1 - term2).mean()


class GaussianCRPS(ScoringRule):
    """Closed-form one-dimensional Gaussian CRPS."""

    @property
    def name(self) -> str:
        return "crps"

    def score(self, forecast: Forecast, observation: torch.Tensor) -> torch.Tensor:
        if forecast.mean is None or forecast.covariance is None:
            raise ValueError("GaussianCRPS requires analytic mean and covariance")
        mean = forecast.mean.reshape(-1)[-1]
        sigma = torch.sqrt(forecast.covariance.reshape(-1)[0])
        y = observation.reshape(-1)[-1]
        z = (y - mean) / sigma
        phi = (2 * math.pi) ** -0.5 * torch.exp(-0.5 * z * z)
        Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        return sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))
