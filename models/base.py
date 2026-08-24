"""OOP contracts for SDE models and optional model capabilities."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import torch

from domain import GaussianTransition, ModelContext
from numerics import safe_cholesky


class SDEModel(torch.nn.Module, ABC):
    """Common dynamics contract for every trainable SDE model.

    Condition and latent regime are always passed through ``ModelContext`` so
    subclasses never change call signatures or keep an active mode internally.
    """

    @property
    @abstractmethod
    def state_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def noise_dim(self) -> int:
        ...

    @abstractmethod
    def drift(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        context: ModelContext,
    ) -> torch.Tensor:
        """Return drift with shape ``(..., state_dim)``."""

    @abstractmethod
    def diffusion(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        context: ModelContext,
    ) -> torch.Tensor:
        """Return diffusion with shape ``(..., state_dim, noise_dim)``."""


class ExactTransitionProvider(ABC):
    """Capability implemented only by models with an analytic transition."""

    @abstractmethod
    def exact_transition(
        self,
        x: torch.Tensor,
        dt: torch.Tensor | float,
        context: ModelContext,
    ) -> GaussianTransition:
        ...


class AffineGaussianTransitionProvider(ExactTransitionProvider, ABC):
    """Exact transition represented as ``F x + c`` and covariance ``S``."""

    @abstractmethod
    def affine_transition(
        self,
        dt: torch.Tensor | float,
        context: ModelContext,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...


@dataclass(frozen=True)
class RegimeParameterUpdate:
    """An EM update that the model applies to its own parameters."""

    regime: int
    drift_matrix: torch.Tensor
    constant_drift: torch.Tensor
    diffusion_matrix: torch.Tensor
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class LatentRegimeModel(ABC):
    """Capability required by segment-level latent-regime estimators."""

    @property
    @abstractmethod
    def n_regimes(self) -> int:
        ...

    @abstractmethod
    def segment_log_likelihood(self, segment: torch.Tensor, dt: float) -> torch.Tensor:
        """Return one log likelihood per regime."""

    @abstractmethod
    def regime_log_prior(self) -> torch.Tensor:
        ...

    @abstractmethod
    def apply_em_update(self, update: RegimeParameterUpdate) -> None:
        ...

    @abstractmethod
    def update_regime_prior(self, probabilities: torch.Tensor) -> None:
        ...


class ParameterRole(Enum):
    DRIFT = "drift"
    DIFFUSION = "diffusion"
    CONDITION_ENCODER = "condition_encoder"


class ParameterGroupProvider(ABC):
    @abstractmethod
    def parameter_groups(
        self,
    ) -> Mapping[ParameterRole, Sequence[torch.nn.Parameter]]:
        ...


class ExactGaussianKernelMixin(AffineGaussianTransitionProvider, ABC):
    """Exact kernel for ``dX = (A X + b) dt + B dW``."""

    @staticmethod
    def covariance_from_coefficients(
        A: torch.Tensor,
        B: torch.Tensor,
        dt: torch.Tensor | float,
    ) -> torch.Tensor:
        d = A.shape[0]
        Q = B @ B.T
        M = torch.zeros((2 * d, 2 * d), dtype=A.dtype, device=A.device)
        M[:d, :d] = A
        M[:d, d:] = Q
        M[d:, d:] = -A.T
        E = torch.linalg.matrix_exp(M * dt)
        S = E[:d, d:] @ torch.linalg.matrix_exp(A.T * dt)
        return 0.5 * (S + S.T)

    @abstractmethod
    def constant_drift(self, context: ModelContext) -> torch.Tensor:
        ...

    @abstractmethod
    def drift_matrix(self, context: ModelContext) -> torch.Tensor:
        ...

    @abstractmethod
    def diffusion_matrix(self, context: ModelContext) -> torch.Tensor:
        ...

    def affine_transition(
        self,
        dt: torch.Tensor | float,
        context: ModelContext,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        A = self.drift_matrix(context)
        B = self.diffusion_matrix(context)
        b = self.constant_drift(context)
        d = A.shape[0]
        augmented = torch.zeros((d + 1, d + 1), dtype=A.dtype, device=A.device)
        augmented[:d, :d] = A
        augmented[:d, d] = b
        exponential = torch.linalg.matrix_exp(augmented * dt)
        F = exponential[:d, :d]
        c = exponential[:d, d]
        covariance = self.covariance_from_coefficients(A, B, dt)
        return F, c, covariance

    def exact_transition(
        self,
        x: torch.Tensor,
        dt: torch.Tensor | float,
        context: ModelContext,
    ) -> GaussianTransition:
        F, c, covariance = self.affine_transition(dt, context)
        mean = x @ F.T + c if x.ndim >= 2 else F @ x + c
        return GaussianTransition(mean, covariance)

    def transition_nll(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        dt: torch.Tensor | float,
        context: ModelContext,
    ) -> torch.Tensor:
        transition = self.exact_transition(x, dt, context)
        d = y.shape[-1]
        difference = y - transition.mean
        L = safe_cholesky(transition.covariance)
        whitened = torch.linalg.solve_triangular(
            L,
            difference.unsqueeze(-1),
            upper=False,
        ).squeeze(-1)
        logdet = 2.0 * torch.sum(torch.log(torch.diag(L)))
        nll = 0.5 * (
            d * math.log(2 * math.pi)
            + logdet
            + (whitened * whitened).sum(-1)
        )
        return nll.mean()


# Import-name compatibility; method compatibility is intentionally not kept.
SDE = SDEModel


__all__ = [
    "AffineGaussianTransitionProvider",
    "ExactGaussianKernelMixin",
    "ExactTransitionProvider",
    "LatentRegimeModel",
    "ParameterGroupProvider",
    "ParameterRole",
    "RegimeParameterUpdate",
    "SDE",
    "SDEModel",
]
