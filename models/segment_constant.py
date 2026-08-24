"""Segment-level latent-regime underdamped Langevin SDE."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

import torch

from domain import ModelContext
from .base import (
    ExactGaussianKernelMixin,
    LatentRegimeModel,
    ParameterGroupProvider,
    ParameterRole,
    RegimeParameterUpdate,
    SDEModel,
)


def _structure_project(A: torch.Tensor) -> torch.Tensor:
    """Project a 2x2 drift matrix to ``[[0, 1], [a-kappa, -Gamma]]``."""
    projected = A.clone()
    projected[0, 0] = 0.0
    projected[0, 1] = 1.0
    projected[1, 1] = torch.clamp(projected[1, 1], max=-1e-6)
    return projected


class SegmentConstantSDE(
    ExactGaussianKernelMixin,
    SDEModel,
    LatentRegimeModel,
    ParameterGroupProvider,
):
    """Exact affine SDE whose latent regime remains fixed within a segment.

    State is ``[position, velocity]``.  The regime is supplied explicitly as
    ``ModelContext.regime`` for every dynamics and transition call.
    """

    def __init__(
        self,
        n_modes: int = 3,
        kappa: float = 0.0,
        dt_ref: float = 60.0,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if n_modes <= 0:
            raise ValueError("n_modes must be positive")
        self.n_modes = n_modes
        self.kappa = float(kappa)
        self.dt_ref = float(dt_ref)
        target = torch.device(device)
        self.Gamma = torch.nn.Parameter(torch.full((n_modes,), 0.01, dtype=dtype, device=target))
        self.a = torch.nn.Parameter(torch.zeros(n_modes, dtype=dtype, device=target))
        self.c = torch.nn.Parameter(torch.zeros(n_modes, dtype=dtype, device=target))
        self.g = torch.nn.Parameter(torch.full((n_modes,), 0.01, dtype=dtype, device=target))
        self.prior_logits = torch.nn.Parameter(torch.zeros(n_modes, dtype=dtype, device=target))

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def noise_dim(self) -> int:
        return 1

    @property
    def n_regimes(self) -> int:
        return self.n_modes

    @property
    def dtype(self) -> torch.dtype:
        return self.Gamma.dtype

    @property
    def device(self) -> torch.device:
        return self.Gamma.device

    def regime_index(self, context: ModelContext) -> int:
        if context.regime is None:
            raise ValueError("SegmentConstantSDE requires ModelContext.regime")
        if isinstance(context.regime, torch.Tensor):
            if context.regime.numel() != 1:
                raise ValueError("scalar regime required for an affine kernel")
            regime = int(context.regime.item())
        else:
            regime = int(context.regime)
        if not 0 <= regime < self.n_modes:
            raise ValueError(f"regime {regime} outside [0, {self.n_modes})")
        return regime

    def constant_drift(self, context: ModelContext) -> torch.Tensor:
        regime = self.regime_index(context)
        b = torch.zeros(2, dtype=self.dtype, device=self.device)
        b[1] = self.c[regime]
        return b

    def drift_matrix(self, context: ModelContext) -> torch.Tensor:
        regime = self.regime_index(context)
        A = torch.zeros((2, 2), dtype=self.dtype, device=self.device)
        A[0, 1] = 1.0
        A[1, 0] = self.a[regime] - self.kappa
        A[1, 1] = -self.Gamma[regime]
        return A

    def diffusion_matrix(self, context: ModelContext) -> torch.Tensor:
        regime = self.regime_index(context)
        B = torch.zeros((2, 1), dtype=self.dtype, device=self.device)
        B[1, 0] = self.g[regime]
        return B

    def drift(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        context: ModelContext,
    ) -> torch.Tensor:
        regime = self.regime_index(context)
        velocity = x[..., 1]
        acceleration = (
            -self.Gamma[regime] * velocity
            + (self.a[regime] - self.kappa) * x[..., 0]
            + self.c[regime]
        )
        return torch.stack([velocity, acceleration], dim=-1)

    def diffusion(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        context: ModelContext,
    ) -> torch.Tensor:
        regime = self.regime_index(context)
        result = torch.zeros(x.shape + (1,), dtype=x.dtype, device=x.device)
        result[..., 1, 0] = self.g[regime]
        return result

    def segment_nll(self, z: torch.Tensor, dt: float) -> torch.Tensor:
        total = torch.zeros(self.n_modes, dtype=self.dtype, device=self.device)
        for regime in range(self.n_modes):
            total[regime] = self.transition_nll(
                z[:-1],
                z[1:],
                dt,
                ModelContext(regime=regime),
            )
        return total

    def segment_log_likelihood(self, segment: torch.Tensor, dt: float) -> torch.Tensor:
        return -self.segment_nll(segment, dt)

    def regime_log_prior(self) -> torch.Tensor:
        return torch.log_softmax(self.prior_logits, dim=0)

    def segment_posterior(self, z: torch.Tensor, dt: float) -> torch.Tensor:
        return torch.softmax(self.regime_log_prior() + self.segment_log_likelihood(z, dt), dim=0)

    def update_regime_prior(self, probabilities: torch.Tensor) -> None:
        if probabilities.shape != (self.n_modes,):
            raise ValueError("regime prior shape mismatch")
        with torch.no_grad():
            self.prior_logits.copy_(torch.log(probabilities.to(self.prior_logits) + 1e-12))

    def apply_em_update(self, update: RegimeParameterUpdate) -> None:
        regime = update.regime
        A, b, B = update.drift_matrix, update.constant_drift, update.diffusion_matrix
        with torch.no_grad():
            self.Gamma[regime].copy_(-A[1, 1])
            self.a[regime].copy_(A[1, 0] + self.kappa)
            self.c[regime].copy_(b[1])
            self.g[regime].copy_(B[1, 0])

    def set_regime_parameters(
        self,
        regime: int,
        *,
        gamma: Optional[float] = None,
        linear_drift: Optional[float] = None,
        constant_drift: Optional[float] = None,
        diffusion: Optional[float] = None,
    ) -> None:
        """Public initialization/checkpoint hook; never exposes ``.data`` writes."""
        if not 0 <= regime < self.n_modes:
            raise ValueError("regime outside model range")
        values = (
            (self.Gamma, gamma),
            (self.a, linear_drift),
            (self.c, constant_drift),
            (self.g, diffusion),
        )
        with torch.no_grad():
            for parameter, value in values:
                if value is not None:
                    parameter[regime].copy_(torch.as_tensor(value, dtype=self.dtype, device=self.device))

    def discrete_to_continuous(
        self,
        F: torch.Tensor,
        cvec: torch.Tensor,
        covariance: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        A = _structure_project(_matrix_logm(F) / dt)
        identity = torch.eye(2, dtype=self.dtype, device=self.device)
        if torch.det(F - identity).abs() > 1e-12:
            b = torch.linalg.solve(F - identity, A @ cvec)
        else:
            b = A @ cvec
        unit_B = torch.zeros((2, 1), dtype=self.dtype, device=self.device)
        unit_B[1, 0] = 1.0
        unit_covariance = self.covariance_from_coefficients(A, unit_B, dt)
        g2 = covariance[1, 1] / (unit_covariance[1, 1] + 1e-12)
        B = torch.zeros((2, 1), dtype=self.dtype, device=self.device)
        B[1, 0] = torch.sqrt(torch.clamp(g2, min=1e-12))
        return A, b, B

    def parameter_groups(
        self,
    ) -> Mapping[ParameterRole, Sequence[torch.nn.Parameter]]:
        return {
            ParameterRole.DRIFT: (self.Gamma, self.a, self.c),
            ParameterRole.DIFFUSION: (self.g,),
        }


def _matrix_logm(F: torch.Tensor) -> torch.Tensor:
    """2x2 matrix logarithm retained for numerical-equivalence migration."""
    eigenvalues, eigenvectors = torch.linalg.eig(F)
    result = (
        eigenvectors
        @ torch.diag(torch.log(eigenvalues))
        @ torch.linalg.inv(eigenvectors)
    )
    return result.real


__all__ = ["SegmentConstantSDE"]
