"""Inference strategy contracts and reusable exact/Euler engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from domain import CapabilityError, Forecast, ForecastRequest, ModelContext
from models.base import ExactTransitionProvider, SDEModel
from numerics import safe_cholesky


@dataclass(frozen=True)
class InferenceContext:
    generator: torch.Generator
    device: torch.device
    dtype: torch.dtype


class InferenceEngine(ABC):
    @abstractmethod
    def supports(self, model: SDEModel) -> bool:
        ...

    @abstractmethod
    def forecast(
        self,
        model: SDEModel,
        request: ForecastRequest,
        context: InferenceContext,
    ) -> Forecast:
        ...


class ExactGaussianEngine(InferenceEngine):
    """Sample paths from an analytic transition provider."""

    def supports(self, model: SDEModel) -> bool:
        return isinstance(model, ExactTransitionProvider)

    def rollout(
        self,
        model: SDEModel,
        initial_state: torch.Tensor,
        dts: torch.Tensor,
        n_samples: int,
        model_context: ModelContext,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if not self.supports(model):
            raise CapabilityError(
                f"{type(model).__name__} 不提供 ExactTransitionProvider"
            )
        state = initial_state.detach().clone().unsqueeze(0).expand(n_samples, -1).clone()
        paths = [state.clone()]
        for dt in dts:
            transition = model.exact_transition(state, dt, model_context)
            transition.validate()
            L = safe_cholesky(transition.covariance)
            noise = torch.randn(
                (n_samples, model.state_dim),
                dtype=state.dtype,
                device=state.device,
                generator=generator,
            )
            state = transition.mean + noise @ L.T
            paths.append(state.clone())
        return torch.stack(paths, dim=1)

    def forecast(self, model, request, context):
        request.validate()
        horizons = request.horizons.to(device=context.device, dtype=context.dtype)
        dts = torch.diff(
            torch.cat(
                [torch.zeros(1, device=context.device, dtype=context.dtype), horizons]
            )
        )
        paths = self.rollout(
            model,
            request.initial_state.to(device=context.device, dtype=context.dtype),
            dts,
            request.n_samples,
            request.context,
            context.generator,
        )
        samples = paths[:, 1:, :]
        final = samples[:, -1, :]
        covariance = (
            torch.cov(final.T)
            if request.n_samples > 1
            else torch.zeros(
                (model.state_dim, model.state_dim),
                dtype=final.dtype,
                device=final.device,
            )
        )
        return Forecast(
            samples=samples,
            mean=samples.mean(dim=0),
            covariance=covariance,
            metadata={"engine": "exact_gaussian"},
        )


class EulerMaruyamaEngine(InferenceEngine):
    """Generic numerical fallback selected explicitly, never implicitly."""

    def __init__(self, max_step: float = 1.0) -> None:
        if max_step <= 0:
            raise ValueError("max_step must be positive")
        self.max_step = max_step

    def supports(self, model: SDEModel) -> bool:
        return isinstance(model, SDEModel)

    def forecast(self, model, request, context):
        request.validate()
        state = request.initial_state.to(context.device, context.dtype)
        state = state.unsqueeze(0).expand(request.n_samples, -1).clone()
        previous = 0.0
        samples = []
        for horizon_tensor in request.horizons:
            horizon = float(horizon_tensor)
            interval = horizon - previous
            n_steps = max(1, int(torch.ceil(torch.tensor(interval / self.max_step))))
            step = interval / n_steps
            for _ in range(n_steps):
                t = torch.tensor(previous, device=context.device, dtype=context.dtype)
                drift = model.drift(t, state, request.context)
                diffusion = model.diffusion(t, state, request.context)
                noise = torch.randn(
                    (request.n_samples, model.noise_dim),
                    device=context.device,
                    dtype=context.dtype,
                    generator=context.generator,
                )
                state = state + drift * step + torch.einsum(
                    "...dn,...n->...d", diffusion, noise
                ) * step ** 0.5
                previous += step
            samples.append(state.clone())
        stacked = torch.stack(samples, dim=1)
        return Forecast(
            samples=stacked,
            mean=stacked.mean(dim=0),
            metadata={"engine": "euler_maruyama", "max_step": self.max_step},
        )


class SplitStepEngine(InferenceEngine):
    """J-1 structure-preserving engine for SegmentConstantSDE."""

    def __init__(self, max_step: float = 1.0) -> None:
        self.max_step = max_step

    def supports(self, model: SDEModel) -> bool:
        from models.segment_constant import SegmentConstantSDE

        return isinstance(model, SegmentConstantSDE)

    def forecast(self, model, request, context):
        from inference.integrator import SplitIntegrator

        request.validate()
        if not self.supports(model):
            raise CapabilityError("SplitStepEngine requires SegmentConstantSDE")
        regime = model.regime_index(request.context)
        integrator = SplitIntegrator(
            float(model.Gamma[regime]),
            float(model.a[regime] - model.kappa),
            float(model.c[regime]),
            float(model.g[regime]),
            dtype=context.dtype,
            device=str(context.device),
        )
        state = request.initial_state.to(context.device, context.dtype)
        state = state.unsqueeze(0).expand(request.n_samples, -1).clone()
        previous = 0.0
        outputs = []
        for horizon_tensor in request.horizons:
            interval = float(horizon_tensor) - previous
            n_steps = max(1, int(torch.ceil(torch.tensor(interval / self.max_step))))
            step = interval / n_steps
            for _ in range(n_steps):
                noise = torch.randn(
                    request.n_samples,
                    device=context.device,
                    dtype=context.dtype,
                    generator=context.generator,
                )
                state = integrator.step(state, step, noise)
            previous = float(horizon_tensor)
            outputs.append(state.clone())
        samples = torch.stack(outputs, dim=1)
        return Forecast(
            samples=samples,
            mean=samples.mean(dim=0),
            metadata={"engine": "split_step", "max_step": self.max_step},
        )


class CommonRandomNumberEngine(ExactGaussianEngine):
    """J-3 shared-path multi-horizon sampling for exact linear models."""

    def forecast(self, model, request, context):
        result = super().forecast(model, request, context)
        return Forecast(
            samples=result.samples,
            mean=result.mean,
            covariance=result.covariance,
            metadata={"engine": "common_random_numbers", "shared_paths": True},
        )


__all__ = [
    "EulerMaruyamaEngine",
    "CommonRandomNumberEngine",
    "ExactGaussianEngine",
    "InferenceContext",
    "InferenceEngine",
    "SplitStepEngine",
]
