"""Contract tests for the clean target architecture in ``learnable_sde``.

These tests deliberately use tiny dummy implementations.  They pin the OOP
boundaries before legacy scientific algorithms are migrated to them.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from application.registry import ComponentRegistry
from application.runtime import RandomStreams, RunContext
from domain import (
    CapabilityError,
    ConfigurationError,
    DataValidationError,
    FitResult,
    Forecast,
    ForecastRequest,
    GaussianTransition,
    ModelContext,
    TrajectoryDataset,
    TrajectorySegment,
    TransitionBatch,
)
from estimation import Estimator, FitContext
from inference import InferenceContext, InferenceEngine
from models import ExactTransitionProvider, SDEModel


class _ExactOU(SDEModel, ExactTransitionProvider):
    """Minimal exact model used only to exercise the contracts."""

    def __init__(self) -> None:
        super().__init__()
        self.rate = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
        self.sigma = torch.nn.Parameter(torch.tensor(0.3, dtype=torch.float64))

    @property
    def state_dim(self) -> int:
        return 1

    @property
    def noise_dim(self) -> int:
        return 1

    def drift(self, t, x, context):
        return -self.rate * x

    def diffusion(self, t, x, context):
        return self.sigma.expand(x.shape + (1,))

    def exact_transition(self, x, dt, context):
        dt_value = torch.as_tensor(dt, dtype=x.dtype, device=x.device)
        decay = torch.exp(-self.rate * dt_value)
        mean = decay * x
        variance = self.sigma.square() * (1.0 - decay.square()) / (2.0 * self.rate)
        covariance = variance.reshape(1, 1)
        return GaussianTransition(mean=mean, covariance=covariance)


class _EulerOnlyModel(SDEModel):
    @property
    def state_dim(self) -> int:
        return 1

    @property
    def noise_dim(self) -> int:
        return 1

    def drift(self, t, x, context):
        return torch.zeros_like(x)

    def diffusion(self, t, x, context):
        return torch.ones(x.shape + (1,), dtype=x.dtype, device=x.device)


class _ExactEngine(InferenceEngine):
    def supports(self, model: SDEModel) -> bool:
        return isinstance(model, ExactTransitionProvider)

    def forecast(self, model, request, context):
        request.validate()
        if not self.supports(model):
            raise CapabilityError("exact transition capability is required")
        transition = model.exact_transition(
            request.initial_state,
            request.horizons[-1],
            request.context,
        )
        transition.validate()
        samples = transition.mean.expand(request.n_samples, -1).clone()
        return Forecast(
            samples=samples,
            mean=transition.mean,
            covariance=transition.covariance,
        )


class _NoOpEstimator(Estimator[_ExactOU, TransitionBatch]):
    def fit(self, model, data, context):
        data.validate()
        return FitResult(converged=True, iterations=0)


def _segment(segment_id: str = "segment-1") -> TrajectorySegment:
    return TrajectorySegment(
        t=torch.tensor([0.0, 1.0], dtype=torch.float64),
        x=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
        dt=1.0,
        meta={"segment_id": segment_id},
    )


def _transitions() -> TransitionBatch:
    return TransitionBatch(
        x=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
        y=torch.tensor([[1.0], [2.0]], dtype=torch.float64),
        dt=torch.tensor([1.0, 1.0], dtype=torch.float64),
    )


def test_sde_model_has_standard_torch_module_lifecycle():
    model = _ExactOU()

    assert isinstance(model, torch.nn.Module)
    assert list(model.parameters()) == [model.rate, model.sigma]

    restored = _ExactOU()
    restored.load_state_dict(model.state_dict())
    assert torch.equal(restored.rate, model.rate)


def test_model_context_replaces_hidden_mode_state_and_is_immutable():
    context = ModelContext(regime=1)
    model: SDEModel = _ExactOU()
    x = torch.tensor([2.0], dtype=torch.float64)

    assert model.drift(torch.tensor(0.0), x, context).shape == x.shape
    with pytest.raises(FrozenInstanceError):
        context.regime = 2


def test_optional_exact_transition_is_a_separate_capability():
    assert isinstance(_ExactOU(), ExactTransitionProvider)
    assert not isinstance(_EulerOnlyModel(), ExactTransitionProvider)


def test_inference_engine_fails_fast_instead_of_falling_back():
    engine = _ExactEngine()
    request = ForecastRequest(
        initial_state=torch.tensor([0.0], dtype=torch.float64),
        horizons=torch.tensor([1.0], dtype=torch.float64),
        n_samples=4,
    )
    context = InferenceContext(
        generator=torch.Generator().manual_seed(3),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    with pytest.raises(CapabilityError):
        engine.forecast(_EulerOnlyModel(), request, context)


def test_exact_engine_returns_the_shared_forecast_type():
    request = ForecastRequest(
        initial_state=torch.tensor([1.0], dtype=torch.float64),
        horizons=torch.tensor([1.0], dtype=torch.float64),
        n_samples=3,
    )
    context = InferenceContext(
        generator=torch.Generator().manual_seed(4),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    result = _ExactEngine().forecast(_ExactOU(), request, context)

    assert result.samples.shape == (3, 1)
    assert result.mean is not None
    assert result.covariance is not None


def test_estimator_receives_model_data_and_runtime_explicitly():
    model = _ExactOU()
    context = FitContext(
        generator=torch.Generator().manual_seed(5),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    result = _NoOpEstimator().fit(model, _transitions(), context)

    assert result.converged is True
    assert result.iterations == 0


def test_trajectory_segment_rejects_non_monotonic_time():
    segment = TrajectorySegment(
        t=torch.tensor([0.0, 0.0]),
        x=torch.zeros((2, 1)),
        dt=1.0,
        meta={"segment_id": "bad-time"},
    )

    with pytest.raises(DataValidationError):
        segment.validate()


def test_dataset_rejects_split_leakage_by_segment_id():
    dataset = TrajectoryDataset(
        train=(_segment("same"),),
        evaluation=(_segment("same"),),
    )

    with pytest.raises(DataValidationError):
        dataset.validate()


def test_transition_batch_rejects_non_positive_dt():
    batch = TransitionBatch(
        x=torch.zeros((1, 1)),
        y=torch.ones((1, 1)),
        dt=torch.tensor([0.0]),
    )

    with pytest.raises(DataValidationError):
        batch.validate()


def test_gaussian_transition_rejects_asymmetric_covariance():
    transition = GaussianTransition(
        mean=torch.zeros(2),
        covariance=torch.tensor([[1.0, 1.0], [0.0, 1.0]]),
    )

    with pytest.raises(DataValidationError):
        transition.validate()


def test_registry_rejects_duplicate_and_unknown_components():
    registry: ComponentRegistry[int, str] = ComponentRegistry()
    registry.register("model", lambda value: f"model-{value}")

    assert registry.create("model", 2) == "model-2"
    with pytest.raises(ConfigurationError):
        registry.register("model", str)
    with pytest.raises(ConfigurationError):
        registry.create("baseline", 2)


def test_random_stream_creation_does_not_pollute_global_rng():
    torch.manual_seed(123)
    expected = torch.rand(4)
    torch.manual_seed(123)

    streams = RandomStreams.from_seed(999)
    actual = torch.rand(4)

    assert torch.equal(actual, expected)
    assert not torch.equal(
        torch.rand(4, generator=streams.training),
        torch.rand(4, generator=streams.inference),
    )


def test_run_context_owns_runtime_policy_and_independent_streams():
    context = RunContext.create(seed=7, dtype=torch.float32)

    assert context.device == torch.device("cpu")
    assert context.dtype == torch.float32
    assert context.random.training is not context.random.inference
