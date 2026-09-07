"""Pure use-case pipelines for prediction, conditioning, and evaluation."""

from __future__ import annotations

from typing import Protocol

from domain import (
    CapabilityError,
    ConditionedForecast,
    DataValidationError,
    Forecast,
    ForecastRequest,
    ObservationSet,
    SearchEvidence,
)
from evaluation import EvaluationReport, Evaluator
from inference import InferenceContext, InferenceEngine
from models import SDEModel

from .runtime import RunContext


class EvidenceConditioner(Protocol):
    """Optional evidence component selected explicitly by the application."""

    def supports(self, evidence: SearchEvidence) -> bool:
        ...

    def condition(self, forecast: Forecast, evidence: SearchEvidence) -> Forecast:
        ...


class EvaluationPipeline:
    """Compute FP, CF, and ER values without persistence or CLI concerns."""

    def __init__(
        self,
        inference_engine: InferenceEngine,
        evaluator: Evaluator,
        conditioner: EvidenceConditioner | None = None,
    ) -> None:
        self.inference_engine = inference_engine
        self.evaluator = evaluator
        self.conditioner = conditioner

    def predict(
        self,
        model: SDEModel,
        request: ForecastRequest,
        runtime: RunContext,
    ) -> Forecast:
        if not self.inference_engine.supports(model):
            raise CapabilityError(
                f"{type(self.inference_engine).__name__} 不支持 {type(model).__name__}"
            )
        return self.inference_engine.forecast(
            model,
            request,
            InferenceContext(
                runtime.random.inference,
                runtime.device,
                runtime.dtype,
            ),
        )

    def condition(
        self,
        forecast: Forecast,
        evidence: SearchEvidence,
    ) -> ConditionedForecast:
        evidence.validate()
        if self.conditioner is None or not self.conditioner.supports(evidence):
            component = (
                "none" if self.conditioner is None else type(self.conditioner).__name__
            )
            raise CapabilityError(
                f"conditioning component {component} 不支持 evidence kind={evidence.kind!r}"
            )
        result = self.conditioner.condition(forecast, evidence)
        return ConditionedForecast(result, (evidence.evidence_id,))

    def evaluate(
        self,
        forecast: Forecast | ConditionedForecast,
        truth: ObservationSet,
    ) -> EvaluationReport:
        truth.validate()
        value = forecast.forecast if isinstance(forecast, ConditionedForecast) else forecast
        expected_shape = value.samples.shape[1:]
        if truth.values.shape != expected_shape:
            raise DataValidationError(
                "observation values 形状必须匹配 forecast horizon/state 维: "
                f"expected={tuple(expected_shape)}, actual={tuple(truth.values.shape)}"
            )
        return self.evaluator.evaluate((value,), (truth.values,))


__all__ = ["EvaluationPipeline", "EvidenceConditioner"]
