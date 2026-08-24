"""Application service orchestrating train, checkpoint, and forecast use cases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from config import Config
from domain import CapabilityError, FitResult, Forecast, ForecastRequest
from estimation.base import FitContext
from estimation.em import SegmentEMData
from inference.base import InferenceContext, InferenceEngine
from infrastructure import TorchModelStore
from models.base import SDEModel
from registry import build_estimator, build_inference_engine, build_model

from .runtime import RunContext


@dataclass(frozen=True)
class TrainingRun:
    model: SDEModel
    fit: FitResult


class ExperimentApplication:
    """One explicitly-owned application run; no module-level state."""

    def __init__(
        self,
        config: Config,
        model: SDEModel,
        estimator,
        inference_engine: InferenceEngine,
        runtime: RunContext,
        model_store: TorchModelStore,
    ) -> None:
        self.config = config
        self.model = model
        self.estimator = estimator
        self.inference_engine = inference_engine
        self.runtime = runtime
        self.model_store = model_store

    @classmethod
    def from_config(cls, config: Config) -> "ExperimentApplication":
        config.validate()
        runtime = RunContext.create(
            config.seed,
            device=config.device,
            dtype={"float32": torch.float32, "float64": torch.float64}[config.dtype],
        )
        return cls(
            config=config,
            model=build_model(config),
            estimator=build_estimator(config),
            inference_engine=build_inference_engine(config),
            runtime=runtime,
            model_store=TorchModelStore(),
        )

    def train(self, data: SegmentEMData) -> TrainingRun:
        prepared = SegmentEMData(
            tuple(
                segment.to(device=self.runtime.device, dtype=self.runtime.dtype)
                for segment in data.segments
            ),
            data.dts,
        )
        fit_context = FitContext(
            self.runtime.random.training,
            self.runtime.device,
            self.runtime.dtype,
        )
        result = self.estimator.fit(self.model, prepared, fit_context)
        return TrainingRun(self.model, result)

    def forecast(self, request: ForecastRequest) -> Forecast:
        if not self.inference_engine.supports(self.model):
            raise CapabilityError(
                f"{type(self.inference_engine).__name__} 不支持 {type(self.model).__name__}"
            )
        return self.inference_engine.forecast(
            self.model,
            request,
            InferenceContext(
                self.runtime.random.inference,
                self.runtime.device,
                self.runtime.dtype,
            ),
        )

    def save_checkpoint(
        self,
        destination: Path,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        base = {
            "model_kind": self.config.components.model,
            "dtype": self.config.dtype,
            "device": self.config.device,
            "seed": self.config.seed,
        }
        base.update(metadata or {})
        self.model_store.save(self.model, base, destination)

    def load_checkpoint(self, source: Path) -> Mapping[str, Any]:
        state, metadata = self.model_store.load_state(source, self.runtime.device)
        expected = metadata.get("model_kind")
        if expected is not None and expected != self.config.components.model:
            raise CapabilityError(
                f"checkpoint model={expected!r}, config model={self.config.components.model!r}"
            )
        self.model.load_state_dict(state)
        return metadata
