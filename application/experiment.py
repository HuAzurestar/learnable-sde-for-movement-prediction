"""Application service orchestrating training, prediction, and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from config import Config
from domain import (
    CapabilityError,
    ConditionedForecast,
    DataValidationError,
    EvidenceId,
    FitResult,
    Forecast,
    ForecastRequest,
    ObservationSet,
    RunRecord,
    SearchEvidence,
    TrainingData,
)
from estimation.base import FitContext
from estimation.em import SegmentEMData
from evaluation import EnergyScore, EvaluationReport, Evaluator
from inference.base import InferenceContext, InferenceEngine
from infrastructure import ArtifactWriter, AtomicRunStore, TorchModelStore
from models.base import SDEModel
from registry import build_estimator, build_inference_engine, build_model

from .pipelines import EvidenceConditioner, EvaluationPipeline
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
        evaluator: Evaluator | None = None,
        conditioner: EvidenceConditioner | None = None,
        run_store: AtomicRunStore | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.estimator = estimator
        self.inference_engine = inference_engine
        self.runtime = runtime
        self.model_store = model_store
        self.evaluator = evaluator if evaluator is not None else Evaluator([EnergyScore()])
        self.conditioner = conditioner
        self.run_store = (
            run_store
            if run_store is not None
            else AtomicRunStore(Path(config.paths.get("output_root", ".local/outputs")))
        )
        self.evaluation_pipeline = EvaluationPipeline(
            inference_engine,
            self.evaluator,
            conditioner,
        )

    @classmethod
    def from_config(
        cls,
        config: Config,
        *,
        evaluator: Evaluator | None = None,
        conditioner: EvidenceConditioner | None = None,
        run_store: AtomicRunStore | None = None,
    ) -> "ExperimentApplication":
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
            evaluator=evaluator,
            conditioner=conditioner,
            run_store=run_store,
        )

    def train(self, data: TrainingData | SegmentEMData) -> TrainingRun:
        if isinstance(data, TrainingData):
            data.validate()
            legacy_data = SegmentEMData(
                tuple(segment.state for segment in data.train),
                tuple(float(segment.dt) for segment in data.train),
            )
        elif isinstance(data, SegmentEMData):
            legacy_data = data
        else:
            raise DataValidationError(
                "training data must be TrajectoryDataset or SegmentEMData"
            )
        prepared = SegmentEMData(
            tuple(
                segment.to(device=self.runtime.device, dtype=self.runtime.dtype)
                for segment in legacy_data.segments
            ),
            legacy_data.dts,
        )
        prepared.validate()
        fit_context = FitContext(
            self.runtime.random.training,
            self.runtime.device,
            self.runtime.dtype,
        )
        result = self.estimator.fit(self.model, prepared, fit_context)
        return TrainingRun(self.model, result)

    def predict(self, model: SDEModel, request: ForecastRequest) -> Forecast:
        return self.evaluation_pipeline.predict(model, request, self.runtime)

    def submit_evidence(self, evidence: SearchEvidence) -> EvidenceId:
        """Validate evidence; persistence is an explicit ``commit_run`` step."""

        evidence.validate()
        return evidence.evidence_id

    def condition(
        self,
        forecast: Forecast,
        evidence: SearchEvidence,
    ) -> ConditionedForecast:
        return self.evaluation_pipeline.condition(forecast, evidence)

    def evaluate(
        self,
        forecast: Forecast | ConditionedForecast,
        truth: ObservationSet,
    ) -> EvaluationReport:
        return self.evaluation_pipeline.evaluate(forecast, truth)

    def commit_run(
        self,
        record: RunRecord,
        write_artifacts: ArtifactWriter | None = None,
    ) -> RunRecord:
        """Atomically publish one run's artifacts and audit record."""

        return self.run_store.commit(record, write_artifacts)

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
