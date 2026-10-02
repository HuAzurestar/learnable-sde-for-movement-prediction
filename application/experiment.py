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
    to_phase_space_1d,
)
from estimation.base import FitContext
from estimation.em import SegmentEMData
from evaluation import EnergyScore, EvaluationReport, Evaluator
from inference.base import InferenceEngine
from infrastructure import ArtifactWriter, AtomicRunStore, TorchModelStore
from models.base import SDEModel
from registry import build_estimator, build_inference_engine, build_model

from .pipelines import EvidenceConditioner, EvaluationPipeline
from .runtime import RunContext
from dataclasses import asdict
from copy import deepcopy
import math
import json
from infrastructure.research_store import ResearchError, encode


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
        self._component_plan_document = None
        self._component_input_document = None
        self.model = model
        self.estimator = estimator
        self.runtime = runtime
        self.model_store = model_store
        self.run_store = (
            run_store
            if run_store is not None
            else AtomicRunStore(Path(config.paths.get("output_root", ".local/outputs")))
        )
        self.evaluation_pipeline = EvaluationPipeline(
            inference_engine,
            evaluator if evaluator is not None else Evaluator([EnergyScore()]),
            conditioner,
        )

    @property
    def inference_engine(self) -> InferenceEngine:
        return self.evaluation_pipeline.inference_engine

    @property
    def evaluator(self) -> Evaluator:
        return self.evaluation_pipeline.evaluator

    @property
    def conditioner(self) -> EvidenceConditioner | None:
        return self.evaluation_pipeline.conditioner

    @property
    def component_plan(self):
        return json.loads(self._component_plan_document) if self._component_plan_document is not None else None

    def _component_inputs(self):
        return json.loads(self._component_input_document) if self._component_input_document is not None else None

    def _check_model_profile(self, model):
        profile = self._component_inputs()
        if profile is not None and (model.state_dim != profile["state_dim"] or model.noise_dim != profile["noise_dim"]
                or str(model.dtype).removeprefix("torch.") != profile["dtype"] or str(model.device) != profile["device"]):
            raise ResearchError("CONTRACT_MISMATCH", "constructed or supplied model differs from frozen input profile")

    @classmethod
    def from_config(
        cls,
        config: Config,
        *,
        evaluator: Evaluator | None = None,
        conditioner: EvidenceConditioner | None = None,
        run_store: AtomicRunStore | None = None,
        component_bindings=None,
        matrix_cells=1,
    ) -> "ExperimentApplication":
        component_plan = None
        if component_bindings is not None:
            from registry import MODEL_REGISTRY, ESTIMATOR_REGISTRY, INFERENCE_REGISTRY, plan_components
            component_plan = plan_components(config, component_bindings, matrix_cells=matrix_cells)
            component_bindings = deepcopy(component_bindings)
            config = Config.from_dict(deepcopy(asdict(config)))
        config.validate()
        runtime = RunContext.create(
            config.seed,
            device=config.device,
            dtype={"float32": torch.float32, "float64": torch.float64}[config.dtype],
        )
        components = None
        if component_bindings is not None:
            components = {role: registry.create_bound(component_bindings[role], matrix_cells=matrix_cells)
                for role, registry in (("model", MODEL_REGISTRY), ("trainer", ESTIMATOR_REGISTRY), ("predictor", INFERENCE_REGISTRY))}
        application = cls(
            config=config,
            model=components["model"] if components is not None else build_model(config),
            estimator=components["trainer"] if components is not None else build_estimator(config),
            inference_engine=components["predictor"] if components is not None else build_inference_engine(config),
            runtime=runtime,
            model_store=TorchModelStore(),
            evaluator=evaluator,
            conditioner=conditioner,
            run_store=run_store,
        )
        if component_plan is not None:
            application._component_plan_document = encode(component_plan)
            application._component_input_document = encode(component_bindings["model"]["inputs"])
            application._check_model_profile(application.model)
            if not application.inference_engine.supports(application.model):
                raise ResearchError("CONTRACT_MISMATCH", "constructed predictor does not support the actual model")
        return application

    def train(self, data: TrainingData | SegmentEMData) -> TrainingRun:
        if isinstance(data, TrainingData):
            data.validate()
            legacy_data = SegmentEMData(
                tuple(to_phase_space_1d(segment) for segment in data.train),
                tuple(float(segment.dt) for segment in data.train),
            )
        elif isinstance(data, SegmentEMData):
            legacy_data = data
        else:
            raise DataValidationError(
                "training data must be TrajectoryDataset or SegmentEMData"
            )
        profile = self._component_inputs()
        if profile is not None:
            self._check_model_profile(self.model)
            if (len(legacy_data.segments) > profile["components"]
                    or sum(segment.shape[0] for segment in legacy_data.segments) > profile["observations"]):
                raise ResearchError("RESOURCE_PLAN_REJECTED", "actual training inputs exceed frozen observation/segment bound")
            if any(segment.ndim != 2 or segment.shape[1] != profile["state_dim"] for segment in legacy_data.segments):
                raise ResearchError("CONTRACT_MISMATCH", "actual training state shape differs")
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
        profile = self._component_inputs()
        if profile is not None:
            self._check_model_profile(model)
            request.validate()
            if type(request.n_samples) is not int or request.n_samples > profile["paths"]:
                raise ResearchError("RESOURCE_PLAN_REJECTED", "actual forecast paths exceed frozen sample bound")
            if request.initial_state.numel() != profile["state_dim"]:
                raise ResearchError("CONTRACT_MISMATCH", "forecast initial state shape differs")
            steps = len(request.horizons) + 1
            if hasattr(self.inference_engine, "max_step"):
                max_step = self.inference_engine.max_step
                if not math.isfinite(max_step) or max_step <= 0:
                    raise ResearchError("CONTRACT_MISMATCH", "predictor step must be positive and finite")
                steps, previous = 1, 0.0
                for value in request.horizons:
                    horizon = float(value)
                    ratio = (horizon - previous) / max_step
                    if not math.isfinite(ratio) or ratio > profile["steps"]:
                        raise ResearchError("RESOURCE_PLAN_REJECTED", "forecast time grid exceeds frozen step bound")
                    # Conservative extra rounding step avoids undercounting
                    # legacy float accumulation without simulating a rollout.
                    steps += max(1, math.ceil(math.nextafter(ratio, math.inf)))
                    previous = horizon
            if steps > profile["steps"]:
                raise ResearchError("RESOURCE_PLAN_REJECTED", "forecast time grid exceeds frozen step bound")
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
        """Compatibility wrapper for the migrated :meth:`predict` use case."""

        return self.predict(self.model, request)

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
