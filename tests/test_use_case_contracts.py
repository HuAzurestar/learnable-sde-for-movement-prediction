"""PIRC-3 Slice A contracts for pure prediction/evaluation pipelines."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

from application.experiment import ExperimentApplication
from application.pipelines import EvaluationPipeline
from application.runtime import RunContext
from config import Components, Config
from domain import (
    CapabilityError,
    ConditionedForecast,
    DataValidationError,
    Forecast,
    ForecastRequest,
    ModelContext,
    ObservationSet,
    SearchEvidence,
    TrainingData,
    TrajectoryDataset,
)
from evaluation import EnergyScore, Evaluator
from inference import ExactGaussianEngine
from models.segment_constant import SegmentConstantSDE


def _i1_config() -> Config:
    return Config(
        seed=314159,
        components=Components(model="I1", estimator="EM", inference="exact"),
        model={"I1": {"n_modes": 1, "kappa": 0.0, "dt_ref": 1.0}},
    )


def _request() -> ForecastRequest:
    return ForecastRequest(
        initial_state=torch.tensor([1.0, 0.25], dtype=torch.float64),
        horizons=torch.tensor([1.0, 2.0], dtype=torch.float64),
        n_samples=3,
        context=ModelContext(regime=0),
    )


def _evidence() -> SearchEvidence:
    return SearchEvidence(
        evidence_id="evidence-1",
        kind="existence",
        region=(1.0, 2.0),
        time_window=(0.0, 2.0),
        spatial_unit="m",
        crs=None,
        confidence=None,
    )


class _IdentityConditioner:
    def supports(self, evidence: SearchEvidence) -> bool:
        return evidence.kind == "existence"

    def condition(self, forecast: Forecast, evidence: SearchEvidence) -> Forecast:
        return Forecast(
            samples=forecast.samples,
            mean=forecast.mean,
            covariance=forecast.covariance,
            metadata={**forecast.metadata, "conditioned_by": evidence.evidence_id},
        )


class _UnsupportedConditioner(_IdentityConditioner):
    def supports(self, evidence: SearchEvidence) -> bool:
        return False


def _pipeline(conditioner=None) -> EvaluationPipeline:
    return EvaluationPipeline(
        inference_engine=ExactGaussianEngine(),
        evaluator=Evaluator([EnergyScore()]),
        conditioner=conditioner,
    )


def test_training_data_reuses_canonical_trajectory_dataset():
    assert TrainingData is TrajectoryDataset


def test_search_evidence_is_immutable_and_validates_declared_context():
    evidence = _evidence()
    evidence.validate()

    with pytest.raises(FrozenInstanceError):
        evidence.kind = "exclusion"
    with pytest.raises(DataValidationError):
        SearchEvidence(
            evidence_id="bad-window",
            kind="existence",
            region=(1.0, 2.0),
            time_window=(2.0, 1.0),
            spatial_unit="m",
        ).validate()
    with pytest.raises(DataValidationError):
        SearchEvidence(
            evidence_id="bad-confidence",
            kind="soft",
            region=(1.0, 2.0),
            time_window=(0.0, 1.0),
            spatial_unit="m",
            confidence=1.1,
        ).validate()


def test_observation_set_rejects_non_finite_truth():
    with pytest.raises(DataValidationError):
        ObservationSet(torch.tensor([[float("nan"), 0.0]])).validate()


def test_pipeline_rejects_truth_shape_that_does_not_match_forecast():
    forecast = Forecast(samples=torch.zeros((3, 2, 2), dtype=torch.float64))

    with pytest.raises(DataValidationError):
        _pipeline().evaluate(forecast, ObservationSet(torch.zeros(2)))


def test_conditioning_fails_fast_when_component_is_incompatible():
    forecast = Forecast(samples=torch.zeros((2, 1, 2), dtype=torch.float64))

    with pytest.raises(CapabilityError):
        _pipeline(_UnsupportedConditioner()).condition(forecast, _evidence())


def test_pipeline_predict_condition_and_evaluate_are_independently_callable():
    model = SegmentConstantSDE(n_modes=1, dt_ref=1.0)
    runtime = RunContext.create(seed=11)
    pipeline = _pipeline(_IdentityConditioner())

    forecast = pipeline.predict(model, _request(), runtime)
    conditioned = pipeline.condition(forecast, _evidence())
    report = pipeline.evaluate(
        conditioned,
        ObservationSet(torch.tensor([[1.0, 0.25], [1.5, 0.25]], dtype=torch.float64)),
    )

    assert isinstance(conditioned, ConditionedForecast)
    assert conditioned.evidence_ids == ("evidence-1",)
    assert conditioned.forecast.metadata["conditioned_by"] == "evidence-1"
    assert set(report.aggregate) == {"energy_score"}


def test_pipeline_methods_do_not_perform_file_io(monkeypatch):
    def unexpected_io(*args, **kwargs):
        raise AssertionError("pure use-case pipeline attempted file I/O")

    monkeypatch.setattr(Path, "open", unexpected_io)
    monkeypatch.setattr(Path, "write_text", unexpected_io)
    model = SegmentConstantSDE(n_modes=1, dt_ref=1.0)
    pipeline = _pipeline(_IdentityConditioner())
    forecast = pipeline.predict(model, _request(), RunContext.create(seed=11))
    conditioned = pipeline.condition(forecast, _evidence())
    pipeline.evaluate(conditioned, ObservationSet(torch.ones((2, 2))))


def test_public_i1_fixture_matches_locked_legacy_forecast():
    config = _i1_config()
    legacy = ExperimentApplication.from_config(config).forecast(_request())
    model = SegmentConstantSDE(n_modes=1, kappa=0.0, dt_ref=1.0)
    current = _pipeline().predict(model, _request(), RunContext.create(config.seed))
    expected_samples = torch.tensor(
        [
            [
                [1.2467525212121509, 0.2467378370937614],
                [1.4976220613098403, 0.24421219321512297],
            ],
            [
                [1.248988301161218, 0.24550241436898573],
                [1.4935481039239347, 0.23615978919047914],
            ],
            [
                [1.2537541996227461, 0.2574549452534233],
                [1.51532318157925, 0.2599961572893715],
            ],
        ],
        dtype=torch.float64,
    )

    torch.testing.assert_close(legacy.samples, expected_samples, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(current.samples, legacy.samples, rtol=0.0, atol=0.0)
