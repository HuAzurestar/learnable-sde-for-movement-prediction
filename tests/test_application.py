"""Integration tests for the real application, checkpoint, and inference path."""

from __future__ import annotations

import json

import pytest
import torch

from cli import train as train_cli
from cli import predict as predict_cli
from application.experiment import ExperimentApplication
from application.synthetic import make_synthetic_em_data
from config import Components, Config
from domain import (
    CapabilityError,
    ConditionedForecast,
    ConfigurationError,
    DataValidationError,
    FitResult,
    Forecast,
    ForecastRequest,
    ModelContext,
    ObservationSet,
    SearchEvidence,
    TrajectoryDataset,
    TrajectorySegment,
)
from estimation.em import SegmentEMData
from evaluation import EnergyScore, Evaluator
from inference import CommonRandomNumberEngine, SplitStepEngine
from registry import build_inference_engine


def _config(inference: str = "exact") -> Config:
    return Config(
        seed=19,
        components=Components(model="I1", estimator="EM", inference=inference),
        model={"I1": {"n_modes": 2, "kappa": 0.0, "dt_ref": 1.0}},
        protocol={"em": {"max_iter": 2}},
    )


def _request() -> ForecastRequest:
    return ForecastRequest(
        initial_state=torch.tensor([1.0, 0.25], dtype=torch.float64),
        horizons=torch.tensor([1.0, 2.0], dtype=torch.float64),
        n_samples=3,
        context=ModelContext(regime=0),
    )


def _evidence(evidence_id: str = "evidence-1") -> SearchEvidence:
    return SearchEvidence(
        evidence_id=evidence_id,
        kind="existence",
        region=(1.0, 2.0),
        time_window=(0.0, 2.0),
        spatial_unit="m",
    )


def _training_data() -> TrajectoryDataset:
    return TrajectoryDataset(
        train=(
            TrajectorySegment(
                t=torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64),
                x=torch.tensor(
                    [[0.0, 10.0], [2.0, 20.0], [8.0, 30.0]],
                    dtype=torch.float64,
                ),
                dt=1.0,
                meta={"segment_id": "train-1"},
            ),
        )
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


class _CapturingEstimator:
    def __init__(self) -> None:
        self.data: SegmentEMData | None = None

    def fit(self, model, data, context) -> FitResult:
        data.validate()
        self.data = data
        return FitResult(converged=True, iterations=0)


def test_application_train_checkpoint_predict_round_trip(tmp_path):
    config = _config()
    data, _ = make_synthetic_em_data(
        n_segments=4,
        length=10,
        dt=1.0,
        seed=config.seed,
    )
    app = ExperimentApplication.from_config(config)
    run = app.train(data)
    checkpoint = tmp_path / "model.pt"
    app.save_checkpoint(checkpoint, {"iterations": run.fit.iterations})

    restored = ExperimentApplication.from_config(config)
    metadata = restored.load_checkpoint(checkpoint)
    forecast = restored.forecast(
        ForecastRequest(
            initial_state=torch.zeros(2, dtype=torch.float64),
            horizons=torch.tensor([1.0, 2.0], dtype=torch.float64),
            n_samples=5,
            context=ModelContext(regime=0),
        )
    )

    assert metadata["iterations"] == run.fit.iterations
    assert forecast.samples.shape == (5, 2, 2)
    assert torch.equal(restored.model.Gamma, app.model.Gamma)


def test_inference_registry_builds_declared_j1_and_j3_engines():
    assert isinstance(build_inference_engine(_config("J1_split")), SplitStepEngine)
    assert isinstance(
        build_inference_engine(_config("J3_CRN")),
        CommonRandomNumberEngine,
    )


def test_unimplemented_neural_and_j2_components_fail_fast():
    neural = _config()
    neural.components.model = "neural"
    with pytest.raises(ConfigurationError):
        ExperimentApplication.from_config(neural)

    with pytest.raises(ConfigurationError):
        build_inference_engine(_config("J2_FP"))


def test_evaluator_uses_canonical_sample_energy_score():
    forecast = Forecast(
        samples=torch.tensor([[[0.0]], [[2.0]]], dtype=torch.float64)
    )
    report = Evaluator([EnergyScore()]).evaluate(
        [forecast],
        [torch.tensor([[1.0]], dtype=torch.float64)],
    )

    assert report.aggregate["energy_score"] == 0.5


def test_application_exposes_composable_predict_condition_and_evaluate_use_cases():
    config = _config()
    legacy = ExperimentApplication.from_config(config).forecast(_request())
    app = ExperimentApplication.from_config(
        config,
        conditioner=_IdentityConditioner(),
    )
    app.estimator = _CapturingEstimator()

    training = app.train(_training_data())
    forecast = app.predict(training.model, _request())
    evidence = _evidence()
    evidence_id = app.submit_evidence(evidence)
    conditioned = app.condition(forecast, evidence)
    report = app.evaluate(
        conditioned,
        ObservationSet(torch.tensor([[1.0, 0.25], [1.5, 0.25]], dtype=torch.float64)),
    )

    torch.testing.assert_close(forecast.samples, legacy.samples, rtol=0.0, atol=0.0)
    assert evidence_id == evidence.evidence_id
    assert isinstance(conditioned, ConditionedForecast)
    assert conditioned.forecast.metadata["conditioned_by"] == evidence.evidence_id
    assert set(report.aggregate) == {"energy_score"}


def test_application_fails_fast_for_invalid_or_unsupported_evidence():
    app = ExperimentApplication.from_config(_config())
    forecast = Forecast(samples=torch.zeros((3, 2, 2), dtype=torch.float64))

    with pytest.raises(DataValidationError):
        app.submit_evidence(_evidence(""))
    with pytest.raises(CapabilityError):
        app.condition(forecast, _evidence())


def test_application_adapts_canonical_training_data_to_legacy_estimator_input():
    app = ExperimentApplication.from_config(_config())
    estimator = _CapturingEstimator()
    app.estimator = estimator
    data = _training_data()
    segment = data.train[0]

    run = app.train(data)

    assert run.model is app.model
    assert isinstance(estimator.data, SegmentEMData)
    assert estimator.data.dts == (1.0,)
    torch.testing.assert_close(
        estimator.data.segments[0],
        torch.tensor(
            [[0.0, 2.0], [2.0, 6.0], [8.0, 6.0]],
            dtype=torch.float64,
        ),
    )
    assert not torch.equal(estimator.data.segments[0], segment.x)


def test_real_data_cli_returns_canonical_training_data_for_application_adaptation(
    monkeypatch,
):
    data = _training_data()

    class _Source:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load(self):
            return list(data.train)

    monkeypatch.setattr(train_cli, "TrajectorySource", _Source)

    loaded = train_cli._real_data("train", 1, 19)

    assert isinstance(loaded, TrajectoryDataset)
    assert len(loaded.train) == 1
    assert loaded.train[0] is data.train[0]


def test_application_forwards_strategy_ownership_to_pipeline():
    app = ExperimentApplication.from_config(_config())
    replacement = SplitStepEngine(max_step=0.5)

    assert app.inference_engine is app.evaluation_pipeline.inference_engine
    assert app.evaluator is app.evaluation_pipeline.evaluator
    assert app.conditioner is app.evaluation_pipeline.conditioner
    with pytest.raises(AttributeError):
        app.inference_engine = replacement

    app.evaluation_pipeline.inference_engine = replacement

    assert app.inference_engine is replacement
    assert app.predict(app.model, _request()).metadata["engine"] == "split_step"


def test_legacy_predict_cli_uses_new_application_entrypoint(
    tmp_path,
    monkeypatch,
    capsys,
):
    config = _config()
    checkpoint = tmp_path / "model.pt"
    ExperimentApplication.from_config(config).save_checkpoint(checkpoint)
    monkeypatch.setattr(predict_cli.Config, "from_yaml", lambda path: config)

    def legacy_path_was_used(*args, **kwargs):
        raise AssertionError("predict CLI called the legacy forecast entrypoint")

    monkeypatch.setattr(ExperimentApplication, "forecast", legacy_path_was_used)

    result = predict_cli.main(
        [
            "--config",
            str(tmp_path / "config.yaml"),
            "--checkpoint",
            str(checkpoint),
            "--x0",
            "1.0",
            "0.25",
            "--horizons",
            "1.0",
            "2.0",
            "--samples",
            "3",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert result == 0
    assert report["command"] == "predict"
    assert report["sample_shape"] == [3, 2, 2]
