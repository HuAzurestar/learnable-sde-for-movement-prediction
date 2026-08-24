"""Integration tests for the real application, checkpoint, and inference path."""

from __future__ import annotations

import torch
import pytest

from application.experiment import ExperimentApplication
from application.synthetic import make_synthetic_em_data
from config import Components, Config
from domain import ConfigurationError, Forecast, ForecastRequest, ModelContext
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
