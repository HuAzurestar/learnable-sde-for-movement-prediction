"""P2 OOP 骨干测试：Estimator ABC、Config 校验、注册表、路径解析。

运行：python -m pytest tests/test_oop.py
"""

from pathlib import Path

import pytest
import torch

from config import Config, Components
from domain import ConfigurationError
from data.paths import resolve
from estimation.base import Estimator
from estimation.em import SegmentEM
from estimation.score import CRPSEstimator
from registry import build_estimator, build_inference_engine, build_model
from inference import ExactGaussianEngine
from models import SDEModel
from models.segment_constant import SegmentConstantSDE

_REPO_ROOT = Path(__file__).resolve().parents[1]


# -- Estimator ABC -----------------------------------------------------------
def test_segment_em_is_estimator():
    assert issubclass(SegmentEM, Estimator)


def test_crps_estimator_is_estimator():
    assert issubclass(CRPSEstimator, Estimator)


# -- Config -----------------------------------------------------------
def test_real_config_loads_and_validates():
    cfg = Config.from_yaml(_REPO_ROOT / "config.yaml")
    cfg.validate()
    assert cfg.components.model == "I1"
    assert cfg.seed == 20260814


def test_config_validate_rejects_bad_seed():
    with pytest.raises(ValueError):
        Config(seed="abc").validate()


def test_config_validate_rejects_bad_dtype():
    with pytest.raises(ValueError):
        Config(dtype="float16").validate()


def test_config_validate_rejects_bad_component():
    cfg = Config(components=Components(model="I11"),
                 ablation_matrix={"model": ["I1", "neural"]})
    with pytest.raises(ValueError):
        cfg.validate()


# -- registry -----------------------------------------------------------
def test_build_model_returns_sde():
    cfg = Config(seed=20260814, model={"I1": {"n_modes": 1, "kappa": 0.0, "dt_ref": 60.0}})
    sde = build_model(cfg)
    assert isinstance(sde, SegmentConstantSDE)
    assert isinstance(sde, SDEModel)
    assert isinstance(sde, torch.nn.Module)


def test_build_model_rejects_unknown():
    cfg = Config(seed=1, components=Components(model="does_not_exist"))
    with pytest.raises(ConfigurationError):
        build_model(cfg)


def test_builds_all_configured_runtime_components():
    cfg = Config()
    assert isinstance(build_estimator(cfg), SegmentEM)
    assert isinstance(build_inference_engine(cfg), ExactGaussianEngine)


# -- paths -----------------------------------------------------------
def test_resolve_returns_safe_local_absolute_path(monkeypatch):
    monkeypatch.delenv("LEARNABLE_SDE_DATA_ROOT", raising=False)
    path = resolve("data_root")
    assert path.is_absolute()
    assert ".local" in path.parts


def test_resolve_rejects_unknown_name():
    with pytest.raises(KeyError):
        resolve("no_such_path")
