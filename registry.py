"""The composition root's model, estimator, and inference registries."""

from __future__ import annotations

from application.registry import ComponentRegistry
from config import Config
from estimation.base import Estimator
from estimation.em import SegmentEM
from inference.base import (
    CommonRandomNumberEngine,
    EulerMaruyamaEngine,
    ExactGaussianEngine,
    InferenceEngine,
    SplitStepEngine,
)
from models.base import SDEModel
from models.segment_constant import SegmentConstantSDE


def _torch_dtype(name: str):
    import torch

    return {"float32": torch.float32, "float64": torch.float64}[name]


def _build_segment_constant(cfg: Config) -> SDEModel:
    model = cfg.model.get("I1", {}) or {}
    return SegmentConstantSDE(
        n_modes=model.get("n_modes", 3),
        kappa=model.get("kappa", 0.0),
        dt_ref=model.get("dt_ref", 60.0),
        dtype=_torch_dtype(cfg.dtype),
        device=cfg.device,
    )


def _build_em(cfg: Config) -> Estimator:
    options = cfg.protocol.get("em", {}) or {}
    return SegmentEM(
        max_iter=int(options.get("max_iter", 50)),
        tol=float(options.get("tol", 1e-5)),
    )


def _build_exact(cfg: Config) -> InferenceEngine:
    return ExactGaussianEngine()


def _build_euler(cfg: Config) -> InferenceEngine:
    options = cfg.protocol.get("euler", {}) or {}
    return EulerMaruyamaEngine(max_step=float(options.get("max_step", 1.0)))


def _build_split(cfg: Config) -> InferenceEngine:
    options = cfg.protocol.get("split", {}) or {}
    return SplitStepEngine(max_step=float(options.get("max_step", 1.0)))


def _build_crn(cfg: Config) -> InferenceEngine:
    return CommonRandomNumberEngine()


MODEL_REGISTRY: ComponentRegistry[Config, SDEModel] = ComponentRegistry()
MODEL_REGISTRY.register("I1", _build_segment_constant)

ESTIMATOR_REGISTRY: ComponentRegistry[Config, Estimator] = ComponentRegistry()
ESTIMATOR_REGISTRY.register("EM", _build_em)

INFERENCE_REGISTRY: ComponentRegistry[Config, InferenceEngine] = ComponentRegistry()
INFERENCE_REGISTRY.register("exact", _build_exact)
INFERENCE_REGISTRY.register("euler", _build_euler)
INFERENCE_REGISTRY.register("J1_split", _build_split)
INFERENCE_REGISTRY.register("J3_CRN", _build_crn)


def build_model(cfg: Config) -> SDEModel:
    cfg.validate()
    return MODEL_REGISTRY.create(cfg.components.model, cfg)


def build_estimator(cfg: Config) -> Estimator:
    cfg.validate()
    return ESTIMATOR_REGISTRY.create(cfg.components.estimator, cfg)


def build_inference_engine(cfg: Config) -> InferenceEngine:
    cfg.validate()
    return INFERENCE_REGISTRY.create(cfg.components.inference, cfg)


__all__ = [
    "ESTIMATOR_REGISTRY",
    "INFERENCE_REGISTRY",
    "MODEL_REGISTRY",
    "build_estimator",
    "build_inference_engine",
    "build_model",
]
