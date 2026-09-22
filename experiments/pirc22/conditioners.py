"""Deterministic compact conditioner families for the PIRC-22 benchmark."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping

import numpy as np
import torch
from torch import nn


CHECKPOINT_SCHEMA_VERSION = "pirc22-conditioner-checkpoint-v1"


class ConditionerError(ValueError):
    """A conditioner configuration, tensor, or checkpoint is invalid."""


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ConditionerSpec:
    conditioner_id: str
    hidden_widths: tuple[int, ...]
    activation: str = "tanh"

    @property
    def layer_count(self) -> int:
        return len(self.hidden_widths)


CONDITIONER_SPECS = (
    ConditionerSpec("linear", ()),
    ConditionerSpec("mlp-small-16", (16,)),
    ConditionerSpec("mlp-small-32", (32,)),
    ConditionerSpec("mlp-medium-32-16", (32, 16)),
    ConditionerSpec("mlp-medium-64-32", (64, 32)),
)
CONDITIONER_BY_ID = {spec.conditioner_id: spec for spec in CONDITIONER_SPECS}


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    maximum_epochs: int = 200
    patience: int = 20
    minimum_delta: float = 1e-8

    def validate(self) -> None:
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ConditionerError("learning_rate must be positive and finite")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ConditionerError("weight_decay must be nonnegative and finite")
        for name in ("batch_size", "maximum_epochs", "patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConditionerError(f"{name} must be a positive integer")
        if not np.isfinite(self.minimum_delta) or self.minimum_delta < 0:
            raise ConditionerError("minimum_delta must be nonnegative and finite")


@dataclass(frozen=True)
class LearningCurvePoint:
    epoch: int
    train_loss: float
    validation_loss: float


@dataclass(frozen=True)
class ConditionerFitResult:
    model: "TerrainConditioner"
    learning_curve: tuple[LearningCurvePoint, ...]
    best_epoch: int
    checkpoint: Mapping[str, object]


class _BiasOnly(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(output_dim, dtype=torch.float64))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.bias.unsqueeze(0).expand(len(values), -1)


class TerrainConditioner(nn.Module):
    """Terrain-only additive drift conditioner; the base SDE remains untouched."""

    def __init__(
        self,
        spec: ConditionerSpec,
        continuous_input_dim: int,
        output_dim: int,
        *,
        categorical_cardinality: int | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        super().__init__()
        if continuous_input_dim < 0 or output_dim < 1:
            raise ConditionerError("conditioner dimensions are invalid")
        if (categorical_cardinality is None) != (embedding_dim is None):
            raise ConditionerError(
                "categorical_cardinality and embedding_dim must be supplied together"
            )
        if categorical_cardinality is not None and (
            categorical_cardinality < 2 or embedding_dim is None or embedding_dim < 1
        ):
            raise ConditionerError("categorical embedding dimensions are invalid")
        self.spec = spec
        self.continuous_input_dim = continuous_input_dim
        self.output_dim = output_dim
        self.categorical_cardinality = categorical_cardinality
        self.embedding_dim = embedding_dim
        self.embedding = (
            nn.Embedding(categorical_cardinality, embedding_dim, dtype=torch.float64)
            if categorical_cardinality is not None and embedding_dim is not None
            else None
        )
        effective_input_dim = continuous_input_dim + (embedding_dim or 0)
        dimensions = (effective_input_dim, *spec.hidden_widths, output_dim)
        layers: list[nn.Module] = []
        for index, (source, target) in enumerate(zip(dimensions, dimensions[1:])):
            layers.append(
                _BiasOnly(target)
                if source == 0
                else nn.Linear(source, target, dtype=torch.float64)
            )
            if index < len(dimensions) - 2:
                if spec.activation != "tanh":
                    raise ConditionerError(
                        f"unsupported conditioner activation: {spec.activation}"
                    )
                layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    @property
    def effective_input_dim(self) -> int:
        return self.continuous_input_dim + (self.embedding_dim or 0)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def initialize(self, seed: int) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        with torch.no_grad():
            if self.embedding is not None:
                nn.init.normal_(self.embedding.weight, mean=0.0, std=0.05, generator=generator)
            for module in self.network:
                if isinstance(module, nn.Linear):
                    gain = nn.init.calculate_gain("tanh") if self.spec.hidden_widths else 1.0
                    if module.weight.numel():
                        nn.init.xavier_uniform_(
                            module.weight, gain=gain, generator=generator
                        )
                    nn.init.zeros_(module.bias)
                elif isinstance(module, _BiasOnly):
                    nn.init.zeros_(module.bias)

    def forward(
        self, continuous: torch.Tensor, categorical_tokens: torch.Tensor | None = None
    ) -> torch.Tensor:
        if continuous.ndim != 2 or continuous.shape[1] != self.continuous_input_dim:
            raise ConditionerError("continuous conditioner input has the wrong shape")
        parts = [continuous]
        if self.embedding is not None:
            if categorical_tokens is None or categorical_tokens.shape != (len(continuous),):
                raise ConditionerError("categorical conditioner input has the wrong shape")
            parts.append(self.embedding(categorical_tokens))
        elif categorical_tokens is not None:
            raise ConditionerError("categorical tokens were supplied without an embedding")
        combined = torch.cat(parts, dim=1)
        return self.network(combined)


def build_conditioner(
    conditioner_id: str,
    continuous_input_dim: int,
    output_dim: int = 2,
    *,
    seed: int,
    categorical_cardinality: int | None = None,
    embedding_dim: int | None = None,
) -> TerrainConditioner:
    try:
        spec = CONDITIONER_BY_ID[conditioner_id]
    except KeyError as error:
        raise ConditionerError(f"unknown conditioner: {conditioner_id}") from error
    model = TerrainConditioner(
        spec,
        continuous_input_dim,
        output_dim,
        categorical_cardinality=categorical_cardinality,
        embedding_dim=embedding_dim,
    )
    model.initialize(seed)
    return model


def _arrays(
    features: np.ndarray,
    targets: np.ndarray,
    tokens: np.ndarray | None,
    *,
    feature_dim: int | None = None,
    output_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    feature_values = np.asarray(features, dtype=np.float64)
    target_values = np.asarray(targets, dtype=np.float64)
    if feature_values.ndim != 2 or target_values.ndim != 2:
        raise ConditionerError("features and targets must be matrices")
    if len(feature_values) != len(target_values) or not len(feature_values):
        raise ConditionerError("features and targets must have equal nonzero rows")
    if feature_dim is not None and feature_values.shape[1] != feature_dim:
        raise ConditionerError("feature dimension does not match checkpoint")
    if output_dim is not None and target_values.shape[1] != output_dim:
        raise ConditionerError("target dimension does not match checkpoint")
    if not np.isfinite(feature_values).all() or not np.isfinite(target_values).all():
        raise ConditionerError("features and targets must be finite")
    token_tensor: torch.Tensor | None = None
    if tokens is not None:
        token_values = np.asarray(tokens)
        if token_values.shape != (len(feature_values),):
            raise ConditionerError("categorical tokens must have one value per row")
        if not np.issubdtype(token_values.dtype, np.integer):
            raise ConditionerError("categorical tokens must be integers")
        token_tensor = torch.as_tensor(token_values.astype(np.int64), dtype=torch.long)
    return (
        torch.as_tensor(feature_values, dtype=torch.float64),
        torch.as_tensor(target_values, dtype=torch.float64),
        token_tensor,
    )


def _state_record(model: TerrainConditioner) -> dict[str, object]:
    return {
        name: tensor.detach().cpu().numpy().tolist()
        for name, tensor in sorted(model.state_dict().items())
    }


def _checkpoint(
    model: TerrainConditioner,
    *,
    seed: int,
    config: TrainingConfig,
    curve: tuple[LearningCurvePoint, ...],
    best_epoch: int,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "conditioner_id": model.spec.conditioner_id,
        "hidden_widths": list(model.spec.hidden_widths),
        "activation": model.spec.activation,
        "continuous_input_dim": model.continuous_input_dim,
        "effective_input_dim": model.effective_input_dim,
        "output_dim": model.output_dim,
        "categorical_cardinality": model.categorical_cardinality,
        "embedding_dim": model.embedding_dim,
        "seed": seed,
        "trainable_parameter_count": model.trainable_parameter_count,
        "training_config": asdict(config),
        "training_scope": {
            "gradient_roles": ["train"],
            "checkpoint_selection_roles": ["validation"],
        },
        "best_epoch": best_epoch,
        "learning_curve": [asdict(point) for point in curve],
        "state": _state_record(model),
    }
    payload["checkpoint_identity_sha256"] = _canonical_hash(payload)
    return payload


def fit_conditioner(
    conditioner_id: str,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    *,
    seed: int,
    config: TrainingConfig = TrainingConfig(),
    train_tokens: np.ndarray | None = None,
    validation_tokens: np.ndarray | None = None,
    categorical_cardinality: int | None = None,
    embedding_dim: int | None = None,
) -> ConditionerFitResult:
    """Fit only on train rows and use validation only for checkpoint selection."""

    config.validate()
    train_x, train_y, train_token_tensor = _arrays(
        train_features, train_targets, train_tokens
    )
    validation_x, validation_y, validation_token_tensor = _arrays(
        validation_features,
        validation_targets,
        validation_tokens,
        feature_dim=train_x.shape[1],
        output_dim=train_y.shape[1],
    )
    if (train_token_tensor is None) != (validation_token_tensor is None):
        raise ConditionerError("train and validation categorical inputs disagree")
    model = build_conditioner(
        conditioner_id,
        train_x.shape[1],
        train_y.shape[1],
        seed=seed,
        categorical_cardinality=categorical_cardinality,
        embedding_dim=embedding_dim,
    )
    if model.embedding is not None:
        if train_token_tensor is None or validation_token_tensor is None:
            raise ConditionerError("embedding model requires categorical tokens")
        if (
            train_token_tensor.min() < 0
            or validation_token_tensor.min() < 0
            or train_token_tensor.max() >= model.categorical_cardinality
            or validation_token_tensor.max() >= model.categorical_cardinality
        ):
            raise ConditionerError("categorical token is outside the vocabulary")
    elif train_token_tensor is not None:
        raise ConditionerError("categorical tokens require an embedding model")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    curve: list[LearningCurvePoint] = []
    for epoch in range(1, config.maximum_epochs + 1):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(order), config.batch_size):
            indexes = order[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            predictions = model(
                train_x[indexes],
                train_token_tensor[indexes] if train_token_tensor is not None else None,
            )
            loss = torch.mean((predictions - train_y[indexes]) ** 2)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss = float(
                torch.mean((model(train_x, train_token_tensor) - train_y) ** 2).item()
            )
            validation_loss = float(
                torch.mean(
                    (
                        model(validation_x, validation_token_tensor)
                        - validation_y
                    )
                    ** 2
                ).item()
            )
        if not np.isfinite(train_loss) or not np.isfinite(validation_loss):
            raise ConditionerError("conditioner training produced a non-finite loss")
        curve.append(LearningCurvePoint(epoch, train_loss, validation_loss))
        if validation_loss < best_loss - config.minimum_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break
    if best_state is None:
        raise ConditionerError("conditioner training produced no checkpoint")
    model.load_state_dict(best_state)
    frozen_curve = tuple(curve)
    checkpoint = _checkpoint(
        model,
        seed=seed,
        config=config,
        curve=frozen_curve,
        best_epoch=best_epoch,
    )
    return ConditionerFitResult(model, frozen_curve, best_epoch, checkpoint)


def load_conditioner(checkpoint: Mapping[str, object]) -> TerrainConditioner:
    payload = dict(checkpoint)
    identity = payload.pop("checkpoint_identity_sha256", None)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ConditionerError("unsupported conditioner checkpoint schema")
    if identity != _canonical_hash(payload):
        raise ConditionerError("conditioner checkpoint identity mismatch")
    model = build_conditioner(
        str(payload["conditioner_id"]),
        int(payload["continuous_input_dim"]),
        int(payload["output_dim"]),
        seed=int(payload["seed"]),
        categorical_cardinality=(
            int(payload["categorical_cardinality"])
            if payload.get("categorical_cardinality") is not None
            else None
        ),
        embedding_dim=(
            int(payload["embedding_dim"])
            if payload.get("embedding_dim") is not None
            else None
        ),
    )
    state = payload.get("state")
    if not isinstance(state, Mapping) or set(state) != set(model.state_dict()):
        raise ConditionerError("conditioner checkpoint state is incomplete")
    converted = {
        name: torch.as_tensor(value, dtype=model.state_dict()[name].dtype)
        for name, value in state.items()
    }
    try:
        model.load_state_dict(converted, strict=True)
    except RuntimeError as error:
        raise ConditionerError("conditioner checkpoint tensor shape is invalid") from error
    if model.trainable_parameter_count != payload.get("trainable_parameter_count"):
        raise ConditionerError("conditioner checkpoint parameter count mismatch")
    model.eval()
    return model


def predict_conditioner(
    model: TerrainConditioner,
    features: np.ndarray,
    *,
    categorical_tokens: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    dummy_targets = np.zeros((len(values), model.output_dim), dtype=np.float64)
    feature_tensor, _, token_tensor = _arrays(
        values,
        dummy_targets,
        categorical_tokens,
        feature_dim=model.continuous_input_dim,
        output_dim=model.output_dim,
    )
    model.eval()
    with torch.no_grad():
        return model(feature_tensor, token_tensor).cpu().numpy()


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CONDITIONER_BY_ID",
    "CONDITIONER_SPECS",
    "ConditionerError",
    "ConditionerFitResult",
    "ConditionerSpec",
    "LearningCurvePoint",
    "TerrainConditioner",
    "TrainingConfig",
    "build_conditioner",
    "fit_conditioner",
    "load_conditioner",
    "predict_conditioner",
]
