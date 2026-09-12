"""Small, real, coupled Gaussian transition models for the NEX326 runner.

The implementation is intentionally CPU-only and bounded so every registered arm can
exercise the complete process in CI.  It is not a replacement for the formal large
cohort: the RunRecord keeps the cohort purpose and therefore cannot turn a fixture run
into a scientific verdict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from .cohort import Segment


RIDGE = 1e-6
COVARIANCE_JITTER = 1e-8
MIN_META_TASK_SEGMENTS = 3
REPTILE_META_EPOCHS = 4
REPTILE_INITIAL_STEP = 0.35
REPTILE_INNER_STEPS = 5
REPTILE_INNER_RATE = 0.5
ESTIMATOR_SCALE_CANDIDATES = (0.55, 0.75, 1.0, 1.3, 1.7)
ESTIMATOR_DRIFT_FRACTIONS = (0.0, 0.1, 0.25, 0.5)


@dataclass(frozen=True)
class TransitionData:
    features: np.ndarray
    targets: np.ndarray
    mode_labels: np.ndarray
    segment_ids: tuple[str, ...]

    @property
    def count(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True)
class ModelState:
    model_kind: str
    condition_names: tuple[str, ...]
    weights: tuple[np.ndarray, ...]
    covariances: tuple[np.ndarray, ...]
    mode_probabilities: np.ndarray
    training_sample_count: int
    validation_objective_before: float
    validation_objective_after: float
    covariance_scale: float
    meta_task_count: int = 0
    adaptation_parameter_delta: float = 0.0
    estimator_method: str = "unassigned"
    estimator_optimization_scope: str = "unassigned"
    estimator_drift_fraction: float = 0.0
    estimator_candidate_count: int = 0
    transfer_method: str = "unassigned"
    finetune_method: str = "unassigned"
    meta_algorithm: str = "none"
    meta_outer_epochs: int = 0
    meta_inner_steps: int = 0
    meta_inner_rate: float = 0.0
    meta_inner_objective_before: float = 0.0
    meta_inner_objective_after: float = 0.0

    @property
    def n_modes(self) -> int:
        return len(self.weights)

    @property
    def feature_count(self) -> int:
        return int(self.weights[0].shape[0])

    def to_dict(self) -> dict[str, object]:
        return {
            "model_kind": self.model_kind,
            "condition_names": list(self.condition_names),
            "weights": [value.tolist() for value in self.weights],
            "covariances": [value.tolist() for value in self.covariances],
            "mode_probabilities": self.mode_probabilities.tolist(),
            "training_sample_count": self.training_sample_count,
            "validation_objective_before": self.validation_objective_before,
            "validation_objective_after": self.validation_objective_after,
            "covariance_scale": self.covariance_scale,
            "meta_task_count": self.meta_task_count,
            "adaptation_parameter_delta": self.adaptation_parameter_delta,
            "estimator_method": self.estimator_method,
            "estimator_optimization_scope": self.estimator_optimization_scope,
            "estimator_drift_fraction": self.estimator_drift_fraction,
            "estimator_candidate_count": self.estimator_candidate_count,
            "transfer_method": self.transfer_method,
            "finetune_method": self.finetune_method,
            "meta_algorithm": self.meta_algorithm,
            "meta_outer_epochs": self.meta_outer_epochs,
            "meta_inner_steps": self.meta_inner_steps,
            "meta_inner_rate": self.meta_inner_rate,
            "meta_inner_objective_before": self.meta_inner_objective_before,
            "meta_inner_objective_after": self.meta_inner_objective_after,
        }


def _segment_angle(segment: Segment) -> float:
    displacement = segment.state[-1] - segment.state[0]
    return math.atan2(float(displacement[1]), float(displacement[0]))


def _balanced_segment_modes(segments: Sequence[Segment]) -> dict[str, int]:
    """Assign deterministic heading tertiles without assuming fixed angle coverage."""
    ranked = sorted(segments, key=lambda segment: (_segment_angle(segment), segment.segment_id))
    count = len(ranked)
    return {
        segment.segment_id: min(2, index * 3 // count)
        for index, segment in enumerate(ranked)
    }


def feature_vector(
    segment: Segment,
    index: int,
    condition_names: Sequence[str],
    model_kind: str,
) -> np.ndarray:
    state = segment.state[index]
    values = [1.0, float(state[0]), float(state[1])]
    if model_kind == "explicit_decomp":
        if index:
            velocity = (segment.state[index] - segment.state[index - 1]) / (
                segment.time[index] - segment.time[index - 1]
            )
        else:
            velocity = np.zeros(2)
        radius = max(float(np.linalg.norm(state)), 1e-9)
        values.extend([float(np.linalg.norm(velocity)), state[0] / radius, state[1] / radius])
    values.extend(float(segment.conditions[name][index]) for name in condition_names)
    return np.asarray(values, dtype=float)


def build_transition_data(
    segments: Sequence[Segment],
    condition_names: Sequence[str],
    model_kind: str,
) -> TransitionData:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    modes: list[int] = []
    segment_ids: list[str] = []
    segment_modes = _balanced_segment_modes(segments)
    for segment in segments:
        mode = segment_modes[segment.segment_id]
        for index in range(len(segment.time) - 1):
            dt = float(segment.time[index + 1] - segment.time[index])
            features.append(feature_vector(segment, index, condition_names, model_kind))
            targets.append((segment.state[index + 1] - segment.state[index]) / dt)
            modes.append(mode)
            segment_ids.append(segment.segment_id)
    if not features:
        raise ValueError("transition builder received no samples")
    return TransitionData(
        features=np.stack(features),
        targets=np.stack(targets),
        mode_labels=np.asarray(modes, dtype=int),
        segment_ids=tuple(segment_ids),
    )


def _ridge_fit(features: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gram = features.T @ features + RIDGE * np.eye(features.shape[1])
    weights = np.linalg.solve(gram, features.T @ targets)
    residuals = targets - features @ weights
    if len(residuals) < 2:
        raise ValueError("at least two residuals are required")
    covariance = np.cov(residuals, rowvar=False)
    covariance = np.asarray(covariance, dtype=float).reshape(2, 2)
    covariance = 0.5 * (covariance + covariance.T) + COVARIANCE_JITTER * np.eye(2)
    return weights, covariance


def _base_fit(data: TransitionData, model_kind: str, condition_names: Sequence[str]) -> ModelState:
    n_modes = 1 if model_kind in {"single_gaussian", "explicit_decomp"} else 3
    if model_kind == "gmm_kernel":
        pooled_weights, _ = _ridge_fit(data.features, data.targets)
        residuals = data.targets - data.features @ pooled_weights
        # A transition-kernel mixture is identified from residual direction, not segment identity.
        labels = np.argmax(
            np.column_stack([residuals[:, 0], residuals[:, 1], -residuals.sum(axis=1)]),
            axis=1,
        )
    else:
        labels = data.mode_labels
    weights: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    counts: list[int] = []
    for mode in range(n_modes):
        mask = np.ones(data.count, dtype=bool) if n_modes == 1 else labels == mode
        if int(mask.sum()) < 2:
            raise ValueError(f"mode {mode} has fewer than two training transitions")
        weight, covariance = _ridge_fit(data.features[mask], data.targets[mask])
        weights.append(weight)
        covariances.append(covariance)
        counts.append(int(mask.sum()))
    probabilities = np.asarray(counts, dtype=float)
    probabilities /= probabilities.sum()
    return ModelState(
        model_kind=model_kind,
        condition_names=tuple(condition_names),
        weights=tuple(weights),
        covariances=tuple(covariances),
        mode_probabilities=probabilities,
        training_sample_count=data.count,
        validation_objective_before=float("nan"),
        validation_objective_after=float("nan"),
        covariance_scale=1.0,
    )


def _one_step_residuals(model: ModelState, data: TransitionData) -> np.ndarray:
    predictions = np.zeros_like(data.targets)
    for index, mode in enumerate(data.mode_labels):
        selected = min(int(mode), model.n_modes - 1)
        predictions[index] = data.features[index] @ model.weights[selected]
    return data.targets - predictions


def _energy_proxy(residuals: np.ndarray, covariances: Sequence[np.ndarray], scale: float) -> float:
    """Deterministic Gauss-Hermite evaluation of the proper d=2 energy score."""
    covariance = sum(covariances) / len(covariances) * scale * scale
    roots, weights = np.polynomial.hermite.hermgauss(10)
    grid = np.asarray([(x, y) for x in roots for y in roots])
    grid_weights = np.asarray([wx * wy for wx in weights for wy in weights]) / math.pi
    chol = np.linalg.cholesky(covariance + COVARIANCE_JITTER * np.eye(2))
    forecast_errors = math.sqrt(2.0) * grid @ chol.T
    pair_differences = 2.0 * grid @ chol.T
    second = float(np.sum(grid_weights * np.linalg.norm(pair_differences, axis=1)))
    scores = [
        float(np.sum(grid_weights * np.linalg.norm(forecast_errors - residual, axis=1)))
        - 0.5 * second
        for residual in residuals
    ]
    return float(np.mean(scores))


def _mean_log_likelihood(residuals: np.ndarray, covariance: np.ndarray) -> float:
    covariance = covariance + COVARIANCE_JITTER * np.eye(2)
    inverse = np.linalg.pinv(covariance)
    logdet = np.linalg.slogdet(covariance)[1]
    quadratic = np.einsum("ni,ij,nj->n", residuals, inverse, residuals)
    return float(np.mean(-0.5 * (2 * math.log(2 * math.pi) + logdet + quadratic)))


def calibrate_estimator(
    model: ModelState,
    validation: TransitionData,
    estimator: str,
    objective_lambda: object = None,
) -> ModelState:
    residuals = _one_step_residuals(model, validation)
    baseline_energy = _energy_proxy(residuals, model.covariances, 1.0)
    baseline_ll = _mean_log_likelihood(residuals, sum(model.covariances) / model.n_modes)
    if estimator == "qmle":
        return replace(
            model,
            validation_objective_before=-baseline_ll,
            validation_objective_after=-baseline_ll,
            estimator_method="qmle",
            estimator_optimization_scope="closed_form_linear_gaussian_fit",
        )
    candidates = ESTIMATOR_SCALE_CANDIDATES
    if estimator in {"mixed", "pure_es"}:
        validation_fit = _base_fit(
            validation,
            model.model_kind,
            model.condition_names,
        )
        scored_joint: list[tuple[float, float, float, tuple[np.ndarray, ...]]] = []
        for drift_fraction in ESTIMATOR_DRIFT_FRACTIONS:
            candidate_weights = tuple(
                (1.0 - drift_fraction) * trained + drift_fraction * validation_weight
                for trained, validation_weight in zip(model.weights, validation_fit.weights)
            )
            candidate_model = replace(model, weights=candidate_weights)
            candidate_residuals = _one_step_residuals(candidate_model, validation)
            for scale in candidates:
                energy = _energy_proxy(candidate_residuals, model.covariances, scale)
                scaled_covariance = (
                    sum(model.covariances) / model.n_modes * scale * scale
                )
                nll = -_mean_log_likelihood(candidate_residuals, scaled_covariance)
                if estimator == "mixed":
                    lam = float(objective_lambda if objective_lambda is not None else 0.5)
                    objective = (
                        lam * energy / max(baseline_energy, 1e-12)
                        + (1.0 - lam) * nll / max(-baseline_ll, 1e-12)
                    )
                else:
                    objective = energy
                scored_joint.append(
                    (objective, drift_fraction, scale, candidate_weights)
                )
        objective, drift_fraction, scale, selected_weights = min(
            scored_joint,
            key=lambda item: (item[0], item[1], item[2]),
        )
        scaled = tuple(covariance * scale * scale for covariance in model.covariances)
        baseline_objective = (
            baseline_energy
            if estimator == "pure_es"
            else next(
                value
                for value, fraction, item_scale, _ in scored_joint
                if fraction == 0.0 and item_scale == 1.0
            )
        )
        return replace(
            model,
            weights=selected_weights,
            covariances=scaled,
            covariance_scale=scale,
            validation_objective_before=float(baseline_objective),
            validation_objective_after=float(objective),
            estimator_method=estimator,
            estimator_optimization_scope="joint_drift_covariance_validation_grid",
            estimator_drift_fraction=drift_fraction,
            estimator_candidate_count=len(scored_joint),
        )
    scored: list[tuple[float, float]] = []
    for scale in candidates:
        energy = _energy_proxy(residuals, model.covariances, scale)
        scaled_covariance = sum(model.covariances) / model.n_modes * scale * scale
        nll = -_mean_log_likelihood(residuals, scaled_covariance)
        if estimator == "mixed":
            lam = float(objective_lambda if objective_lambda is not None else 0.5)
            objective = lam * energy / max(baseline_energy, 1e-12) + (1.0 - lam) * nll / max(-baseline_ll, 1e-12)
        else:
            objective = energy
        scored.append((objective, scale))
    objective, scale = min(scored)
    scaled = tuple(covariance * scale * scale for covariance in model.covariances)
    baseline_objective = baseline_energy if estimator != "mixed" else next(value for value, item_scale in scored if item_scale == 1.0)
    return replace(
        model,
        covariances=scaled,
        covariance_scale=scale,
        validation_objective_before=float(baseline_objective),
        validation_objective_after=float(objective),
        estimator_method=estimator,
        estimator_optimization_scope="registered_covariance_scale_grid",
        estimator_candidate_count=len(scored),
    )


def _blend_models(
    pretrained: ModelState,
    adapted: ModelState,
    *,
    finetune: str,
    weight: float = 0.65,
) -> ModelState:
    if pretrained.n_modes != adapted.n_modes or pretrained.feature_count != adapted.feature_count:
        raise ValueError("pretraining and adaptation model shapes differ")
    weights = tuple(
        (1.0 - weight) * old + weight * new
        for old, new in zip(pretrained.weights, adapted.weights)
    )
    if finetune == "drift_only":
        covariances = pretrained.covariances
    elif finetune in {"two_step", "all"}:
        covariances = tuple(
            (1.0 - weight) * old + weight * new
            for old, new in zip(pretrained.covariances, adapted.covariances)
        )
    else:
        raise ValueError(f"unknown finetune method: {finetune}")
    delta = math.sqrt(sum(float(np.sum((new - old) ** 2)) for old, new in zip(pretrained.weights, weights)))
    return replace(
        pretrained,
        weights=weights,
        covariances=covariances,
        mode_probabilities=(1.0 - weight) * pretrained.mode_probabilities + weight * adapted.mode_probabilities,
        training_sample_count=pretrained.training_sample_count + adapted.training_sample_count,
        adaptation_parameter_delta=delta,
    )


def _reptile_inner_adapt(initialization: ModelState, task: TransitionData) -> ModelState:
    """Take stable task-loss gradient steps starting from the current initialization."""
    weights = [value.copy() for value in initialization.weights]
    for _ in range(REPTILE_INNER_STEPS):
        for mode in range(initialization.n_modes):
            mask = (
                np.ones(task.count, dtype=bool)
                if initialization.n_modes == 1
                else task.mode_labels == mode
            )
            if int(mask.sum()) < 2:
                continue
            features = task.features[mask]
            targets = task.targets[mask]
            lipschitz = (
                float(np.linalg.norm(features, ord=2) ** 2) / len(features) + RIDGE
            )
            gradient = (
                features.T @ (features @ weights[mode] - targets) / len(features)
                + RIDGE * weights[mode]
            )
            weights[mode] = weights[mode] - REPTILE_INNER_RATE / lipschitz * gradient

    covariances: list[np.ndarray] = []
    counts: list[int] = []
    for mode in range(initialization.n_modes):
        mask = (
            np.ones(task.count, dtype=bool)
            if initialization.n_modes == 1
            else task.mode_labels == mode
        )
        count = int(mask.sum())
        counts.append(count)
        if count < 2:
            covariances.append(initialization.covariances[mode])
            continue
        residuals = task.targets[mask] - task.features[mask] @ weights[mode]
        covariance = np.asarray(np.cov(residuals, rowvar=False), dtype=float).reshape(2, 2)
        covariances.append(
            0.5 * (covariance + covariance.T) + COVARIANCE_JITTER * np.eye(2)
        )
    probabilities = np.asarray(counts, dtype=float)
    probabilities /= probabilities.sum()
    return replace(
        initialization,
        weights=tuple(weights),
        covariances=tuple(covariances),
        mode_probabilities=probabilities,
        training_sample_count=task.count,
    )


def _reptile_initialization(
    initialization: ModelState,
    tasks: Sequence[TransitionData],
) -> ModelState:
    """Run first-order Reptile with task inner loops and outer interpolation."""
    current = initialization
    inner_objectives_before: list[float] = []
    inner_objectives_after: list[float] = []
    for epoch in range(REPTILE_META_EPOCHS):
        step = REPTILE_INITIAL_STEP / math.sqrt(epoch + 1.0)
        for task in tasks:
            before_residuals = _one_step_residuals(current, task)
            adapted = _reptile_inner_adapt(current, task)
            after_residuals = _one_step_residuals(adapted, task)
            inner_objectives_before.append(
                float(np.mean(np.sum(before_residuals**2, axis=1)))
            )
            inner_objectives_after.append(
                float(np.mean(np.sum(after_residuals**2, axis=1)))
            )
            weights = tuple(
                old + step * (target - old)
                for old, target in zip(current.weights, adapted.weights)
            )
            covariances = tuple(
                old + step * (target - old)
                for old, target in zip(current.covariances, adapted.covariances)
            )
            probabilities = current.mode_probabilities + step * (
                adapted.mode_probabilities - current.mode_probabilities
            )
            probabilities /= probabilities.sum()
            current = replace(
                current,
                weights=weights,
                covariances=covariances,
                mode_probabilities=probabilities,
            )
    return replace(
        current,
        training_sample_count=initialization.training_sample_count
        + REPTILE_META_EPOCHS * sum(task.count for task in tasks),
        meta_task_count=len(tasks),
        meta_algorithm="first_order_reptile_gradient_inner_loop",
        meta_outer_epochs=REPTILE_META_EPOCHS,
        meta_inner_steps=REPTILE_INNER_STEPS,
        meta_inner_rate=REPTILE_INNER_RATE,
        meta_inner_objective_before=float(np.mean(inner_objectives_before)),
        meta_inner_objective_after=float(np.mean(inner_objectives_after)),
    )


def train_model(
    train_segments: Sequence[Segment],
    validation_segments: Sequence[Segment],
    adapt_segments: Sequence[Segment],
    animal_segments: Sequence[Segment],
    config: Mapping[str, object],
) -> ModelState:
    model_kind = str(config["model"])
    condition_names = tuple(str(name) for name in config.get("condition", []))
    transfer = str(config.get("transfer", "full_finetune"))
    finetune = str(config.get("finetune", "all"))

    def fit(segments: Sequence[Segment]) -> ModelState:
        return _base_fit(
            build_transition_data(segments, condition_names, model_kind),
            model_kind,
            condition_names,
        )

    if transfer == "scratch":
        model = fit(adapt_segments)
    elif transfer == "animal_pretrain":
        model = _blend_models(fit(animal_segments), fit(adapt_segments), finetune=finetune)
    elif transfer == "meta_reptile":
        task_segments = {
            region: [segment for segment in train_segments if segment.region == region]
            for region in sorted({segment.region for segment in train_segments})
        }
        eligible_tasks = [
            segments
            for segments in task_segments.values()
            if len(segments) >= MIN_META_TASK_SEGMENTS
        ]
        if len(eligible_tasks) < 2:
            raise ValueError("meta_reptile requires at least two eligible tasks")
        tasks = [
            build_transition_data(segments, condition_names, model_kind)
            for segments in eligible_tasks
        ]
        meta_initialization = _reptile_initialization(fit(train_segments), tasks)
        model = _blend_models(meta_initialization, fit(adapt_segments), finetune=finetune)
    else:
        model = _blend_models(fit(train_segments), fit(adapt_segments), finetune=finetune)
    model = replace(
        model,
        transfer_method=transfer,
        finetune_method=finetune,
    )
    validation = build_transition_data(validation_segments, condition_names, model_kind)
    return calibrate_estimator(
        model,
        validation,
        str(config.get("estimator", "crps_energy")),
        config.get("objective_lambda"),
    )


__all__ = [
    "ModelState",
    "MIN_META_TASK_SEGMENTS",
    "ESTIMATOR_DRIFT_FRACTIONS",
    "ESTIMATOR_SCALE_CANDIDATES",
    "REPTILE_INITIAL_STEP",
    "REPTILE_INNER_RATE",
    "REPTILE_INNER_STEPS",
    "REPTILE_META_EPOCHS",
    "TransitionData",
    "build_transition_data",
    "calibrate_estimator",
    "feature_vector",
    "train_model",
]
