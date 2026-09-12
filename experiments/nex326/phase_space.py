"""Supplemental four-dimensional underdamped benchmark for NEX326 cohorts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np

from .cohort import Cohort, Segment, load_cohort


SPEC_PATH = Path(__file__).with_name("phase_space_benchmark.json")
REPORT_SCHEMA_VERSION = "nex326-phase-space-benchmark-report-v1"
IMPLEMENTATION_FILES = (
    "cohort.py",
    "environment.lock.json",
    "phase_space.py",
    "phase_space_benchmark.json",
    "phase_space_terrain_benchmark.json",
    "spatial_conditions.py",
)


class PhaseSpaceError(ValueError):
    """A phase-space execution violates the registered four-dimensional contract."""


class ConditionField(Protocol):
    """Spatial condition lookup used during off-observation rollout."""

    @property
    def names(self) -> tuple[str, ...]: ...

    def evaluate(self, position: np.ndarray, time: float) -> np.ndarray: ...


class ConditionFieldResolver(Protocol):
    """Provide the coordinate-aware field belonging to one trajectory file."""

    @property
    def names(self) -> tuple[str, ...]: ...

    def for_segment(self, segment: Segment) -> ConditionField: ...

    def identity(self) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class AffineVelocityModel:
    """Coupled affine velocity SDE with diffusion confined to velocity."""

    condition_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    diffusion_covariance: np.ndarray
    transition_count: int

    def acceleration(
        self,
        velocity: np.ndarray,
        conditions: np.ndarray | None = None,
    ) -> np.ndarray:
        velocity_values = np.asarray(velocity, dtype=float)
        one = velocity_values.ndim == 1
        velocity_matrix = np.atleast_2d(velocity_values)
        if velocity_matrix.shape[1] != 2:
            raise PhaseSpaceError("velocity must have shape (..., 2)")
        if self.condition_names:
            if conditions is None:
                raise PhaseSpaceError("registered conditions require a spatial field")
            condition_matrix = np.atleast_2d(np.asarray(conditions, dtype=float))
            if condition_matrix.shape != (len(velocity_matrix), len(self.condition_names)):
                raise PhaseSpaceError("condition field returned an incompatible shape")
            raw = np.column_stack([velocity_matrix, condition_matrix])
        else:
            raw = velocity_matrix
        normalized = (raw - self.feature_mean) / self.feature_scale
        design = np.column_stack([np.ones(len(normalized)), normalized])
        result = design @ self.weights
        return result[0] if one else result

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "coupled_affine_velocity_ou",
            "condition_names": list(self.condition_names),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "weights": self.weights.tolist(),
            "diffusion_covariance": self.diffusion_covariance.tolist(),
            "transition_count": self.transition_count,
            "diffusion_state_support": ["vx", "vy"],
        }


@dataclass(frozen=True)
class PhaseSpacePrediction:
    segment_id: str
    target_state: np.ndarray
    paths: np.ndarray
    cutoff: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_identity() -> dict[str, object]:
    root = Path(__file__).resolve().parent
    files = [
        {"path": name, "sha256": _sha256(root / name)}
        for name in IMPLEMENTATION_FILES
    ]
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    lock = json.loads((root / "environment.lock.json").read_text(encoding="utf-8"))
    runtime = {
        "python": platform.python_version(),
        **{
            package: importlib.metadata.version(package)
            for package in sorted(lock["packages"])
        },
    }
    return {
        "source_bundle_sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
        "runtime": runtime,
        "environment_lock_conformant": runtime
        == {"python": lock["python"], **dict(sorted(lock["packages"].items()))},
    }


def load_phase_space_spec(path: Path | str = SPEC_PATH) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "nex326-phase-space-benchmark-spec-v1":
        raise PhaseSpaceError("unsupported phase-space specification")
    state = payload.get("state_contract")
    if not isinstance(state, Mapping) or state.get("layout") != ["x", "y", "vx", "vy"]:
        raise PhaseSpaceError("phase-space state layout must be [x,y,vx,vy]")
    if state.get("position_dynamics") != "dX=Vdt" or state.get("noise_target") != "velocity_only":
        raise PhaseSpaceError("phase-space kinematic structure is not registered")
    condition = payload.get("condition_contract")
    if not isinstance(condition, Mapping) or condition.get("condition_is_dynamic_state") is not False:
        raise PhaseSpaceError("conditions must remain external to the dynamic state")
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("fit_splits") != ["train", "adapt"]:
        raise PhaseSpaceError("unexpected phase-space split protocol")
    return payload


def phase_space_state(segment: Segment) -> np.ndarray:
    """Create causal ``[x,y,vx,vy]`` states from irregular position observations."""
    if len(segment.time) < 3:
        raise PhaseSpaceError("phase-space conversion requires at least three observations")
    elapsed = np.diff(segment.time)
    if not np.isfinite(elapsed).all() or np.any(elapsed <= 0.0):
        raise PhaseSpaceError("phase-space conversion requires positive finite intervals")
    interval_velocity = np.diff(segment.state, axis=0) / elapsed[:, None]
    velocity = np.empty_like(segment.state)
    velocity[1:] = interval_velocity
    velocity[0] = interval_velocity[0]
    state = np.column_stack([segment.state, velocity])
    if state.shape != (len(segment.time), 4) or not np.isfinite(state).all():
        raise PhaseSpaceError("derived phase-space state is invalid")
    return state


def _transition_rows(
    segments: Sequence[Segment],
    condition_names: Sequence[str],
    condition_resolver: ConditionFieldResolver | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    increments: list[np.ndarray] = []
    elapsed_values: list[float] = []
    for segment in segments:
        phase = phase_space_state(segment)
        field = (
            condition_resolver.for_segment(segment)
            if condition_resolver is not None
            else None
        )
        for index in range(1, len(segment.time) - 1):
            elapsed = float(segment.time[index + 1] - segment.time[index])
            if field is not None:
                condition = field.evaluate(
                    phase[index : index + 1, :2], float(segment.time[index])
                )[0]
            else:
                condition = np.asarray(
                    [segment.conditions[name][index] for name in condition_names],
                    dtype=float,
                )
            features.append(np.concatenate([phase[index, 2:], condition]))
            increments.append(phase[index + 1, 2:] - phase[index, 2:])
            elapsed_values.append(elapsed)
    if len(features) < 3:
        raise PhaseSpaceError("phase-space fit requires at least three velocity transitions")
    return np.stack(features), np.stack(increments), np.asarray(elapsed_values)


def fit_affine_velocity_model(
    segments: Sequence[Segment],
    *,
    condition_names: Sequence[str] = (),
    condition_resolver: ConditionFieldResolver | None = None,
    ridge: float = 1e-6,
) -> AffineVelocityModel:
    """Fit ``dV=f(V,C)dt+GdW`` without learning the kinematic position row."""
    if ridge <= 0.0:
        raise PhaseSpaceError("ridge must be positive")
    names = tuple(str(name) for name in condition_names)
    if condition_resolver is not None and condition_resolver.names != names:
        raise PhaseSpaceError("condition resolver names do not match the registered model")
    if condition_resolver is None and any(
        any(name not in segment.conditions for name in names) for segment in segments
    ):
        raise PhaseSpaceError("a registered training condition is unavailable")
    raw, velocity_increment, elapsed = _transition_rows(
        segments, names, condition_resolver
    )
    feature_mean = raw.mean(axis=0)
    feature_scale = raw.std(axis=0)
    feature_scale[feature_scale < 1e-10] = 1.0
    normalized = (raw - feature_mean) / feature_scale
    rate_design = np.column_stack([np.ones(len(normalized)), normalized])
    increment_design = rate_design * elapsed[:, None]
    penalty = ridge * np.eye(increment_design.shape[1])
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(
        increment_design.T @ increment_design + penalty,
        increment_design.T @ velocity_increment,
    )
    residual = velocity_increment - increment_design @ weights
    diffusion = np.einsum("ni,nj->ij", residual, residual) / elapsed.sum()
    diffusion = 0.5 * (diffusion + diffusion.T) + 1e-10 * np.eye(2)
    if float(np.linalg.eigvalsh(diffusion).min()) <= 0.0:
        raise PhaseSpaceError("velocity diffusion fit is not positive definite")
    return AffineVelocityModel(
        condition_names=names,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        weights=weights,
        diffusion_covariance=diffusion,
        transition_count=len(raw),
    )


def rollout_phase_space(
    model: AffineVelocityModel,
    segment: Segment,
    *,
    cutoff: int,
    n_samples: int,
    rng: np.random.Generator,
    condition_field: ConditionField | None = None,
) -> np.ndarray:
    """Roll out four-dimensional paths while preserving ``dX=Vdt`` by construction."""
    if n_samples < 2:
        raise PhaseSpaceError("phase-space rollout requires at least two samples")
    if not 2 <= cutoff < len(segment.time) - 1:
        raise PhaseSpaceError("phase-space evaluation cutoff is invalid")
    if model.condition_names and (
        condition_field is None or condition_field.names != model.condition_names
    ):
        raise PhaseSpaceError("rollout requires the registered spatial condition field")
    initial = phase_space_state(segment)[cutoff]
    step_count = len(segment.time) - cutoff - 1
    paths = np.empty((n_samples, step_count + 1, 4), dtype=float)
    paths[:, 0, :] = initial
    for offset, index in enumerate(range(cutoff, len(segment.time) - 1), start=1):
        elapsed = float(segment.time[index + 1] - segment.time[index])
        current = paths[:, offset - 1, :]
        conditions = (
            condition_field.evaluate(current[:, :2], float(segment.time[index]))
            if condition_field is not None
            else None
        )
        acceleration = model.acceleration(current[:, 2:], conditions)
        noise = rng.multivariate_normal(
            np.zeros(2), model.diffusion_covariance * elapsed, size=n_samples
        )
        next_velocity = current[:, 2:] + acceleration * elapsed + noise
        next_position = current[:, :2] + 0.5 * (
            current[:, 2:] + next_velocity
        ) * elapsed
        paths[:, offset, :2] = next_position
        paths[:, offset, 2:] = next_velocity
    if not np.isfinite(paths).all():
        raise PhaseSpaceError("phase-space rollout produced non-finite states")
    return paths


def _energy_score(samples: np.ndarray, target: np.ndarray) -> float:
    first = np.linalg.norm(samples - target, axis=1).mean()
    second = np.linalg.norm(samples - np.roll(samples, 1, axis=0), axis=1).mean()
    return float(first - 0.5 * second)


def _validation_velocity_rmse(
    model: AffineVelocityModel,
    segments: Sequence[Segment],
    condition_resolver: ConditionFieldResolver | None = None,
) -> float:
    raw, increments, elapsed = _transition_rows(
        segments, model.condition_names, condition_resolver
    )
    conditions = raw[:, 2:] if model.condition_names else None
    predicted = model.acceleration(raw[:, :2], conditions) * elapsed[:, None]
    return float(np.sqrt(np.mean((predicted - increments) ** 2)))


def _prediction_metrics(
    predictions: Sequence[PhaseSpacePrediction],
) -> tuple[dict[str, float | int], list[dict[str, object]]]:
    energy: list[float] = []
    coverage: list[float] = []
    position_error: list[float] = []
    velocity_error: list[float] = []
    rows: list[dict[str, object]] = []
    for prediction in predictions:
        endpoints = prediction.paths[:, -1, :]
        position = endpoints[:, :2]
        target_position = prediction.target_state[:2]
        center = position.mean(axis=0)
        radii = np.linalg.norm(position - center, axis=1)
        target_radius = float(np.linalg.norm(target_position - center))
        segment_energy = _energy_score(position, target_position)
        segment_position_error = target_radius
        segment_velocity_error = float(
            np.linalg.norm(endpoints[:, 2:].mean(axis=0) - prediction.target_state[2:])
        )
        energy.append(segment_energy)
        coverage.append(float(target_radius <= np.quantile(radii, 0.9)))
        position_error.append(segment_position_error)
        velocity_error.append(segment_velocity_error)
        rows.append(
            {
                "segment_id": prediction.segment_id,
                "cutoff": prediction.cutoff,
                "position_energy_score_d2": segment_energy,
                "position_endpoint_error": segment_position_error,
                "velocity_endpoint_error": segment_velocity_error,
            }
        )
    return (
        {
            "position_energy_score_d2": float(np.mean(energy)),
            "position_hdr90_coverage": float(np.mean(coverage)),
            "position_cep50_error": float(np.median(position_error)),
            "velocity_endpoint_rmse": float(
                np.sqrt(np.mean(np.square(velocity_error)))
            ),
            "evaluation_segment_count": len(predictions),
        },
        rows,
    )


def run_phase_space_benchmark(
    cohort: Cohort,
    spec: Mapping[str, object],
    *,
    n_samples: int,
    seed: int,
    condition_resolver: ConditionFieldResolver | None = None,
) -> dict[str, object]:
    """Run a supplemental four-dimensional benchmark with optional spatial fields."""
    velocity_config = spec["velocity_model"]
    condition_config = spec["condition_contract"]
    condition_names = tuple(condition_config["registered_condition_names"])
    if condition_names and condition_resolver is None:
        raise PhaseSpaceError("registered conditions require a spatial condition resolver")
    if condition_resolver is not None and condition_resolver.names != condition_names:
        raise PhaseSpaceError("condition resolver names do not match the specification")
    fit_segments = cohort.splits["train"] + cohort.splits["adapt"]
    model = fit_affine_velocity_model(
        fit_segments,
        condition_names=condition_names,
        condition_resolver=condition_resolver,
        ridge=float(velocity_config["ridge"]),
    )
    rng = np.random.default_rng(seed)
    predictions: list[PhaseSpacePrediction] = []
    for segment in cohort.splits["evaluation"]:
        cutoff = max(2, len(segment.time) // 2 - 1)
        condition_field = (
            condition_resolver.for_segment(segment)
            if condition_resolver is not None
            else None
        )
        paths = rollout_phase_space(
            model,
            segment,
            cutoff=cutoff,
            n_samples=n_samples,
            rng=rng,
            condition_field=condition_field,
        )
        predictions.append(
            PhaseSpacePrediction(
                segment_id=segment.segment_id,
                target_state=phase_space_state(segment)[-1],
                paths=paths,
                cutoff=cutoff,
            )
        )
    metrics, per_segment = _prediction_metrics(predictions)
    # Recompute the exact irregular-dt identity with the registered segment times.
    exact_errors: list[float] = []
    for prediction, segment in zip(predictions, cohort.splits["evaluation"]):
        elapsed = np.diff(segment.time[prediction.cutoff :])
        delta_position = np.diff(prediction.paths[:, :, :2], axis=1)
        average_velocity = 0.5 * (
            prediction.paths[:, :-1, 2:] + prediction.paths[:, 1:, 2:]
        )
        exact_errors.append(
            float(np.max(np.abs(delta_position - average_velocity * elapsed[None, :, None])))
        )
    metrics["kinematic_identity_max_error"] = float(max(exact_errors))
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "benchmark_id": spec["benchmark_id"],
        "scientific_role": spec["scientific_role"],
        "verdict": "not_assessed",
        "base_benchmark": spec["base_benchmark"],
        "implementation": _implementation_identity(),
        "dataset": {
            "dataset_id": cohort.dataset_id,
            "data_version": cohort.data_version,
            "fingerprint": cohort.fingerprint,
            "purpose": cohort.purpose,
            "split_counts": {name: len(values) for name, values in cohort.splits.items()},
        },
        "state_contract": spec["state_contract"],
        "condition_contract": spec["condition_contract"],
        "condition_field": (
            condition_resolver.identity() if condition_resolver is not None else None
        ),
        "protocol": {
            **spec["protocol"],
            "executed_seed": seed,
            "executed_prediction_samples": n_samples,
        },
        "model": model.to_dict(),
        "validation": {
            "velocity_increment_rmse": _validation_velocity_rmse(
                model, cohort.splits["validation"], condition_resolver
            )
        },
        "metrics": metrics,
        "per_segment": per_segment,
    }


def write_phase_space_report(
    cohort_path: Path | str,
    output_path: Path | str,
    *,
    spec_path: Path | str = SPEC_PATH,
    n_samples: int | None = None,
    seed: int | None = None,
    condition_resolver: ConditionFieldResolver | None = None,
) -> dict[str, object]:
    destination = Path(output_path)
    if destination.exists():
        raise PhaseSpaceError(f"output already exists: {destination}")
    spec = load_phase_space_spec(spec_path)
    protocol = spec["protocol"]
    selected_samples = int(
        protocol["prediction_samples"] if n_samples is None else n_samples
    )
    selected_seed = int(protocol["seed"] if seed is None else seed)
    report = run_phase_space_benchmark(
        load_cohort(cohort_path),
        spec,
        n_samples=selected_samples,
        seed=selected_seed,
        condition_resolver=condition_resolver,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = [
    "AffineVelocityModel",
    "ConditionField",
    "ConditionFieldResolver",
    "PhaseSpaceError",
    "fit_affine_velocity_model",
    "load_phase_space_spec",
    "phase_space_state",
    "rollout_phase_space",
    "run_phase_space_benchmark",
    "write_phase_space_report",
]
