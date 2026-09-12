"""Auditable end-to-end runner for all NEX326 arm/subconfig executions."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .cohort import Cohort, Segment, load_cohort, select_segments
from .model import (
    MIN_META_TASK_SEGMENTS,
    ModelState,
    build_transition_data,
    feature_vector,
    train_model,
)
from .schrodinger import SchrodingerBridgeError, solve_particle_schrodinger_bridge
from .specification import ArmSpec, ExperimentSpec, load_experiment_spec


RUN_RECORD_VERSION = "nex326-run-record-v2"
STAGES = (
    "load_versioned_data",
    "build_transition_samples",
    "adapt_features",
    "initialize_model",
    "train",
    "checkpoint",
    "inference",
    "metrics",
    "mechanism_gates",
    "run_record",
    "tsde_ready",
)
IMPLEMENTATION_FILES = (
    "cohort.py",
    "environment.lock.json",
    "experiment.json",
    "model.py",
    "runner.py",
    "schrodinger.py",
    "specification.py",
)


class RunError(RuntimeError):
    """One execution cannot complete the registered process."""


class DataUnavailable(RunError):
    """Registered external data are absent, while the implementation remains usable."""

    def __init__(self, stage: str, reason: str) -> None:
        super().__init__(reason)
        self.stage = stage
        self.reason = reason


@dataclass(frozen=True)
class Prediction:
    segment_id: str
    target: np.ndarray
    samples: np.ndarray
    prior_mean: np.ndarray | None
    prior_source: str | None
    unbridged_prior_distance: float | None
    bridged_prior_distance: float | None
    bridge_path: np.ndarray | None
    bridge_diagnostics: Mapping[str, object] | None

    def to_dict(self) -> dict[str, object]:
        return {
            "segment_id": self.segment_id,
            "target": self.target.tolist(),
            "samples": self.samples.tolist(),
            "endpoint_prior_mean": self.prior_mean.tolist() if self.prior_mean is not None else None,
            "endpoint_prior_source": self.prior_source,
            "bridge_path": self.bridge_path.tolist() if self.bridge_path is not None else None,
            "bridge_diagnostics": (
                dict(self.bridge_diagnostics)
                if self.bridge_diagnostics is not None
                else None
            ),
        }


def _json_dump(payload: object, destination: Path) -> None:
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(base: int, arm_id: int, subconfig_id: str) -> int:
    suffix = int(hashlib.sha256(subconfig_id.encode()).hexdigest()[:8], 16)
    return (base + arm_id * 1009 + suffix) % (2**32 - 1)


def implementation_identity() -> dict[str, object]:
    """Hash the exact local sources and runtime versions that determine an execution."""
    root = Path(__file__).resolve().parent
    files = [
        {"path": name, "sha256": _sha256(root / name)}
        for name in IMPLEMENTATION_FILES
    ]
    canonical_sources = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    source_bundle = hashlib.sha256(canonical_sources).hexdigest()
    lock = json.loads((root / "environment.lock.json").read_text(encoding="utf-8"))
    runtime = {
        "python": platform.python_version(),
        **{
            package: importlib.metadata.version(package)
            for package in sorted(lock["packages"])
        },
    }
    lock_conformant = runtime == {
        "python": lock["python"],
        **dict(sorted(lock["packages"].items())),
    }
    canonical_execution = json.dumps(
        {"source_bundle_sha256": source_bundle, "runtime": runtime},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "source_bundle_sha256": source_bundle,
        "execution_identity_sha256": hashlib.sha256(canonical_execution).hexdigest(),
        "files": files,
        "environment_lock": {
            "path": "environment.lock.json",
            "sha256": _sha256(root / "environment.lock.json"),
            "conformant": lock_conformant,
        },
        "runtime": runtime,
    }


def validate_run_record(record: Mapping[str, object]) -> None:
    """Validate the runtime subset of ``run_record.schema.json`` before commit."""
    required = {
        "schema_version",
        "experiment_id",
        "spec_version",
        "run_id",
        "result_id",
        "arm_id",
        "subconfig_id",
        "group",
        "lineage",
        "implementation_status",
        "implementation",
        "run_status",
        "verdict",
        "dataset",
        "config",
        "runtime",
        "seed",
        "replicate_seed",
        "stages",
        "artifacts",
        "metrics",
        "mechanism_gates",
    }
    missing = required - record.keys()
    if missing:
        raise RunError(f"RunRecord missing fields: {sorted(missing)}")
    if record["schema_version"] != RUN_RECORD_VERSION:
        raise RunError("RunRecord schema version mismatch")
    if record["experiment_id"] != "NEX326" or record["spec_version"] != "nex326-process-v2":
        raise RunError("RunRecord experiment identity mismatch")
    arm_id = int(record["arm_id"])
    if arm_id not in range(1, 23):
        raise RunError("RunRecord arm_id is outside 1..22")
    if record["implementation_status"] != "implemented":
        raise RunError("PIRC-19 records may not claim implementation_missing")
    implementation = record["implementation"]
    if not isinstance(implementation, Mapping):
        raise RunError("RunRecord implementation identity must be a mapping")
    source_bundle = implementation.get("source_bundle_sha256")
    execution_identity = implementation.get("execution_identity_sha256")
    source_files = implementation.get("files")
    if (
        not isinstance(source_bundle, str)
        or len(source_bundle) != 64
        or any(character not in "0123456789abcdef" for character in source_bundle)
        or not isinstance(source_files, list)
        or not source_files
        or not isinstance(execution_identity, str)
        or len(execution_identity) != 64
        or any(character not in "0123456789abcdef" for character in execution_identity)
    ):
        raise RunError("RunRecord implementation identity is incomplete")
    if any(
        not isinstance(source, Mapping)
        or not isinstance(source.get("path"), str)
        or not source["path"]
        or not isinstance(source.get("sha256"), str)
        or len(source["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in source["sha256"])
        for source in source_files
    ):
        raise RunError("RunRecord implementation source references are invalid")
    canonical_sources = json.dumps(
        source_files, sort_keys=True, separators=(",", ":")
    ).encode()
    if hashlib.sha256(canonical_sources).hexdigest() != source_bundle:
        raise RunError("RunRecord implementation source bundle hash does not match its files")
    runtime_versions = implementation.get("runtime")
    if (
        not isinstance(runtime_versions, Mapping)
        or not isinstance(runtime_versions.get("python"), str)
        or not all(isinstance(value, str) and value for value in runtime_versions.values())
    ):
        raise RunError("RunRecord implementation runtime versions are incomplete")
    canonical_execution = json.dumps(
        {"source_bundle_sha256": source_bundle, "runtime": dict(runtime_versions)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if hashlib.sha256(canonical_execution).hexdigest() != execution_identity:
        raise RunError("RunRecord execution identity does not match source and runtime")
    environment_lock = implementation.get("environment_lock")
    if (
        not isinstance(environment_lock, Mapping)
        or environment_lock.get("path") != "environment.lock.json"
        or not isinstance(environment_lock.get("sha256"), str)
        or len(environment_lock["sha256"]) != 64
        or not isinstance(environment_lock.get("conformant"), bool)
    ):
        raise RunError("RunRecord environment lock identity is incomplete")
    lock_sources = [
        source
        for source in source_files
        if source.get("path") == environment_lock["path"]
    ]
    if (
        len(lock_sources) != 1
        or lock_sources[0]["sha256"] != environment_lock["sha256"]
    ):
        raise RunError("RunRecord environment lock does not match its source reference")
    replicate_seed = record["replicate_seed"]
    if (
        isinstance(replicate_seed, bool)
        or not isinstance(replicate_seed, int)
        or not 0 <= replicate_seed < 2**32
    ):
        raise RunError("RunRecord replicate_seed must be an integer in [0, 2**32)")
    status = str(record["run_status"])
    if status not in {"succeeded", "failed", "data_unavailable"}:
        raise RunError(f"invalid run status: {status}")
    if record["verdict"] not in {
        "not_assessed",
        "retain",
        "redundant",
        "harmful",
        "inconclusive",
        "unavailable",
    }:
        raise RunError("invalid scientific verdict")
    runtime = record["runtime"]
    if not isinstance(runtime, Mapping):
        raise RunError("RunRecord runtime must be a mapping")
    requested_samples = runtime.get("requested_prediction_samples")
    effective_samples = runtime.get("effective_prediction_samples")
    if (
        isinstance(requested_samples, bool)
        or not isinstance(requested_samples, int)
        or requested_samples <= 0
    ):
        raise RunError("RunRecord requested prediction samples must be a positive integer")
    if status == "succeeded":
        if (
            isinstance(effective_samples, bool)
            or not isinstance(effective_samples, int)
            or effective_samples < requested_samples
        ):
            raise RunError(
                "successful RunRecord effective prediction samples must be at least requested"
            )
    elif effective_samples is not None:
        raise RunError("non-successful RunRecord cannot claim effective prediction samples")
    stages = record["stages"]
    if (
        not isinstance(stages, list)
        or not all(isinstance(item, Mapping) for item in stages)
        or [item.get("name") for item in stages] != list(STAGES)
    ):
        raise RunError("RunRecord stages do not match the standard process")
    artifacts = record["artifacts"]
    gates = record["mechanism_gates"]
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise RunError("RunRecord artifacts must be a non-empty mapping")
    if not isinstance(gates, list) or not gates:
        raise RunError("RunRecord mechanism_gates must be a non-empty list")
    if status == "succeeded":
        if any(item.get("status") != "completed" for item in stages):
            raise RunError("successful RunRecord contains an incomplete stage")
        if not {"checkpoint", "predictions", "metrics", "mechanism"} <= artifacts.keys():
            raise RunError("successful RunRecord lacks required artifacts")
        if not record["metrics"] or not gates:
            raise RunError("successful RunRecord lacks metrics or mechanism gates")
        if record["verdict"] == "unavailable":
            raise RunError("successful RunRecord cannot have an unavailable verdict")
    elif status == "data_unavailable":
        failure = record.get("failure")
        if not isinstance(failure, Mapping) or not failure.get("reason"):
            raise RunError("data_unavailable RunRecord requires a structured reason")
        if "availability" not in artifacts:
            raise RunError("data_unavailable RunRecord requires an availability artifact")
        if record["verdict"] != "unavailable":
            raise RunError("data_unavailable RunRecord requires an unavailable verdict")
    for name, reference in artifacts.items():
        if (
            not isinstance(reference, Mapping)
            or not isinstance(reference.get("path"), str)
            or not reference["path"]
            or not isinstance(reference.get("sha256"), str)
            or len(reference["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in reference["sha256"])
            or isinstance(reference.get("size_bytes"), bool)
            or not isinstance(reference.get("size_bytes"), int)
            or reference["size_bytes"] <= 0
        ):
            raise RunError(f"artifact {name} has an invalid integrity reference")
    for gate in gates:
        if not isinstance(gate, Mapping) or not {
            "statistic",
            "value",
            "operator",
            "threshold",
            "passed",
            "sample_size",
        } <= gate.keys():
            raise RunError("mechanism gates must contain recomputable statistics")
        if (
            isinstance(gate["value"], bool)
            or not isinstance(gate["value"], Real)
            or isinstance(gate["threshold"], bool)
            or not isinstance(gate["threshold"], Real)
            or not isinstance(gate["passed"], bool)
            or isinstance(gate["sample_size"], bool)
            or not isinstance(gate["sample_size"], int)
            or gate["sample_size"] < 0
        ):
            raise RunError("mechanism gate contains invalid statistic values")


def _transition_matrices(
    model: ModelState,
    mode: int,
    segment: Segment,
    index: int,
    dt: float,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    proxy = Segment(
        segment_id=segment.segment_id,
        source_domain=segment.source_domain,
        region=segment.region,
        time=segment.time,
        state=segment.state.copy(),
        conditions=segment.conditions,
        has_terrain=segment.has_terrain,
        endpoint_prior_mean=segment.endpoint_prior_mean,
        endpoint_prior_covariance=segment.endpoint_prior_covariance,
        endpoint_prior_source=segment.endpoint_prior_source,
    )
    proxy.state[index] = state
    features = feature_vector(proxy, index, model.condition_names, model.model_kind)
    rate = features @ model.weights[mode]
    linear = model.weights[mode][1:3].T
    matrix = np.eye(2) + dt * linear
    intercept = state + dt * rate - matrix @ state
    covariance = model.covariances[mode] * dt * dt
    return matrix, intercept, covariance


def _propagate_gaussian(
    model: ModelState,
    mode: int,
    segment: Segment,
    start: int,
    integrator: str,
) -> tuple[np.ndarray, np.ndarray]:
    mean = segment.state[start].copy()
    covariance = np.zeros((2, 2), dtype=float)
    for index in range(start, len(segment.time) - 1):
        dt = float(segment.time[index + 1] - segment.time[index])
        substeps = 4 if integrator == "euler_maruyama" else (2 if integrator == "split" else 1)
        step = dt / substeps
        for _ in range(substeps):
            matrix, intercept, noise = _transition_matrices(model, mode, segment, index, step, mean)
            mean = matrix @ mean + intercept
            covariance = matrix @ covariance @ matrix.T + noise / substeps
    covariance = 0.5 * (covariance + covariance.T) + 1e-10 * np.eye(2)
    return mean, covariance


def _mc_rollout(
    model: ModelState,
    segment: Segment,
    start: int,
    integrator: str,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    states = np.repeat(segment.state[start][None, :], n_samples, axis=0)
    fixed_modes = rng.choice(model.n_modes, n_samples, p=model.mode_probabilities)
    for index in range(start, len(segment.time) - 1):
        dt = float(segment.time[index + 1] - segment.time[index])
        modes = (
            rng.choice(model.n_modes, n_samples, p=model.mode_probabilities)
            if model.model_kind in {"pointwise_mixture", "gmm_kernel"}
            else fixed_modes
        )
        substeps = 4 if integrator == "euler_maruyama" else (2 if integrator == "split" else 1)
        step = dt / substeps
        for _ in range(substeps):
            updated = states.copy()
            for mode in range(model.n_modes):
                mask = modes == mode
                if not mask.any():
                    continue
                for sample_index in np.flatnonzero(mask):
                    matrix, intercept, noise = _transition_matrices(
                        model, mode, segment, index, step, states[sample_index]
                    )
                    updated[sample_index] = (
                        matrix @ states[sample_index]
                        + intercept
                        + rng.multivariate_normal(np.zeros(2), noise / substeps)
                    )
            states = updated
    return states


def _fp_rollout(
    model: ModelState,
    segment: Segment,
    start: int,
    integrator: str,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    counts = rng.multinomial(n_samples, model.mode_probabilities)
    samples: list[np.ndarray] = []
    for mode, count in enumerate(counts):
        if not count:
            continue
        mean, covariance = _propagate_gaussian(model, mode, segment, start, integrator)
        samples.append(rng.multivariate_normal(mean, covariance, size=count))
    return np.concatenate(samples, axis=0)


def _apply_bridge(
    samples: np.ndarray,
    segment: Segment,
    bridge: str,
    rng: np.random.Generator,
    *,
    epsilon_scale: float = 0.5,
    time_steps: int = 8,
    max_iterations: int = 500,
    tolerance: float = 1e-8,
) -> tuple[
    np.ndarray,
    float | None,
    float | None,
    np.ndarray | None,
    Mapping[str, object] | None,
]:
    if bridge == "none":
        return samples, None, None, None, None
    if segment.endpoint_prior_mean is None or segment.endpoint_prior_covariance is None:
        raise RunError(f"bridge requires endpoint prior: {segment.segment_id}")
    if segment.endpoint_prior_derived_from_truth:
        raise RunError("evaluation truth may not be used as an endpoint prior")
    prior_mean = segment.endpoint_prior_mean
    before = float(np.linalg.norm(samples.mean(axis=0) - prior_mean))
    if bridge == "doob":
        result = rng.multivariate_normal(prior_mean, segment.endpoint_prior_covariance * 0.05, len(samples))
    elif bridge == "soft_endpoint":
        result = samples + 0.5 * (prior_mean - samples)
    elif bridge == "gaussian_schrodinger":
        try:
            solved = solve_particle_schrodinger_bridge(
                samples,
                prior_mean,
                segment.endpoint_prior_covariance,
                rng,
                epsilon_scale=epsilon_scale,
                time_steps=time_steps,
                max_iterations=max_iterations,
                tolerance=tolerance,
            )
        except SchrodingerBridgeError as exc:
            raise RunError(f"Schrodinger bridge failed: {exc}") from exc
        result = solved.endpoint_samples
        bridge_path = solved.paths
        bridge_diagnostics = solved.diagnostics
    else:
        raise RunError(f"unsupported bridge: {bridge}")
    after = float(np.linalg.norm(result.mean(axis=0) - prior_mean))
    if bridge != "gaussian_schrodinger":
        bridge_path = None
        bridge_diagnostics = None
    return result, before, after, bridge_path, bridge_diagnostics


def predict_segments(
    model: ModelState,
    segments: Sequence[Segment],
    config: Mapping[str, object],
    *,
    seed: int,
    n_samples: int,
) -> tuple[Prediction, ...]:
    rng = np.random.default_rng(seed)
    predictions: list[Prediction] = []
    for segment in segments:
        start = max(1, len(segment.time) // 2 - 1)
        poa = str(config.get("poa", "fp"))
        integrator = str(config.get("integrator", "split"))
        if poa == "fp":
            samples = _fp_rollout(model, segment, start, integrator, n_samples, rng)
        else:
            samples = _mc_rollout(model, segment, start, integrator, n_samples, rng)
        bridged, before, after, bridge_path, bridge_diagnostics = _apply_bridge(
            samples,
            segment,
            str(config.get("bridge", "none")),
            rng,
            epsilon_scale=float(config.get("bridge_epsilon_scale", 0.5)),
            time_steps=int(config.get("bridge_time_steps", 8)),
            max_iterations=int(config.get("bridge_sinkhorn_iterations", 500)),
            tolerance=float(config.get("bridge_sinkhorn_tolerance", 1e-8)),
        )
        predictions.append(
            Prediction(
                segment_id=segment.segment_id,
                target=segment.state[-1].copy(),
                samples=bridged,
                prior_mean=segment.endpoint_prior_mean,
                prior_source=segment.endpoint_prior_source,
                unbridged_prior_distance=before,
                bridged_prior_distance=after,
                bridge_path=bridge_path,
                bridge_diagnostics=bridge_diagnostics,
            )
        )
    return tuple(predictions)


def _energy_score(samples: np.ndarray, target: np.ndarray) -> float:
    first = np.linalg.norm(samples - target, axis=1).mean()
    shifted = np.roll(samples, 1, axis=0)
    second = np.linalg.norm(samples - shifted, axis=1).mean()
    return float(first - 0.5 * second)


def compute_metrics(
    predictions: Sequence[Prediction],
    score: str = "d2_mc",
) -> dict[str, float | int]:
    energies: list[float] = []
    coverages: list[float] = []
    cep_errors: list[float] = []
    for prediction in predictions:
        energies.append(
            _gaussian_quadrature_energy(prediction.samples, prediction.target)
            if score == "d2_closed"
            else _energy_score(prediction.samples, prediction.target)
        )
        center = prediction.samples.mean(axis=0)
        radii = np.linalg.norm(prediction.samples - center, axis=1)
        target_radius = float(np.linalg.norm(prediction.target - center))
        coverages.append(float(target_radius <= np.quantile(radii, 0.9)))
        cep_errors.append(float(np.linalg.norm(prediction.target - center)))
    return {
        "energy_score_d2": float(np.mean(energies)),
        "hdr90_coverage": float(np.mean(coverages)),
        "cep50_error": float(np.median(cep_errors)),
        "n_evaluation_segments": len(predictions),
    }


def _gaussian_quadrature_energy(samples: np.ndarray, target: np.ndarray) -> float:
    mean = samples.mean(axis=0)
    covariance = np.cov(samples, rowvar=False) + 1e-9 * np.eye(2)
    roots, weights = np.polynomial.hermite.hermgauss(12)
    grid = np.asarray([(x, y) for x in roots for y in roots])
    grid_weights = np.asarray([wx * wy for wx in weights for wy in weights]) / math.pi
    chol = np.linalg.cholesky(covariance)
    first_points = mean + math.sqrt(2.0) * grid @ chol.T
    first = float(np.sum(grid_weights * np.linalg.norm(first_points - target, axis=1)))
    second_points = 2.0 * grid @ chol.T
    second = float(np.sum(grid_weights * np.linalg.norm(second_points, axis=1)))
    return first - 0.5 * second


def _variance_reduction_factor(model: ModelState, seed: int) -> float:
    rng = np.random.default_rng(seed)
    covariance = sum(model.covariances) / model.n_modes
    chol = np.linalg.cholesky(covariance + 1e-10 * np.eye(2))
    shared = rng.normal(size=(256, 2)) @ chol.T
    later = shared + rng.normal(size=(256, 2)) @ chol.T * 0.5
    independent_a = rng.normal(size=(256, 2)) @ chol.T
    independent_b = rng.normal(size=(256, 2)) @ chol.T * math.sqrt(1.25)
    shared_delta = later[:, 0] - shared[:, 0]
    independent_delta = independent_b[:, 0] - independent_a[:, 0]
    return float(np.var(independent_delta) / max(np.var(shared_delta), 1e-15))


def mechanism_statistics(
    arm: ArmSpec,
    model: ModelState,
    predictions: Sequence[Prediction],
    evaluation_segments: Sequence[Segment],
    config: Mapping[str, object],
    seed: int,
    gate_definition: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    gate = gate_definition or arm.mechanism_gate
    statistic = str(gate["statistic"])
    if statistic == "mode_entropy":
        value = float(-np.sum(model.mode_probabilities * np.log(model.mode_probabilities + 1e-15)))
    elif statistic == "mode_switch_rate":
        value = float(1.0 - np.sum(model.mode_probabilities**2))
    elif statistic == "covariance_min_eigenvalue":
        value = float(min(np.linalg.eigvalsh(covariance).min() for covariance in model.covariances))
    elif statistic == "component_separation":
        intercepts = [weights[0] for weights in model.weights]
        value = float(max(np.linalg.norm(a - b) for a in intercepts for b in intercepts))
    elif statistic == "decomposition_coefficient_norm":
        value = float(sum(np.linalg.norm(weights[3:6]) for weights in model.weights))
    elif statistic == "effective_mode_count":
        value = float(np.sum(model.mode_probabilities >= 0.05))
    elif statistic in {"validation_energy_improvement", "objective_improvement"}:
        value = float(model.validation_objective_before - model.validation_objective_after)
    elif statistic == "mean_log_likelihood":
        value = float(-model.validation_objective_after)
    elif statistic == "closed_mc_relative_error":
        prediction = predictions[0]
        closed = _gaussian_quadrature_energy(prediction.samples, prediction.target)
        mean = prediction.samples.mean(axis=0)
        covariance = np.cov(prediction.samples, rowvar=False) + 1e-9 * np.eye(2)
        rng = np.random.default_rng(seed + 481)
        diagnostic_samples = rng.multivariate_normal(mean, covariance, size=8192)
        mc = _energy_score(diagnostic_samples, prediction.target)
        value = float(abs(closed - mc) / max(abs(closed), 1e-9))
    elif statistic == "adaptation_parameter_delta":
        value = float(model.adaptation_parameter_delta)
    elif statistic == "adaptation_sample_count":
        value = float(model.training_sample_count)
    elif statistic == "animal_pretrain_sample_count":
        value = float(model.training_sample_count)
    elif statistic == "meta_task_count":
        value = float(model.meta_task_count)
    elif statistic == "condition_coefficient_norm":
        start = 6 if model.model_kind == "explicit_decomp" else 3
        value = float(sum(np.linalg.norm(weights[start:]) for weights in model.weights))
    elif statistic == "condition_feature_variance":
        names = tuple(str(name) for name in config.get("condition", []))
        values = [segment.conditions[name] for segment in evaluation_segments for name in names]
        value = float(np.mean([np.var(item) for item in values])) if values else 0.0
    elif statistic == "condition_feature_count":
        value = float(len(tuple(config.get("condition", []))))
    elif statistic in {"split_exact_error", "euler_discretization_error"}:
        segment = evaluation_segments[0]
        start = max(1, len(segment.time) // 2 - 1)
        exact, _ = _propagate_gaussian(model, 0, segment, start, "exact")
        method = "split" if statistic == "split_exact_error" else "euler_maruyama"
        approximate, _ = _propagate_gaussian(model, 0, segment, start, method)
        value = float(np.linalg.norm(approximate - exact))
    elif statistic == "density_mass_error":
        value = float(abs(model.mode_probabilities.sum() - 1.0))
    elif statistic == "variance_reduction_factor":
        value = _variance_reduction_factor(model, seed)
    elif statistic == "endpoint_prior_distance_reduction":
        reductions = [
            prediction.unbridged_prior_distance - prediction.bridged_prior_distance
            for prediction in predictions
            if prediction.unbridged_prior_distance is not None
            and prediction.bridged_prior_distance is not None
        ]
        value = float(np.mean(reductions))
    else:
        raise RunError(f"mechanism statistic is not implemented: {statistic}")
    threshold = float(gate["threshold"])
    operator = str(gate.get("operator", "ge"))
    passed = value >= threshold if operator == "ge" else value <= threshold
    return [
        {
            "statistic": statistic,
            "value": value,
            "operator": operator,
            "threshold": threshold,
            "passed": bool(passed),
            "sample_size": len(predictions),
        }
    ]


class NEX326Runner:
    """Execute every arm through one invariant, artifact-producing process."""

    def __init__(
        self,
        spec: ExperimentSpec,
        cohort: Cohort,
        output_root: Path | str,
        *,
        n_samples: int = 64,
        replicate_seed: int | None = None,
        strict_environment: bool = False,
    ) -> None:
        self.spec = spec
        self.cohort = cohort
        self.output_root = Path(output_root)
        if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")
        self.n_samples = n_samples
        self.implementation = implementation_identity()
        if strict_environment and not self.implementation["environment_lock"]["conformant"]:
            raise RunError("runtime does not conform to environment.lock.json")
        self.replicate_seed = spec.seed if replicate_seed is None else replicate_seed
        if (
            isinstance(self.replicate_seed, bool)
            or not isinstance(self.replicate_seed, int)
            or not 0 <= self.replicate_seed < 2**32
        ):
            raise ValueError("replicate_seed must be an integer in [0, 2**32)")

    @classmethod
    def from_paths(
        cls,
        cohort_path: Path | str,
        output_root: Path | str,
        *,
        spec_path: Path | str | None = None,
        n_samples: int = 64,
        replicate_seed: int | None = None,
        strict_environment: bool = False,
    ) -> "NEX326Runner":
        spec = load_experiment_spec(spec_path) if spec_path is not None else load_experiment_spec()
        return cls(
            spec,
            load_cohort(cohort_path),
            output_root,
            n_samples=n_samples,
            replicate_seed=replicate_seed,
            strict_environment=strict_environment,
        )

    def _config(self, subconfig: Mapping[str, object]) -> dict[str, object]:
        return {**self.spec.full_components, **subconfig.get("components", {})}

    def run_one(self, arm: ArmSpec, subconfig: Mapping[str, object]) -> dict[str, object]:
        try:
            return self._run_available(arm, subconfig)
        except DataUnavailable as unavailable:
            return self._record_unavailable(arm, subconfig, unavailable)

    def _run_available(self, arm: ArmSpec, subconfig: Mapping[str, object]) -> dict[str, object]:
        subconfig_id = str(subconfig["subconfig_id"])
        seed = _stable_seed(self.replicate_seed, arm.arm_id, subconfig_id)
        config = self._config(subconfig)
        dt_seconds = float(config["dt_seconds"])
        coverage_mask = str(config["coverage_mask"]) if config.get("coverage_mask") else None
        for split in ("train", "validation", "adapt", "evaluation"):
            if not self.cohort.splits[split]:
                raise DataUnavailable(
                    "load_versioned_data",
                    self.cohort.unavailable_reasons.get(split, f"required split is empty: {split}"),
                )
        required_conditions = tuple(str(name) for name in config.get("condition", []))
        required_condition_splits = ["train", "validation", "adapt", "evaluation"]
        if arm.arm_id == 13:
            required_condition_splits.append("animal_pretrain")
        for condition in required_conditions:
            missing_from = [
                split
                for split in required_condition_splits
                if any(
                    condition not in segment.conditions
                    for segment in self.cohort.splits[split]
                )
            ]
            if missing_from:
                raise DataUnavailable(
                    "adapt_features",
                    self.cohort.unavailable_reasons.get(
                        f"condition:{condition}",
                        f"condition {condition} is unavailable in splits: {missing_from}",
                    ),
                )
        if arm.arm_id == 13 and not self.cohort.splits["animal_pretrain"]:
            raise DataUnavailable(
                "load_versioned_data",
                self.cohort.unavailable_reasons.get(
                    "animal_pretrain", "licensed animal pretraining data are unavailable"
                ),
            )
        selected: dict[str, tuple[Segment, ...]] = {}
        for split, segments in self.cohort.splits.items():
            if not segments:
                selected[split] = ()
                continue
            try:
                selected[split] = select_segments(
                    segments,
                    dt_seconds=dt_seconds,
                    coverage_mask=(
                        coverage_mask if split in {"adapt", "evaluation"} else None
                    ),
                )
            except ValueError as exc:
                raise DataUnavailable("adapt_features", str(exc)) from exc
        if arm.arm_id == 22 and any(
            segment.endpoint_prior_mean is None
            or segment.endpoint_prior_covariance is None
            or not segment.endpoint_prior_source
            for segment in selected["evaluation"]
        ):
            raise DataUnavailable(
                "inference",
                self.cohort.unavailable_reasons.get(
                    "endpoint_prior", "registered external endpoint priors are unavailable"
                ),
            )
        if config.get("transfer") == "meta_reptile":
            meta_task_counts = Counter(segment.region for segment in selected["train"])
            eligible_meta_tasks = {
                region
                for region, count in meta_task_counts.items()
                if count >= MIN_META_TASK_SEGMENTS
            }
            if len(eligible_meta_tasks) < 2:
                raise DataUnavailable(
                    "adapt_features",
                    "meta_reptile requires at least two task domains with three segments each",
                )
        # Building these samples is a distinct registered stage, not an implicit fit side effect.
        build_transition_data(selected["train"], config.get("condition", []), str(config["model"]))
        model = train_model(
            selected["train"],
            selected["validation"],
            selected["adapt"],
            selected["animal_pretrain"],
            config,
        )
        prediction_sample_count = max(
            self.n_samples,
            int(config.get("score_samples", self.n_samples)),
        )
        predictions = predict_segments(
            model,
            selected["evaluation"],
            config,
            seed=seed,
            n_samples=prediction_sample_count,
        )
        metrics = compute_metrics(predictions, str(config.get("score", "d2_mc")))
        gates = mechanism_statistics(
            arm,
            model,
            predictions,
            selected["evaluation"],
            config,
            seed,
            subconfig.get("mechanism_gate"),
        )
        run_id = f"nex326-v2-arm-{arm.arm_id:02d}-{subconfig_id}-seed-{self.replicate_seed}"
        result_id = hashlib.sha256(
            (
                f"{self.spec.spec_version}:{run_id}:{self.cohort.fingerprint}:"
                f"{self.implementation['execution_identity_sha256']}"
            ).encode()
        ).hexdigest()
        record: dict[str, object] = {
            "schema_version": RUN_RECORD_VERSION,
            "experiment_id": self.spec.experiment_id,
            "spec_version": self.spec.spec_version,
            "run_id": run_id,
            "result_id": result_id,
            "arm_id": arm.arm_id,
            "subconfig_id": subconfig_id,
            "group": arm.group,
            "slot": arm.slot,
            "full_config_id": arm.control["config_id"],
            "is_full_anchor": bool(arm.control.get("full_anchor", False)),
            "lineage": {
                "idea_ids": list(arm.idea_ids),
                "sources": list(arm.sources),
                "role": arm.lineage_role,
                "confidence": arm.lineage_confidence,
                "source_correction": arm.source_correction,
            },
            "implementation_status": arm.implementation_status,
            "implementation": self.implementation,
            "run_status": "succeeded",
            "verdict": "not_assessed",
            "dataset": {
                "dataset_id": self.cohort.dataset_id,
                "schema_version": self.cohort.schema_version,
                "data_version": self.cohort.data_version,
                "fingerprint": self.cohort.fingerprint,
                "purpose": self.cohort.purpose,
                "split_counts": {name: len(values) for name, values in selected.items()},
            },
            "config": config,
            "runtime": {
                "requested_prediction_samples": self.n_samples,
                "effective_prediction_samples": prediction_sample_count,
            },
            "seed": self.spec.seed,
            "replicate_seed": self.replicate_seed,
            "execution_seed": seed,
            "stages": [{"name": name, "status": "completed"} for name in STAGES],
            "artifacts": {},
            "metrics": metrics,
            "mechanism_gates": gates,
            "failure": None,
        }
        self._commit(record, model, predictions, metrics, gates)
        return record

    def _record_unavailable(
        self,
        arm: ArmSpec,
        subconfig: Mapping[str, object],
        unavailable: DataUnavailable,
    ) -> dict[str, object]:
        subconfig_id = str(subconfig["subconfig_id"])
        config = self._config(subconfig)
        run_id = f"nex326-v2-arm-{arm.arm_id:02d}-{subconfig_id}-seed-{self.replicate_seed}"
        result_id = hashlib.sha256(
            (
                f"{self.spec.spec_version}:{run_id}:{self.cohort.fingerprint}:"
                f"{self.implementation['execution_identity_sha256']}"
            ).encode()
        ).hexdigest()
        stage_index = STAGES.index(unavailable.stage)
        stages = [
            {
                "name": name,
                "status": (
                    "completed"
                    if index < stage_index
                    else "data_unavailable"
                    if index == stage_index
                    else "not_run"
                ),
            }
            for index, name in enumerate(STAGES)
        ]
        gate = {
            "statistic": "data_availability",
            "value": 0.0,
            "operator": "ge",
            "threshold": 1.0,
            "passed": False,
            "sample_size": 0,
        }
        record: dict[str, object] = {
            "schema_version": RUN_RECORD_VERSION,
            "experiment_id": self.spec.experiment_id,
            "spec_version": self.spec.spec_version,
            "run_id": run_id,
            "result_id": result_id,
            "arm_id": arm.arm_id,
            "subconfig_id": subconfig_id,
            "group": arm.group,
            "slot": arm.slot,
            "full_config_id": arm.control["config_id"],
            "is_full_anchor": bool(arm.control.get("full_anchor", False)),
            "lineage": {
                "idea_ids": list(arm.idea_ids),
                "sources": list(arm.sources),
                "role": arm.lineage_role,
                "confidence": arm.lineage_confidence,
                "source_correction": arm.source_correction,
            },
            "implementation_status": arm.implementation_status,
            "implementation": self.implementation,
            "run_status": "data_unavailable",
            "verdict": "unavailable",
            "dataset": {
                "dataset_id": self.cohort.dataset_id,
                "schema_version": self.cohort.schema_version,
                "data_version": self.cohort.data_version,
                "fingerprint": self.cohort.fingerprint,
                "purpose": self.cohort.purpose,
                "split_counts": {
                    name: len(values) for name, values in self.cohort.splits.items()
                },
            },
            "config": config,
            "runtime": {
                "requested_prediction_samples": self.n_samples,
                "effective_prediction_samples": None,
            },
            "seed": self.spec.seed,
            "replicate_seed": self.replicate_seed,
            "execution_seed": _stable_seed(self.replicate_seed, arm.arm_id, subconfig_id),
            "stages": stages,
            "artifacts": {},
            "metrics": {},
            "mechanism_gates": [gate],
            "failure": {
                "stage": unavailable.stage,
                "reason": unavailable.reason,
                "category": "external_data_unavailable",
            },
        }
        self._commit_unavailable(record)
        return record

    def _commit(
        self,
        record: dict[str, object],
        model: ModelState,
        predictions: Sequence[Prediction],
        metrics: Mapping[str, object],
        gates: Sequence[Mapping[str, object]],
    ) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        destination = self.output_root / str(record["run_id"])
        if destination.exists():
            raise RunError(f"run already exists: {destination}")
        staging = Path(tempfile.mkdtemp(prefix=".nex326-", dir=self.output_root))
        try:
            artifacts = {
                "checkpoint": ("checkpoint.json", model.to_dict()),
                "predictions": ("predictions.json", {"predictions": [item.to_dict() for item in predictions]}),
                "metrics": ("metrics.json", dict(metrics)),
                "mechanism": ("mechanism.json", {"gates": list(gates)}),
            }
            references: dict[str, object] = {}
            for name, (filename, payload) in artifacts.items():
                path = staging / filename
                _json_dump(payload, path)
                references[name] = {
                    "path": f"{record['run_id']}/{filename}",
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            record["artifacts"] = references
            validate_run_record(record)
            _json_dump(record, staging / "run_record.json")
            staging.replace(destination)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise

    def _commit_unavailable(self, record: dict[str, object]) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        destination = self.output_root / str(record["run_id"])
        if destination.exists():
            raise RunError(f"run already exists: {destination}")
        staging = Path(tempfile.mkdtemp(prefix=".nex326-", dir=self.output_root))
        try:
            path = staging / "availability.json"
            _json_dump(
                {
                    "run_status": record["run_status"],
                    "failure": record["failure"],
                    "dataset": record["dataset"],
                },
                path,
            )
            record["artifacts"] = {
                "availability": {
                    "path": f"{record['run_id']}/availability.json",
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            }
            validate_run_record(record)
            _json_dump(record, staging / "run_record.json")
            staging.replace(destination)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise

    def run_all(self) -> tuple[dict[str, object], ...]:
        records = tuple(self.run_one(arm, subconfig) for arm, subconfig in self.spec.executions)
        manifest = {
            "schema_version": "nex326-run-manifest-v1",
            "experiment_id": self.spec.experiment_id,
            "spec_version": self.spec.spec_version,
            "record_count": len(records),
            "protocol_seed": self.spec.seed,
            "replicate_seed": self.replicate_seed,
            "requested_prediction_samples": self.n_samples,
            "implementation": self.implementation,
            "arm_ids": sorted({int(record["arm_id"]) for record in records}),
            "run_status": dict(
                sorted(
                    {
                        status: sum(record["run_status"] == status for record in records)
                        for status in {str(record["run_status"]) for record in records}
                    }.items()
                )
            ),
            "run_records": [f"{record['run_id']}/run_record.json" for record in records],
        }
        _json_dump(manifest, self.output_root / "manifest.json")
        return records


__all__ = [
    "DataUnavailable",
    "NEX326Runner",
    "Prediction",
    "RUN_RECORD_VERSION",
    "RunError",
    "STAGES",
    "compute_metrics",
    "implementation_identity",
    "mechanism_statistics",
    "predict_segments",
    "validate_run_record",
]
