"""Target-free NEX326 forecasting; no cohort, Segment, labels or file access.

The legacy fit estimates velocity-rate residual covariance R [m^2/s^2]. An
EXPLICIT reference interval tau embeds that discrete model as Q=R*tau [m^2/s]:
over tau a constant-coefficient step has covariance Q*tau=R*tau^2. This is a
declared continuous-time embedding, not proof that the legacy fit is an SDE MLE.
tau, history/mode clocks and numerical h are different parameters. No training
interval or numerical budget is inferred here; their empirical binding remains
the protocol/training adapter's responsibility.

FP means conditional Gaussian-moment propagation given sampled latent-mode
histories, not a spatial PDE solver. MC/CRN evaluate nonlinear features on each
sample. FP evaluates them on the conditional mean, an assumed-density closure.
The locally affine exact kernel is exact only for fixed affine coefficients.
CRN uses an explicit pair identity shared with its paired control; it does not
change the marginal law or establish variance reduction by itself.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Callable

import numpy as np
import torch

from experiments.nex326.model import ModelState
from .origins import Origin, frozen_array
from .rollout import Forecast, _horizons

VERSION = "pirc17-causal-method-rollout-v1"
MODEL_KINDS = {"seg_constant_mode", "pointwise_mixture", "single_gaussian",
               "gmm_kernel", "explicit_decomp"}
SWITCHING_MODELS = {"pointwise_mixture", "gmm_kernel"}
INTEGRATORS = {"exact", "split", "euler_maruyama"}


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"positive finite {name} required")
    return float(value)


def _integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"positive integer {name} required")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"nonempty {name} required")
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _psd(value):
    """Reject material indefiniteness; only remove eigenvalue roundoff below 0."""
    value = np.asarray(value, dtype=float)
    if value.shape != (2, 2) or not np.isfinite(value).all():
        raise ValueError("finite 2x2 covariance required")
    tolerance = 128 * np.finfo(float).eps * max(1., float(np.linalg.norm(value, ord=2)))
    if np.max(np.abs(value - value.T)) > tolerance:
        raise ValueError("symmetric covariance required")
    symmetric = .5 * (value + value.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if eigenvalues[0] < -tolerance:
        raise ValueError("covariance is not positive semidefinite")
    root = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.))[None, :]
    if eigenvalues[0] < 0:
        symmetric = root @ root.T
    return symmetric, root


def _exponential(matrix):
    result = torch.linalg.matrix_exp(torch.tensor(matrix, dtype=torch.float64)).numpy()
    if not np.isfinite(result).all():
        raise ValueError("nonfinite affine transition; no clipping or step retry")
    return result


def affine_kernel(linear, diffusion_covariance, seconds, integrator):
    """Return F, J, S for x_next=F*x+J*b+Normal(0,S).

EM: F=I+hA, J=hI, S=hQ. Strang: exact half drift, Qh noise,
exact half drift. Exact: augmented matrix exponential and the Van Loan block
integral. There is no substep-dependent rescaling of Q, covariance floor,
Euler alias for exact, or fallback on failure.
"""
    h = _positive(seconds, "step seconds")
    linear = frozen_array(linear, (2, 2))
    noise, _ = _psd(diffusion_covariance)
    if integrator not in INTEGRATORS:
        raise ValueError("unregistered method integrator")
    eye = np.eye(2)
    if integrator == "euler_maruyama":
        transition, integral, covariance = eye + h * linear, h * eye, h * noise
    else:
        augmented = np.zeros((4, 4))
        augmented[:2, :2], augmented[:2, 2:] = linear, eye
        if integrator == "split":
            half = _exponential(augmented * (h / 2))
            fh, jh = half[:2, :2], half[:2, 2:]
            transition, integral = fh @ fh, (eye + fh) @ jh
            covariance = fh @ (h * noise) @ fh.T
        else:
            deterministic = _exponential(augmented * h)
            transition, integral = deterministic[:2, :2], deterministic[:2, 2:]
            van_loan = np.zeros((4, 4))
            van_loan[:2, :2], van_loan[:2, 2:] = linear, noise
            van_loan[2:, 2:] = -linear.T
            stochastic = _exponential(van_loan * h)
            covariance = stochastic[:2, 2:] @ transition.T
    covariance = .5 * (covariance + covariance.T)
    _psd(covariance)
    return frozen_array(transition), frozen_array(integral), frozen_array(covariance)


@dataclass(frozen=True)
class MethodDynamics:
    model: ModelState
    reference_interval_seconds: float
    fit_identity: str
    noise_binding_rationale: str
    model_sha256: str

    @classmethod
    def bind(cls, model, *, reference_interval_seconds, fit_identity, noise_binding_rationale):
        if not isinstance(model, ModelState) or model.model_kind not in MODEL_KINDS:
            raise ValueError("supported fitted NEX326 ModelState required")
        tau = _positive(reference_interval_seconds, "model reference interval")
        _text(fit_identity, "immutable fit identity")
        _text(noise_binding_rationale, "noise binding rationale")
        names = tuple(model.condition_names)
        if len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
            raise ValueError("unique nonempty condition names required")
        count = len(model.weights)
        expected_modes = 1 if model.model_kind in {"single_gaussian", "explicit_decomp"} else 3
        if count != expected_modes or len(model.covariances) != count:
            raise ValueError("mode count does not match frozen model family")
        features = (6 if model.model_kind == "explicit_decomp" else 3) + len(names)
        weights = tuple(frozen_array(w, (features, 2)) for w in model.weights)
        covariances = tuple(frozen_array(_psd(c)[0]) for c in model.covariances)
        probabilities = frozen_array(model.mode_probabilities, (count,))
        if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1., rtol=0, atol=1e-12):
            raise ValueError("normalized nonnegative mode probabilities required")
        copied = replace(model, condition_names=names, weights=weights, covariances=covariances,
                         mode_probabilities=probabilities)
        identity = _digest(copied.to_dict())
        return cls(copied, tau, fit_identity, noise_binding_rationale, identity)

    def validate(self):
        # Also validate manually constructed dataclass instances, not just bind().
        checked = self.bind(self.model, reference_interval_seconds=self.reference_interval_seconds,
                            fit_identity=self.fit_identity, noise_binding_rationale=self.noise_binding_rationale)
        if checked.model_sha256 != self.model_sha256:
            raise ValueError("bound model was mutated")
        _positive(self.reference_interval_seconds, "model reference interval")
        _text(self.fit_identity, "immutable fit identity")
        _text(self.noise_binding_rationale, "noise binding rationale")

    def identity(self):
        self.validate()
        return {"model_sha256": self.model_sha256, "fit_identity": self.fit_identity,
                "reference_interval_seconds": self.reference_interval_seconds,
                "noise_embedding": "Q_m2_per_s = R_rate_m2_per_s2 * reference_interval_seconds",
                "noise_binding_rationale": self.noise_binding_rationale,
                "embedding_is_continuous_time_fit_qualification": False}


def causal_features(model, positions, velocities, condition_values):
    """Same feature columns as the fitted model; state/history provided causally."""
    positions, velocities = np.asarray(positions), np.asarray(velocities)
    conditions = np.asarray(condition_values)
    if positions.ndim != 2 or positions.shape[1:] != (2,) or velocities.shape != positions.shape:
        raise ValueError("particle-by-xy position and velocity required")
    if conditions.shape != (len(positions), len(model.condition_names)):
        raise ValueError("prediction-time condition columns mismatch")
    if any(not np.isfinite(a).all() for a in (positions, velocities, conditions)):
        raise ValueError("nonfinite prediction-time feature; no hidden-target fallback")
    parts = [np.ones((len(positions), 1)), positions]
    if model.model_kind == "explicit_decomp":
        speed = np.linalg.norm(velocities, axis=1, keepdims=True)
        radial = positions / np.maximum(np.linalg.norm(positions, axis=1, keepdims=True), 1e-9)
        parts.extend([speed, radial])
    parts.append(conditions)
    return np.column_stack(parts)


def _grid(horizons, numerical_step, history_step, mode_step, max_steps):
    """Physical mode/history clocks do not accelerate when numerical h shrinks."""
    ticks = [1, 1, 1]
    periods = [numerical_step, history_step, mode_step]
    output = 0
    elapsed = 0.
    grid = []
    while output < len(horizons):
        boundaries = [ticks[i] * periods[i] for i in range(3)]
        end = min(*boundaries, float(horizons[output]))
        if end <= elapsed:
            raise ValueError("step below floating-point resolution")
        is_history, is_mode = end == boundaries[1], end == boundaries[2]
        is_output = end == horizons[output]
        grid.append((elapsed, end, is_history, is_mode, is_output))
        if len(grid) > max_steps:
            raise ValueError("registered method step budget exhausted before prediction")
        for i in range(3):
            if end == boundaries[i]:
                ticks[i] += 1
        if is_output:
            output += 1
        elapsed = end
    return grid


def _rng(seed, key, purpose):
    digest = hashlib.sha256(json.dumps([key, purpose], separators=(",", ":")).encode()).digest()
    words = [int.from_bytes(digest[i:i + 4], "little") for i in range(0, 32, 4)]
    return np.random.default_rng(np.random.SeedSequence([seed, *words]))


@dataclass(frozen=True)
class MethodForecast:
    forecast: Forecast
    conditional_means_m: np.ndarray | None
    conditional_covariances_m2: np.ndarray | None
    diagnostics: dict


def forecast_method(dynamics: MethodDynamics, origin: Origin, horizons_seconds, *,
                    propagation: str, integrator: str, particles: int, seed: int,
                    max_step_seconds: float, history_step_seconds: float, max_steps: int,
                    max_particle_steps: int,
                    origin_id: str, run_id: str, condition_names=(),
                    condition_at: Callable | None = None, crn_pair_id: str | None = None):
    """Forecast all requested horizons in one pass with bounded fixed settings.

Only prediction-time providers receive predicted positions and time. FP can
share a crn_pair_id with a CRN candidate; ordinary MC must use its independent
run_id. Same pair+origin+seed+grid gives the same indexed Gaussian/uniform draws.
Different grids are NOT a coupled-Brownian refinement oracle. No run is retried
and no grid, particles, seeds or horizon is expanded on failure.
"""
    if not isinstance(dynamics, MethodDynamics) or not isinstance(origin, Origin):
        raise ValueError("bound method dynamics and target-free Origin required")
    dynamics.validate()
    model = dynamics.model
    horizons = _horizons(horizons_seconds)
    _integer(particles, "particle count")
    _integer(max_steps, "step budget")
    _integer(max_particle_steps, "particle-step budget")
    h = _positive(max_step_seconds, "numerical step")
    history_h = _positive(history_step_seconds, "history clock")
    if type(seed) is not int or seed < 0:
        raise ValueError("nonnegative integer seed required")
    _text(origin_id, "origin identity")
    _text(run_id, "run identity")
    if propagation not in {"fp", "mc", "crn"} or integrator not in INTEGRATORS:
        raise ValueError("unregistered method propagation/integration path")
    if crn_pair_id is not None:
        _text(crn_pair_id, "CRN pair identity")
        if propagation == "mc":
            raise ValueError("ordinary MC must use an independent run stream")
    elif propagation == "crn":
        raise ValueError("CRN requires an explicit pair identity, not just an alias")
    if tuple(condition_names) != model.condition_names or bool(model.condition_names) != (condition_at is not None):
        raise ValueError("exact named prediction-time condition provider required")
    grid = _grid(horizons, h, history_h, dynamics.reference_interval_seconds, max_steps)
    if particles * len(grid) > max_particle_steps:
        raise ValueError("registered particle-step budget exhausted before prediction")
    key = ([origin_id, "paired", crn_pair_id] if crn_pair_id
           else [origin_id, "independent", propagation, run_id])
    states = np.broadcast_to(origin.position_m, (particles, 2)).copy()
    centers = states.copy()
    velocity = origin.sample_velocities(particles, _rng(seed, key, "initial-velocity"))
    previous_tick = centers.copy()
    covariances = np.zeros((particles, 2, 2))
    cdf = np.cumsum(model.mode_probabilities)
    cdf[-1] = 1.

    def modes_at(tick):
        return np.searchsorted(cdf, _rng(seed, key, ["modes", tick]).random(particles), side="right")

    mode_tick = 0
    modes = modes_at(mode_tick)
    paths, mean_outputs, covariance_outputs = [], [], []
    kernels = {}
    mode_changes = mode_opportunities = feature_rows = 0
    history_ticks = []
    mode_ticks = []
    for start, end, is_history, is_mode, is_output in grid:
        dt = end - start
        evaluation_state = centers if propagation == "fp" else states
        conditions = (np.asarray(condition_at(frozen_array(evaluation_state), origin.epoch_seconds + start))
                      if condition_at is not None else np.empty((particles, 0)))
        features = causal_features(model, evaluation_state, velocity, conditions)
        feature_rows += particles if condition_at is not None else 0
        z = _rng(seed, key, ["normal", float(start).hex(), float(end).hex()]).standard_normal((particles, 2))
        for mode in range(model.n_modes):
            selected = modes == mode
            if not selected.any():
                continue
            linear = model.weights[mode][1:3].T
            offset = features[selected] @ model.weights[mode] - evaluation_state[selected] @ linear.T
            cache_key = mode, dt
            if cache_key not in kernels:
                f, j, q = affine_kernel(linear, model.covariances[mode] * dynamics.reference_interval_seconds,
                                        dt, integrator)
                kernels[cache_key] = f, j, q, _psd(q)[1]
            f, j, q, root = kernels[cache_key]
            intercept = offset @ j.T
            states[selected] = states[selected] @ f.T + intercept + z[selected] @ root.T
            if propagation == "fp":
                centers[selected] = centers[selected] @ f.T + intercept
                covariances[selected] = f @ covariances[selected] @ f.T + q
        if not np.isfinite(states).all() or not np.isfinite(centers).all() or not np.isfinite(covariances).all():
            raise ValueError("nonfinite method forecast; no clipping or automatic retry")
        if is_history:
            current = centers if propagation == "fp" else states
            velocity = (current - previous_tick) / history_h
            previous_tick = current.copy()
            history_ticks.append(end)
        if is_output:
            paths.append(states.copy())
            if propagation == "fp":
                mean_outputs.append(centers.copy())
                covariance_outputs.append(covariances.copy())
        if is_mode and end < horizons[-1] and model.model_kind in SWITCHING_MODELS:
            mode_tick += 1
            updated = modes_at(mode_tick)
            mode_changes += int(np.count_nonzero(updated != modes))
            mode_opportunities += particles
            modes = updated
            mode_ticks.append(end)
    return MethodForecast(
        Forecast(horizons, frozen_array(np.stack(paths, axis=1)), 0, feature_rows, VERSION),
        frozen_array(np.stack(mean_outputs, axis=1)) if propagation == "fp" else None,
        frozen_array(np.stack(covariance_outputs, axis=1)) if propagation == "fp" else None,
        {"version": VERSION, "dynamics": dynamics.identity(), "propagation": propagation,
         "model_kind": model.model_kind, "condition_names": list(model.condition_names),
         "origin_mode": origin.mode, "origin_id": origin_id, "run_id": run_id,
         "origin_epoch_seconds": origin.epoch_seconds, "velocity_source": origin.velocity_source,
         "velocity_observed_at_seconds": origin.velocity_observed_at_seconds,
         "velocity_error_mps": origin.velocity_error_mps,
         "integrator": integrator, "particles": particles, "seed": seed,
         "max_step_seconds": h, "history_step_seconds": history_h,
         "mode_step_seconds": dynamics.reference_interval_seconds,
         "integration_steps": len(grid), "registered_max_steps": max_steps,
         "registered_max_particle_steps": max_particle_steps,
         "history_ticks_seconds": history_ticks, "mode_resampling_times_seconds": mode_ticks,
         "mode_changes": mode_changes, "mode_change_opportunities": mode_opportunities,
         "random_stream_sha256": _digest(key), "crn_pair_id": crn_pair_id,
         "different_grids_have_coupled_Brownian_paths": False,
         "history_rule": "initial Origin velocity, then last fixed-clock predicted secant",
         "fp_semantics": "conditional Gaussian moments given sampled mode histories; mean-feature assumed-density closure, not spatial PDE",
         "nonlinear_exactness_claimed": False, "variance_reduction_qualified": False,
         "numerically_qualified": False, "scientific_claim_authorized": False})
