"""Explicitly approximate split-step Gaussian switching filter and smoother.

Each substep propagates in its starting mode, connects modes with exp(Q dt),
then moment-matches incoming components. This is not the exact CTMC switching
likelihood. No mode/event truth is accepted by this interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from experiments.pirc28.synthetic import ObservationBatch, simulate_path
from infrastructure.research_store import digest
from models.switching import SwitchingError, SwitchingSDEModel, finite_array, gaussian_sample, psd_array


def covariance_checked(value):
    value = (value + value.T) / 2
    if not np.isfinite(value).all():
        raise SwitchingError("NONFINITE", "approximate covariance is nonfinite")
    eigenvalues, vectors = np.linalg.eigh(value)
    if not np.isfinite(value).all() or eigenvalues.min() < -1e-8 * max(1.0, float(abs(eigenvalues).max())):
        raise SwitchingError("INFERENCE_INAPPLICABLE", "approximate covariance lost positive semidefiniteness")
    return (vectors * np.maximum(eigenvalues, 0)) @ vectors.T


def moment_match(weights, means, covariances):
    total = weights.sum()
    if total <= 0:
        raise SwitchingError("INFERENCE_INAPPLICABLE", "mode has no incoming probability")
    weights = weights / total
    mean = weights @ means
    difference = means - mean
    covariance = np.einsum('i,ijk->jk', weights, covariances) + np.einsum('i,ij,ik->jk', weights, difference, difference)
    return mean, covariance_checked(covariance)


def measurement_update(mean, covariance, y, h, r):
    innovation = y - h @ mean
    variance = covariance_checked(h @ covariance @ h.T + r)
    sign, logdet = np.linalg.slogdet(variance)
    if sign <= 0:
        raise SwitchingError("INFERENCE_INAPPLICABLE", "observation likelihood is singular")
    gain = np.linalg.solve(variance, h @ covariance).T
    mean = mean + gain @ innovation
    identity = np.eye(4) - gain @ h
    covariance = covariance_checked(identity @ covariance @ identity.T + gain @ r @ gain.T)
    log_likelihood = -0.5 * (len(y) * math.log(2 * math.pi) + logdet + innovation @ np.linalg.solve(variance, innovation))
    return mean, covariance, float(log_likelihood)


def normalize_log_weights(log_weights):
    maximum = float(log_weights.max())
    if not math.isfinite(maximum):
        raise SwitchingError("INFERENCE_INAPPLICABLE", "all posterior components vanished")
    values = np.exp(log_weights - maximum)
    total = float(values.sum())
    return values / total, maximum + math.log(total)


@dataclass(frozen=True)
class InferenceSpec:
    max_step_seconds: float = 0.25
    max_rate_step: float = 0.05
    max_substeps: int = 100_000
    max_iterations: int = 30
    max_initializations: int = 4
    objective_tolerance: float = 1e-5
    min_occupancy: float = 0.05
    identical_mode_tolerance: float = 1e-10
    min_diffusion_variance: float = 1e-5
    ridge: float = 1e-5
    rate_prior_count: float = 0.1
    rate_prior_dwell: float = 1.0
    line_search_steps: int = 6

    def validate(self):
        for name in ('max_step_seconds', 'max_rate_step', 'objective_tolerance',
                     'min_diffusion_variance', 'identical_mode_tolerance', 'ridge', 'rate_prior_count', 'rate_prior_dwell'):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise SwitchingError("CONTRACT_MISMATCH", f"{name} must be finite and positive")
        if (self.max_rate_step > 0.05 or type(self.min_occupancy) not in (int, float)
                or not math.isfinite(self.min_occupancy) or not 0 <= self.min_occupancy < 0.5):
            raise SwitchingError("CONTRACT_MISMATCH", "rate-step or occupancy constraint differs")
        for name, upper in (('max_substeps', 100_000), ('max_iterations', 100),
                            ('max_initializations', 32), ('line_search_steps', 16)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= upper:
                raise SwitchingError("CONTRACT_MISMATCH", f"{name} exceeds the bounded inference contract")
        return self

    def to_dict(self):
        self.validate()
        return {"schema_version": "pirc28-inference-spec-v1", "algorithm": "split-step-IMM-generalized-EM-v1",
            "accuracy": "approximate", "gaussians_per_mode": 1, "discarded_probability_mass": 0.0,
            "merge_rule": "all-incoming-moment-match", **self.__dict__}


@dataclass(frozen=True)
class FilteringPosterior:
    time: float
    probability: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    model_hash: str
    prefix_hash: str
    role: str = "filtering"

    def to_dict(self):
        return {"schema_version": "pirc28-joint-posterior-v1", "time": self.time, "role": self.role,
            "probability": self.probability.tolist(), "means": self.means.tolist(),
            "covariances": self.covariances.tolist(), "model_hash": self.model_hash, "prefix_hash": self.prefix_hash}


@dataclass
class FilterResult:
    times: np.ndarray
    probabilities: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    steps: list[dict]
    observation_indices: tuple[int, ...]
    posteriors: tuple[FilteringPosterior, ...]
    log_likelihood: float
    inference_spec: dict

    def to_dict(self):
        return {"schema_version": "pirc28-filter-result-v1", "role": "filtering", "accuracy": "approximate",
            "objective": "split-step-moment-matched-observation-log-likelihood",
            "log_likelihood": self.log_likelihood, "inference_spec": self.inference_spec,
            "posteriors": [posterior.to_dict() for posterior in self.posteriors]}


@dataclass
class SmootherResult:
    probabilities: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    pair_probabilities: np.ndarray
    pair_means: np.ndarray
    pair_covariances: np.ndarray
    lag_covariances: np.ndarray
    role: str = "smoothing-diagnostic-only"


def filter_switching(observations: ObservationBatch, model: SwitchingSDEModel,
                     initial_probability, initial_means, initial_covariances,
                     spec: InferenceSpec, *, path_index=0, check_budget: Callable[[], None]):
    spec.validate()
    if type(observations) is not ObservationBatch:
        raise SwitchingError("CONTRACT_MISMATCH", "filter accepts observation-only batches")
    times = np.asarray(observations.times, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or times[0] != 0 or np.any(np.diff(times) <= 0)):
        raise SwitchingError("CONTRACT_MISMATCH", "invalid observation schedule")
    h = np.asarray(observations.H, dtype=float)
    if h.ndim != 2 or not 1 <= len(h) <= 4 or h.shape[1] != 4:
        raise SwitchingError("CONTRACT_MISMATCH", "observation H differs from the four-state schema")
    h = finite_array(h, h.shape, "H")
    r = psd_array(observations.R, (len(h), len(h)), "R")
    values = np.asarray(observations.values, dtype=float)
    if (values.ndim != 3 or values.shape[1:] != (len(times), len(h))
            or type(path_index) is not int or not 0 <= path_index < len(values)):
        raise SwitchingError("CONTRACT_MISMATCH", "observation batch dimensions or path index differ")
    y = finite_array(values[path_index], (len(times), len(h)), "observations")
    k = model.n_modes
    probability = finite_array(initial_probability, (k,), "initial probability")
    if np.any(probability < 0) or not np.isclose(probability.sum(), 1, rtol=0, atol=1e-12):
        raise SwitchingError("INVALID_PROBABILITY", "initial mode probability differs")
    means = finite_array(initial_means, (k, 4), "initial means")
    covariances = psd_array(initial_covariances, (k, 4, 4), "initial covariances")
    max_rate = float(-model.Q.diagonal().min())
    delta_limit = min(spec.max_step_seconds, spec.max_rate_step / max_rate) if max_rate else spec.max_step_seconds
    substeps = [max(1, math.ceil(float(dt) / delta_limit)) for dt in np.diff(times)]
    if sum(substeps) > spec.max_substeps or sum(substeps) * k*k * 4*4 * 6 > 20_000_000:
        raise SwitchingError("INFERENCE_INAPPLICABLE", "filter/smoother history exceeds the frozen quota")
    check_budget()
    log_weights = np.full(k, -np.inf)
    for mode in range(k):
        if probability[mode] > 0:
            means[mode], covariances[mode], score = measurement_update(means[mode], covariances[mode], y[0], h, r)
            log_weights[mode] = math.log(probability[mode]) + score
    probability, objective = normalize_log_weights(log_weights)
    all_times, all_p, all_m, all_c = [times[0]], [probability.copy()], [means.copy()], [covariances.copy()]
    steps, observation_indices, posteriors = [], [0], []
    model_hash = digest(model.to_dict())

    def posterior(index):
        prefix = {"times": times[:index+1].tolist(), "values": y[:index+1].tolist(), "H": h.tolist(), "R": r.tolist()}
        return FilteringPosterior(float(times[index]), probability.copy(), means.copy(), covariances.copy(),
                                  model_hash, digest(prefix))

    posteriors.append(posterior(0))
    transition_cache = {}
    for observation_index, count in enumerate(substeps, 1):
        dt = float(times[observation_index] - times[observation_index - 1]) / count
        if dt not in transition_cache:
            kernels = [tuple(value.detach().cpu().numpy() for value in model.conditional_affine_transition(mode, dt)) for mode in range(k)]
            transition = model.mode_transition(dt).detach().cpu().numpy()
            transition_cache[dt] = (kernels, transition)
        kernels, transition = transition_cache[dt]
        for substep in range(count):
            check_budget()
            previous_p, previous_m, previous_c = probability.copy(), means.copy(), covariances.copy()
            predicted_m = np.array([f @ means[mode] + c for mode, (f, c, _) in enumerate(kernels)])
            predicted_c = np.array([covariance_checked(f @ covariances[mode] @ f.T + s) for mode, (f, _, s) in enumerate(kernels)])
            component_m = np.tile(predicted_m[:, None, :], (1, k, 1))
            component_c = np.tile(predicted_c[:, None, :, :], (1, k, 1, 1))
            log_weights = np.full((k, k), -np.inf)
            for source in range(k):
                for target in range(k):
                    weight = probability[source] * transition[source, target]
                    if weight > 0:
                        log_weights[source, target] = math.log(weight)
                        if substep == count - 1:
                            m, c, score = measurement_update(predicted_m[source], predicted_c[source], y[observation_index], h, r)
                            component_m[source, target], component_c[source, target] = m, c
                            log_weights[source, target] += score
            weights, increment = normalize_log_weights(log_weights)
            if substep == count - 1:
                objective += increment
            probability = weights.sum(axis=0)
            means, covariances = np.zeros((k, 4)), np.zeros((k, 4, 4))
            for target in range(k):
                if probability[target] > 0:
                    means[target], covariances[target] = moment_match(weights[:, target], component_m[:, target], component_c[:, target])
                else:
                    means[target], covariances[target] = previous_m[target], previous_c[target]
            current_time = float(times[observation_index - 1]) + (substep+1) * dt
            steps.append({"dt": dt, "previous_probability": previous_p, "previous_means": previous_m,
                "previous_covariances": previous_c, "predicted_means": predicted_m,
                "predicted_covariances": predicted_c, "F": np.array([item[0] for item in kernels]),
                "pair_weights": weights, "transition": transition})
            all_times.append(current_time)
            all_p.append(probability.copy())
            all_m.append(means.copy())
            all_c.append(covariances.copy())
        observation_indices.append(len(all_p)-1)
        posteriors.append(posterior(observation_index))
    check_budget()
    return FilterResult(np.asarray(all_times), np.asarray(all_p), np.asarray(all_m), np.asarray(all_c),
        steps, tuple(observation_indices), tuple(posteriors), float(objective), spec.to_dict())


def smooth_switching(filtered: FilterResult, *, check_budget: Callable[[], None]):
    """Moment-based backward diagnostic; never a permissible forecast origin."""
    probability, means, covariance = (value.copy() for value in
        (filtered.probabilities, filtered.means, filtered.covariances))
    n, k = len(filtered.steps), probability.shape[1]
    pairs = np.zeros((n, k, k))
    pair_means = np.zeros((n, k, k, 4))
    pair_covariances = np.zeros((n, k, k, 4, 4))
    lag_covariances = np.zeros_like(pair_covariances)
    for index in range(n-1, -1, -1):
        check_budget()
        step = filtered.steps[index]
        ratio = np.divide(probability[index+1], filtered.probabilities[index+1],
                          out=np.zeros(k), where=filtered.probabilities[index+1] > 0)
        weights = step['pair_weights'] * ratio[None, :]
        if weights.sum() <= 0:
            raise SwitchingError("INFERENCE_INAPPLICABLE", "backward mode probability vanished")
        pairs[index] = weights / weights.sum()
        probability[index] = pairs[index].sum(axis=1)
        for source in range(k):
            previous_c = step['previous_covariances'][source]
            predicted_c = step['predicted_covariances'][source]
            gain = previous_c @ step['F'][source].T @ np.linalg.pinv(predicted_c, rcond=1e-12)
            for target in range(k):
                pair_means[index, source, target] = step['previous_means'][source] + gain @ (means[index+1, target] - step['predicted_means'][source])
                pair_covariances[index, source, target] = covariance_checked(previous_c + gain @ (covariance[index+1, target] - predicted_c) @ gain.T)
                lag_covariances[index, source, target] = gain @ covariance[index+1, target]
            if probability[index, source] > 0:
                means[index, source], covariance[index, source] = moment_match(pairs[index, source],
                    pair_means[index, source], pair_covariances[index, source])
    return SmootherResult(probability, means, covariance, pairs, pair_means, pair_covariances, lag_covariances)


def forecast_switching(model, posterior: FilteringPosterior, horizons, *, sample_count,
                       seed, max_jumps=1000, check_budget: Callable[[], None]):
    if type(posterior) is not FilteringPosterior or posterior.role != 'filtering':
        raise SwitchingError("FUTURE_INFORMATION", "forecast origin must be a filtering joint posterior")
    if posterior.model_hash != digest(model.to_dict()):
        raise SwitchingError("IDENTITY_CONFLICT", "posterior and forecast model identities differ")
    horizons = np.asarray(horizons, dtype=float)
    if (horizons.ndim != 1 or not 1 <= len(horizons) <= 100_000 or not np.isfinite(horizons).all()
            or np.any(horizons <= 0) or np.any(np.diff(horizons) <= 0)):
        raise SwitchingError("CONTRACT_MISMATCH", "forecast horizons must be finite, positive and increasing")
    if type(sample_count) is not int or not 1 <= sample_count <= 100_000:
        raise SwitchingError("RESOURCE_LIMIT", "forecast sample quota differs")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise SwitchingError("CONTRACT_MISMATCH", "forecast seed must be unsigned 64-bit")
    if type(max_jumps) is not int or not 1 <= max_jumps <= 100_000:
        raise SwitchingError("CONTRACT_MISMATCH", "forecast jump quota differs")
    if (sample_count * (len(horizons)*5 + max_jumps*2) > 20_000_000
            or float(-model.Q.diagonal().min()) * horizons[-1] > max_jumps):
        raise SwitchingError("RESOURCE_LIMIT", "forecast path or jump quota exceeded")
    k = model.n_modes
    probability = finite_array(posterior.probability, (k,), "posterior probability")
    means = finite_array(posterior.means, (k, 4), "posterior means")
    covariance = psd_array(posterior.covariances, (k, 4, 4), "posterior covariances")
    if np.any(probability < 0) or not np.isclose(probability.sum(), 1, rtol=0, atol=1e-12):
        raise SwitchingError("INVALID_PROBABILITY", "posterior is not normalized")
    samples, modes, jumps, jump_modes, initial_modes = [], [], [], [], []
    for sample_index in range(sample_count):
        check_budget()
        rng_mode, rng_state = [np.random.default_rng(np.random.SeedSequence([seed, sample_index, role])) for role in range(2)]
        mode = int(rng_mode.choice(k, p=probability))
        state = gaussian_sample(means[mode], covariance[mode], rng_state)
        z, s, jt, jm, _ = simulate_path(model, np.concatenate(([0.0], horizons)), mode, state, rng_mode, rng_state,
            max_jumps=max_jumps, check_budget=check_budget)
        initial_modes.append(mode)
        samples.append(z[1:])
        modes.append(s[1:])
        jumps.append(jt)
        jump_modes.append(jm)
    q = model.Q.detach().cpu().numpy()
    exact_marginal, exact_any, exact_endpoint = [], [], []
    for h in horizons:
        transition = model.mode_transition(float(h)).detach().cpu().numpy()
        exact_marginal.append((probability @ transition).tolist())
        exact_any.append(float(1 - probability @ np.exp(np.diag(q) * h)))
        exact_endpoint.append(float(probability @ (1 - np.diag(transition))))
    check_budget()
    return {"schema_version": "pirc28-forecast-v1", "role": "filtering-origin", "origin": posterior.time,
        "posterior_hash": digest(posterior.to_dict()), "model_hash": posterior.model_hash,
        "horizons": horizons.tolist(), "sample_count": sample_count, "seed": seed,
        "samples": np.asarray(samples).tolist(), "mode_paths": np.asarray(modes).tolist(),
        "initial_modes": initial_modes, "jump_times": [list(value) for value in jumps],
        "jump_modes": [list(value) for value in jump_modes],
        "mode_marginal": exact_marginal, "probability_any_jump": exact_any,
        "probability_different_endpoint_mode": exact_endpoint, "hazard_per_second": (-np.diag(q)).tolist(),
        "mean_dwell_seconds": [float(1 / rate) if rate > 0 else None for rate in -np.diag(q)],
        "dwell_model": "homogeneous-CTMC-memoryless", "mean_is_summary_only": True,
        "state_order": ['x', 'y', 'vx', 'vy'], "resume_level": "restart-only",
        "accuracy": "event-driven-sampling-from-approximate-filtering-posterior"}
