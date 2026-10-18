"""Event-driven synthetic truth, separate observation-only training inputs.

The caller supplies a budget poll from the shared supervisor. No production
ledger, store identity, implicit deadline or research authorization is created
here. Unit qualification fixtures can explicitly supply their test poll.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from infrastructure.research_store import digest
from models.switching import SwitchingError, SwitchingSDEModel, finite_array, gaussian_sample, psd_array


@dataclass(frozen=True)
class ObservationBatch:
    times: np.ndarray
    values: np.ndarray
    H: np.ndarray
    R: np.ndarray
    scenario_hash: str

    def to_dict(self):
        return {"schema_version": "pirc28-observations-v1", "scenario_hash": self.scenario_hash,
            **{name: getattr(self, name).tolist() for name in ("times", "values", "H", "R")}}


@dataclass(frozen=True)
class SyntheticTruth:
    states: np.ndarray
    modes: np.ndarray
    jump_times: tuple[tuple[float, ...], ...]
    jump_modes: tuple[tuple[int, ...], ...]
    dwell_by_mode: np.ndarray
    scenario_hash: str

    def to_dict(self):
        return {"schema_version": "pirc28-synthetic-truth-v1", "scenario_hash": self.scenario_hash,
            "states": self.states.tolist(), "modes": self.modes.tolist(),
            "jump_times": [list(value) for value in self.jump_times],
            "jump_modes": [list(value) for value in self.jump_modes],
            "dwell_by_mode": self.dwell_by_mode.tolist(), "visibility": "synthetic-evaluator-only"}


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    model: dict
    times: tuple[float, ...]
    sample_count: int
    seed: int
    initial_mode_probability: tuple[float, ...]
    initial_means: tuple
    initial_covariances: tuple
    H: tuple
    R: tuple
    max_jumps: int = 1000
    label_permutation: tuple[int, ...] = ()

    def validated(self):
        model = SwitchingSDEModel.from_dict(self.model)
        k = model.n_modes
        times = np.asarray(self.times, dtype=float)
        if (times.ndim != 1 or not 2 <= len(times) <= 100_000 or not np.isfinite(times).all()
                or times[0] != 0 or np.any(np.diff(times) <= 0)):
            raise SwitchingError("CONTRACT_MISMATCH", "schedule must start at zero and increase strictly")
        if not isinstance(self.scenario_id, str) or not self.scenario_id or len(self.scenario_id) > 128:
            raise SwitchingError("CONTRACT_MISMATCH", "scenario ID must be a bounded nonempty string")
        if type(self.sample_count) is not int or not 1 <= self.sample_count <= 100_000:
            raise SwitchingError("CONTRACT_MISMATCH", "sample count is outside the registered limit")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise SwitchingError("CONTRACT_MISMATCH", "seed must be an unsigned 64-bit integer")
        if type(self.max_jumps) is not int or not 1 <= self.max_jumps <= 100_000:
            raise SwitchingError("CONTRACT_MISMATCH", "max_jumps must be bounded and positive")
        h = np.asarray(self.H, dtype=float)
        if h.ndim != 2 or h.shape[1] != 4 or not 1 <= h.shape[0] <= 4:
            raise SwitchingError("CONTRACT_MISMATCH", "H must observe 1..4 coordinates of the four-state model")
        h = finite_array(h, h.shape, "H")
        r = psd_array(self.R, (len(h), len(h)), "R")
        p = finite_array(self.initial_mode_probability, (k,), "initial mode probability")
        if np.any(p < 0) or not np.isclose(p.sum(), 1, rtol=0, atol=1e-12):
            raise SwitchingError("INVALID_PROBABILITY", "initial mode probabilities must sum to one")
        means = finite_array(self.initial_means, (k, 4), "initial means")
        covariance = psd_array(self.initial_covariances, (k, 4, 4), "initial covariances")
        labels = self.label_permutation or tuple(range(k))
        if sorted(labels) != list(range(k)) or any(type(i) is not int for i in labels):
            raise SwitchingError("CONTRACT_MISMATCH", "truth label map must be a permutation")
        elements = self.sample_count * (len(times) * (4 + len(h) + 1) + 2 * self.max_jumps + k)
        if elements > 20_000_000:
            raise SwitchingError("RESOURCE_LIMIT", "scenario tensor elements exceed the frozen quota")
        rate = float(-model.Q.diagonal().min())
        if rate * times[-1] > self.max_jumps:
            raise SwitchingError("JUMP_LIMIT", "expected jump upper bound exceeds the registered quota")
        return model, times.copy(), p, means, covariance, h, r, np.asarray(labels)

    def to_dict(self):
        model, times, p, means, covariance, h, r, labels = self.validated()
        return {"schema_version": "pirc28-scenario-v1", "scenario_id": self.scenario_id,
            "model": model.to_dict(), "times": times.tolist(), "sample_count": self.sample_count,
            "seed": self.seed, "initial_mode_probability": p.tolist(), "initial_means": means.tolist(),
            "initial_covariances": covariance.tolist(), "H": h.tolist(), "R": r.tolist(),
            "max_jumps": self.max_jumps, "label_permutation": labels.tolist(),
            "generator": "event-driven-conditional-affine-v1", "coupling": "PCG64-per-path-role-v1",
            "truth_visibility": "synthetic-evaluator-only", "context": "registered-no-terrain"}

    @property
    def scenario_hash(self):
        return digest(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or value.get('schema_version') != 'pirc28-scenario-v1':
            raise SwitchingError('CONTRACT_MISMATCH', 'scenario schema differs')
        try:
            spec = cls(**{name: value[name] for name in cls.__dataclass_fields__})
        except (KeyError, TypeError) as error:
            raise SwitchingError('CONTRACT_MISMATCH', 'scenario fields are incomplete') from error
        if digest(spec.to_dict()) != digest(value):
            raise SwitchingError('IDENTITY_CONFLICT', 'scenario declarations are not canonical')
        return spec


def simulate_path(model, times, initial_mode, initial_state, rng_modes, rng_state,
                  *, max_jumps: int, check_budget: Callable[[], None]):
    """Zero or many jumps per observation interval, with no drift averaging."""
    times = np.asarray(times, dtype=np.float64)
    if (times.ndim != 1 or not 2 <= len(times) <= 100_000 or not np.isfinite(times).all()
            or times[0] != 0 or np.any(np.diff(times) <= 0)):
        raise SwitchingError("CONTRACT_MISMATCH", "path schedule must start at zero and increase strictly")
    if type(max_jumps) is not int or not 1 <= max_jumps <= 100_000:
        raise SwitchingError("CONTRACT_MISMATCH", "path jump quota must be bounded and positive")
    mode = model._mode(initial_mode)
    state = finite_array(initial_state, (4,), "initial state")
    q = model.Q.detach().cpu().numpy()
    states, modes = [state.copy()], [mode]
    jumps, jump_modes = [], [mode]
    dwell = np.zeros(model.n_modes)
    time = float(times[0])

    def next_jump(now, current):
        rate = -q[current, current]
        if rate == 0:
            return math.inf
        wait = float(rng_modes.exponential(1 / rate))
        if wait <= 0 or now + wait <= now:
            raise SwitchingError("NONFINITE", "jump clock lost floating-point resolution")
        return now + wait

    jump = next_jump(time, mode)
    for observation_time in times[1:]:
        check_budget()
        while time < observation_time:
            check_budget()
            end = min(jump, float(observation_time))
            dt = end - time
            if dt <= 0:
                raise SwitchingError("NONFINITE", "nonpositive event segment")
            f, c, covariance = model.conditional_affine_transition(mode, dt)
            state = gaussian_sample(f.detach().cpu().numpy() @ state + c.detach().cpu().numpy(),
                                    covariance.detach().cpu().numpy(), rng_state)
            if not np.isfinite(state).all():
                raise SwitchingError("NONFINITE", "continuous state overflowed")
            dwell[mode] += dt
            time = end
            if time == jump:
                if len(jumps) >= max_jumps:
                    raise SwitchingError("JUMP_LIMIT", "actual jump quota exceeded; path is incomplete")
                weights = q[mode].copy()
                weights[mode] = 0
                weights /= weights.sum()
                mode = int(rng_modes.choice(model.n_modes, p=weights))
                jumps.append(time)
                jump_modes.append(mode)
                jump = next_jump(time, mode)
        states.append(state.copy())
        modes.append(mode)
    return np.asarray(states), np.asarray(modes), tuple(jumps), tuple(jump_modes), dwell


def generate_scenario(spec: ScenarioSpec, *, check_budget: Callable[[], None]):
    model, times, p, means, covariance, h, r, labels = spec.validated()
    identity = spec.scenario_hash
    # Independent role streams keep observation noise out of the latent path.
    # Child seeds are indexed by path, so chunk/restart boundaries do not reuse RNG.
    states, modes, observations, jumps, jump_modes, dwell = [], [], [], [], [], []
    for path_id in range(spec.sample_count):
        check_budget()
        streams = [np.random.default_rng(np.random.SeedSequence([spec.seed, path_id, role])) for role in range(3)]
        rng_modes, rng_state, rng_observation = streams
        mode = int(rng_modes.choice(model.n_modes, p=p))
        initial = gaussian_sample(means[mode], covariance[mode], rng_state)
        z, s, jt, jm, dt = simulate_path(model, times, mode, initial, rng_modes, rng_state,
            max_jumps=spec.max_jumps, check_budget=check_budget)
        y = np.asarray([gaussian_sample(h @ state, r, rng_observation) for state in z])
        states.append(z)
        modes.append(labels[s])
        observations.append(y)
        jumps.append(jt)
        jump_modes.append(tuple(int(labels[mode]) for mode in jm))
        relabeled_dwell = np.zeros_like(dt)
        relabeled_dwell[labels] = dt
        dwell.append(relabeled_dwell)
    check_budget()
    observation = ObservationBatch(times, np.asarray(observations), h, r, identity)
    truth = SyntheticTruth(np.asarray(states), np.asarray(modes), tuple(jumps), tuple(jump_modes),
                           np.asarray(dwell), identity)
    manifest = {"schema_version": "pirc28-generation-v1", "scenario_hash": identity,
        "scenario": spec.to_dict(), "observations_hash": digest(observation.to_dict()),
        "truth_hash": digest(truth.to_dict()), "status": "SUCCEEDED", "resume_level": "restart-only",
        "configured_K": model.n_modes, "occupied_K": len(np.unique(truth.modes))}
    return observation, truth, manifest
