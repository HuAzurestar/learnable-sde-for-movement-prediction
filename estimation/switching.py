"""Approximate generalized EM for the frozen K<=2 switching model.

Conditional CTMC bridge moments count repeat jumps, unlike endpoint changes.
Continuous sufficient statistics use the declared Gaussian moment approximation.
Every candidate is line-searched on the *approximate* observation objective;
this never promises monotonic exact switching likelihood or a global optimum.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
import torch

from experiments.pirc28.synthetic import ObservationBatch
from experiments.pirc28.failures import NUMERICAL_EXCEPTIONS, NumericalCellFailure, numerical_diagnostic
from inference.switching import InferenceSpec, filter_switching, smooth_switching
from infrastructure.research_store import digest
from models.switching import SwitchingError, SwitchingSDEModel


def ctmc_bridge_moments(model, dt):
    """E[dwell, jump counts | initial mode, endpoint mode] for exp(Q dt)."""
    q = model.Q.detach().cpu().numpy()
    k = len(q)
    p = model.mode_transition(dt).detach().cpu().numpy()
    dwell = np.zeros((k, k, k))
    counts = np.zeros((k, k, k, k))
    for source in range(k):
        for target in range(k):
            e = np.zeros_like(q)
            e[source, target] = 1 if source == target else q[source, target]
            augmented = np.block([[q, e], [np.zeros_like(q), q]])
            integral = torch.linalg.matrix_exp(torch.from_numpy(augmented) * dt).numpy()[:k, k:]
            conditional = np.divide(integral, p, out=np.zeros_like(integral), where=p > 0)
            if source == target:
                dwell[source] = conditional
            else:
                counts[source, target] = conditional
    return dwell, counts


@dataclass
class SwitchingFit:
    model: SwitchingSDEModel
    objective_trace: tuple[float, ...]
    stopping_reason: str
    occupancy: np.ndarray
    collapsed: bool
    filtered: tuple
    initialization_id: str
    inference_spec: dict
    rejected_objectives: tuple[float, ...]
    indistinguishable_modes: bool = False

    def to_dict(self, *, include_posteriors=True):
        value = {"schema_version": "pirc28-fit-v1", "model": self.model.to_dict(),
            "model_hash": digest(self.model.to_dict()), "initialization_id": self.initialization_id,
            "objective": "split-step-moment-matched-observation-log-likelihood", "accuracy": "approximate",
            "objective_trace": list(self.objective_trace), "rejected_objectives": list(self.rejected_objectives),
            "stopping_reason": self.stopping_reason, "posterior_occupancy": self.occupancy.tolist(),
            "configured_K": self.model.n_modes, "occupied_K": int(np.sum(self.occupancy >= self.inference_spec['min_occupancy'])),
            "collapsed": self.collapsed, "inference_spec": self.inference_spec,
            "indistinguishable_modes": self.indistinguishable_modes,
            "resume_level": "numerical-tolerance"}
        if include_posteriors:
            value['posteriors'] = [result.to_dict() for result in self.filtered]
        return value


def _filters(observations, model, probability, means, covariance, spec, check_budget):
    rate = float(-model.Q.diagonal().min())
    limit = min(spec.max_step_seconds, spec.max_rate_step / rate) if rate else spec.max_step_seconds
    steps = sum(math.ceil(float(dt) / limit) for dt in np.diff(observations.times))
    if steps * len(observations.values) * model.n_modes**2 * 4**2 * 12 > 20_000_000:
        raise SwitchingError("INFERENCE_INAPPLICABLE", "multi-path candidate workspace exceeds the frozen quota")
    return tuple(filter_switching(observations, model, probability, means, covariance, spec,
        path_index=index, check_budget=check_budget) for index in range(len(observations.values)))


def _proposal(filtered, model, spec, check_budget):
    k = model.n_modes
    xx, ax, residual_sums = np.zeros((k, 5, 5)), np.zeros((k, 2, 5)), np.zeros((k, 2, 2))
    dwell, counts, occupancy, mass = np.zeros(k), np.zeros((k, k)), np.zeros(k), np.zeros(k)
    bridge_cache, transitions = {}, []
    for result in filtered:
        smoothed = smooth_switching(result, check_budget=check_budget)
        for step_index, step in enumerate(result.steps):
            check_budget()
            dt = step['dt']
            if dt not in bridge_cache:
                bridge_cache[dt] = ctmc_bridge_moments(model, dt)
            dwell_moment, jump_moment = bridge_cache[dt]
            weights = smoothed.pair_probabilities[step_index]
            dwell += np.einsum('ij,aij->a', weights, dwell_moment)
            counts += np.einsum('ij,abij->ab', weights, jump_moment)
            occupancy += smoothed.probabilities[step_index] * dt
            for source in range(k):
                for target in range(k):
                    weight = weights[source, target]
                    if weight <= 0:
                        continue
                    m0, c0 = smoothed.pair_means[step_index, source, target], smoothed.pair_covariances[step_index, source, target]
                    m1, c1 = smoothed.means[step_index+1, target], smoothed.covariances[step_index+1, target]
                    cross = smoothed.lag_covariances[step_index, source, target]
                    x = np.concatenate((m0, [1.0]))
                    exx = np.outer(x, x)
                    exx[:4, :4] += c0
                    change = m1[2:] - m0[2:]
                    edx = np.outer(change, x)
                    edx[:, :4] += cross.T[2:] - c0[2:]
                    edd = (c1[2:, 2:] + c0[2:, 2:] - cross[2:, 2:] - cross[2:, 2:].T
                           + np.outer(change, change))
                    xx[source] += weight * dt * exx
                    ax[source] += weight * edx
                    transitions.append((source, weight, dt, exx, edx, edd))
                    mass[source] += weight * dt
    coefficients = np.array([np.concatenate((model.A[i].numpy(), model.B[i].numpy(), model.a[i].numpy()[:, None]), axis=1) for i in range(k)])
    for source in range(k):
        if mass[source] > 0:
            coefficients[source] = np.linalg.solve(xx[source] + np.eye(5) * spec.ridge, ax[source].T).T
    for source, weight, dt, exx, edx, edd in transitions:
        coefficient = coefficients[source]
        residual = edd - dt * (coefficient @ edx.T + edx @ coefficient.T) + dt**2 * coefficient @ exx @ coefficient.T
        residual_sums[source] += weight * residual
    noise = model.L.detach().cpu().numpy().copy()
    for source in range(k):
        if mass[source] > 0:
            covariance = residual_sums[source] / mass[source]
            values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
            noise[source] = (vectors * np.sqrt(np.maximum(values, spec.min_diffusion_variance))) @ vectors.T
    q = np.zeros((k, k))
    lower, upper = model.rate_bounds
    for source in range(k):
        for target in range(k):
            if target != source:
                q[source, target] = np.clip((counts[source, target] + spec.rate_prior_count)
                    / (dwell[source] + spec.rate_prior_dwell), lower, upper)
        q[source, source] = -q[source].sum()
    proposed = SwitchingSDEModel(q, coefficients[:, :, :2], coefficients[:, :, 2:4], coefficients[:, :, 4], noise,
                                rate_bounds=model.rate_bounds)
    occupancy /= occupancy.sum()
    return proposed, occupancy


def _interpolate(old, new, fraction):
    arrays = [(1-fraction) * getattr(old, name).numpy() + fraction * getattr(new, name).numpy()
              for name in ('Q', 'A', 'B', 'a')]
    noise = []
    for left, right in zip(old.L.numpy(), new.L.numpy()):
        covariance = (1-fraction) * (left @ left.T) + fraction * (right @ right.T)
        values, vectors = np.linalg.eigh(covariance)
        noise.append((vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T)
    return SwitchingSDEModel(*arrays, np.asarray(noise), rate_bounds=old.rate_bounds)


def fit_switching(observations: ObservationBatch, model: SwitchingSDEModel,
                  initial_probability, initial_means, initial_covariances, spec: InferenceSpec,
                  *, initialization_id: str, check_budget: Callable[[], None],
                  progress_callback=None, resume_state=None):
    spec.validate()
    if not isinstance(initialization_id, str) or not initialization_id:
        raise SwitchingError("CONTRACT_MISMATCH", "initialization identity is required")
    if type(observations) is not ObservationBatch or not 1 <= len(observations.values) <= 10_000:
        raise SwitchingError("CONTRACT_MISMATCH", "fit accepts bounded observation-only paths")
    rate = float(-model.Q.diagonal().min())
    limit = min(spec.max_step_seconds, spec.max_rate_step / rate) if rate else spec.max_step_seconds
    steps = sum(math.ceil(float(dt) / limit) for dt in np.diff(observations.times))
    # Include backward pair means/covariances, lag moments and proposal records.
    if steps * len(observations.values) * model.n_modes**2 * 4**2 * 12 > 20_000_000:
        raise SwitchingError("INFERENCE_INAPPLICABLE", "multi-path EM workspace exceeds the frozen quota")
    identity = {"data_hash": digest(observations.to_dict()), "spec_hash": digest(spec.to_dict()),
        "initial_distribution_hash": digest([np.asarray(value).tolist()
            for value in (initial_probability, initial_means, initial_covariances)]),
        "initialization_id": initialization_id, "initial_model_hash": digest(model.to_dict())}
    current = SwitchingSDEModel.from_dict(model.to_dict())
    trace, rejected, completed_iterations, complete_reason = [], [], 0, None
    if resume_state is not None:
        required = {'schema_version', 'identity', 'model', 'objective_trace', 'rejected_objectives',
                    'completed_iterations', 'complete_reason'}
        if (type(resume_state) is not dict or set(resume_state) != required
                or resume_state['schema_version'] != 'pirc28-em-checkpoint-v1' or resume_state['identity'] != identity):
            raise SwitchingError('IDENTITY_CONFLICT', 'EM checkpoint data, model, objective or initialization differs')
        completed_iterations = resume_state['completed_iterations']
        trace, rejected = list(resume_state['objective_trace']), list(resume_state['rejected_objectives'])
        complete_reason = resume_state['complete_reason']
        if (type(completed_iterations) is not int or not 0 <= completed_iterations <= spec.max_iterations
                or len(trace) != completed_iterations + 1
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in trace + rejected)
                or np.any(np.diff(trace) < 0)
                or complete_reason not in {None, 'MAX_ITERATIONS', 'REJECTED_STEP', 'CONVERGED_APPROXIMATE_OBJECTIVE'}):
            raise SwitchingError('CONTRACT_MISMATCH', 'EM checkpoint progress is inconsistent')
        current = SwitchingSDEModel.from_dict(resume_state['model'])
        if current.n_modes != model.n_modes or current.rate_bounds != model.rate_bounds:
            raise SwitchingError('IDENTITY_CONFLICT', 'EM checkpoint model family differs')
    filtered = _filters(observations, current, initial_probability, initial_means, initial_covariances, spec, check_budget)
    objective = sum(result.log_likelihood for result in filtered)
    if trace and not math.isclose(trace[-1], objective, rel_tol=1e-9, abs_tol=1e-9):
        raise SwitchingError('IDENTITY_CONFLICT', 'recomputed checkpoint objective differs')
    if not trace:
        trace = [objective]

    def report(reason=None):
        if progress_callback is not None:
            progress_callback({"schema_version": 'pirc28-em-checkpoint-v1', 'identity': identity,
                'model': current.to_dict(), 'objective_trace': list(trace), 'rejected_objectives': list(rejected),
                'completed_iterations': completed_iterations, 'complete_reason': reason})

    reason = complete_reason or 'MAX_ITERATIONS'
    report(complete_reason)
    for _ in range(completed_iterations, spec.max_iterations) if complete_reason is None else ():
        check_budget()
        proposed, _ = _proposal(filtered, current, spec, check_budget)
        accepted = None
        for search_step in range(spec.line_search_steps):
            check_budget()
            candidate = _interpolate(current, proposed, 0.5**search_step)
            results = _filters(observations, candidate, initial_probability, initial_means, initial_covariances, spec, check_budget)
            objective = sum(result.log_likelihood for result in results)
            if objective >= trace[-1]:
                accepted = candidate, results, objective
                break
            rejected.append(objective)
        if accepted is None:
            reason = 'REJECTED_STEP'
            break
        current, filtered, objective = accepted
        improvement = objective - trace[-1]
        trace.append(objective)
        completed_iterations += 1
        report()
        if improvement <= spec.objective_tolerance * (1 + abs(objective)):
            reason = 'CONVERGED_APPROXIMATE_OBJECTIVE'
            break
    occupancy = np.zeros(current.n_modes)
    for result in filtered:
        smooth = smooth_switching(result, check_budget=check_budget)
        occupancy += np.sum(smooth.probabilities[:-1] * np.diff(result.times)[:, None], axis=0)
    occupancy /= occupancy.sum()
    indistinguishable = False
    if current.n_modes == 2:
        groups = [current.A.numpy(), current.B.numpy(), current.a.numpy(),
                  current.L.numpy() @ np.swapaxes(current.L.numpy(), -1, -2)]
        indistinguishable = all(np.max(np.abs(group[0]-group[1])) <= spec.identical_mode_tolerance for group in groups)
    check_budget()
    report(reason)
    return SwitchingFit(current, tuple(trace), reason, occupancy,
        bool(current.n_modes > 1 and (occupancy.min() < spec.min_occupancy or indistinguishable)), filtered,
        initialization_id, spec.to_dict(), tuple(rejected), indistinguishable)


def select_initialization(fits, selection: ObservationBatch, initial_probability,
                          initial_means, initial_covariances, spec, *, split_role,
                          check_budget: Callable[[], None]):
    """Retain every candidate; choose on explicit selection inputs, never test."""
    spec.validate()
    if split_role != 'selection':
        raise SwitchingError("UNAUTHORIZED_DATA", "initialization selection requires the selection split")
    if not 1 <= len(fits) <= spec.max_initializations:
        raise SwitchingError("CONTRACT_MISMATCH", "initialization count exceeds preregistered quota")
    if len({fit.initialization_id for fit in fits}) != len(fits):
        raise SwitchingError("IDENTITY_CONFLICT", "initialization IDs must be unique")
    candidates, eligible = [], []
    for fit in fits:
        check_budget()
        candidate = {'initialization_id': fit.initialization_id, 'collapsed': fit.collapsed,
                     'stopping_reason': fit.stopping_reason}
        try:
            results = _filters(selection, fit.model, initial_probability, initial_means, initial_covariances, spec, check_budget)
            score = float(sum(result.log_likelihood for result in results))
            if not math.isfinite(score):
                raise SwitchingError('NONFINITE', 'selection objective is nonfinite')
        except NUMERICAL_EXCEPTIONS as error:
            candidate.update(numerical_diagnostic(error, 'selection', fit.initialization_id), score=None)
        else:
            candidate.update(state='SUCCEEDED', score=score)
            eligible.append((score, -len(candidates), fit))
        candidates.append(candidate)
    receipt = {"schema_version": "pirc28-initialization-selection-v1", "split_role": split_role,
        "selection_hash": digest(selection.to_dict()), "rule": "largest-approximate-selection-objective-stable-tie",
        "selected_initialization": None, "candidates": candidates}
    if not eligible:
        raise NumericalCellFailure('selection', [fit.to_dict(include_posteriors=False) for fit in fits], receipt)
    winner = max(eligible, key=lambda value: value[:2])[2]
    receipt['selected_initialization'] = winner.initialization_id
    return winner, receipt
