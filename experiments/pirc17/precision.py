"""Conditional particle Monte Carlo error, distinct from independent-block inference.

Delete one entire simulated path at a time. This preserves dependence across
scoring times and, for paired ensembles, across configurations/integration steps.
Jackknife standard errors are asymptotic diagnostics, not error guarantees.
"""
from __future__ import annotations

import numpy as np

PRECISION_VERSION = "pirc17-energy-path-jackknife-v1"


def energy_delete_one(particles, target, *, chunk_size=128):
    x, y = np.asarray(particles, dtype=float), np.asarray(target, dtype=float)
    if x.ndim != 3 or y.shape != x.shape[1:] or len(x) < 3 or not x.shape[1] or not x.shape[2]:
        raise ValueError("at least three paths (N,T,D) and matching observed targets required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("finite paths and targets required")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("positive integer chunk size required")
    n = len(x)
    scores, omitted = [], []
    for time in range(x.shape[1]):
        samples = x[:, time]
        to_target = np.linalg.norm(samples-y[time], axis=-1)
        distance_rows = np.empty(n)
        for start in range(0, n, chunk_size):
            distance_rows[start:start+chunk_size] = np.linalg.norm(
                samples[start:start+chunk_size, None]-samples[None], axis=-1,
            ).sum(axis=1)
        unordered_sum = distance_rows.sum()/2
        scores.append(to_target.mean()-unordered_sum/(n*(n-1)))
        omitted.append((to_target.sum()-to_target)/(n-1)
                       -(unordered_sum-distance_rows)/((n-1)*(n-2)))
    return np.asarray(scores), np.stack(omitted, axis=1)


def summarize_delete_one(scores, deleted, time_weights):
    values, leave_one = np.asarray(scores, dtype=float), np.asarray(deleted, dtype=float)
    weights = np.asarray(time_weights, dtype=float)
    if (values.ndim != 1 or weights.shape != values.shape or not len(values)
            or leave_one.ndim != 2 or leave_one.shape[1:] != values.shape or len(leave_one) < 3):
        raise ValueError("matching score/time/path dimensions required")
    if (not all(np.isfinite(v).all() for v in (values, leave_one, weights)) or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12)):
        raise ValueError("finite scores and normalized nonnegative weights required")
    n = len(leave_one)
    centered = leave_one-leave_one.mean(axis=0)
    covariance = (n-1)/n * centered.T@centered
    return {"schema_version": PRECISION_VERSION, "particle_count": n,
            "by_time_energy_score_m": values.tolist(),
            "time_weighted_energy_score_m": float(weights@values),
            "by_time_standard_error_m": np.sqrt(np.maximum(0, covariance.diagonal())).tolist(),
            "time_weighted_standard_error_m": float(np.sqrt(max(0, weights@covariance@weights))),
            "time_covariance_m2": covariance.tolist(),
            "scope": "conditional particle Monte Carlo error; not block sampling or training uncertainty",
            "method": "delete-one-path jackknife of unbiased energy U-statistic; asymptotic diagnostic"}


def energy_precision(particles, target, time_weights):
    scores, deleted = energy_delete_one(particles, target)
    return summarize_delete_one(scores, deleted, time_weights)


def paired_energy_precision(candidate, control, target, time_weights):
    """Caller must first verify the exact shared origin/seed/Brownian identity.

    Path index k must denote the same random draw in both arrays. Never estimate
    paired Monte Carlo error by treating the two ensembles as independent.
    """
    if np.shape(candidate) != np.shape(control):
        raise ValueError("paired ensembles must have the same path/time dimensions")
    first, first_deleted = energy_delete_one(candidate, target)
    second, second_deleted = energy_delete_one(control, target)
    result = summarize_delete_one(first-second, first_deleted-second_deleted, time_weights)
    result["sign_convention"] = "candidate_ES_minus_control_ES"
    return result
