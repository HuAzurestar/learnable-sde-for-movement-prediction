"""Shared-prefix particle-budget uncertainty via independent compound paths.

For N = m*n, group i is (i, i+n, ..., i+(m-1)*n). Each group contains
one shared path and m-1 extra paths. Delete that whole group from N and its
shared path from n together. The statistic is symmetric in these iid groups.
The jackknife is an asymptotic conditional diagnostic, not a finite-sample
error bound. Groups here are simulated draws, NOT research-independent blocks.
"""
from __future__ import annotations

import numpy as np

from .precision import energy_delete_one, summarize_delete_one

NESTED_PRECISION_VERSION = "pirc17-energy-nested-group-jackknife-v1"


def energy_delete_groups(particles, target, groups, *, chunk_size=128):
    """Exact recomputation of ES after each fixed strided group deletion."""
    x, y = np.asarray(particles, dtype=float), np.asarray(target, dtype=float)
    if (x.ndim != 3 or y.shape != x.shape[1:] or not x.shape[1] or not x.shape[2]
            or type(groups) is not int or groups < 3 or len(x) < groups or len(x) % groups):
        raise ValueError("at least three equal nonempty groups of finite paths required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("finite paths and targets required")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("positive integer chunk size required")
    count, group_size = len(x), len(x)//groups
    indexes = np.arange(count).reshape(group_size, groups).T
    remaining = count-group_size
    scores, deleted = [], []
    for time in range(x.shape[1]):
        samples = x[:, time]
        to_target = np.linalg.norm(samples-y[time], axis=-1)
        distance_rows = np.empty(count)
        for start in range(0, count, chunk_size):
            distance_rows[start:start+chunk_size] = np.linalg.norm(
                samples[start:start+chunk_size, None]-samples[None], axis=-1).sum(axis=1)
        unordered_sum = distance_rows.sum()/2
        grouped = samples[indexes]
        within_unordered = np.linalg.norm(grouped[:, :, None]-grouped[:, None, :], axis=-1).sum(axis=(1, 2))/2
        retained_unordered = unordered_sum-distance_rows[indexes].sum(axis=1)+within_unordered
        scores.append(to_target.mean()-unordered_sum/(count*(count-1)))
        deleted.append((to_target.sum()-to_target[indexes].sum(axis=1))/remaining
                       -retained_unordered/(remaining*(remaining-1)))
    return np.asarray(scores), np.stack(deleted, axis=1)


def nested_energy_precision(candidate, control, target, time_weights):
    """Caller must also verify common origin/seed/Brownian and scoring identities."""
    large, small = np.asarray(candidate, dtype=float), np.asarray(control, dtype=float)
    if (large.ndim != 3 or small.ndim != 3 or large.shape[1:] != small.shape[1:]
            or len(small) < 3 or len(large) < len(small) or len(large) % len(small)):
        raise ValueError("nested budgets must have an integer size ratio and at least three shared paths")
    if not np.array_equal(large[:len(small)], small):
        raise ValueError("nested particle paths must preserve the exact shared prefix")
    first, first_deleted = energy_delete_groups(large, target, len(small))
    second, second_deleted = energy_delete_one(small, target)
    report = summarize_delete_one(first-second, first_deleted-second_deleted, time_weights)
    report.pop("particle_count")
    report.update(schema_version=NESTED_PRECISION_VERSION, candidate_particle_count=len(large),
        control_particle_count=len(small), jackknife_group_count=len(small), paths_per_group=len(large)//len(small),
        sign_convention="larger_budget_ES_minus_smaller_budget_ES",
        grouping="group i contains path i+k*n for k=0..N/n-1; fixed before values",
        method="delete-one-iid-compound-path-group jackknife; asymptotic conditional diagnostic",
        scope="conditional particle Monte Carlo error with shared prefix; not research-block or training uncertainty")
    return report
