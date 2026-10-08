"""Software fixtures, not invented research samples."""
import numpy as np
import pytest

from experiments.pirc17.metrics import energy_score
from experiments.pirc17.nested_precision import energy_delete_groups, nested_energy_precision
from experiments.pirc17.precision import energy_delete_one


@pytest.mark.parametrize("groups,multiple", [(3, 1), (3, 2), (5, 3)])
def test_group_deletions_match_every_brute_force_energy_score(groups, multiple):
    rng = np.random.default_rng(20261001)
    paths, targets = rng.normal(size=(groups*multiple, 4, 2)), rng.normal(size=(4, 2))
    scores, deleted = energy_delete_groups(paths, targets, groups, chunk_size=2)
    np.testing.assert_allclose(scores, [energy_score(paths[:, t], targets[t]) for t in range(4)], atol=1e-12)
    for group in range(groups):
        remaining = np.delete(paths, np.arange(group, len(paths), groups), axis=0)
        np.testing.assert_allclose(deleted[group], [energy_score(remaining[:, t], targets[t]) for t in range(4)], atol=1e-12)


def test_shared_prefix_covariance_matches_explicit_joint_deletions():
    rng = np.random.default_rng(24)
    large, targets = rng.normal(size=(16, 3, 2)), rng.normal(size=(3, 2))
    small = large[:8]
    report = nested_energy_precision(large, small, targets, [.2, .3, .5])
    values = []
    for i in range(8):
        left, right = np.delete(large, [i, i+8], axis=0), np.delete(small, i, axis=0)
        values.append([energy_score(left[:, t], targets[t])-energy_score(right[:, t], targets[t]) for t in range(3)])
    values = np.asarray(values)
    centered = values-values.mean(axis=0)
    covariance = 7/8*centered.T@centered
    np.testing.assert_allclose(report["time_covariance_m2"], covariance, atol=1e-12)
    weights = np.array([.2, .3, .5])
    assert report["time_weighted_standard_error_m"] == pytest.approx(np.sqrt(weights@covariance@weights))
    assert report["jackknife_group_count"] == 8 and report["candidate_particle_count"] == 16


def test_same_budget_is_exact_zero_and_one_group_path_matches_original():
    paths = np.random.default_rng(11).normal(size=(8, 2, 2))
    targets = np.zeros((2, 2))
    a, b = energy_delete_groups(paths, targets, 8)
    x, y = energy_delete_one(paths, targets)
    np.testing.assert_array_equal(a, x)
    np.testing.assert_array_equal(b, y)
    report = nested_energy_precision(paths, paths, targets, [.5, .5])
    assert report["time_weighted_standard_error_m"] == 0
    assert report["time_weighted_energy_score_m"] == 0


def test_conditional_variance_agrees_with_uniform_u_statistic_oracle():
    # Software-only X~Uniform(0,1), target=1. ES is the U-statistic with
    # h(x,z)=1-max(x,z); Hoeffding component variances are 1/45 and 1/90.
    # Var(ES_n)=4/(45*n)+1/(45*n*(n-1)). For a shared prefix,
    # Cov(ES_n,ES_N)=Var(ES_N), so Var(ES_N-ES_n) is the difference,
    # not the sum. This checks an asymptotic SE, not finite-sample coverage.
    rng = np.random.default_rng(20260928)
    values, variances = [], []
    for _ in range(2000):
        large = rng.uniform(size=(32, 1, 1))
        report = nested_energy_precision(large, large[:16], np.ones((1, 1)), [1.])
        values.append(report["time_weighted_energy_score_m"])
        variances.append(report["time_weighted_standard_error_m"]**2)
    def variance(n):
        return 4/(45*n)+1/(45*n*(n-1))
    exact = variance(16)-variance(32)
    assert .85 < np.var(values, ddof=1)/exact < 1.15
    assert .85 < np.mean(variances)/exact < 1.15


@pytest.mark.parametrize("change", ["prefix", "ratio", "target", "nonfinite", "dimensions"])
def test_unsupported_or_unpaired_evidence_rejected(change):
    large = np.random.default_rng(5).normal(size=(8, 2, 2))
    small, targets = large[:4].copy(), np.zeros((2, 2))
    if change == "prefix":
        small[0, 0, 0] += 1
    elif change == "ratio":
        large = large[:7]
    elif change == "target":
        targets = targets[:, :1]
    elif change == "nonfinite":
        large[-1, 0, 0] = np.nan
    else:
        small = small[:, :, :1]
    with pytest.raises(ValueError):
        nested_energy_precision(large, small, targets, [.5, .5])
