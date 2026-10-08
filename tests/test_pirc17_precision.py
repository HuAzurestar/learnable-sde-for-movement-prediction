import numpy as np
import pytest

from experiments.pirc17.metrics import energy_score
from experiments.pirc17.precision import energy_delete_one, energy_precision, paired_energy_precision


def test_deletions_match_direct_energy_recomputation_and_pseudovalue_variance():
    rng = np.random.default_rng(17)
    paths, truth = rng.normal(size=(11, 3, 2)), rng.normal(size=(3, 2))
    weights = np.array([.2, .3, .5])
    scores, deleted = energy_delete_one(paths, truth, chunk_size=4)
    expected = np.array([[energy_score(np.delete(paths[:, t], k, axis=0), truth[t])
                         for t in range(3)] for k in range(11)])
    np.testing.assert_allclose(deleted, expected, rtol=1e-13, atol=1e-13)
    direct_scores = np.array([energy_score(paths[:, t], truth[t]) for t in range(3)])
    np.testing.assert_allclose(scores, direct_scores, rtol=1e-13, atol=1e-13)
    pseudo = len(paths)*(direct_scores@weights)-(len(paths)-1)*(expected@weights)
    report = energy_precision(paths, truth, weights)
    assert report["time_weighted_standard_error_m"] == pytest.approx(pseudo.std(ddof=1)/np.sqrt(len(paths)))


def test_paired_paths_preserve_covariance_and_identical_forecasts_have_zero_error():
    rng = np.random.default_rng(91)
    control = rng.normal(size=(17, 2, 2))
    candidate = control*np.array([1., 2.])+.4
    truth = np.zeros((2, 2))
    weights = np.array([.25, .75])
    result = paired_energy_precision(candidate, control, truth, weights)
    deleted = np.array([[energy_score(np.delete(candidate[:, t], k, axis=0), truth[t])
                         -energy_score(np.delete(control[:, t], k, axis=0), truth[t])
                         for t in range(2)] for k in range(17)])@weights
    assert result["time_weighted_standard_error_m"] == pytest.approx(np.sqrt(16/17*np.sum((deleted-deleted.mean())**2)))
    same = paired_energy_precision(control, control, truth, weights)
    assert same["time_weighted_standard_error_m"] == 0
    assert same["time_weighted_energy_score_m"] == 0


def test_physical_units_scale_the_score_and_error_together():
    paths = np.arange(24, dtype=float).reshape(6, 2, 2)
    truth = np.ones((2, 2))
    a = energy_precision(paths, truth, [.5, .5])
    b = energy_precision(paths*1000, truth*1000, [.5, .5])
    assert b["time_weighted_standard_error_m"] == pytest.approx(a["time_weighted_standard_error_m"]*1000)


@pytest.mark.parametrize("paths,truth,weights", [
    (np.zeros((2, 2, 2)), np.zeros((2, 2)), [.5, .5]),
    (np.full((4, 2, 2), np.nan), np.zeros((2, 2)), [.5, .5]),
    (np.zeros((4, 2, 2)), np.zeros((2, 2)), [.5, .6]),
])
def test_invalid_ensembles_or_weights_cannot_be_qualified(paths, truth, weights):
    with pytest.raises(ValueError):
        energy_precision(paths, truth, weights)


def test_particle_artifact_is_exact_exclusive_and_hash_bound(tmp_path):
    import hashlib
    from experiments.pirc17.development_rollout import save_particle_artifact
    row = dict(sample_id="origin", configuration="base", seed=20260814, particles=3, max_step_seconds=.625)
    paths = np.arange(12, dtype=float).reshape(3, 2, 2)
    targets, horizons = np.zeros((2, 2)), np.array([1., 2.])
    ledger = tmp_path/"pilot.jsonl"
    identity = save_particle_artifact(ledger, row, paths, targets, horizons)
    artifact = tmp_path/identity["path"]
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == identity["sha256"]
    with np.load(artifact, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["positions_m"], paths)
        np.testing.assert_array_equal(saved["target_positions_m"], targets)
        np.testing.assert_array_equal(saved["elapsed_seconds"], horizons)
    with pytest.raises(FileExistsError):
        save_particle_artifact(ledger, row, paths+1, targets, horizons)
