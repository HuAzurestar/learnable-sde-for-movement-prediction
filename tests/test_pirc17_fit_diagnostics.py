"""Small parser/statistics checks; no model fitting or SDE rollouts."""
import copy

import pytest

from experiments.pirc17.fit_diagnostics import terrain_diagnostics, transition_counts


def training():
    rows = [{"role": role, "reference_interval_seconds": 60,
             "transition_count": number, "removed_partial_tail": tail,
             "scoring_observations_changed": False, "segment_id": "private"}
            for role, number, tail in (("train", 4, True), ("adapt", 2, False),
                                       ("validation", 3, True))]
    return {"per_segment": rows, "reference_interval_seconds": 60,
            "sample_counts": {"train": 1, "adapt": 1, "validation": 1},
            "transitions_by_role": {"train": 4, "adapt": 2, "validation": 3},
            "removed_tails_by_role": {"train": 1, "adapt": 0, "validation": 1}}


def terrain():
    return {"configuration": "base", "train_transition_count": 4,
            "validation_transition_count": 1,
            "training_policy": {"base_ridge": 1e-6},
            "diffusion_covariance_m2_per_s": [[2., 0.], [0., 1.]],
            "conditioner_checkpoint": {"input_dim": 2, "solver_report": {
                "train_rows": 4, "input_dim": 2, "output_dim": 2,
                "weight_decay": 1e-4, "normal_equation_penalty": .0004,
                "augmented_rank": 3, "augmented_condition": 10.,
                "relative_gradient_norm": 1e-14}},
            "diagnostics": {role: {"base_only_mse_m2_per_s2": 1.,
                                   "fitted_mse_m2_per_s2": .9}
                            for role in ("train", "validation")}}


def test_counts_are_recomputed_without_identifiers():
    result = transition_counts(training())
    assert result["transitions"]["train"] == 4
    assert "private" not in str(result)


@pytest.mark.parametrize("field", ["sample_counts", "transitions_by_role", "removed_tails_by_role"])
def test_bad_aggregate_rejected(field):
    record = training()
    record[field]["train"] += 1
    with pytest.raises(ValueError, match="aggregate"):
        transition_counts(record)


@pytest.mark.parametrize("key,value", [("role", "final"), ("transition_count", -1),
                                      ("transition_count", 1.5), ("transition_count", True),
                                      ("reference_interval_seconds", 30),
                                      ("scoring_observations_changed", True),
                                      ("removed_partial_tail", 1)])
def test_bad_transition_rejected(key, value):
    record = training()
    record["per_segment"][0][key] = value
    with pytest.raises(ValueError):
        transition_counts(record)


def test_diffusion_spectrum_and_augmented_rank_label():
    result = terrain_diagnostics(terrain())
    assert result["diffusion_eigenvalues_m2_per_s"] == [1., 2.]
    assert "raw_feature_rank" not in result
    assert result["augmented_ridge_rank"] == 3


@pytest.mark.parametrize("q", [[[1., 2.], [0., 1.]], [[-1., 0.], [0., 1.]],
                             [[1.]], [[float('nan'), 0.], [0., 1.]]])
def test_invalid_covariance_rejected(q):
    record = terrain()
    record["diffusion_covariance_m2_per_s"] = copy.deepcopy(q)
    with pytest.raises(ValueError):
        terrain_diagnostics(record)


def test_wrong_ridge_normalization_rejected():
    record = terrain()
    record["conditioner_checkpoint"]["solver_report"]["normal_equation_penalty"] *= 2
    with pytest.raises(ValueError, match="normalization"):
        terrain_diagnostics(record)
