"""Synthetic clock/covariance arrays only, not new trajectory experiments."""
import copy

import pytest

from experiments.pirc17.terrain_fit_scope_description import (
    describe_intervals, first_interval_ns, saved_q_description)


def sample_rows():
    sample = {"split": "train", "file_id": "file", "segment_id": "segment",
              "independent_block_id": "block", "history_end": 4, "target_start": 5, "target_end": 6}
    rows = [{"dataset_version": "dataset", "file_id": "file", "segment_id": "segment",
             "split": "train", "independent_block_id": "block", "point_index": 100 + index,
             "absolute_epoch_ns": 1_700_000_000_000_000_000 + seconds * 1_000_000_000}
            for index, seconds in enumerate([-20, -15, -10, -5, 0, 7, 14])]
    return sample, rows


def test_first_interval_is_actual_adjacent_ticks_not_tau_or_float_epoch_subtraction():
    s, rows = sample_rows()
    rows[5]["absolute_epoch_ns"] += 1
    assert first_interval_ns(s, rows, "dataset") == 7_000_000_001


@pytest.mark.parametrize("field,value", [("split", "final_eval"), ("history_end", True),
                                       ("target_start", 6)])
def test_wrong_role_or_nonadjacent_sample(field, value):
    s, rows = sample_rows(); s[field] = value
    with pytest.raises(ValueError): first_interval_ns(s, rows, "dataset")


@pytest.mark.parametrize("field,value", [("dataset_version", "other"), ("file_id", "other"),
                                       ("split", "validation"), ("segment_id", "other"),
                                       ("independent_block_id", "other"), ("absolute_epoch_ns", 1.2),
                                       ("point_index", True)])
def test_wrong_original_row_binding(field, value):
    s, rows = sample_rows(); rows[4][field] = value
    with pytest.raises(ValueError): first_interval_ns(s, rows, "dataset")


@pytest.mark.parametrize("delta", [0, -1, 60_000_000_001])
def test_clock_support_rejects_nonpositive_or_over_sixty_seconds(delta):
    s, rows = sample_rows()
    rows[5]["absolute_epoch_ns"] = rows[4]["absolute_epoch_ns"] + delta
    with pytest.raises(ValueError, match="support"): first_interval_ns(s, rows, "dataset")


@pytest.mark.parametrize("duplicate", [False, True])
def test_missing_or_duplicate_first_tick_is_not_imputed(duplicate):
    s, rows = sample_rows()
    rows = rows + [copy.deepcopy(rows[5])] if duplicate else rows[:5]
    with pytest.raises(ValueError, match="missing or duplicate"): first_interval_ns(s, rows, "dataset")


def test_quantiles_and_bins_are_descriptive_and_preserve_both_weights():
    r = describe_intervals([1_000_000_000, 5_000_000_000, 10_000_000_000, 60_000_000_000])
    assert r["transitions"] == 4 and r["sum_seconds"] == 76.
    assert r["median_seconds"] == 7.5 and r["q25_seconds"] == 4.
    assert r["q75_seconds"] == 22.5
    assert r["equal_drift_row_weight"] == .25
    assert r["rate_outer_product_Q_coefficient_minimum_seconds"] == .25
    assert r["rate_outer_product_Q_coefficient_maximum_seconds"] == 15.
    assert r["interval_bin_counts_seconds"] == {"(0,1]": 1, "(1,5]": 1, "(5,10]": 1, "(10,30]": 0, "(30,60]": 1}


def test_original_file_indices_are_not_segment_offsets_and_input_order_is_irrelevant():
    s, rows = sample_rows()
    assert rows[4]["point_index"] != s["history_end"]
    assert first_interval_ns(s, list(reversed(rows)), "dataset") == 7_000_000_000


@pytest.mark.parametrize("values", [[], [True], [1.5], [-1], [60_000_000_001]])
def test_invalid_duration_lists(values):
    with pytest.raises(ValueError): describe_intervals(values)


def test_saved_terrain_Q_is_not_rescaled_or_reestimated():
    r = saved_q_description({"configuration": "base", "diffusion_covariance_m2_per_s": [[2., 0.], [0., 1.]]})
    assert r["saved_eigenvalues_m2_per_s"] == [1., 2.]
    assert not r["negative_roundoff_clamp_required_for_L"]
    assert not r["positive_floor_or_jitter_added_by_terrain_diffusion_estimator"]


def test_terrain_negative_roundoff_rule_is_not_method_relative_tolerance():
    r = saved_q_description({"configuration": "base", "diffusion_covariance_m2_per_s": [[-1e-11, 0.], [0., 1.]]})
    assert r["negative_roundoff_clamp_required_for_L"]
    assert r["saved_eigenvalues_m2_per_s"][0] == -1e-11
    assert r["negative_eigenvalue_rejection_threshold_m2_per_s"] == -1e-10


@pytest.mark.parametrize("q", [[[1.]], [[1., .001], [0., 1.]], [[-1e-9, 0.], [0., 1.]],
                             [[float("inf"), 0.], [0., 1.]]])
def test_invalid_saved_terrain_covariance(q):
    with pytest.raises(ValueError): saved_q_description({"configuration": "base", "diffusion_covariance_m2_per_s": q})
