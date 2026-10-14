"""Synthetic numeric fixtures only; no saved source data, fitting or forecasts."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.pirc17.source_sequence_description import ROLES, fingerprints, summarize


def sample(role, file_id, xy=None, clock=None):
    return {"role": role, "file_id": file_id, "recording_hash_block": file_id,
            **fingerprints([[1., 2.], [3., 4.]] if xy is None else xy,
                           np.array([1, 2] if clock is None else clock, dtype=np.int64))}


def test_coordinates_can_match_without_clocks_matching():
    first = sample("train", "a")
    second = sample("validation", "b", clock=[2, 3])
    assert first["coordinate_fingerprint"] == second["coordinate_fingerprint"]
    assert first["coordinate_clock_fingerprint"] != second["coordinate_clock_fingerprint"]


@pytest.mark.parametrize("xy", [
    [[3., 4.], [1., 2.]],  # Row order is not normalized.
    [[11., 12.], [13., 14.]],  # No translation normalization.
    [[1., 2.], [3., np.nextafter(4., 5.)]],  # No tolerance or rounding.
    [[1., 2.], [3., 4.], [3., 4.]],  # No repeated-point removal.
])
def test_numeric_agreement_is_not_approximate_route_matching(xy):
    base = sample("train", "a")
    changed = sample("validation", "b", xy=xy, clock=list(range(len(xy))))
    assert base["coordinate_fingerprint"] != changed["coordinate_fingerprint"]


def test_signed_zero_and_nan_are_explicit_normalizations_not_dropped_points():
    first = fingerprints([[0., np.nan], [np.inf, 3.]], np.array([1, 2], dtype=np.int64))
    second = fingerprints([[-0., -np.nan], [np.inf, 3.]], np.array([1, 2], dtype=np.int64))
    assert first == second
    assert first["points"] == 2
    assert first["nonfinite_coordinate_rows"] == 2
    changed = fingerprints([[0., np.nan], [-np.inf, 3.]], np.array([1, 2], dtype=np.int64))
    assert first["coordinate_fingerprint"] != changed["coordinate_fingerprint"]


def test_nat_is_counted_and_retained_not_repaired():
    first = fingerprints([[1., 2.], [3., 4.]], np.array([1, np.iinfo(np.int64).min]))
    assert first["missing_clock_rows"] == 1
    assert first["coordinate_clock_fingerprint"] != sample("train", "a")["coordinate_clock_fingerprint"]


@pytest.mark.parametrize("xy,ticks", [
    ([], np.array([], dtype=np.int64)),
    ([1., 2.], np.array([1, 2], dtype=np.int64)),
    ([[1., 2.]], np.array([1, 2], dtype=np.int64)),
    ([[1., 2.]], np.array([1.], dtype=float)),
    ([[1., 2.]], np.array([[1]], dtype=np.int64)),
])
def test_bad_shape_or_noninteger_clocks_are_not_silently_repaired(xy, ticks):
    with pytest.raises(ValueError):
        fingerprints(xy, ticks)


def test_file_pairs_and_shared_groups_have_different_denominators():
    records = [sample(role, role) for role in ROLES]
    records.append(sample("train", "extra"))
    result = summarize(records)
    assert result["roles"]["train"]["source_files"] == 2
    assert result["roles"]["train"]["coordinate_fingerprint"] == {
        "equal_sequence_file_pairs": 1, "equal_sequence_groups": 1,
        "files_in_equal_sequence_groups": 2,
        "equal_sequence_pairs_outside_same_recording_hash_block": 1}
    pair = result["cross_role"]["train:validation"]
    assert pair["possible_source_file_pairs"] == 2
    assert pair["coordinate_fingerprint"] == {
        "equal_sequence_file_pairs": 2, "shared_sequence_groups": 1,
        "matching_files_left": 2, "matching_files_right": 1}
    assert len(result["cross_role"]) == 6
    assert "file_id" not in str(result) and "extra" not in str(result)


def test_full_clock_matching_is_subset_of_coordinate_matching():
    records = [sample(role, role) for role in ROLES]
    records.append(sample("train", "later", clock=[11, 12]))
    pair = summarize(records)["cross_role"]["train:validation"]
    assert pair["coordinate_fingerprint"]["equal_sequence_file_pairs"] == 2
    assert pair["coordinate_clock_fingerprint"]["equal_sequence_file_pairs"] == 1


@pytest.mark.parametrize("change", ["missing", "bad_role", "duplicate"])
def test_missing_roles_or_duplicate_file_identities_do_not_create_successful_subset(change):
    records = [sample(role, role) for role in ROLES]
    if change == "missing":
        records.pop()
    elif change == "bad_role":
        records[-1]["role"] = "other"
    else:
        records[-1]["file_id"] = "train"
    with pytest.raises(ValueError):
        summarize(records)


def test_pair_denominators_are_all_files_not_equal_fingerprint_subset():
    records = [sample(role, role) for role in ROLES]
    records[0] = sample("train", "train", xy=[[9., 10.], [11., 12.]])
    pair = summarize(records)["cross_role"]["train:final_eval"]
    assert pair["possible_source_file_pairs"] == 1
    assert pair["coordinate_fingerprint"]["equal_sequence_file_pairs"] == 0


def test_summary_does_not_mutate_source_fixture():
    records = [sample(role, role) for role in ROLES]
    before = deepcopy(records)
    summarize(records)
    assert records == before


def test_already_grouped_recording_duplicates_are_not_reported_as_extra_block_matches():
    records = [sample(role, role) for role in ROLES]
    records.append(sample("train", "alias"))
    records[-1]["recording_hash_block"] = "train"
    row = summarize(records)["roles"]["train"]["coordinate_fingerprint"]
    assert row["equal_sequence_file_pairs"] == 1
    assert row["equal_sequence_pairs_outside_same_recording_hash_block"] == 0
