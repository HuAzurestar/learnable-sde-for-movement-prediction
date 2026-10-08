"""Synthetic clock-only reader contracts; no new trajectory experiments."""
import copy

import pytest

from experiments.pirc17.development_history_description import (
    NS, describe_role, describe_spans, prefix_clock_record)
from tests.test_pirc17_terrain_fit_scope_description import sample_rows


def fixture():
    sample, rows = sample_rows()
    sample["history_start"] = 0
    return sample, rows


def test_last_three_visible_points_ignore_earlier_history_and_future_clock():
    s, rows = fixture()
    rows[0]["absolute_epoch_ns"] -= 900 * NS
    record = prefix_clock_record(s, rows, "dataset")
    assert record == {"points": 3, "origin_span_ns": 10 * NS,
                      "last_gap_ns": 5 * NS, "first_tick_span_ns": 10 * NS}
    # Saved first-future interval is seven seconds, NOT the five-second
    # model history tick; no future timestamp contributes to the prefix.
    rows[5]["absolute_epoch_ns"] += NS
    assert prefix_clock_record(s, rows, "dataset") == record


def test_two_visible_points_are_not_fabricated_into_three():
    s, rows = fixture(); s["history_start"] = 3
    assert prefix_clock_record(s, rows, "dataset") == {
        "points": 2, "origin_span_ns": 5 * NS, "last_gap_ns": 5 * NS,
        "first_tick_span_ns": 10 * NS}


def test_order_and_original_file_indices_are_not_segment_offsets():
    s, rows = fixture()
    assert rows[4]["point_index"] != s["history_end"]
    assert prefix_clock_record(s, rows[::-1], "dataset") == prefix_clock_record(s, rows, "dataset")


def test_nanosecond_precision_and_strict_ten_second_classification():
    s, rows = fixture(); rows[2]["absolute_epoch_ns"] -= 1
    assert prefix_clock_record(s, rows, "dataset")["origin_span_ns"] == 10 * NS + 1
    spans = describe_spans([10 * NS - 1, 10 * NS, 10 * NS + 1], maximum_ns=120 * NS)
    assert spans["compared_with_stable_10s"] == {"shorter": 1, "equal": 1, "longer": 1}


@pytest.mark.parametrize("start", [True, -1, 4, 5, 1.5])
def test_invalid_or_one_point_history_bounds(start):
    s, rows = fixture(); s["history_start"] = start
    with pytest.raises(ValueError): prefix_clock_record(s, rows, "dataset")


@pytest.mark.parametrize("key,value", [("dataset_version", "other"), ("split", "validation"),
                                      ("file_id", "other"), ("segment_id", "other"),
                                      ("independent_block_id", "other"), ("absolute_epoch_ns", 1.2)])
def test_each_selected_prefix_row_is_bound(key, value):
    s, rows = fixture(); rows[2][key] = value
    with pytest.raises(ValueError): prefix_clock_record(s, rows, "dataset")


@pytest.mark.parametrize("gap", [0, -1, 60 * NS + 1])
def test_nonpositive_or_excessive_prefix_gaps(gap):
    s, rows = fixture(); rows[2]["absolute_epoch_ns"] = rows[3]["absolute_epoch_ns"] - gap
    with pytest.raises(ValueError): prefix_clock_record(s, rows, "dataset")


def test_final_eval_and_duplicate_rows_are_not_admitted():
    s, rows = fixture(); s["split"] = "final_eval"
    with pytest.raises(ValueError): prefix_clock_record(s, rows, "dataset")
    s, rows = fixture(); rows.append(copy.deepcopy(rows[3]))
    with pytest.raises(ValueError): prefix_clock_record(s, rows, "dataset")


def test_maximum_two_gap_span_and_first_tick_are_distinct():
    s, rows = fixture()
    rows[2]["absolute_epoch_ns"] = rows[4]["absolute_epoch_ns"] - 120 * NS
    rows[3]["absolute_epoch_ns"] = rows[4]["absolute_epoch_ns"] - 60 * NS
    r = prefix_clock_record(s, rows, "dataset")
    assert r["origin_span_ns"] == 120 * NS and r["first_tick_span_ns"] == 65 * NS


def test_quantiles_counts_and_point_population_are_descriptive():
    records = [{"points": n, "origin_span_ns": span * NS, "last_gap_ns": gap * NS,
                "first_tick_span_ns": (gap + 5) * NS}
               for n, span, gap in [(2, 2, 2), (3, 10, 5), (3, 60, 30), (3, 120, 60)]]
    r = describe_role(records)
    assert r["windows"] == 4 and r["retained_point_counts"] == {"2": 1, "3": 3}
    assert r["origin_span"]["median_seconds"] == 35.
    assert r["origin_span"]["q25_seconds"] == 8.
    assert r["origin_span"]["q75_seconds"] == 75.
    assert r["origin_span"]["compared_with_stable_10s"] == {"shorter": 1, "equal": 1, "longer": 2}
    assert sum(r["origin_span"]["span_bin_counts_seconds"].values()) == 4


@pytest.mark.parametrize("values", [[], [True], [1.5], [0], [-1], [120 * NS + 1]])
def test_invalid_span_statistics(values):
    with pytest.raises(ValueError): describe_spans(values, maximum_ns=120 * NS)


def test_empty_role_is_not_summarized_as_success():
    with pytest.raises(ValueError): describe_role([])
