"""Pure timestamp aggregation contracts; no new scientific rollouts."""
from copy import deepcopy

import pytest

from experiments.pirc17.cohort_duration_description import NS, describe, summarize, window_spans


def fixture():
    sample = dict(sample_id="private", data_version="data", file_id="file", segment_id="segment",
                  independent_block_id="block", split="final_eval", history_start=0, history_end=1,
                  target_start=2, target_end=3)
    rows = [dict(dataset_version="data", file_id="file", segment_id="segment", split="final_eval",
                 independent_block_id="block", point_index=p, absolute_epoch_ns=t*NS)
            for p, t in zip((7, 9, 11, 13), (10, 11, 15, 20))]
    return sample, rows


def test_segment_offsets_not_original_file_point_indexes():
    sample, rows = fixture()
    assert window_spans(sample, list(reversed(rows)), "data") == {
        "visible_history_ns": NS, "followup_ns": 9*NS, "whole_window_ns": 10*NS}


def test_zero_and_negative_spans_are_retained_not_clipped_or_filtered():
    sample, rows = fixture()
    for ticks in ((10, 10, 10, 10), (20, 15, 11, 10)):
        current = deepcopy(rows)
        for row, tick in zip(current, ticks): row["absolute_epoch_ns"] = tick*NS
        spans = window_spans(sample, current, "data")
        assert spans["followup_ns"] == (0 if ticks[0] == 10 else -5*NS)
        assert spans["whole_window_ns"] == (0 if ticks[0] == 10 else -10*NS)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "float_clock", "extra_coordinate",
                                  "wrong_file", "wrong_split", "wrong_data", "wrong_bounds"])
def test_clock_metadata_mismatch_rejected(kind):
    sample, rows = fixture()
    if kind == "missing": rows.pop()
    elif kind == "duplicate": rows[1]["point_index"] = rows[0]["point_index"]
    elif kind == "float_clock": rows[0]["absolute_epoch_ns"] = 10.
    elif kind == "extra_coordinate": rows[0]["longitude"] = 0.
    elif kind == "wrong_file": rows[0]["file_id"] = "other"
    elif kind == "wrong_split": sample["split"] = "train"
    elif kind == "wrong_data": rows[0]["dataset_version"] = "other"
    else: sample["target_start"] = 3
    with pytest.raises(ValueError): window_spans(sample, rows, "data")


def test_empty_and_nonpositive_groups_have_explicit_counts_not_nan_or_drop():
    assert describe([])["median_seconds"] is None
    result = describe([-NS, 0, NS, 1800*NS, 1801*NS])
    assert result["windows"] == 5 and result["nonpositive_count"] == 2
    assert result["median_seconds"] == 1 and result["at_least_1800s_count"] == 2
    assert result["q25_seconds"] == 0 and result["q75_seconds"] == 1800
    with pytest.raises(ValueError): describe([1.])


def population():
    rows = [dict(sample_id=str(i), independent_block_id=str(i), split="final_eval",
                 eligible=not reasons, reasons=reasons) for i, reasons in enumerate(([],
                 ["followup-shorter-than-1800s"], ["invalid-road-coverage-origin-through-target-end"], []))]
    clocks = {str(i): dict(visible_history_ns=NS, followup_ns=followup*NS,
                           whole_window_ns=(followup+1)*NS)
              for i, followup in enumerate((1800, 5, 1900, 2000))}
    return rows, clocks, {"0"}


def test_all_original_dispositions_not_clock_requalification():
    rows, clocks, selected = population()
    clocks["0"]["followup_ns"] = 0  # statistic does not change saved selection
    result = summarize(rows, clocks, selected)
    assert result["released"]["windows"] == 4 and result["selected_primary"]["windows"] == 1
    assert result["temporal_excluded"]["followup"]["median_seconds"] == 5
    assert result["coverage_excluded_after_temporal"]["followup"]["median_seconds"] == 1900
    assert result["eligible_not_selected"]["windows"] == 1
    assert result["selected_primary"]["followup"]["nonpositive_count"] == 1


@pytest.mark.parametrize("kind", ["missing_clock", "extra_clock", "duplicate", "wrong_status",
                                  "unknown_reason", "ineligible_selected", "missing_selected"])
def test_missing_or_changed_population_rejected(kind):
    rows, clocks, selected = population()
    if kind == "missing_clock": clocks.pop("1")
    elif kind == "extra_clock": clocks["extra"] = clocks["0"]
    elif kind == "duplicate": rows.append(rows[0])
    elif kind == "wrong_status": rows[1]["eligible"] = True
    elif kind == "unknown_reason": rows[1]["reasons"] = ["error-based-choice"]
    elif kind == "ineligible_selected": selected.add("1")
    else: selected.add("missing")
    with pytest.raises(ValueError): summarize(rows, clocks, selected)
