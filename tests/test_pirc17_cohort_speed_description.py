"""Synthetic identity/scalar fixtures only; no scientific forecasts or fits."""
from copy import deepcopy
import json

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.pirc17.cohort_speed_description import describe, joined_statistics, summarize, window_statistic


def sample(sid="private", segment="later"):
    return dict(sample_id=sid, file_id="file", segment_id=segment, independent_block_id=sid,
                split="final_eval", history_start=0, history_end=1, target_start=2, target_end=3)


def aggregate():
    return dict(n=4, unique_indexes=4, first_index=0, last_index=3, file_count=1, file_id="file",
                parent_count=1, missing_source=0, wrong_source_file=0, parent_count_mismatch=0,
                parent_order_invalid=0, invalid_speed=0, median_speed=0.0)


def test_zero_speed_valid_and_no_missing_subset_repair():
    assert window_statistic(sample(), aggregate()) == {"median_saved_speed": 0., "reason": None}
    assert window_statistic(sample(), None) == {"median_saved_speed": None, "reason": "missing-alignment-window"}
    result = describe([dict(median_saved_speed=1., reason=None), dict(median_saved_speed=None, reason="missing")])
    assert result["windows"] == 2 and result["statistic_successes"] == 1 and result["statistic_failures"] == 1
    assert result["median"] is None and not result["full_stage_distribution_available"]


@pytest.mark.parametrize("kind", ["n", "unique_indexes", "first_index", "last_index", "file_count", "file_id",
                                  "parent_count", "missing_source", "wrong_source_file", "parent_count_mismatch",
                                  "parent_order_invalid", "invalid_speed"])
def test_incomplete_or_ambiguous_mapping_is_explicit_failure(kind):
    row = aggregate()
    row[kind] = "other" if kind == "file_id" else row[kind] + 1
    result = window_statistic(sample(), row)
    assert result["reason"] is not None and result["median_saved_speed"] is None


def test_linear_quartiles_equal_windows_empty_and_invalid_scalars():
    records = [dict(median_saved_speed=v, reason=None) for v in (0., 1., 3., 10.)]
    result = describe(records)
    assert [result[k] for k in ("minimum", "q25", "median", "q75", "maximum")] == [0., .75, 2., 4.75, 10.]
    assert describe([])["windows"] == 0 and describe([])["median"] is None
    assert not describe([])["full_stage_distribution_available"]
    for v in (-1., float("nan"), float("inf")):
        with pytest.raises(ValueError): describe([dict(median_saved_speed=v, reason=None)])
    with pytest.raises(ValueError): describe([dict(median_saved_speed=1., reason="bad")])


def mapping_fixture(tmp_path, kind="valid"):
    alignment = [dict(file_id="file", segment_id="early" if i < 4 else "later", source_segment_id="parent",
                      segment_point_index=i % 4, source_point_index=10+i*2) for i in range(8)]
    source = [dict(file_id="file", segment_id="parent", t=i*.5, speed=float(i)) for i in range(8)]
    # Deliberately reverse input order. Correct source parent ranking must put
    # later[0] at original parent ordinal4, not reset it to ordinal0.
    if kind == "missing_alignment_parent": alignment.pop(0)
    elif kind == "duplicate_source_time": source[1]["t"] = source[0]["t"]
    elif kind == "duplicate_source_index": alignment[1]["source_point_index"] = alignment[0]["source_point_index"]
    elif kind == "negative_speed": source[-1]["speed"] = -1.
    elif kind == "nan_speed": source[-1]["speed"] = float("nan")
    elif kind == "wrong_file": source[-1]["file_id"] = "wrong"
    path = tmp_path / "alignment.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in reversed(alignment)) + "\n", encoding="utf-8")
    trajectory = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist(list(reversed(source))), trajectory)
    connection = duckdb.connect(config={"threads": 1})
    try:
        return joined_statistics(path, trajectory, {"private": sample()}, connection)
    finally:
        connection.close()


def test_parent_ordinal_before_refined_window_and_saved_scalars_not_vectors(tmp_path):
    assert mapping_fixture(tmp_path)["private"] == {"median_saved_speed": 5.5, "reason": None}


@pytest.mark.parametrize("kind,reason", [
    ("missing_alignment_parent", "source-parent-count-mismatch"),
    ("duplicate_source_time", "ambiguous-source-parent-order"),
    ("duplicate_source_index", "ambiguous-source-parent-order"),
    ("negative_speed", "nonfinite-or-negative-saved-speed"),
    ("nan_speed", "nonfinite-or-negative-saved-speed"),
    ("wrong_file", "source-file-identity-mismatch"),
])
def test_join_guards_keep_failure_instead_of_subsetting(tmp_path, kind, reason):
    assert mapping_fixture(tmp_path, kind)["private"] == {"median_saved_speed": None, "reason": reason}


def population():
    rows = [dict(sample_id=str(i), independent_block_id=str(i), split="final_eval",
                 eligible=not reasons, reasons=reasons) for i, reasons in enumerate(([],
                 ["followup-shorter-than-1800s"], ["invalid-road-coverage-origin-through-target-end"], []))]
    stats = {str(i): dict(median_saved_speed=float(i), reason=None) for i in range(4)}
    return rows, stats, {"0"}


def test_all_dispositions_retained_without_speed_based_selection():
    rows, stats, selected = population()
    stats["0"] = dict(median_saved_speed=None, reason="missing-source-point")
    result = summarize(rows, stats, selected)
    assert result["released"]["windows"] == 4
    assert result["selected_primary"]["windows"] == 1
    assert result["selected_primary"]["window_median_saved_speed"]["statistic_failures"] == 1
    assert result["temporal_excluded"]["window_median_saved_speed"]["median"] == 1
    assert result["eligible_not_selected"]["window_median_saved_speed"]["median"] == 3


@pytest.mark.parametrize("kind", ["missing", "extra", "duplicate", "status", "reason", "ineligible_selected"])
def test_changed_or_selected_only_population_rejected(kind):
    rows, stats, selected = population()
    if kind == "missing": stats.pop("1")
    elif kind == "extra": stats["extra"] = stats["0"]
    elif kind == "duplicate": rows.append(deepcopy(rows[0]))
    elif kind == "status": rows[1]["eligible"] = True
    elif kind == "reason": rows[1]["reasons"] = ["error-based-selection"]
    else: selected.add("1")
    with pytest.raises(ValueError): summarize(rows, stats, selected)
