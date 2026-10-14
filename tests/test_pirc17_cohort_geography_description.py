"""Pure metadata aggregation contracts, not new scientific experiments."""
from copy import deepcopy
import json

import pytest

from experiments.pirc17.cohort_geography_description import summarize


def fixture():
    rows = []
    labels = {}
    for sid, block, country, reasons in (
        ("a", "block-a", "GB", []), ("b", "block-a", "GB", ["followup-shorter-than-1800s"]),
        ("c", "block-c", "BE", []), ("d", "block-d", "HT", ["followup-shorter-than-1800s"]),
        ("e", "block-e", "CN", ["invalid-road-coverage-origin-through-target-end"]),
        ("f", "block-f", "DE", ["invalid-integer-point-metadata", "nonpositive-observation-gap"]),
        ("g", "block-g", "JP", []),
    ):
        rows.append(dict(sample_id=sid, independent_block_id=block, split="final_eval",
                         reasons=reasons, eligible=not reasons))
        labels[sid] = [country, "source-region", "source-area"]
    return rows, labels, {"a", "c"}


def test_stage_partition_counts_windows_and_blocks_separately():
    result = summarize(*fixture())
    stages = result["stages"]
    assert (stages["released"]["windows"], stages["released"]["recording_hash_blocks"]) == (7, 6)
    assert stages["released"]["countries"]["GB"] == {"windows": 2, "recording_hash_blocks": 1}
    assert [stages[x]["windows"] for x in ("metadata_complete", "temporal_support",
            "joint_feature_validity", "selected_primary", "eligible_not_selected")] == [6, 4, 3, 2, 1]
    assert stages["metadata_excluded"]["windows"] == 1
    assert stages["temporal_excluded"]["windows"] == 2  # metadata failure takes precedence
    assert stages["coverage_excluded_after_temporal"]["windows"] == 1
    assert result["rejection_block_counts_not_additive_to_admitted_block_counts"]


def test_display_contains_all_selected_and_largest_unselected_without_outcome_filter():
    result = summarize(*fixture())
    names = [r["source_country"] for r in result["display_rows"]]
    assert names == ["BE", "CN", "DE", "GB", "Other source countries", "Total"]
    # Equal released counts are resolved alphabetically, not by performance.
    assert result["countries_absent_from_selected"] == ["CN", "DE", "HT", "JP"]
    for stage in ("released", "temporal_support", "joint_feature_validity", "selected_primary"):
        for key in ("windows", "recording_hash_blocks"):
            assert sum(r[stage][key] for r in result["display_rows"][:-1]) == result["display_rows"][-1][stage][key]


def test_private_identities_and_labels_not_exported():
    text = json.dumps(summarize(*fixture()))
    assert "block-a" not in text and "source-region" not in text and "source-area" not in text
    assert "sample_id" not in text


def test_missing_city_label_is_retained_and_reported_not_dropped():
    rows, labels, selected = fixture()
    labels["d"][2] = ""
    result = summarize(rows, labels, selected)
    assert result["stages"]["released"]["countries"]["HT"]["windows"] == 1
    assert result["stages"]["released"]["source_harvest_area_missing_windows"] == 1
    assert result["stages"]["selected_primary"]["source_harvest_area_missing_windows"] == 0


@pytest.mark.parametrize("mutation", [
    "duplicate", "wrong_split", "missing_label", "extra_label", "bad_label", "bad_country",
    "block_conflict", "unknown_reason", "duplicate_reason", "inconsistent_eligible",
    "not_bool", "missing_selected", "selected_ineligible", "same_block_selected",
])
def test_reject_incomplete_or_ambiguous_metadata(mutation):
    rows, labels, selected = deepcopy(fixture())
    if mutation == "duplicate": rows.append(rows[0])
    elif mutation == "wrong_split": rows[0]["split"] = "train"
    elif mutation == "missing_label": labels.pop("a")
    elif mutation == "extra_label": labels["extra"] = ["GB", "source-region", "source-area"]
    elif mutation == "bad_label": labels["a"] = ["GB", "source-region"]
    elif mutation == "bad_country": labels["a"][0] = "gb"
    elif mutation == "block_conflict": labels["b"][0] = "BE"
    elif mutation == "unknown_reason": rows[1]["reasons"] = ["unknown"]
    elif mutation == "duplicate_reason": rows[1]["reasons"] *= 2
    elif mutation == "inconsistent_eligible": rows[1]["eligible"] = True
    elif mutation == "not_bool": rows[0]["eligible"] = 1
    elif mutation == "missing_selected": selected.add("missing")
    elif mutation == "selected_ineligible": selected.add("b")
    elif mutation == "same_block_selected":
        rows[1]["reasons"] = []
        rows[1]["eligible"] = True
        selected.add("b")
    with pytest.raises(ValueError): summarize(rows, labels, selected)
