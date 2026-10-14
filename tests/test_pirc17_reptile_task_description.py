"""Pure label/accounting fixtures; no Reptile update or scientific experiment."""
from copy import deepcopy

import pytest

from experiments.pirc17.reptile_task_description import summarize, task_key


@pytest.mark.parametrize("city,region,key,kind", [
    (" London ", "Area", "London", "city"), ("NaN", " Area ", " Area ", "region"),
    (" nan ", "nan", "nan", "region"), (None, "Area", "Area", "region"),
    ("", None, "unknown", "unknown"), ("  ", "", "unknown", "unknown"),
    ("london", "London", "london", "city"), ("unknown", "Area", "unknown", "city"),
])
def test_literal_city_first_adapter_rule_no_extra_strip_or_casefold(city, region, key, kind):
    assert task_key(city, region) == (key, kind)


def test_typed_source_labels_and_whitespace_region_preserved():
    assert task_key(None, " ") == (" ", "region")
    with pytest.raises(ValueError): task_key(1, None)


def fixture():
    rows = [dict(segment_id=str(i), role="train", transition_count=i+3) for i in range(9)]
    rows += [dict(segment_id="a", role="adapt", transition_count=3),
             dict(segment_id="v", role="validation", transition_count=5)]
    labels = {str(i): ("A" if i < 3 else "B" if i < 6 else "C" if i < 8 else "D", "city") for i in range(9)}
    model = dict(meta_task_count=2, meta_outer_epochs=4, meta_inner_steps=5, training_sample_count=198)
    return rows, labels, model


def test_all_training_groups_eligible_and_small_group_counts_reconcile():
    rows, labels, model = fixture()
    result = summarize(rows, labels, model)
    assert result["all_training_groups"] == dict(label_groups=4, segments=9, transitions_per_pass=63)
    assert result["eligible_tasks"] == dict(label_groups=2, segments=6, transitions_per_pass=33)
    assert result["ineligible_small_groups"] == dict(label_groups=2, segments=3, transitions_per_pass=30)
    assert result["exposure_reconciliation"] == dict(global_initialization=63, repeated_meta_tasks=132, target_adaptation=3, total=198)
    assert result["task_visits"] == 8 and result["training_role_transitions"]["validation"] == 5
    assert result["tasks"] == [dict(task="T01", segments=3, transitions_per_pass=12, segment_rank_cardinalities=[1,1,1]),
                               dict(task="T02", segments=3, transitions_per_pass=21, segment_rank_cardinalities=[1,1,1])]
    assert result["eligible_task_size_histogram"] == {"3": 2}


def test_input_order_invariant_and_exact_case_labels_not_geographic_merging():
    rows, labels, model = fixture()
    first = summarize(rows, labels, model)
    assert first == summarize(list(reversed(rows)), dict(reversed(list(labels.items()))), model)
    labels = {sid: (label.lower() if label == "B" else label, kind) for sid, (label, kind) in labels.items()}
    assert summarize(rows, labels, model)["eligible_tasks"]["label_groups"] == 2


@pytest.mark.parametrize("kind", ["missing_label", "extra_label", "duplicate", "invalid_role", "short_count", "float_count",
                                  "missing_adapt", "wrong_task_count", "wrong_epochs", "wrong_inner_steps", "wrong_exposure", "float_metadata"])
def test_incomplete_changed_or_misinterpreted_accounting_rejected(kind):
    rows, labels, model = fixture()
    if kind == "missing_label": labels.pop("0")
    elif kind == "extra_label": labels["extra"] = ("A", "city")
    elif kind == "duplicate": rows.append(deepcopy(rows[0]))
    elif kind == "invalid_role": rows[0]["role"] = "final_eval"
    elif kind == "short_count": rows[0]["transition_count"] = 2
    elif kind == "float_count": rows[0]["transition_count"] = 3.
    elif kind == "missing_adapt": rows = [r for r in rows if r["role"] != "adapt"]
    elif kind == "wrong_task_count": model["meta_task_count"] = 3
    elif kind == "wrong_epochs": model["meta_outer_epochs"] = 5
    elif kind == "wrong_inner_steps": model["meta_inner_steps"] = 6
    elif kind == "wrong_exposure": model["training_sample_count"] += 1
    else: model["meta_task_count"] = 2.
    with pytest.raises(ValueError): summarize(rows, labels, model)


def test_export_contains_no_raw_labels_or_members_but_has_full_binding_digest():
    rows, labels, model = fixture()
    result = summarize(rows, labels, model)
    assert len(result["membership_identity_sha256"]) == 64
    assert all(set(task) == {"task", "segments", "transitions_per_pass", "segment_rank_cardinalities"} for task in result["tasks"])
