"""Synthetic metadata/cardinality contracts, not new model experiments."""
import copy

import pytest

from experiments.pirc17.mode_partition_description import ROLES, role_counts, segment_rank_counts


def training():
    rows = [{"role": role, "segment_id": role + str(i), "transition_count": 3 + i % 7,
             "reference_interval_seconds": 60., "scoring_observations_changed": False}
            for role, n in ROLES.items() for i in range(n)]
    return {"per_segment": rows, "sample_counts": dict(ROLES), "reference_interval_seconds": 60.,
            "transitions_by_role": {role: sum(r["transition_count"] for r in rows if r["role"] == role)
                                    for role in ROLES}}


@pytest.mark.parametrize("n,expected", [(1, [1, 0, 0]), (2, [1, 1, 0]), (3, [1, 1, 1]),
                                       (76, [26, 25, 25]), (81, [27, 27, 27]), (328, [110, 109, 109])])
def test_rank_cardinalities(n, expected):
    assert segment_rank_counts(n) == expected and sum(expected) == n


@pytest.mark.parametrize("n", [0, -1, True, 2.5])
def test_invalid_segment_counts(n):
    with pytest.raises(ValueError): segment_rank_counts(n)


def test_role_totals_are_checked_without_fabricating_transition_frequencies():
    t = training(); r = role_counts(t)
    for role in ROLES:
        assert r[role]["segments"] == ROLES[role]
        assert r[role]["all_transitions"] == t["transitions_by_role"][role]
        assert r[role]["multimode_transition_frequencies"] is None


def test_balanced_segment_counts_do_not_balance_transition_counts():
    # Three ranks, one segment each, but lengths differ. This is arithmetic
    # on metadata, not a fabricated empirical mode assignment or trajectory.
    assert segment_rank_counts(3) == [1, 1, 1]
    assert [3, 6, 12] != [7, 7, 7]


@pytest.mark.parametrize("key,value", [("role", "final_eval"), ("transition_count", True),
                                      ("transition_count", 2), ("reference_interval_seconds", 30.),
                                      ("scoring_observations_changed", True)])
def test_invalid_original_row(key, value):
    t = training(); t["per_segment"][0][key] = value
    with pytest.raises(ValueError): role_counts(t)


@pytest.mark.parametrize("field", ["sample_counts", "transitions_by_role"])
def test_saved_aggregate_must_match_per_segment_accounting(field):
    t = training(); t[field]["train"] += 1
    with pytest.raises(ValueError): role_counts(t)


def test_duplicate_or_missing_prepared_segment_is_not_silently_counted():
    t = training(); t["per_segment"].append(copy.deepcopy(t["per_segment"][0]))
    with pytest.raises(ValueError): role_counts(t)
    t = training(); t["per_segment"].pop()
    with pytest.raises(ValueError): role_counts(t)
