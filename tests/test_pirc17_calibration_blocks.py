"""Synthetic saved-score arithmetic only; no fitted models or simulated paths."""
import copy
import math

import pytest

from experiments.pirc17.calibration_description import HORIZONS, LEVELS, SEEDS, SUBJECTS, distribution, summarize


def grid():
    rows = []
    for subject in SUBJECTS:
        for rank in range(46):
            for seed in SEEDS:
                times = [{"elapsed_seconds": horizon, "region": {
                    "region": "ensemble_mean_radial_quantile_disk", "levels": [
                        {"level": level, "radius_m": index + 1,
                         "area_m2": math.pi * (index + 1) ** 2,
                         "covered": index >= 2, "empirical_mass": level}
                        for index, level in enumerate(LEVELS)]}}
                    for horizon in HORIZONS]
                rows.append({"configuration": subject, "origin_rank": rank, "seed": seed,
                             "status": "success", "matrix": "NEX326-methods",
                             "origin_mode": "causal_prefix", "partition": "final_eval",
                             "scientific": True, "scores": {"particle_count": 512,
                                                             "by_time": times}})
    return rows


def test_default_projection_aggregate_is_unchanged():
    original = summarize(grid())
    enriched = summarize(grid(), include_blocks=True)
    for profile in enriched:
        for time in profile["horizons"]:
            for level in time["levels"]:
                for key in ("blocks", "coverage_distribution", "area_km2_distribution",
                            "radius_m_distribution", "blocks_by_covered_seed_count"):
                    level.pop(key)
    assert enriched == original


def test_block_seeds_are_not_flattened_into_independent_targets():
    rows = grid()
    rows[0]["scores"]["by_time"][0]["region"]["levels"][2]["covered"] = False
    value = summarize(rows, include_blocks=True)[0]["horizons"][0]["levels"][2]
    assert value["blocks_by_covered_seed_count"] == [0, 0, 0, 0, 1, 45]
    assert sum(value["blocks_by_covered_seed_count"]) == len(value["blocks"]) == 46
    assert value["blocks"][0]["covered_seed_count"] == 4
    assert value["blocks"][0]["seed_coverage_fraction"] == .8
    assert value["coverage_distribution"] == {
        "minimum": .8, "q25": 1, "median": 1, "q75": 1, "maximum": 1}
    assert value["empirical_coverage"] == pytest.approx(229 / 230)


def test_linear_quartiles_and_exact_unit_conversion():
    assert distribution(list(range(46))) == {
        "minimum": 0, "q25": 11.25, "median": 22.5, "q75": 33.75, "maximum": 45}
    value = summarize(grid(), include_blocks=True)[0]["horizons"][0]["levels"][2]
    assert value["area_km2_distribution"]["median"] == pytest.approx(9 * math.pi / 1e6)
    assert value["radius_m_distribution"]["median"] == 3


def test_shuffled_input_preserves_paired_anonymous_block_order():
    rows = grid()
    before = copy.deepcopy(rows)
    actual = summarize(list(reversed(rows)), include_blocks=True)
    assert actual == summarize(rows, include_blocks=True)
    assert rows == before
    for profile in actual:
        for time in profile["horizons"]:
            for level in time["levels"]:
                assert [b["block_number"] for b in level["blocks"]] == list(range(1, 47))
                assert set(level["blocks"][0]) == {
                    "block_number", "covered_seed_count", "seed_coverage_fraction",
                    "mean_disk_area_km2", "mean_disk_radius_m"}


@pytest.mark.parametrize("values", [[], list(range(45)), list(range(47)),
                                   [float("nan")] + [0] * 45,
                                   [float("inf")] + [0] * 45])
def test_incomplete_or_nonfinite_distribution_is_rejected(values):
    with pytest.raises(ValueError):
        distribution(values)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "failure", "wrong_mode"])
def test_block_mode_retains_whole_grid_failure_policy(kind):
    rows = grid()
    if kind == "missing": rows.pop()
    if kind == "duplicate": rows.append(copy.deepcopy(rows[0]))
    if kind == "failure": rows[0]["status"] = "failed"
    if kind == "wrong_mode": rows[0]["origin_mode"] = "point_only"
    with pytest.raises(ValueError):
        summarize(rows, include_blocks=True)
