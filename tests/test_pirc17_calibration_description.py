"""Synthetic cached-number fixtures only; no fits, particles or trajectories."""
import copy
import math

import pytest

from experiments.pirc17.calibration_description import HORIZONS, LEVELS, SEEDS, SUBJECTS, summarize, validate_regions


def row(subject=SUBJECTS[0], rank=0, seed=SEEDS[0]):
    by_time = []
    for t in HORIZONS:
        entries = [{"level": level, "radius_m": index + 1,
                    "area_m2": math.pi * (index + 1)**2,
                    "covered": index >= 2, "empirical_mass": level}
                   for index, level in enumerate(LEVELS)]
        by_time.append({"elapsed_seconds": t, "region": {
            "region": "ensemble_mean_radial_quantile_disk", "levels": entries}})
    return {"configuration": subject, "origin_rank": rank, "seed": seed,
            "status": "success", "matrix": "NEX326-methods", "origin_mode": "causal_prefix",
            "partition": "final_eval", "scientific": True,
            "scores": {"particle_count": 512, "by_time": by_time}}


def grid():
    return [row(s, r, seed) for s in SUBJECTS for r in range(46) for seed in SEEDS]


def test_complete_grid_block_seed_weights_and_units():
    rows = grid()
    rows[0]["scores"]["by_time"][0]["region"]["levels"][2]["covered"] = False
    result = summarize(rows)
    level = result[0]["horizons"][0]["levels"][2]
    assert level["empirical_coverage"] == pytest.approx(229/230)
    assert level["mean_disk_area_km2"] == pytest.approx(9*math.pi/1e6)
    assert level["mean_disk_radius_m"] == 3


@pytest.mark.parametrize("kind", ["missing", "duplicate", "failure", "wrong_mode", "wrong_seed"])
def test_no_partial_population(kind):
    rows = grid()
    if kind == "missing": rows.pop()
    if kind == "duplicate": rows.append(copy.deepcopy(rows[0]))
    if kind == "failure": rows[0]["status"] = "failed"
    if kind == "wrong_mode": rows[0]["origin_mode"] = "point_only"
    if kind == "wrong_seed": rows[0]["seed"] = 9
    with pytest.raises(ValueError): summarize(rows)


@pytest.mark.parametrize("key,value", [("area_m2", -1), ("area_m2", 3),
                                      ("radius_m", float("nan")), ("radius_m", True),
                                      ("covered", 1), ("empirical_mass", 2)])
def test_bad_saved_region_rejected(key, value):
    r = row()
    r["scores"]["by_time"][0]["region"]["levels"][0][key] = value
    with pytest.raises(ValueError): validate_regions(r)


def test_non_nested_containment_rejected():
    r = row()
    r["scores"]["by_time"][0]["region"]["levels"][0]["covered"] = True
    with pytest.raises(ValueError, match="nested"): validate_regions(r)


@pytest.mark.parametrize("kind", ["missing_level", "missing_time", "outside_tolerance", "wrong_particles"])
def test_exact_registered_levels_horizons_and_ensemble(kind):
    r = row()
    if kind == "missing_level": r["scores"]["by_time"][0]["region"]["levels"].pop()
    if kind == "missing_time": r["scores"]["by_time"].pop()
    if kind == "outside_tolerance": r["scores"]["by_time"][0]["elapsed_seconds"] += 31
    if kind == "wrong_particles": r["scores"]["particle_count"] = 100
    with pytest.raises(ValueError): validate_regions(r)


def test_nonfinite_time_rejected():
    r = row()
    r["scores"]["by_time"][0]["elapsed_seconds"] = float("nan")
    with pytest.raises(ValueError): validate_regions(r)
