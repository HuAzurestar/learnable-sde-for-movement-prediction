"""Synthetic saved-number fixtures;no models,particles or forecasts."""
import copy
import math

import pytest

from experiments.pirc17.all_method_regions import SUBJECTS, agree, summarize
from experiments.pirc17.calibration_description import HORIZONS, LEVELS, SEEDS


def grid():
    rows = []
    for subject in SUBJECTS:
        for rank in range(46):
            for seed in SEEDS:
                times = []
                for horizon in HORIZONS:
                    entries = [dict(level=p,radius_m=10*(i+1),area_m2=math.pi*(10*(i+1))**2,
                                    covered=i>=2,empirical_mass=p) for i,p in enumerate(LEVELS)]
                    times.append(dict(elapsed_seconds=horizon,region=dict(
                        region="ensemble_mean_radial_quantile_disk",levels=entries)))
                rows.append(dict(configuration=subject,origin_rank=rank,seed=seed,status="success",
                    matrix="NEX326-methods",origin_mode="causal_prefix",partition="final_eval",
                    scientific=True,scores=dict(particle_count=512,by_time=times)))
    return rows


def test_all_identities_times_levels_and_block_seed_means():
    rows = grid()
    rows[0]["scores"]["by_time"][0]["region"]["levels"][2]["covered"] = False
    profiles = summarize(rows)
    assert len(SUBJECTS)==len(set(SUBJECTS))==len(profiles)==28
    assert [p["configuration"] for p in profiles] == list(SUBJECTS)
    assert all(p["forecast_rows"]==230 and p["independent_blocks"]==46 for p in profiles)
    level = profiles[0]["horizons"][0]["levels"][2]
    assert level["empirical_coverage"] == pytest.approx(229/230)
    assert level["mean_disk_area_km2"] == pytest.approx(math.pi*900/1e6)
    assert level["mean_disk_radius_m"]==30
    assert [t["nominal_horizon_seconds"] for t in profiles[0]["horizons"]]==list(HORIZONS)
    assert all([x["nominal_level"] for x in t["levels"]]==list(LEVELS)
               for p in profiles for t in p["horizons"])


@pytest.mark.parametrize("kind", ["missing","duplicate","failed","wrong_mode","wrong_seed",
                                  "wrong_subject","wrong_rank","wrong_partition","wrong_matrix"])
def test_complete_original_grid_or_fail(kind):
    rows=grid()
    if kind=="missing": rows.pop()
    if kind=="duplicate": rows.append(copy.deepcopy(rows[0]))
    changes={"failed":("status","failed"),"wrong_mode":("origin_mode","point_only"),
             "wrong_seed":("seed",9),"wrong_subject":("configuration","new-model"),
             "wrong_rank":("origin_rank",46),"wrong_partition":("partition","training"),
             "wrong_matrix":("matrix","terrain")}
    if kind in changes: k,v=changes[kind]; rows[0][k]=v
    with pytest.raises(ValueError): summarize(rows)


def test_mean_area_is_mean_of_areas_not_area_of_mean_radius():
    rows=grid()
    for time in rows[0]["scores"]["by_time"]:
        for x in time["region"]["levels"]:
            x["radius_m"]*=2
            x["area_m2"]*=4
    level=summarize(rows)[0]["horizons"][0]["levels"][2]
    assert level["mean_disk_area_km2"] > math.pi*level["mean_disk_radius_m"]**2/1e6


@pytest.mark.parametrize("actual,expected",[(float("nan"),1),(1,float("inf")),(True,1),(1,2)])
def test_nonfinite_boolean_or_changed_original_metric_fails(actual,expected):
    with pytest.raises(ValueError): agree(actual,expected,"synthetic metric")


def test_roundoff_tolerance_is_not_a_new_scientific_equivalence_test():
    agree(1,1+1e-13,"synthetic arithmetic")
