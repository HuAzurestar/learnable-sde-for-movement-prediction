"""Synthetic software fixtures, not further PIRC17 scientific experiments."""
import math
import pytest

from experiments.pirc17.origin_environment import FACTORS,SEEDS,SUBJECTS,describe_origins,descriptive_scores


def fixtures():
    origins=[dict(rank=i,sample_id=str(i),independent_block_id='b'+str(i),worldcover_class=10 if i<39 else 50,
        slope_radians=.1,road_distance_m=10.,river_distance_m=100.,**{s:'valid' for s in FACTORS.values()}) for i in range(46)]
    rows=[dict(origin_rank=o['rank'],sample_id=o['sample_id'],independent_block_id=o['independent_block_id'],seed=seed,
        configuration=subject,status='success',origin_mode='causal_prefix',matrix='NEX326-methods',score_m=float(o['rank']),
        scores=dict(fde_m=float(o['rank']+10))) for o in origins for seed in SEEDS for subject in SUBJECTS]
    return origins,rows


def test_descriptor_counts_and_units_are_per_independent_origin():
    origins,_=fixtures();d=describe_origins(origins)
    assert sum(x['blocks'] for x in d['worldcover_at_origin'])==46
    assert d['continuous_at_origin']['slope_degrees']['median']==pytest.approx(180*.1/math.pi)
    assert d['origin_class_is_whole_route_environment'] is False


def test_empty_categories_are_not_zero_error_and_means_reconcile():
    origins,rows=fixtures();r=descriptive_scores(rows,origins)
    empty=next(x for x in r['group_means'] if x['code']==80)
    assert empty['status']=='not_represented' and all(x['fde_m'] is None for x in empty['subjects'])
    assert r['overall'][0]['weighted_es_m']==22.5
    assert r['cached_forecast_rows']==690 and r['hypothesis_tests']==0


def test_no_missing_failed_duplicate_or_success_only_grid():
    origins,rows=fixtures()
    for invalid in (rows[:-1],rows+[rows[0]],[dict(rows[0],status='failed')]+rows[1:]):
        with pytest.raises(ValueError):descriptive_scores(invalid,origins)


def test_invalid_descriptor_cannot_be_imputed_or_removed():
    origins,_=fixtures()
    with pytest.raises(ValueError):describe_origins(origins[:-1])
    origins[0]['worldcover_status']='source_missing'
    with pytest.raises(ValueError):describe_origins(origins)
