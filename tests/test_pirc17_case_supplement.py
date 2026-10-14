"""Synthetic report assembly fixtures; not extra PIRC17 experiments."""
import copy

import fitz
import pytest

from experiments.pirc17.case_supplement import RANKS, SUBJECTS, Writer, check_body, validate_sources


def fixture():
    common=dict(private_local_review_only=True,public_route_release_authorized=False,
                independent_saved_output_audit=False,cases=[])
    cases=copy.deepcopy(common);comparisons=copy.deepcopy(common)
    origins=[dict(rank=i,sample_id=str(i),independent_block_id=str(i),worldcover_class=10,
                  slope_radians=.1,road_distance_m=1.,river_distance_m=2.,worldcover_status='valid',
                  dem_surface_status='valid',overture_road_status='valid',hydrorivers_river_status='valid')
             for i in range(46)]
    for rank in RANKS:
        geo=dict(rank=rank,sample_id=str(rank),country='fixture',region='fixture',harvest_area='fixture')
        original=dict(rank=rank,status='success',seed=20260814,newly_generated_forecasts=0,
                      geography=geo,sample_id=str(rank),independent_block_id=str(rank),
                      record_sha256='record'+str(rank),target_elapsed_seconds=[60.,300.,900.,1800.],
                      point_mean_errors_m=[1.]*4)
        models=[]
        for subject in SUBJECTS:
            forecast=dict(status='success',rank=rank,seed=20260814,subject=subject,newly_generated_forecasts=0,
                          sample_id=str(rank),independent_block_id=str(rank),record_sha256='record'+str(rank),
                          prediction_seconds=.5)
            models.append(dict(subject=subject,status='success',forecast=forecast,
                               score=dict(status='success',weighted_es_m=1.,fde_m=1.,es_by_time_m=[1.]*4),
                               region=dict(radius_m=2.,covered=True)))
        cases['cases'].append(original);comparisons['cases'].append(dict(rank=rank,geography=geo,models=models))
    return cases,comparisons,dict(origins=origins)


def test_all_original_cases_and_models_are_required():
    cases,comparisons,environment=fixture()
    assert [r['original']['rank'] for r in validate_sources(cases,comparisons,environment)]==list(RANKS)
    comparisons['cases'].pop()
    with pytest.raises(ValueError):validate_sources(cases,comparisons,environment)


@pytest.mark.parametrize('field,value',[('seed',20260815),('sample_id','other'),('status','failed')])
def test_seed_source_and_failure_cannot_be_substituted(field,value):
    cases,comparisons,environment=fixture()
    comparisons['cases'][0]['models'][1]['forecast'][field]=value
    with pytest.raises(ValueError):validate_sources(cases,comparisons,environment)


def test_original_metrics_must_remain_consistent():
    cases,comparisons,environment=fixture()
    comparisons['cases'][0]['models'][0]['score']['weighted_es_m']=2.
    with pytest.raises(ValueError):validate_sources(cases,comparisons,environment)
    cases,comparisons,environment=fixture()
    comparisons['cases'][0]['models'][0]['region']['covered']=False
    with pytest.raises(ValueError):validate_sources(cases,comparisons,environment)


def test_invalid_environment_or_audit_claim_rejected():
    cases,comparisons,environment=fixture();environment['origins'][0]['worldcover_status']='missing'
    with pytest.raises(ValueError):validate_sources(cases,comparisons,environment)
    cases,comparisons,environment=fixture();comparisons['independent_saved_output_audit']=True
    with pytest.raises(ValueError):validate_sources(cases,comparisons,environment)


def test_copied_body_preserves_all_pages_and_rejects_changed_text():
    source=fitz.open();source.new_page().insert_text((40,40),'Original manuscript page')
    copied=fitz.open();copied.insert_pdf(source);check_body(source,copied)
    copied[0].insert_text((40,80),'Changed')
    with pytest.raises(ValueError):check_body(source,copied)
    source.close();copied.close()


def test_english_notes_use_builtin_font_and_clipped_text_is_rejected():
    document=fitz.open();writer=Writer(document,'en','not-needed-for-english.ttf')
    page=writer.page('Local review source credits')
    writer.text(page,'Readable English source attribution',80,height=40)
    assert 'Readable English' in page.get_text()
    with pytest.raises(ValueError):writer.text(page,'Do not clip this text '*100,100,height=10)
    document.close()
