"""Software-only case selection/projection tests, not new scientific trials."""
import json
from pathlib import Path

import pytest

from experiments.pirc17.case_review import check_output, geographical_summary, select_cases, tile_name


def test_selection_preserves_failed_first_case_without_outcome_filter():
    rows=[dict(rank=0,country='BE',status='failed'),dict(rank=1,country='BE',status='success'),
          dict(rank=2,country='GB',status='unavailable'),dict(rank=3,country='US',status='success')]
    assert [r['rank'] for r in select_cases(rows)]==[0,2,3]


def test_public_projection_has_no_recording_or_route_identifiers():
    rows=[dict(sample_id=f'private-{i}',file_id='private-route',country='BE' if i<10 else 'GB',
               region='source-region',harvest_area='source-area',longitude=123.) for i in range(46)]
    result=geographical_summary(rows)
    assert result['countries']=={'BE':10,'GB':36}
    assert sum(r['blocks'] for r in result['harvest_areas'])==46
    assert 'private-route' not in json.dumps(result)
    assert 'longitude' not in json.dumps(result)
    with pytest.raises(ValueError):geographical_summary(rows[:-1])
    with pytest.raises(ValueError):geographical_summary(rows[:-1]+[rows[0]])


def test_private_destination_cannot_be_public_or_existing(tmp_path):
    with pytest.raises(ValueError):check_output(tmp_path/'review')
    with pytest.raises(ValueError):check_output(Path(__file__).resolve().parents[1]/'.local')


def test_signed_tile_grid():
    assert tile_name(51,-3)=='N51W003'
    assert tile_name(-6,114)=='S06E114'
