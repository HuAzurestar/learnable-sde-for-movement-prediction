from copy import deepcopy
from datetime import datetime, timezone

from experiments.pirc17 import checkpoint_observe as observer
from experiments.pirc17.protocol_core import read_json


def sample():
    return dict(session='science', report_utc='2026-10-06T00:00:00+00:00',
        science_success=100, science_runnable=20,
        conditional_forecast_eta_hours=2., lanes={'0':dict(success=4), '1':dict(success=6)})


def test_rate_uses_producer_interval_and_lane_counts():
    before = sample()
    after = deepcopy(before)
    after.update(report_utc='2026-10-06T00:30:00+00:00', science_success=110)
    after['lanes'] = {'0':dict(success=8), '1':dict(success=12)}
    value = observer.summary(after, before, event='forecast_summary')
    assert value['interval_science_per_hour'] == 20
    assert value['lane_added'] == {'0':4, '1':6}
    unchanged = observer.summary(before, before, event='forecast_summary')
    assert unchanged['interval_science_per_hour'] is None


def test_completed_queue_does_not_recommend_more_lanes():
    current = sample()
    current['science_runnable'] = 0
    value = observer.summary(current, None, event='eight_hour_review', review_due=True)
    assert value['eight_hour_review_due'] is True
    assert 'no extra forecast lanes' in value['concurrency_review']


def test_resume_keeps_original_eight_hour_deadline(tmp_path, monkeypatch):
    moment = datetime(2026, 10, 6, tzinfo=timezone.utc)
    monkeypatch.setattr(observer, 'utcnow', lambda:moment)
    monkeypatch.setattr(observer, 'snapshot', lambda *_:sample())
    observer.observe(tmp_path/'science', tmp_path/'scores', tmp_path/'observations', once=True)
    first = read_json(tmp_path/'observations'/'schedule.json')
    assert first['next_summary_utc'] == '2026-10-06T00:30:00+00:00'
    assert first['review_due_utc'] == '2026-10-06T08:00:00+00:00'
    moment = datetime(2026, 10, 6, 8, 1, tzinfo=timezone.utc)
    observer.observe(tmp_path/'science', tmp_path/'scores', tmp_path/'observations', once=True)
    resumed = read_json(tmp_path/'observations'/'schedule.json')
    assert resumed['review_due_utc'] == first['review_due_utc']
    assert resumed['review_reported'] is True
    assert read_json(tmp_path/'observations'/'latest.json')['event'] == 'eight_hour_review'
