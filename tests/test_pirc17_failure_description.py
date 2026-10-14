"""Pure metadata fixtures only; no fitting, arrays or simulated predictions."""
from copy import deepcopy

import pytest

from experiments.pirc17.failure_description import affected_families, failure_entries, literal_assignment


def fixture():
    workloads = [dict(work_id=str(i), kind='scientific_forecast', generated_forecasts=1,
        subject=c, matrix=m, origin_mode=o, seed=20260814+i, phase='forecast', max_active_seconds=180)
        for i, (c, m, o) in enumerate((('dt30', 'NEX326-methods', 'known_velocity'),
            ('gaussian', 'NEX326-methods', 'known_velocity'), ('lio-river', 'terrain', 'causal_prefix'),
            ('all-terrain', 'terrain', 'causal_prefix'), ('lio-road', 'terrain', 'causal_prefix')))]
    failures = {str(i): 'interrupted/deadline/error; output retained' if i < 3
                else 'interrupted; full reserved cost retained' for i in range(5)}
    previous = dict(failures={k:v for k,v in failures.items() if int(k) < 3}, parallel_pending={
        str(i-3): dict(work_id=str(i), phase='forecast', reserved_ns=180_000_000_000) for i in (3,4)})
    return workloads, failures, previous


def test_preserved_generic_reason_not_model_divergence():
    rows = failure_entries(*fixture())
    assert len(rows) == 5
    assert sum(r['evidence_class'] == 'predecessor_inflight_without_completion_reply' for r in rows) == 2
    assert not any(r['model_divergence_established'] or r['item_exception_available'] for r in rows)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'success', 'wrong_phase', 'refund', 'overlap'])
def test_changed_failure_or_pending_contract_rejected(change):
    w, f, p = deepcopy(fixture())
    if change == 'missing': del f['4']
    if change == 'duplicate': w.append(w[0])
    if change == 'success': f['4'] = 'success'
    if change == 'wrong_phase': p['parallel_pending']['0']['phase'] = 'other'
    if change == 'refund': p['parallel_pending']['0']['reserved_ns'] = 1
    if change == 'overlap': p['parallel_pending']['0']['work_id'] = '0'
    with pytest.raises(ValueError): failure_entries(w, f, p)


def test_whole_family_disposition_no_cross_mode_contamination():
    rows = failure_entries(*fixture())
    primary = {'whole': ('all-terrain','base'), 'road': ('all-terrain','loo-road'),
               'river': ('all-terrain','loo-river')}
    families = {'weighted-es-primary': primary, 'weighted-es-lio': {'road': ('lio-road','base'),
        'river': ('lio-river','base'), 'surface': ('lio-surface','base')},
        'method-structure': {'gaussian': ('gaussian','full'), 'other': ('other','full')},
        'method-dt': {'dt30': ('dt30','full'), 'dt600': ('dt600','full')}}
    result = affected_families(rows, families)
    assert len(result) == 4
    assert all(not r['successful_subset_inference_allowed'] for r in result)
    assert len(result[0]['affected_contrasts']) == 3
    assert len(result[1]['affected_contrasts']) == 3  # even the unfailed surface pair
    assert {r['origin_mode'] for r in result[2:]} == {'known_velocity'}
    assert not any(r['origin_mode'] == 'point_only' for r in result)


def test_frozen_literals_parsed_without_importing_execution():
    assert literal_assignment("FAMILY={'x':('A','B')}\nraise RuntimeError('never execute')", 'FAMILY') == {'x':('A','B')}
    with pytest.raises(ValueError): literal_assignment('OTHER=1', 'FAMILY')
