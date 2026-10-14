"""Portable software negatives; these are not real history or launch approval.

The positive production proof uses the actual source-pinned private inventory
separately. No portable fixture fabricates a complete accepted predecessor.
"""
from copy import deepcopy

import pytest

from experiments.pirc17 import formal_input_interruption as module
from experiments.pirc17.protocol_core import canonical, digest, envelope, unpack


def save(path, payload):
    record = envelope(payload)
    path.write_bytes(canonical(record)+b'\n')
    return record


def test_catalog_binds_all_bytes_and_empty_directories(tmp_path):
    (tmp_path/'empty').mkdir()
    (tmp_path/'writer.lock').write_bytes(b'0')
    save(tmp_path/'a.json',dict(software_fixture_only=True))
    first = module._catalog(tmp_path)
    assert first['empty'] == dict(kind='directory')
    assert first['writer.lock']['bytes'] == 1
    (tmp_path/'a.json').write_bytes((tmp_path/'a.json').read_bytes()+b'\n')
    assert digest(module._catalog(tmp_path)) != digest(first)
    (tmp_path/'extra-empty').mkdir()
    assert set(module._catalog(tmp_path)) != set(first)


@pytest.mark.parametrize('limit',['MAX_NODES','MAX_FILE_BYTES','MAX_TOTAL_BYTES'])
def test_catalog_has_fixed_traversal_and_byte_bounds(tmp_path,monkeypatch,limit):
    (tmp_path/'a').write_bytes(b'123')
    (tmp_path/'b').write_bytes(b'123')
    monkeypatch.setattr(module,limit,1 if limit != 'MAX_TOTAL_BYTES' else 4)
    with pytest.raises(ValueError,match='bounded|budget'): module._catalog(tmp_path)


def test_catalog_rejects_relative_and_non_directory(tmp_path):
    with pytest.raises(ValueError): module._catalog('relative')
    p = tmp_path/'file'; p.write_bytes(b'x')
    with pytest.raises(ValueError): module._catalog(p)


@pytest.fixture
def access(tmp_path):
    contract = {k:digest(k) for k in ('protocol_sha256','execution_sha256','approval_sha256')}
    for kind,n in [('final_eval_eligibility',1),('final_eval_positions',1),('final_eval_features',2)]:
        for i in range(n):
            row = dict(schema_version='pirc17-final-access-event-v1', **contract,
                access_kind=kind, event='started', attempt_id=str(i),
                population_sha256=None if kind == 'final_eval_eligibility' else digest('population'))
            start = envelope(row)
            save(tmp_path/(start['sha256']+'.json'),row)
            end = dict(row,event='returned',started_sha256=start['sha256'],possible_reads_retained=True)
            record = envelope(end)
            save(tmp_path/(record['sha256']+'.json'),end)
    return tmp_path,contract


def test_access_counts_four_reads_not_eight_and_never_claims_zero(access):
    root,contract = access
    v = module._access_evidence(root,contract)
    assert v['event_count'] == 8 and v['started_reads'] == v['returned_reads'] == 4
    assert v['historically_exposed'] is True and v['possible_reads_retained'] is True
    assert v['kinds']['final_eval_features'] == 2


@pytest.mark.parametrize('change',['authority','missing','extra','unretained','wrong_start','duplicate_return','kind'])
def test_access_rejects_changed_authority_partial_or_unlinked_returns(access,change):
    root,contract = access
    path = next(p for p in root.iterdir() if unpack(module.costs._read_bound(p,module.costs.MAX_RECORD_BYTES)[0])['event']=='returned')
    row = unpack(module.costs._read_bound(path,module.costs.MAX_RECORD_BYTES)[0])
    if change == 'missing': path.unlink()
    elif change == 'extra': (root/'extra').write_bytes(b'x')
    else:
        row = deepcopy(row)
        if change == 'authority': row['approval_sha256'] = digest('other approval')
        elif change == 'unretained': row['possible_reads_retained'] = False
        elif change == 'wrong_start': row['started_sha256'] = digest('missing start')
        elif change == 'kind': row['access_kind'] = 'unregistered'
        elif change == 'duplicate_return':
            other = next(unpack(module.costs._read_bound(p,module.costs.MAX_RECORD_BYTES)[0]) for p in root.iterdir()
                if p != path and unpack(module.costs._read_bound(p,module.costs.MAX_RECORD_BYTES)[0])['event']=='returned')
            row.update(started_sha256=other['started_sha256'],attempt_id=other['attempt_id'],access_kind=other['access_kind'])
        path.unlink()
        save(root/(digest(row)+'.json'),row)
    with pytest.raises(ValueError): module._access_evidence(root,contract)


@pytest.fixture
def scope():
    work = digest('software input fixture only')
    c = dict(charged_total_ns=module.REGISTERED_CHARGED_NS,event_types=module.EVENT_TYPES,
        generated_forecasts_reserved=0,halted_reason='supervision_error',work_inventory_count=11659,
        work_dispositions={work:'failure'},charged_ns_by_phase={module.PHASE:module.REGISTERED_CHARGED_NS,'science':0},
        phase_caps_ns={module.PHASE:3600_000_000_000,'science':1_000_000_000})
    contract = dict(workloads=[dict(work_id=work,phase=module.PHASE)])
    return c,contract


@pytest.mark.parametrize('change',['free','first_only','double_first','forecast','bool_forecast','fit_cost','success','extra_work','extra_event'])
def test_scope_rejects_cheaper_double_charged_or_scientific_history(scope,change):
    c,contract = scope; c = deepcopy(c)
    if change == 'free': c['charged_total_ns']=0
    elif change == 'first_only': c['charged_total_ns']=module.startup.REGISTERED_CHARGED_NS
    elif change == 'double_first': c['charged_total_ns']+=module.startup.REGISTERED_CHARGED_NS
    elif change == 'forecast': c['generated_forecasts_reserved']=1
    elif change == 'bool_forecast': c['generated_forecasts_reserved']=False
    elif change == 'fit_cost': c['charged_ns_by_phase']['science']=1
    elif change == 'success': c['work_dispositions'][contract['workloads'][0]['work_id']]='success'
    elif change == 'extra_work': c['work_dispositions'][digest('another')]='failure'
    elif change == 'extra_event': c['event_types'].append('reserve')
    with pytest.raises(ValueError): module._cost_scope(c,contract)


def test_wrong_or_unpinned_directory_never_reaches_writer_science_or_native(tmp_path,monkeypatch):
    (tmp_path/'ledger').mkdir(); (tmp_path/'ledger.launch').mkdir()
    def forbidden(*a,**k): raise AssertionError('unbound history reached work/authority')
    monkeypatch.setattr(module.costs,'inspect_closed_ledger',forbidden)
    monkeypatch.setattr(module.budget.Ledger,'create',forbidden)
    monkeypatch.setattr(module.budget.Ledger,'open',forbidden)
    monkeypatch.setattr(module.startup,'_closed_job',forbidden)
    with pytest.raises(ValueError,match='registered.*inventory'): module.inspect_input_interruption(tmp_path/'ledger')
    with pytest.raises(ValueError,match='absolute'): module.inspect_input_interruption('ledger')
    with pytest.raises(TypeError): module.inspect_input_interruption(tmp_path/'ledger',expected_root_sha256=digest('cheaper'))


def test_wrong_binding_version_is_not_reinterpreted_as_new_authority():
    with pytest.raises(ValueError,match='exact input interruption'):
        module.verify_input_interruption(envelope(dict(schema_version=module.startup.VERSION)))
