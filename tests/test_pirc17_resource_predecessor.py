"""Actual software journal, synthetic full-size sources; NOT formal admission.

Live science/native checks are explicitly replaced only for writer-unit tests.
All 6298 successes and the already-charged dispatched failure are represented.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from tests.test_pirc17_resource_policy import fixture as policy_fixture
from experiments.pirc17 import formal_budget as budget, formal_partial_predecessor as partial
from experiments.pirc17 import formal_partial_science as science, formal_resource_policy as policy
from experiments.pirc17 import formal_resource_predecessor as module
from experiments.pirc17.protocol_core import digest, envelope, unpack

NS = budget.NANOSECONDS


def science_metadata(f):
    """Synthetic ownership inventory only, never domain/admission evidence."""
    cost = f['cost']
    works = unpack(f['matrix'])['workloads']
    complete = [w for w in works if cost['work_dispositions'].get(w['work_id']) == 'success']

    def artifact(w):
        return dict(path='fixture-artifacts/'+w['work_id']+'.json',
            file_sha256=digest(['bytes', w['work_id']]), content_sha256=digest(['artifact', w['work_id']]))

    return dict(schema_version=science.VERSION, read_only=True, authorizes_execution=False,
        cross_execution_admitted=False, new_fits=0, new_forecasts=0,
        ledger_root_sha256=cost['ledger_root_sha256'], committed_tip=cost['ledger_tip'],
        **{'original_'+key+'_sha256': cost[key+'_sha256']
           for key in ('protocol', 'execution', 'matrix', 'approval')},
        completed_sources={w['work_id']: dict(result_sha256=digest(['result', w['work_id']]),
            settlement_sha256=digest(['settle', w['work_id']]),
            observation_sha256=digest(['native', w['work_id']]), controller_elapsed_ns=1,
            artifact=None if w['kind'] == 'input_qualification_and_population' else artifact(w)) for w in complete},
        successful_work_counts=dict(Counter(w['kind'] for w in complete)),
        fit_bindings={w['fit_identity']: artifact(w) for w in complete if w['kind'] in {'method_fit', 'terrain_fit'}},
        forecast_index={w['work_id']: artifact(w) for w in complete if w['kind'] in science.KINDS},
        forecast_kind_counts=dict(Counter(w['kind'] for w in complete if w['kind'] in science.KINDS)),
        forecast_status_counts={'success': 6271}, context_sha256=digest('SOFTWARE absent context'))


@pytest.fixture
def fixture(policy_fixture, monkeypatch):
    f = policy_fixture
    p = policy.build_policy(**f['args'])
    cost = f['cost']
    complete = [w for w in unpack(f['matrix'])['workloads']
                if cost['work_dispositions'].get(w['work_id']) == 'success']
    saved = science_metadata(f)
    ref = f['ref']('software-science', saved)
    binding = module.build_binding(resource_policy=p, science_reference=ref)
    directory = Path(f['contract']['ledger_directory']).parent/'software-resource-successor'
    runtime = envelope(dict(protocol_sha256=cost['protocol_sha256'], matrix_sha256=cost['matrix_sha256'],
        ledger_directory=str(directory), predecessor=binding, software_fixture_only=True))
    contract = budget.contract_for_matrix(f['matrix'], protocol_sha256=cost['protocol_sha256'],
        execution_sha256=digest('software-new-execution'), runtime_manifest_sha256=runtime['sha256'],
        approval_sha256=digest('SOFTWARE-NOT-REAL-APPROVAL'), ledger_directory=directory, resource_policy=p)
    checks = []
    monkeypatch.setattr(module, 'verify_resource_predecessor', lambda record: checks.append(record['sha256']))
    return dict(source=f, policy=p, saved=saved, binding=binding, runtime=runtime,
        contract=contract, directory=directory, checks=checks, complete=complete)


def test_closed_cap_floor_preserves_all_categories_results_and_failed_token_once(fixture):
    f = fixture
    with budget.Ledger.create(f['directory'], f['contract']) as ledger:
        ledger.import_resource_floor(f['runtime'])
        value = ledger.summary()
        for key in ('charged_ns_by_phase', 'measured_ns_by_phase', 'conservatively_charged_ns_by_phase',
                    'control_charged_ns_by_phase', 'control_observed_ns_by_phase'):
            assert value[key] == f['source']['cost'][key]
        assert sum(value['charged_ns_by_phase'].values()) == 13079435_000000
        assert value['generated_forecasts_reserved'] == 6272
        assert value['imported_success_count'] == 6298 and value['unattempted_work_items'] == 5361
        assert ledger._state.imported_success == f['saved']['completed_sources']
        assert ledger._state.status.get(policy.RETRY_WORK_ID) is None
        assert ledger._state.carried_generation_reservations == {}
        assert value['predecessor_floor']['retry_old_reservation_sha256'] == unpack(f['policy'])['retry_old_reservation_sha256']
        assert len((f['directory']/'events/000000.json').read_bytes()) < 64*1024
        before = ledger.summary()
        with pytest.raises(ValueError, match='healthy empty'):
            ledger.import_resource_floor(f['runtime'])
        assert ledger.summary() == before and f['checks'] == [f['binding']['sha256']]
        with pytest.raises(ValueError, match='already attempted'):
            ledger.reserve(f['complete'][0]['work_id'])


def test_approved_retry_buys_new_token_and_time_without_refunding_original_failure(fixture):
    f = fixture
    with budget.Ledger.create(f['directory'], f['contract']) as ledger:
        ledger.import_resource_floor(f['runtime'])
        before = ledger.summary()['charged_ns_by_phase']['method_forecasts']
        reservation = ledger.reserve(policy.RETRY_WORK_ID)
        assert reservation['reserved_ns'] == 30*NS  # Not the old645ms allocation.
        assert ledger.summary()['generated_forecasts_reserved'] == 6273
        assert ledger.summary()['charged_ns_by_phase']['method_forecasts'] == before+30*NS
        assert ledger.summary()['consumed_generation_transfers'] == {}
        ledger.settle(reservation['reservation_sha256'], status='success', elapsed_ns=NS,
            completion_evidence_sha256=digest('SOFTWARE-COMPLETION'),
            result_sha256=digest('SOFTWARE-RESULT'), reason='synthetic journal test')
        assert ledger.summary()['charged_ns_by_phase']['method_forecasts'] == before+NS
        with pytest.raises(ValueError, match='already attempted'):
            ledger.reserve(policy.RETRY_WORK_ID)


def test_cold_replay_requires_no_native_query_or_scientific_reinspection(fixture, monkeypatch):
    f = fixture
    with budget.Ledger.create(f['directory'], f['contract']) as ledger:
        ledger.import_resource_floor(f['runtime'])
        root, tip, expected = ledger.root_sha256, ledger.tip, ledger.summary()
    def forbidden(*args, **kwargs):
        raise AssertionError('cold accounting replay called live source inspection')
    monkeypatch.setattr(module, 'verify_resource_predecessor', forbidden)
    with budget.Ledger.open(f['directory'], expected_root_sha256=root, expected_tip=tip) as ledger:
        assert ledger.summary() == expected
    assert partial._history(f['binding']) == module._history(f['binding'])


def test_failed_live_source_verification_publishes_nothing(fixture, monkeypatch):
    f = fixture
    def fail(*args):
        raise ValueError('source closure or scientific replay failed')
    monkeypatch.setattr(module, 'verify_resource_predecessor', fail)
    with budget.Ledger.create(f['directory'], f['contract']) as ledger:
        before = ledger.summary()
        with pytest.raises(ValueError, match='source closure'):
            ledger.import_resource_floor(f['runtime'])
        assert ledger.summary() == before and list((f['directory']/'events').iterdir()) == []


@pytest.mark.parametrize('change', ['missing_success', 'import_timeout', 'wrong_tip', 'wrong_scope',
    'wrong_artifact', 'missing_fit', 'missing_forecast', 'extra_forecast'])
def test_incomplete_or_relabelled_source_cannot_be_imported(fixture, change):
    f = fixture
    saved = deepcopy(f['saved'])
    key = next(iter(saved['completed_sources']))
    if change == 'missing_success':
        del saved['completed_sources'][key]
    elif change == 'import_timeout':
        saved['completed_sources'][policy.RETRY_WORK_ID] = saved['completed_sources'][key]
    elif change == 'wrong_tip':
        saved['committed_tip']['event_count'] -= 1
    elif change == 'wrong_scope':
        saved['original_execution_sha256'] = digest('not-the-source')
    elif change == 'wrong_artifact':
        saved['completed_sources'][key]['artifact'] = next(iter(saved['forecast_index'].values()))
    elif change == 'missing_fit':
        del saved['fit_bindings'][next(iter(saved['fit_bindings']))]
    elif change == 'missing_forecast':
        del saved['forecast_index'][next(iter(saved['forecast_index']))]
    else:
        saved['forecast_index'][policy.RETRY_WORK_ID] = next(iter(saved['forecast_index'].values()))
    with pytest.raises(ValueError):
        module.build_binding(resource_policy=f['policy'], science_reference=f['source']['ref']('altered-science', saved))


def test_old_predispatch_event_cannot_apply_the_new_closed_cap_binding(fixture):
    f = fixture
    with budget.Ledger.create(f['directory'], f['contract']) as ledger:
        with pytest.raises(ValueError):
            ledger.import_partial_floor(f['runtime'])
        assert ledger.tip['event_count'] == 0 and not f['checks']


def test_resource_contract_cannot_dispatch_or_buy_control_before_its_full_floor(fixture):
    f = fixture
    with budget.Ledger.create(f['directory'], f['contract']) as ledger:
        before = ledger.summary()
        with pytest.raises(ValueError, match='predecessor floor first'):
            ledger.reserve(policy.RETRY_WORK_ID)
        with pytest.raises(ValueError, match='predecessor floor first'):
            ledger.open_control(digest('SOFTWARE-CONTROL'), phase='method_forecasts', credit_ns=NS)
        with pytest.raises(ValueError, match='predecessor floor first'):
            ledger.halt('software attempted substitute first event', digest('SOFTWARE-EVIDENCE'))
        assert ledger.summary() == before and list((f['directory']/'events').iterdir()) == []
