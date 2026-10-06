"""Temporary managed qualification controls, not scientific replicates."""

from dataclasses import replace
import copy
import json

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from experiments.pirc28.computation import (QualificationRunner, computation_plan,
    validate_computation_result)
from experiments.pirc28.evidence import collect_qualification, source_attempts_hash
from infrastructure.research_store import ResearchError, digest
from tests.test_pirc28_evidence import completed_pair, prepared
from tests.test_pirc28_qualification import spec as definition_spec
from tests.test_pirc28_synthetic import fixture_poll


@pytest.fixture(scope='module')
def computed(completed_pair):
    store, study, definition, grant, _ = completed_pair
    before = BudgetLedger(store).balance('pirc28-k2')
    runner = QualificationRunner(store)
    result = runner.run(definition, [study['study_id']], {study['study_id']: grant},
        budget=BudgetSpec(90, category='smoke'), allow_fixture=True)
    if result['state'] != 'SUCCEEDED':
        path = store.path/'artifacts'/('.attempt-'+result['attempt_id'])/'worker.log'
        pytest.fail(str(result)+'\n'+path.read_text(errors='replace'))
    return completed_pair, result, before


def test_actual_managed_qualification_charges_original_arm_and_publishes_receipt(computed):
    data, result, before = computed
    store, study, _, _, _ = data
    assert result['qualification_state'] == 'FIXTURE_ONLY'
    assert result['real_data_authorized'] is False and result['reused'] is False
    receipt = store.manifest(result['receipt_id'])
    assert digest(receipt) == result['receipt_hash']
    assert receipt['cost']['arm_id'] == 'pirc28-k2'
    assert receipt['cost']['charged_ms'] > 0
    assert receipt['cost']['reserved_ms'] == 90_000
    assert receipt['qualified_identifiability'] is receipt['real_data_authorized'] is False
    assert BudgetLedger(store).balance('pirc28-k2')['committed_ms'] == before['committed_ms']+receipt['cost']['charged_ms']
    derived = store.manifest('study-'+receipt['study_id'])['spec']
    assert derived['arms'] == [study['arms'][1]]
    assert 'capability' not in derived['cells'][0]  # No invented motion capability.
    with store.lock():
        value = json.loads(store._verified_artifact_content(store._manifest('artifact-'+result['artifact_id'])))
    assert value['report']['source_attempts_hash'] == source_attempts_hash(store, [study])
    assert value['report']['gate']['state'] == 'INCOMPLETE'
    assert len(value['report']['rows']) == 4
    assert value['report']['lineage'][0]['attempts'][0]['cost']['charged_ms'] == 2000


def test_successful_computation_reuse_has_no_new_reservation(computed):
    data, first, _ = computed
    store, study, definition, grant, _ = data
    reserves = [event for event in store.events() if event['event_kind'] == 'RESERVE']
    result = QualificationRunner(store).run(definition, [study['study_id']], {study['study_id']: grant},
        budget=BudgetSpec(90, category='smoke'), allow_fixture=True)
    assert result['reused'] is True and result['attempt_id'] == first['attempt_id']
    assert result['artifact_id'] == first['artifact_id'] and result['receipt_hash'] == first['receipt_hash']
    assert [event for event in store.events() if event['event_kind'] == 'RESERVE'] == reserves


@pytest.mark.parametrize('damage', ['gate', 'report-hash', 'source-snapshot', 'request', 'real-authorization', 'fixture-promotion'])
def test_result_bindings_refuse_tampered_disposition(computed, damage):
    data, result, _ = computed
    store = data[0]
    with store.lock():
        value = json.loads(store._verified_artifact_content(store._manifest('artifact-'+result['artifact_id'])))
    request = store.manifest('pirc28-qualification-request-'+result['attempt_id'])
    if damage == 'gate':
        value['report']['qualified_identifiability'] = True
        value['report_hash'] = digest(value['report'])
    elif damage == 'report-hash':
        value['report_hash'] = digest('wrong')
    elif damage == 'source-snapshot':
        value['report']['source_attempts_hash'] = digest('wrong')
        value['report_hash'] = digest(value['report'])
    elif damage == 'request':
        value['request_hash'] = digest('wrong')
    elif damage == 'real-authorization':
        value['report']['real_data_authorized'] = True
        value['report_hash'] = digest(value['report'])
    else:
        value['report']['state'] = value['report']['gate']['state'] = 'PASSED'
        value['report']['qualified_identifiability'] = True
        value['report_hash'] = digest(value['report'])
    with pytest.raises(ResearchError, match='CONTRACT_MISMATCH'):
        validate_computation_result(store, value, request)


def test_wrong_stage_and_forged_grants_do_not_reserve(tmp_path):
    store, study, definition, grant, _ = prepared(tmp_path/'scope')
    runner = QualificationRunner(store)
    with pytest.raises(ResearchError, match='CONTRACT_MISMATCH'):
        runner.run(definition, [study['study_id']], {study['study_id']: grant},
                   budget=BudgetSpec(90, category='pilot'), allow_fixture=True)
    with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
        runner.run(definition, [study['study_id']], {study['study_id']: {**grant, 'purposes': ['preview']}},
                   budget=BudgetSpec(90, category='smoke'), allow_fixture=True)
    assert not any(event['event_kind'] == 'RESERVE' for event in store.events())


def test_closed_original_arm_refuses_computation_without_launch(tmp_path):
    store, study, definition, grant, _ = prepared(tmp_path/'closed')
    store.append('ARM_CLOSED', {'arm_id': 'pirc28-k2', 'reason': 'explicit test closure'})
    with pytest.raises(ResearchError, match='BUDGET_EXHAUSTED'):
        QualificationRunner(store).run(definition, [study['study_id']], {study['study_id']: grant},
            budget=BudgetSpec(90, category='smoke'), allow_fixture=True)
    assert not any(event['event_kind'] in ('RESERVE', 'WORKER_STARTED') for event in store.events())
    assert any(attempt['state'] == 'BUDGET_EXHAUSTED' for attempt in store.attempts().values())


def test_successful_reuse_still_requires_current_grants(computed, monkeypatch):
    data, _, _ = computed
    store, study, definition, grant, _ = data
    original = store.authorization
    def expired(identifier, **options):
        value = original(identifier, **options)
        return {**value, 'expires_at': '2000-01-01T00:00:00+00:00'}
    monkeypatch.setattr(store, 'authorization', expired)
    with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
        QualificationRunner(store).run(definition, [study['study_id']], {study['study_id']: grant},
            budget=BudgetSpec(90, category='smoke'), allow_fixture=True)


@pytest.mark.parametrize('damage', ['missing-settlement', 'wrong-launch-reservation', 'missing-tree-stop'])
def test_reuse_requires_actual_computation_cost_and_containment_receipts(computed, monkeypatch, damage):
    data, result, _ = computed
    store, study, definition, grant, _ = data
    events = copy.deepcopy(store.events())
    if damage == 'missing-settlement':
        events = [event for event in events if not (event['event_kind'] == 'SETTLE'
            and event['payload'].get('attempt_id') == result['attempt_id'])]
    elif damage == 'missing-tree-stop':
        events = [event for event in events if not (event['event_kind'] == 'WORKER_TREE_STOPPED'
            and event['payload'].get('attempt_id') == result['attempt_id'])]
    else:
        event = next(event for event in events if event['event_kind'] == 'WORKER_STARTED'
            and event['payload'].get('attempt_id') == result['attempt_id'])
        event['payload']['reservation_id'] = digest('other reservation')
    monkeypatch.setattr(store, 'events', lambda: events)
    with pytest.raises(ResearchError, match='UNQUALIFIED'):
        QualificationRunner(store).run(definition, [study['study_id']], {study['study_id']: grant},
            budget=BudgetSpec(90, category='smoke'), allow_fixture=True)


def test_bounded_bootstrap_and_read_allocations():
    definition = definition_spec(2)
    plan = computation_plan(definition, 1024)
    assert plan['maximum_rows'] == 4 and plan['bootstrap_batch_draws'] == 32
    for quota in (0, True, 2**30+1):
        with pytest.raises(ResearchError, match='RESOURCE_PLAN_REJECTED'):
            computation_plan(definition, quota)
    axes = next(iter(definition.scenario_axes.values()))
    oversized = replace(definition, scenario_axes={'scenario-'+str(index): axes for index in range(2500)}, bootstrap_draws=10_000)
    with pytest.raises(ResearchError, match='RESOURCE_PLAN_REJECTED'):
        computation_plan(oversized, 1024)


def test_source_changes_during_collection_are_not_a_frozen_aggregate(completed_pair, monkeypatch):
    store, study, definition, grant, _ = completed_pair
    original = store.attempts
    calls = 0
    def changing():
        nonlocal calls
        calls += 1
        states = original()
        if calls >= 2:
            states = copy.deepcopy(states)
            selected = next(value for value in states.values() if value['state'] == 'SUCCEEDED')
            selected['error_code'] = 'altered-source-snapshot'
        return states
    monkeypatch.setattr(store, 'attempts', changing)
    with pytest.raises(ResearchError, match='INDEX_STALE'):
        collect_qualification(store, definition, [study['study_id']], {study['study_id']: grant},
            check_budget=fixture_poll, allow_fixture=True)


def test_internal_worker_rejects_standalone_unbudgeted_execution(monkeypatch, tmp_path):
    from experiments.pirc28.qualification_worker import main
    monkeypatch.delenv('PIRC25_WORKER_CONTROL', raising=False)
    with pytest.raises(ResearchError, match='MISSING_INPUT'):
        main([str(tmp_path), 'unused-store', str(tmp_path/'input.json'), str(tmp_path/'result.json')])


def test_final_clock_uses_chronological_expiry_not_offset_text(monkeypatch):
    from experiments.pirc28 import computation
    checked = []
    monkeypatch.setattr(computation, 'require_unexpired_disclosure',
        lambda store, study_id, grant, purpose: checked.append(study_id))
    computation._final_clock(None, {'authorizations': {
        'later': {'expires_at': '2100-01-01T00:30:00+00:00'},
        'earlier': {'expires_at': '2100-01-01T08:00:00+08:00'}}})
    assert checked == ['earlier']


def test_cli_uses_registered_definition_and_existing_grants(computed, monkeypatch, capsys):
    from experiments.pirc28.__main__ import main
    data, first, _ = computed
    store, study, definition, grant, _ = data
    definition_hash = store.publish('pirc28-identifiability-spec-'+digest(definition.to_dict()), definition.to_dict())
    # Native worker command routing is covered above; the CLI invokes that same runner.
    code = main(['--root', str(store.path.parent), '--store-id', store.store_id, '--seconds', '90',
        'qualify', definition_hash, '--study-grant', study['study_id']+'='+grant['authorization_id'], '--allow-fixture'])
    assert code == 0
    value = json.loads(capsys.readouterr().out)
    assert value['attempt_id'] == first['attempt_id'] and value['reused'] is True
