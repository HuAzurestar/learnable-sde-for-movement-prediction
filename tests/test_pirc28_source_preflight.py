"""Disposable admitted-metadata controls, NOT native execution/SCI proof."""

import copy
import time
from types import SimpleNamespace

import pytest

from application.research_admission import AdmissionGate
from application.research_budget import BudgetLedger, BudgetSpec
from experiments.pirc28.runtime import switching_plugin
from experiments.pirc28.source_preflight import validate_scientific_source
from infrastructure.research_store import ResearchError, digest
from tests import test_pirc28_study as study_tests
from tests.test_pirc28_approximation import v2_master


@pytest.fixture(scope='module')
def admitted_sources(tmp_path_factory):
    # Only simulated operator provenance in a disposable store. RUNNING below
    # is metadata for pre-fit checks; no worker/MC result or SCI grant is made.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(study_tests, 'definition_value', v2_master)
        store, master, grants, _, studies = study_tests.prepared(tmp_path_factory.mktemp('preflight')/'store')
    study = store.manifest('study-'+studies[0]['study_id'])['spec']
    sources = []
    for k in (1, 2):
        cell = next(cell for cell in study['cells'] if cell['fit_K'] == k)
        run = store.register_run(study['study_id'], cell)
        attempt_id = store.new_attempt(run)
        BudgetLedger(store).reserve(attempt_id, BudgetSpec(60, category='pilot'))
        AdmissionGate(store).prepare(study, cell, switching_plugin(), attempt_id)
        attempt = store.transition(attempt_id, 'RUNNING')
        sources.append((study, cell, attempt))
    return store, master, grants, sources


def poll():
    pass  # Explicit unfunded metadata-unit fixture, never production numerical work.


@pytest.mark.parametrize('k', [1, 2])
def test_complete_original_preflight_is_not_execution_or_qualification(admitted_sources, k):
    store, master, _, sources = admitted_sources
    study, cell, attempt = sources[k-1]
    result = validate_scientific_source(store, study, cell, attempt, check_budget=poll)
    assert result['study_definition_hash'] == master.definition_hash
    assert result['attempt_id'] == attempt['attempt_id']
    assert result['run_id'] == attempt['run_id']
    assert result['qualified_identifiability'] is False
    assert result['real_data_authorized'] is False
    assert not any(event['event_kind'] == 'WORKER_STARTED' for event in store.events())


@pytest.mark.parametrize('damage', ['master', 'store', 'code', 'seed', 'arm', 'allocation', 'fixture', 'formal', 'attempt'])
def test_changed_source_is_refused(admitted_sources, damage):
    store, _, _, sources = admitted_sources
    study, cell, attempt = copy.deepcopy(sources[1])
    if damage == 'master':
        study.pop('pirc28_study_definition_hash')
    elif damage == 'store':
        study['runtime_binding']['store_id'] = 'another-ledger'
    elif damage == 'code':
        study['code_hash'] = '0'*64
    elif damage == 'seed':
        cell['seed'] += 1
    elif damage == 'arm':
        cell['arm_id'] = 'new-budget-alias'
    elif damage == 'allocation':
        cell['execution']['config']['paths'] += 1
    elif damage in ('fixture', 'formal'):
        study['admission']['mode'] = damage
    else:
        attempt['run_id'] = '0'*64
    with pytest.raises(ResearchError):
        validate_scientific_source(store, study, cell, attempt, check_budget=poll)


@pytest.mark.parametrize('damage', ['missing-study', 'protocol', 'preregistration', 'receipt',
    'command', 'registry', 'resources', 'grant', 'package'])
def test_borrowed_or_incomplete_canonical_metadata_is_refused(admitted_sources, monkeypatch, damage):
    store, master, _, sources = admitted_sources
    study, cell, attempt = sources[1]
    original = store.manifest
    foreign = next(name for name in master.study_ids if name != study['study_id'])
    def manifest(name):
        if damage == 'missing-study' and name == 'study-'+foreign:
            raise ResearchError('MISSING_INPUT', 'disposable missing source')
        value = original(name)
        if damage == 'protocol' and name.startswith('protocol-'):
            value['blocks'][0]['sha256'] = '0'*64
        elif damage == 'preregistration' and name.startswith('preregistration-'):
            value['study_ids'] = [study['study_id']]
        elif name.startswith('admission-') and damage in ('receipt', 'command', 'registry', 'resources', 'grant', 'package'):
            if damage == 'receipt':
                value['cell_hash'] = '0'*64
            elif damage == 'command':
                value['command_hash'] = '0'*64
            elif damage == 'registry':
                value['registry_entry']['version'] = '99.0.0'
            elif damage == 'resources':
                value['resource_plan'] = {}
            elif damage == 'grant':
                value['documents']['authorization']['purposes'] = ['export']
            else:
                value['documents']['package']['code_hash'] = '0'*64
        return value
    monkeypatch.setattr(store, 'manifest', manifest)
    with pytest.raises(ResearchError):
        validate_scientific_source(store, study, cell, attempt, check_budget=poll)


@pytest.mark.parametrize('damage', ['late-master', 'late-preregistration', 'missing-admission', 'missing-reserve', 'settled',
    'admission-before-funding'])
def test_publication_admission_and_original_funding_order(admitted_sources, monkeypatch, damage):
    store, _, _, sources = admitted_sources
    study, cell, attempt = sources[1]
    events = store.events()
    last = events[-1]['sequence']+1
    reserve_sequence = next(event['sequence'] for event in events if event['event_kind'] == 'RESERVE'
        and event['payload'].get('attempt_id') == attempt['attempt_id'])
    modified = []
    for event in events:
        own = event['payload'].get('attempt_id') == attempt['attempt_id']
        if own and ((damage == 'missing-admission' and event['event_kind'] == 'ADMISSION')
                    or (damage == 'missing-reserve' and event['event_kind'] == 'RESERVE')):
            continue
        if event['event_kind'] == 'MANIFEST' and (
            damage == 'late-master' and event['payload']['object_id'].startswith('pirc28-study-definition-')
            or damage == 'late-preregistration' and event['payload']['object_id'].startswith('preregistration-')):
            event['sequence'] = last
        if own and damage == 'admission-before-funding' and event['event_kind'] == 'ADMISSION':
            event['sequence'] = reserve_sequence-1
        modified.append(event)
        if own and damage == 'settled' and event['event_kind'] == 'RESERVE':
            modified.append({**event, 'event_kind': 'SETTLE'})
    monkeypatch.setattr(store, 'events', lambda: modified)
    with pytest.raises(ResearchError):
        validate_scientific_source(store, study, cell, attempt, check_budget=poll)


def test_deadline_authority_propagates_without_numerical_failure(admitted_sources):
    store, _, _, sources = admitted_sources
    def expired():
        raise ResearchError('TIMEOUT', 'disposable deadline')
    with pytest.raises(ResearchError, match='disposable deadline'):
        validate_scientific_source(store, *sources[1], check_budget=expired)


def test_complete_metadata_is_bounded_before_numerical_work(admitted_sources, monkeypatch):
    from experiments.pirc28 import source_preflight
    store, _, _, sources = admitted_sources
    monkeypatch.setattr(source_preflight, 'MAXIMUM_DEFINITION_BYTES', 1)
    with pytest.raises(ResearchError) as error:
        validate_scientific_source(store, *sources[1], check_budget=poll)
    assert error.value.code == 'RESOURCE_LIMIT'


@pytest.mark.parametrize('damage', ['command', 'grant'])
def test_self_consistent_borrowed_receipt_cannot_bypass_specific_binding(admitted_sources, monkeypatch, damage):
    # Inject a self-consistent metadata projection only, never publish it or
    # claim a native run. This passes the hash check so later refusals aren't
    # masked by the deliberate byte-integrity failures in other controls.
    store, _, _, sources = admitted_sources
    study, cell, attempt = sources[1]
    events = store.events()
    admission = next(event for event in events if event['event_kind'] == 'ADMISSION'
        and event['payload']['attempt_id'] == attempt['attempt_id'])
    receipt = store.manifest('admission-'+admission['payload']['admission_hash'])
    if damage == 'command':
        receipt['command_hash'] = '0'*64
    else:
        receipt['documents']['authorization']['purposes'] = ['export']
    receipt.pop('admission_hash')
    reference = digest(receipt)
    receipt['admission_hash'] = reference
    admission['payload']['admission_hash'] = reference
    original = store.manifest
    monkeypatch.setattr(store, 'events', lambda: events)
    monkeypatch.setattr(store, 'manifest', lambda name: copy.deepcopy(receipt)
        if name == 'admission-'+reference else original(name))
    with pytest.raises(ResearchError) as error:
        validate_scientific_source(store, study, cell, attempt, check_budget=poll)
    assert error.value.code == ('IDENTITY_CONFLICT' if damage == 'command' else 'UNAUTHORIZED_DATA')


def test_worker_valid_preflight_supplies_only_explicit_source_diagnostic_executor(admitted_sources, monkeypatch):
    from experiments.pirc28 import pipeline, source_preflight, worker
    store, _, _, sources = admitted_sources
    study, cell, attempt = sources[1]
    directory = store.path.parent/'control-fixture'
    control = SimpleNamespace(directory=directory,
        descriptor={'attempt_id': attempt['attempt_id'], 'deadline': time.monotonic()+60})
    monkeypatch.setattr(worker, 'ResearchStore', lambda *_: store)
    monkeypatch.setattr(worker.WorkerControl, 'from_environment', lambda: control)
    monkeypatch.setattr(worker.sys, 'argv', ['worker', str(store.path.parent), store.store_id,
        study['study_id'], digest(cell), str(directory/'result.json')])
    def forbidden(*_, **__):
        pytest.fail('preflight must not enable V2 generation')
    monkeypatch.setattr(pipeline, 'generate_scenario', forbidden)
    original, checked = source_preflight.validate_scientific_source, []
    def observed(*args, **kwargs):
        value = original(*args, **kwargs)
        checked.append(value)
        return value
    monkeypatch.setattr(source_preflight, 'validate_scientific_source', observed)
    def execute(*args, **kwargs):
        assert callable(kwargs['source_approximation_executor'])
        assert kwargs['fixture_approximation_executor'] is None
        raise ResearchError('TEST_PIPELINE_ENTERED', 'disposable invocation control, no numerical execution')
    monkeypatch.setattr(worker, 'execute_recipe', execute)
    with pytest.raises(ResearchError) as error:
        worker.main()
    assert error.value.code == 'TEST_PIPELINE_ENTERED'
    assert len(checked) == 1 and checked[0]['attempt_id'] == attempt['attempt_id']


def test_worker_preflight_authority_failure_precedes_pipeline(admitted_sources, monkeypatch):
    from experiments.pirc28 import source_preflight, worker
    store, _, _, sources = admitted_sources
    study, cell, attempt = sources[1]
    directory = store.path.parent/'control-fixture'
    control = SimpleNamespace(directory=directory,
        descriptor={'attempt_id': attempt['attempt_id'], 'deadline': time.monotonic()+60})
    monkeypatch.setattr(worker, 'ResearchStore', lambda *_: store)
    monkeypatch.setattr(worker.WorkerControl, 'from_environment', lambda: control)
    monkeypatch.setattr(worker.sys, 'argv', ['worker', str(store.path.parent), store.store_id,
        study['study_id'], digest(cell), str(directory/'result.json')])
    def refused(*_, **__):
        raise ResearchError('UNAUTHORIZED_DATA', 'pre-fit authority refused')
    def forbidden(*_, **__):
        pytest.fail('pipeline must not precede scientific source preflight')
    monkeypatch.setattr(source_preflight, 'validate_scientific_source', refused)
    monkeypatch.setattr(worker, 'execute_recipe', forbidden)
    with pytest.raises(ResearchError, match='pre-fit authority refused'):
        worker.main()
