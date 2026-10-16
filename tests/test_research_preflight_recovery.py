"""Disposable operator-attestation controls, not production permissions/results."""

from copy import deepcopy

import pytest

from application.research_admission import AdmissionGate
from application.research_budget import BudgetLedger, ResourceBudgetSpec
from application.research_preflight_recovery import (approve_preflight_recovery,
    adjudicate_preflight_recovery, DECISION_ID, EVENT)
from application.research_queue import ResearchQueue
from application.research_restart_adjudication import (resolve_retry_adjudication,
    adjudicate_resource_retry, promote_resource_arm)
from experiments.pirc25.affine import code_hash
from experiments.pirc28.runtime import switching_plugin
from experiments.pirc28.source_epoch import register_source_epoch
from experiments.pirc28.study import SyntheticStudyDefinition
from infrastructure.research_store import ResearchError, digest
from tests.test_research_unfunded_source import proposal


def future_epoch(store, old_definition, old_policy, source, monkeypatch):
    value = deepcopy(old_definition)
    value['runtime_code_hash'] = source
    definition = SyntheticStudyDefinition.from_dict(value)
    policy = deepcopy(store.manifest(old_policy['manifest_id']))
    policy['source_code_hash'] = source
    policy_ref = {'manifest_id': 'future-policy-' + digest(policy), 'sha256': digest(policy)}
    store.publish(policy_ref['manifest_id'], policy)
    with monkeypatch.context() as patch:
        patch.setattr('experiments.pirc28.source_epoch.code_hash', lambda: source)
        patch.setattr('experiments.pirc28.study.code_hash', lambda: source)
        epoch = register_source_epoch(store, digest(old_definition), definition, execution_policy=policy_ref)
    return policy_ref, epoch


@pytest.fixture
def case(tmp_path, monkeypatch, request):
    fault = getattr(request, 'param', None)
    # Freeze genuinely different historical version metadata. The fixture's
    # native-stop attestation is explicitly simulated by the existing helpers;
    # no process or real research input is used by this evidence-only scope.
    with monkeypatch.context() as legacy:
        legacy.setattr('experiments.pirc28.runtime.VERSION', '1.1.0')
        store, _, old_policy, old_epoch, _, previous = proposal(tmp_path, legacy)
        previous_receipt = store.manifest(previous['manifest_id'])
        parent = previous_receipt['retry_attempt_id']
        run = store.manifest('run-' + previous_receipt['current_run_id'])
        spec = store.manifest('study-' + run['study_id'])['spec']
        plugin = switching_plugin()
        old_entry = deepcopy(plugin.registry_entry.manifest())
        old_entry['code_hash'] = digest('older retained implementation, fixture only')
        entry_hash = store.publish('registry-entry-' + digest(old_entry), old_entry)
        store.publish('registry-version-' + digest({'id': plugin.plugin_id, 'version': '1.1.0'}),
            {'component_id': plugin.plugin_id, 'component_version': '1.1.0', 'registry_entry_hash': entry_hash})
        reservation = BudgetLedger(store).reserve(parent, ResourceBudgetSpec(old_policy, code_hash()))
        ResearchQueue(store).enqueue(reservation)
        assert ResearchQueue(store).claim(reservation['reservation_id']) is not None
        with pytest.raises(ResearchError, match='IDENTITY_CONFLICT'):
            AdmissionGate(store).prepare(spec, run['cell'], plugin, parent)
        if fault == 'started':
            store.transition(parent, 'RUNNING')
        BudgetLedger(store).settle(reservation['reservation_id'], 1 if fault == 'cost' else 0,
                                   outcome='PREFLIGHT_FAILED')
        store.transition(parent, 'PREFLIGHT_FAILED',
                         error_code='CONTRACT_MISMATCH' if fault == 'error' else 'IDENTITY_CONFLICT')
        old_definition = store.manifest('pirc28-study-definition-' + old_epoch['study_definition_hash'])
    policy, epoch = future_epoch(store, old_definition, old_policy, 'd' * 64, monkeypatch)
    options = {'parent_attempt_id': parent, 'source_bridge': epoch['source_bridge'],
        'execution_policy': policy, 'source_code_hash': 'd' * 64, 'decision_basis': {
            'authority': 'local-human-decision-record', 'solution_point': 'SOL-007', 'result': 'CONFIRMED',
            'solution_commit': '1' * 40, 'task_id': 'synthetic-test-only', 'session_id': 'not-human-authorization'}}
    return store, options, previous, old_definition, old_policy


def register(case):
    store, options, *_ = case
    decision = approve_preflight_recovery(store, **options)
    reference = adjudicate_preflight_recovery(store, decision)
    receipt = resolve_retry_adjudication(store, reference,
        execution_policy=options['execution_policy'], source_code_hash=options['source_code_hash'])
    return decision, reference, receipt


def test_approval_registers_one_child_and_existing_budget_queue_resolve_it(case):
    store, options, previous, *_ = case
    before = store.events()
    attempts = store.attempts()
    decision = approve_preflight_recovery(store, **options)
    assert store.attempts() == attempts  # Approval is not execution or an attempt.
    reference = adjudicate_preflight_recovery(store, decision)
    receipt = resolve_retry_adjudication(store, reference,
        execution_policy=options['execution_policy'], source_code_hash=options['source_code_hash'])
    assert receipt['parent_attempt_id'] == options['parent_attempt_id']
    assert receipt['retry_ordinal'] == 3 and receipt['maximum_infrastructure_retries'] == 2
    assert receipt['maximum_extra_attempts'] == 1 and receipt['retained_cost_ms'] == 47892
    assert receipt['previous_adjudication'] == previous
    assert store.events()[:len(before)] == before
    assert BudgetLedger(store).balance('pirc28-k1')['closed'] is True
    reservation = BudgetLedger(store).reserve(receipt['retry_attempt_id'],
        ResourceBudgetSpec(options['execution_policy'], options['source_code_hash']))
    assert reservation['restart_adjudication'] == reference
    ResearchQueue(store).enqueue(reservation)
    assert ResearchQueue(store).claim(reservation['reservation_id']) is not None
    after = store.events()
    assert approve_preflight_recovery(store, **options) == decision
    assert adjudicate_preflight_recovery(store, decision) == reference
    assert store.events() == after
    assert not any(e['event_kind'] == 'WORKER_STARTED' for e in after[len(before):])


def test_one_time_failure_cannot_register_retry_or_promote_an_arm(case):
    store, options, *_ = case
    _, reference, receipt = register(case)
    child = receipt['retry_attempt_id']
    reservation = BudgetLedger(store).reserve(child,
        ResourceBudgetSpec(options['execution_policy'], options['source_code_hash']))
    BudgetLedger(store).settle(reservation['reservation_id'], 0, outcome='PREFLIGHT_FAILED')
    store.transition(child, 'PREFLIGHT_FAILED', error_code='IDENTITY_CONFLICT')
    before = store.events()
    with pytest.raises(ResearchError, match='RETRY_EXHAUSTED'):
        store.new_attempt(receipt['current_run_id'], parent_attempt_id=child, reason='not permitted')
    with pytest.raises(ResearchError, match='RETRY_EXHAUSTED'):
        adjudicate_resource_retry(store, child, execution_policy=options['execution_policy'],
                                  source_code_hash=options['source_code_hash'])
    with pytest.raises(ResearchError, match='UNQUALIFIED'):
        promote_resource_arm(store, reference)
    assert store.events() == before
    assert BudgetLedger(store).balance('pirc28-k1')['closed'] is True


@pytest.mark.parametrize('case', ['cost', 'started', 'error'], indirect=True)
def test_nonzero_cost_started_or_other_failure_is_not_this_exception(case):
    store, options, *_ = case
    before = store.events()
    with pytest.raises(ResearchError):
        approve_preflight_recovery(store, **options)
    assert store.events() == before


@pytest.mark.parametrize('kind', ['WORKER_STARTED', 'CHECKPOINT', 'READ_STARTED', 'UNKNOWN_ACTIVITY'])
def test_extra_parent_activity_cannot_be_declared_zero_worker(case, kind):
    store, options, *_ = case
    store.append(kind, {'attempt_id': options['parent_attempt_id'], 'fixture': 'not real execution'})
    before = store.events()
    with pytest.raises(ResearchError, match='activity'):
        approve_preflight_recovery(store, **options)
    assert store.events() == before


def test_valid_alternative_source_cannot_reset_global_exception(case, monkeypatch):
    store, options, _, old_definition, old_policy = case
    approve_preflight_recovery(store, **options)
    policy, epoch = future_epoch(store, old_definition, old_policy, 'e' * 64, monkeypatch)
    alternative = {**options, 'source_bridge': epoch['source_bridge'], 'execution_policy': policy,
                   'source_code_hash': 'e' * 64}
    before = store.events()
    with pytest.raises(ResearchError, match='one-time decision cannot change'):
        approve_preflight_recovery(store, **alternative)
    assert store.events() == before


@pytest.mark.parametrize('stage', ['approval', 'attempt'])
def test_partial_publication_replay_keeps_one_decision_and_one_child(case, monkeypatch, stage):
    store, options, *_ = case
    original = store._append
    def fail_once(kind, payload, *args, **kwargs):
        if (stage == 'approval' and kind == 'PREFLIGHT_RECOVERY_APPROVED') or (
                stage == 'attempt' and kind == 'ATTEMPT' and payload.get('parent_attempt_id') == options['parent_attempt_id']):
            raise OSError('simulated operator interruption')
        return original(kind, payload, *args, **kwargs)
    if stage == 'approval':
        with monkeypatch.context() as patch:
            patch.setattr(store, '_append', fail_once)
            with pytest.raises(OSError):
                approve_preflight_recovery(store, **options)
    decision = approve_preflight_recovery(store, **options)
    if stage == 'attempt':
        with monkeypatch.context() as patch:
            patch.setattr(store, '_append', fail_once)
            with pytest.raises(OSError):
                adjudicate_preflight_recovery(store, decision)
    reference = adjudicate_preflight_recovery(store, decision)
    receipt = store.manifest(reference['manifest_id'])
    assert decision['manifest_id'] == DECISION_ID
    assert len([e for e in store.events() if e['event_kind'] == EVENT]) == 1
    assert len([a for a in store.attempts().values() if a['run_id'] == receipt['current_run_id']]) == 1
