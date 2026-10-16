"""Disposable v3-lineage/native controls; never production qualification."""

from copy import deepcopy

import pytest

from application.research_admission import AdmissionGate
from application.research_budget import BudgetLedger, ResourceBudgetSpec
from application.research_preflight_recovery import approve_preflight_recovery, adjudicate_preflight_recovery
from application.research_queue import ResearchQueue
from application.research_restart_adjudication import adjudicate_restart, resolve_retry_adjudication
from application.research_unfunded_source import supersede_unfunded_restart
from experiments.pirc25.affine import code_hash
from experiments.pirc28 import source_preflight, study as definitions
from experiments.pirc28.release import execute_first_restart
from experiments.pirc28.runtime import switching_plugin
from infrastructure.research_store import ResearchError, digest
from tests.test_research_preflight_recovery import future_epoch
from tests.test_research_unfunded_source import proposal


@pytest.fixture
def native_case(tmp_path, monkeypatch, request):
    # Historical PID 123/stop attestation is simulated. Do not stub observations
    # for any worker launched by the real native execution below.
    from infrastructure.process_tree import process_may_be_alive
    monkeypatch.setattr('infrastructure.process_tree.process_may_be_alive',
        lambda pid: False if pid == 123 else process_may_be_alive(pid))
    if getattr(request, 'param', None) == 'v2':
        from tests.test_pirc28_approximation import v2_master
        monkeypatch.setattr('tests.test_pirc28_source_epoch.definition_value', v2_master)
    elif getattr(request, 'param', None) == 'full64':
        from experiments.pirc28.plan import REFERENCE_PLAN, load_plan, prepare_definition
        def full_master():
            value = prepare_definition(load_plan(REFERENCE_PLAN)).to_dict()
            value['runtime_code_hash'] = 'a' * 64
            return value
        monkeypatch.setattr('tests.test_pirc28_source_epoch.definition_value', full_master)
    with monkeypatch.context() as legacy:
        legacy.setattr('experiments.pirc28.runtime.VERSION', '1.1.0')
        for name in ('tests.test_pirc28_study.code_hash', 'tests.test_pirc28_source_epoch.code_hash',
                     'tests.test_research_unfunded_source.code_hash',
                     'experiments.pirc28.source_epoch.code_hash', 'experiments.pirc28.study.code_hash'):
            legacy.setattr(name, lambda: 'a' * 64)
        store, root, old_policy, old_epoch, cancellation, unfunded = proposal(tmp_path, legacy)
        # cancelled_epoch uses its own inner patch. Restore the narrow predicate
        # so native worker liveness will not inherit the historical simulation.
        legacy.setattr('infrastructure.process_tree.process_may_be_alive',
            lambda pid: False if pid == 123 else process_may_be_alive(pid))
        supersession = supersede_unfunded_restart(store, unfunded,
            source_code_hash='f' * 64, task_id='synthetic-test-only', session_id='v3-fixture',
            reason='simulated unfunded source correction, not human authorization')
        old_bridge = store.manifest(old_epoch['source_bridge']['manifest_id'])
        original = store.manifest('pirc28-study-definition-' + old_bridge['historical_definition_hash'])
        parent_policy, parent_epoch = future_epoch(store, original, old_policy, 'f' * 64, legacy)
        previous = adjudicate_restart(store, root, source_bridge=parent_epoch['source_bridge'],
            execution_policy=parent_policy, source_code_hash='f' * 64,
            prelaunch_cancellations=[cancellation], unfunded_supersession=supersession)
        previous_receipt = store.manifest(previous['manifest_id'])
        assert previous_receipt['schema_version'] == 'pirc28-restart-adjudication-v3'
        parent = previous_receipt['retry_attempt_id']
        run = store.manifest('run-' + previous_receipt['current_run_id'])
        spec = store.manifest('study-' + run['study_id'])['spec']
        plugin = switching_plugin()
        retained_entry = deepcopy(plugin.registry_entry.manifest())
        retained_entry['code_hash'] = digest('retained older adapter, disposable fixture')
        entry_hash = store.publish('registry-entry-' + digest(retained_entry), retained_entry)
        store.publish('registry-version-' + digest({'id': plugin.plugin_id, 'version': '1.1.0'}),
            {'component_id': plugin.plugin_id, 'component_version': '1.1.0', 'registry_entry_hash': entry_hash})
        reservation = BudgetLedger(store).reserve(parent, ResourceBudgetSpec(parent_policy, 'f' * 64))
        ResearchQueue(store).enqueue(reservation)
        assert ResearchQueue(store).claim(reservation['reservation_id']) is not None
        # Reconstruct the old operator's preflight at its simulated source;
        # restore actual source hashing before any new native worker exists.
        with legacy.context() as old_operator:
            old_operator.setattr('experiments.pirc25.affine.code_hash', lambda: 'f' * 64)
            with pytest.raises(ResearchError, match='IDENTITY_CONFLICT'):
                AdmissionGate(store).prepare(spec, run['cell'], plugin, parent)
        BudgetLedger(store).settle(reservation['reservation_id'], 0, outcome='PREFLIGHT_FAILED')
        store.transition(parent, 'PREFLIGHT_FAILED', error_code='IDENTITY_CONFLICT')
        old_definition = store.manifest('pirc28-study-definition-' + parent_epoch['study_definition_hash'])
    current_policy, current_epoch = future_epoch(store, old_definition, parent_policy, code_hash(), monkeypatch)
    decision = approve_preflight_recovery(store, parent_attempt_id=parent,
        source_bridge=current_epoch['source_bridge'], execution_policy=current_policy,
        source_code_hash=code_hash(), decision_basis={
            'authority': 'local-human-decision-record', 'solution_point': 'SOL-007', 'result': 'CONFIRMED',
            'solution_commit': '1' * 40, 'task_id': 'synthetic-test-only', 'session_id': 'not-human-authorization'})
    reference = adjudicate_preflight_recovery(store, decision)
    receipt = resolve_retry_adjudication(store, reference,
        execution_policy=current_policy, source_code_hash=code_hash())
    return store, current_policy, current_epoch, reference, receipt, previous, supersession


def test_native_v3_recovery_stops_settles_reuses_and_only_then_promotes(native_case):
    store, policy, _, reference, receipt, previous, supersession = native_case
    before = store.events()
    old_mapping_id = 'registry-version-' + digest({'id': 'pirc28-switching', 'version': '1.1.0'})
    old_mapping = store.manifest(old_mapping_id)
    release = execute_first_restart(store, restart_adjudication=reference)
    assert release['result']['state'] == 'SUCCEEDED'
    assert release['result']['attempt_id'] == receipt['retry_attempt_id']
    assert release['verified']['reused'] is True
    assert store.manifest(old_mapping_id) == old_mapping
    assert store.events()[:len(before)] == before
    assert store.manifest(previous['manifest_id'])['unfunded_supersession'] == supersession
    retired = store.manifest(supersession['manifest_id'])['attempt_id']
    assert store.attempts()[retired]['error_code'] == 'SOURCE_CORRECTED_BEFORE_FUNDING'
    assert not any(e['event_kind'] in {'RESERVE', 'SETTLE', 'WORKER_STARTED'}
                   and e['payload'].get('attempt_id') == retired for e in store.events())
    events = [e for e in store.events() if e['payload'].get('attempt_id') == receipt['retry_attempt_id']]
    for kind in ('RESERVE', 'WORKER_STARTED', 'WORKER_TREE_STOPPED', 'SETTLE'):
        assert len([e for e in events if e['event_kind'] == kind]) == 1
    reserve = next(e for e in events if e['event_kind'] == 'RESERVE')
    launch = next(e for e in events if e['event_kind'] == 'WORKER_STARTED')
    stop = next(e for e in events if e['event_kind'] == 'WORKER_TREE_STOPPED')
    settle = next(e for e in events if e['event_kind'] == 'SETTLE')
    assert reserve['payload']['restart_adjudication'] == reference
    assert launch['payload']['deadline_monotonic'] is None
    assert stop['payload']['confirmation'] == 'native-job-or-process-group-no-running-descendants'
    assert reserve['sequence'] < launch['sequence'] < stop['sequence'] < settle['sequence']
    assert settle['payload']['charged_ms'] == settle['payload']['monotonic_elapsed_ms'] > 0
    all_events = store.events()
    reuse = next(e for e in all_events if e['event_kind'] == 'REUSE_VERIFIED'
                 and e['payload'].get('attempt_id') == receipt['retry_attempt_id'])
    continuation = next(e for e in all_events if e['event_kind'] == 'RESOURCE_ARM_CONTINUATION'
                        and e['payload'] == release['resource_arm_continuation'])
    assert settle['sequence'] < reuse['sequence'] < continuation['sequence']
    ledger = BudgetLedger(store)
    assert ledger.balance('pirc28-k1')['closed'] is True
    assert ledger.balance('pirc28-k1', execution_policy=policy, source_code_hash=code_hash())['closed'] is False
    starts = sum(e['event_kind'] == 'WORKER_STARTED' for e in store.events())
    replay = execute_first_restart(store, restart_adjudication=reference)
    assert replay['verified']['reused'] is True
    assert replay['resource_arm_continuation'] == release['resource_arm_continuation']
    assert sum(e['event_kind'] == 'WORKER_STARTED' for e in store.events()) == starts
    with pytest.raises(ResearchError, match='RETRY_EXHAUSTED'):
        store.new_attempt(receipt['current_run_id'], parent_attempt_id=receipt['retry_attempt_id'], reason='no fourth')


@pytest.mark.parametrize('native_case', ['v2'], indirect=True)
def test_cold_v3_worker_primes_all_definition_epochs_outside_writer(native_case, monkeypatch):
    store, policy, current_epoch, reference, receipt, *_ = native_case
    run = store.manifest('run-' + receipt['current_run_id'])
    spec = store.manifest('study-' + run['study_id'])['spec']
    BudgetLedger(store).reserve(receipt['retry_attempt_id'], ResourceBudgetSpec(policy, code_hash()))
    AdmissionGate(store).prepare(spec, run['cell'], switching_plugin(), receipt['retry_attempt_id'])
    attempt = store.transition(receipt['retry_attempt_id'], 'RUNNING')
    definitions._validated_definition.cache_clear()
    original = definitions._validate
    scopes = []
    def cold(value):
        scopes.append(store._read_snapshot())
        assert store._read_snapshot() is None, 'cold definition validation owns shared writer'
        return original(value)
    monkeypatch.setattr(definitions, '_validate', cold)
    result = source_preflight.validate_scientific_source(store, spec, run['cell'], attempt,
                                                        check_budget=lambda: None)
    assert result['study_definition_hash'] == current_epoch['study_definition_hash']
    assert result['qualified_identifiability'] is False
    assert len(scopes) == 3 and all(scope is None for scope in scopes)
    assert resolve_retry_adjudication(store, reference,
        execution_policy=policy, source_code_hash=code_hash()) == receipt
