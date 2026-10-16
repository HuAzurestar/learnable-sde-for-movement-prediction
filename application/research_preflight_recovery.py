"""One explicitly approved PIRC-28 preflight recovery, not a third-retry policy.

Approval is a trusted local operator attestation, like an execution policy. It
must name the human-approved parent. Workers cannot self-authorize by reporting
an error. Fixed store-wide identities prevent reuse through another source,
decision, run, parent, or directory in the same authoritative store.
"""

from copy import deepcopy
import re

from infrastructure.research_store import ResearchError, digest, encode
from .research_restart_adjudication import (_published, _reference, _require,
    _resolve_receipt as resolve_previous, _prime_restart, prime_source_definitions,
    resolve_source_bridge, scientific_cell_identity, SCHEMA as RESTART_SCHEMA,
    PRELAUNCH_SCHEMA, SUPERSESSION_SCHEMA)

SCHEMA = 'pirc28-preflight-recovery-adjudication-v1'
DECISION_SCHEMA = 'pirc28-preflight-recovery-decision-v1'
DECISION_ID = 'pirc28-sol007-one-preflight-recovery-decision'
ADJUDICATION_ID = 'pirc28-sol007-one-preflight-recovery-adjudication'
APPROVAL_EVENT = 'PREFLIGHT_RECOVERY_APPROVED'
EVENT = 'PREFLIGHT_RECOVERY_ADJUDICATED'
EVENT_ID = 'pirc28-sol007-one-preflight-recovery'
REASON = 'PIRC-28 explicitly approved one-time registry preflight recovery'


def _basis(value):
    fields = {'authority', 'solution_point', 'result', 'solution_commit', 'task_id', 'session_id'}
    _require(type(value) is dict and set(value) == fields
             and value['authority'] == 'local-human-decision-record'
             and value['solution_point'] == 'SOL-007' and value['result'] == 'CONFIRMED'
             and type(value['solution_commit']) is str
             and re.fullmatch('[0-9a-f]{40}', value['solution_commit'])
             and all(type(value[k]) is str and 0 < len(value[k]) <= 128
                     for k in ('task_id', 'session_id')),
             'explicit bounded SOL-007 decision provenance required', 'UNAUTHORIZED_DATA')


def _parent_reference(store, parent_id):
    reserves = [e for e in store._events() if e['event_kind'] == 'RESERVE'
                and e['payload'].get('attempt_id') == parent_id]
    _require(len(reserves) == 1, 'one funded preflight parent required', 'RECOVERY_REQUIRED')
    reference = reserves[0]['payload'].get('restart_adjudication')
    _reference(reference, 'previous adjudication')
    return reference


def _prime_scope(store, previous, bridge):
    if store._read_snapshot() is None:
        _reference(previous, 'previous preflight adjudication')
        value = store.manifest(previous['manifest_id'])
        _require(type(value) is dict and digest(value) == previous['sha256']
                 and value.get('schema_version') in {RESTART_SCHEMA, PRELAUNCH_SCHEMA, SUPERSESSION_SCHEMA},
                 'one-time recovery must follow an original source restart, never another exception')
        _prime_restart(store, previous)
        prime_source_definitions(store, bridge)


def prime_recovery(store, value):
    if store._read_snapshot() is not None:
        return
    _require(type(value) is dict, 'preflight metadata must be an object')
    if value.get('schema_version') == SCHEMA:
        reference = value.get('decision')
        _reference(reference, 'preflight decision')
        value = store.manifest(reference['manifest_id'])
        _require(digest(value) == reference['sha256'], 'preflight decision hash differs')
    _require(type(value) is dict and value.get('schema_version') == DECISION_SCHEMA,
             'preflight decision schema differs')
    scope = value.get('scope')
    _require(type(scope) is dict and {'previous_adjudication', 'source_bridge'} <= set(scope),
             'preflight scope is incomplete')
    _prime_scope(store, scope['previous_adjudication'], scope['source_bridge'])


def _facts(store, parent_id, bridge_ref, policy_ref, source):
    """Re-derive every fact from one fresh physically verified owned prefix."""
    from .research_execution import execution_plan
    from .research_execution_policy import resolve_policy
    from experiments.pirc28.runtime import switching_plugin

    _require(type(parent_id) is str and re.fullmatch('[0-9a-f]{32}', parent_id),
             'exact preflight parent identity required')
    _reference(bridge_ref, 'preflight source bridge')
    _reference(policy_ref, 'preflight execution policy')
    events, attempts = store._events(), store._attempts()
    parent = attempts.get(parent_id)
    _require(parent is not None and parent['state'] == 'PREFLIGHT_FAILED'
             and parent.get('error_code') == 'IDENTITY_CONFLICT'
             and parent.get('started_at') is None and parent.get('artifact_id') is None
             and parent.get('artifact_manifest_hash') is None,
             'exact artifact-free never-started identity preflight failure required', 'RECOVERY_REQUIRED')
    run = store._manifest('run-' + parent['run_id'])
    _require(run['arm_id'] == 'pirc28-k1' and type(run['cell'].get('fit_K')) is int
             and run['cell']['fit_K'] == 1 and run['cell'].get('visibility') == 'synthetic',
             'one-time recovery is scoped to the frozen synthetic K1 cell', 'UNAUTHORIZED_DATA')
    previous_ref = _parent_reference(store, parent_id)
    previous = resolve_previous(store, previous_ref)
    _require(previous['retry_attempt_id'] == parent_id and previous['current_run_id'] == run['run_id']
             and type(previous['retry_ordinal']) is int and previous['retry_ordinal'] == 2
             and type(previous['maximum_infrastructure_retries']) is int
             and previous['maximum_infrastructure_retries'] == 2
             and previous['current_code_hash'] != source,
             'retain the exact exhausted second source retry', 'RETRY_EXHAUSTED')
    related = [e for e in events if e['payload'].get('attempt_id') == parent_id]
    kinds = ['ATTEMPT', 'RESERVE', 'QUEUED', 'QUEUE_CLAIMED', 'SETTLE', 'ATTEMPT']
    _require([e['event_kind'] for e in related] == kinds,
             'preflight parent has extra, missing or possible execution activity', 'RECOVERY_REQUIRED')
    registration, reserve, queued, claimed, settled, terminal = related
    reservation, cost = reserve['payload'], settled['payload']
    rid = reservation['reservation_id']
    _require(registration['payload']['state'] == 'REGISTERED' and terminal['payload'] == parent
             and all(e['payload'].get('reservation_id') == rid for e in related[1:5])
             and reservation.get('restart_adjudication') == previous_ref
             and reservation.get('budget_mode') == 'resource-guarded-no-time-cap'
             and reservation.get('reserved_ms') is None
             and reservation.get('run_id') == run['run_id']
             and all(e['payload'].get('arm_id') == 'pirc28-k1' for e in related[1:5])
             and claimed['payload'].get('resource') == 'cpu'
             and type(claimed['payload'].get('slot')) is int and claimed['payload']['slot'] == 0
             and cost.get('settled') is True and cost.get('outcome') == 'PREFLIGHT_FAILED'
             and all(type(cost.get(k)) is int and cost[k] == 0
                     for k in ('charged_ms', 'monotonic_elapsed_ms'))
             and cost.get('restart_adjudication') == previous_ref,
             'claimed preflight zero-cost settlement bindings differ', 'RECOVERY_REQUIRED')
    # Include reservation-only activity too; unknown scoped events must not be
    # silently treated as proof that no read, checkpoint or worker ever existed.
    _require([e for e in events if e['payload'].get('attempt_id') == parent_id
              or e['payload'].get('reservation_id') == rid] == related,
             'additional reservation activity prevents zero-worker recovery', 'RECOVERY_REQUIRED')
    old_policy = resolve_policy(store, previous['execution_policy'], arm_id=run['arm_id'],
                                source_code_hash=previous['current_code_hash'])
    current_policy = resolve_policy(store, policy_ref, arm_id=run['arm_id'], source_code_hash=source)
    _require(encode(old_policy['resources']) == encode(current_policy['resources']),
             'one-time recovery cannot expand resource guards', 'UNAUTHORIZED_DATA')
    bridge = resolve_source_bridge(store, bridge_ref, execution_policy=policy_ref, source_code_hash=source)
    _require(bridge['bridge']['historical_code_hash'] == previous['current_code_hash'],
             'recovery source bridge does not start at its failed parent', 'IDENTITY_CONFLICT')
    lineage = bridge['by_historical_study'].get(run['study_id'])
    _require(lineage is not None and digest(lineage['historical']) == run['spec_hash'],
             'preflight parent is absent from the complete source bridge', 'IDENTITY_CONFLICT')
    pair = next((p for p in lineage['pair']['cell_pairs'] if p['historical_cell_hash'] == run['cell_hash']), None)
    _require(pair is not None, 'preflight scientific cell is absent from source bridge')
    target_spec = lineage['current']
    _require(target_spec.get('runtime_binding') == {'root': str(store.path.parent), 'store_id': store.store_id},
             'one-time target must use the same authoritative store', 'UNAUTHORIZED_DATA')
    target = next(c for c in target_spec['cells'] if digest(c) == pair['current_cell_hash'])
    _require(encode({k: v for k, v in run['cell'].items() if k != 'execution'})
             == encode({k: v for k, v in target.items() if k != 'execution'}),
             'one-time recovery changes the frozen cell', 'IDENTITY_CONFLICT')
    old_binding, new_binding = run['cell']['execution'], target['execution']
    _require(encode({k: v for k, v in old_binding.items()
                     if k not in {'component_version', 'registry_entry_hash', 'resource_plan_hash'}})
             == encode({k: v for k, v in new_binding.items()
                        if k not in {'component_version', 'registry_entry_hash', 'resource_plan_hash'}}),
             'one-time recovery changes execution inputs or allocation', 'IDENTITY_CONFLICT')
    old_entry = store._manifest('registry-entry-' + old_binding['registry_entry_hash'])
    _require(old_entry.get('component_id') == 'pirc28-switching' and old_entry.get('version') == '1.1.0',
             'retained 1.1.0 adapter evidence required', 'RECOVERY_REQUIRED')
    old_version_id = 'registry-version-' + digest({'id': 'pirc28-switching', 'version': '1.1.0'})
    old_mapping = store._manifest(old_version_id)
    catalog = store._manifest('registry-entry-' + old_mapping['registry_entry_hash'])
    _require(old_mapping == {'component_id': 'pirc28-switching', 'component_version': '1.1.0',
                            'registry_entry_hash': digest(catalog)}
             and digest(old_entry) == old_binding['registry_entry_hash']
             and old_entry['code_hash'] != catalog.get('code_hash')
             and encode({k: v for k, v in old_entry.items() if k != 'code_hash'})
                 == encode({k: v for k, v in catalog.items() if k != 'code_hash'}),
             'actual retained version/code conflict evidence required', 'RECOVERY_REQUIRED')
    plugin = switching_plugin()
    entry = plugin.registry_entry.manifest()
    _require(entry['version'] != '1.1.0'
             and encode({k: v for k, v in entry.items() if k not in {'version', 'code_hash'}})
                 == encode({k: v for k, v in old_entry.items() if k not in {'version', 'code_hash'}}),
             'distinct version with unchanged adapter contract required', 'IDENTITY_CONFLICT')
    execution_plan(target_spec, target, plugin)
    current_id = 'registry-version-' + digest({'id': plugin.plugin_id, 'version': entry['version']})
    if store._manifest_event(current_id, last=True) is not None:
        _require(store._manifest(current_id) == {'component_id': plugin.plugin_id,
            'component_version': entry['version'], 'registry_entry_hash': digest(entry)},
            'replacement version already has a different immutable identity', 'IDENTITY_CONFLICT')
    else:
        _require(not (store.path / 'manifests' / (current_id + '.json')).exists(),
                 'unpublished replacement catalog needs reconciliation', 'RECOVERY_REQUIRED')
    target_run = {'study_id': target_spec['study_id'], 'experiment_id': target_spec['experiment_id'],
        'spec_hash': digest(target_spec), 'arm_id': target['arm_id'], 'cell_hash': digest(target),
        'seed': target.get('seed'), 'cell': target}
    facts = {'parent_attempt_id': parent_id, 'parent_reservation_id': rid,
        'previous_adjudication': previous_ref, 'source_bridge': deepcopy(bridge_ref),
        'execution_policy': deepcopy(policy_ref), 'current_code_hash': source,
        'historical_run_id': run['run_id'], 'current_run_id': digest(target_run),
        'current_cell_hash': digest(target), 'scientific_cell_hash': digest(scientific_cell_identity(target)),
        'retained_cost_ms': previous['retained_cost_ms'],
        'historical_event_hashes': deepcopy(previous['historical_event_hashes']),
        'parent_event_hashes': dict(zip(('registration', 'reserve', 'queued', 'claim', 'settlement', 'terminal'),
                                      (e['hash'] for e in related)))}
    return facts, target_spec, target


def _decision(store, reference):
    _reference(reference, 'one-time preflight decision')
    _require(reference.get('manifest_id') == DECISION_ID, 'fixed one-time decision identity required')
    value = _published(store, reference, 'one-time preflight decision')
    _require(type(value) is dict and set(value) == {'schema_version', 'store_id', 'decision', 'maximum_extra_attempts',
                           'ordinary_maximum_infrastructure_retries', 'decision_basis', 'scope'}
             and value['schema_version'] == DECISION_SCHEMA and value['store_id'] == store.store_id
             and value['decision'] == 'approved'
             and type(value['maximum_extra_attempts']) is int and value['maximum_extra_attempts'] == 1
             and type(value['ordinary_maximum_infrastructure_retries']) is int
             and value['ordinary_maximum_infrastructure_retries'] == 2,
             'one-time preflight decision scope differs', 'UNAUTHORIZED_DATA')
    _basis(value['decision_basis'])
    scope = value['scope']
    _require(type(scope) is dict and {'parent_attempt_id', 'source_bridge', 'execution_policy',
                                    'current_code_hash'} <= set(scope), 'preflight decision scope incomplete')
    facts, _, _ = _facts(store, scope['parent_attempt_id'], scope['source_bridge'],
                          scope['execution_policy'], scope['current_code_hash'])
    _require(digest(scope) == digest(facts), 'approved preflight evidence changed', 'IDENTITY_CONFLICT')
    approved = [e for e in store._events() if e['event_kind'] == APPROVAL_EVENT]
    _require(len(approved) == 1 and approved[0]['payload'] == reference,
             'one global explicit preflight decision required', 'UNAUTHORIZED_DATA')
    return value


def approve_preflight_recovery(store, *, parent_attempt_id, source_bridge, execution_policy,
                              source_code_hash, decision_basis):
    """Trusted operator only: publish one immutable scoped decision, no attempt."""
    _basis(decision_basis)
    with store._read_transaction():
        previous = _parent_reference(store, parent_attempt_id)
    _prime_scope(store, previous, source_bridge)
    with store._read_transaction():
        facts, spec, cell = _facts(store, parent_attempt_id, source_bridge, execution_policy, source_code_hash)
        value = {'schema_version': DECISION_SCHEMA, 'store_id': store.store_id, 'decision': 'approved',
            'maximum_extra_attempts': 1, 'ordinary_maximum_infrastructure_retries': 2,
            'decision_basis': deepcopy(decision_basis), 'scope': facts}
        # Reject changed replay before publishing even unrelated run metadata.
        if (store.path / 'manifests' / (DECISION_ID + '.json')).exists():
            _require(store._manifest(DECISION_ID) == value,
                     'one-time decision cannot change source, parent or provenance', 'IDENTITY_CONFLICT')
            reference = {'manifest_id': DECISION_ID, 'sha256': digest(value)}
            if not any(e['event_kind'] == APPROVAL_EVENT for e in store._events()):
                _active_scope(store, facts)
                _require(not any(a['run_id'] == facts['current_run_id'] for a in store._attempts().values()),
                         'unapproved one-time target has work', 'RECOVERY_REQUIRED')
                store._append(APPROVAL_EVENT, reference, EVENT_ID + '-approval')
            _decision(store, reference)
            return reference
        _active_scope(store, facts)
        _require(not any(a['run_id'] == facts['current_run_id'] for a in store._attempts().values()),
                 'one-time target already has an attempt', 'IDENTITY_CONFLICT')
        _require(store.register_run(spec['study_id'], cell) == facts['current_run_id'], 'target run differs')
        sha = store.publish(DECISION_ID, value)
        reference = {'manifest_id': DECISION_ID, 'sha256': sha}
        store._append(APPROVAL_EVENT, reference, EVENT_ID + '-approval')
        _decision(store, reference)
        return reference


def _child_id(store, facts):
    return digest([store.store_id, EVENT_ID, facts['current_run_id']])[:32]


def _attempt(receipt):
    return {'attempt_id': receipt['retry_attempt_id'], 'run_id': receipt['current_run_id'],
        'parent_attempt_id': receipt['parent_attempt_id'], 'reason': REASON, 'state': 'REGISTERED',
        'started_at': None, 'ended_at': None, 'error_code': None, 'artifact_manifest_hash': None}


def _expected(store, decision_ref, decision):
    scope = decision['scope']
    return {'schema_version': SCHEMA, 'decision': deepcopy(decision_ref), **deepcopy(scope),
        'retry_attempt_id': _child_id(store, scope), 'retry_ordinal': 3,
        'maximum_infrastructure_retries': 2, 'maximum_extra_attempts': 1}


def _active_scope(store, receipt):
    closures = [e['hash'] for e in store._events() if e['event_kind'] == 'ARM_CLOSED'
                and e['payload'].get('arm_id') == 'pirc28-k1']
    _require(closures[-1:] == [receipt['historical_event_hashes']['closure']],
             'one-time recovery cannot cross a later closure', 'BUDGET_EXHAUSTED')
    origin = next(e for e in store._events() if e['hash'] == closures[-1])
    for event in store._events()[origin['sequence']:]:
        if event['event_kind'] in {'ARM_CONTINUATION', 'RESOURCE_ARM_CONTINUATION'}:
            continuation = store._manifest(event['payload']['manifest_id'])
            _require(continuation.get('arm_id') != 'pirc28-k1',
                     'one-time recovery cannot borrow a different continuation', 'BUDGET_EXHAUSTED')


def resolve_preflight_recovery(store, reference, *, allow_missing_attempt=False):
    """Owned-scope resolver used by all existing funding/result boundary paths."""
    _reference(reference, 'one-time preflight adjudication')
    _require(reference.get('manifest_id') == ADJUDICATION_ID, 'fixed one-time adjudication identity required')
    receipt = _published(store, reference, 'one-time preflight adjudication')
    _require(type(receipt) is dict and 'decision' in receipt, 'one-time receipt fields incomplete')
    decision = _decision(store, receipt['decision'])
    _require(digest(receipt) == digest(_expected(store, receipt['decision'], decision)),
             'one-time receipt changes the approved scope', 'IDENTITY_CONFLICT')
    events = store._events()
    publications = [e for e in events if e['event_kind'] == EVENT]
    _require(len(publications) == 1 and publications[0]['payload'] == reference,
             'one-time adjudication must be globally unique', 'IDENTITY_CONFLICT')
    child = store._attempts().get(receipt['retry_attempt_id'])
    others = [a for a in store._attempts().values() if a['run_id'] == receipt['current_run_id']]
    _require(all(a['attempt_id'] == receipt['retry_attempt_id'] for a in others),
             'one-time run has another attempt', 'RETRY_EXHAUSTED')
    if child is None:
        _require(allow_missing_attempt, 'one-time child missing', 'RECOVERY_REQUIRED')
        _active_scope(store, receipt)
    else:
        initial = _attempt(receipt)
        registrations = [e for e in events if e['event_kind'] == 'ATTEMPT' and e['payload'] == initial]
        _require(len(registrations) == 1 and registrations[0]['sequence'] > publications[0]['sequence']
                 and all(child.get(k) == initial[k] for k in ('attempt_id', 'run_id', 'parent_attempt_id', 'reason')),
                 'one-time child origin differs', 'IDENTITY_CONFLICT')
        if child['state'] in {'REGISTERED', 'RUNNING'}:
            _active_scope(store, receipt)
    return deepcopy(receipt)


def adjudicate_preflight_recovery(store, decision_reference):
    """Register exactly one deterministic child; never fund or launch it."""
    _reference(decision_reference, 'one-time preflight decision')
    value = store.manifest(decision_reference['manifest_id'])
    _require(digest(value) == decision_reference['sha256'], 'preflight decision hash differs')
    prime_recovery(store, value)
    with store._read_transaction():
        decision = _decision(store, decision_reference)
        previous = [e for e in store._events() if e['event_id'] == EVENT_ID]
        if previous:
            _require(len(previous) == 1 and previous[0]['event_kind'] == EVENT,
                     'one-time adjudication replay differs', 'IDENTITY_CONFLICT')
            reference = previous[0]['payload']
            receipt = resolve_preflight_recovery(store, reference, allow_missing_attempt=True)
        else:
            receipt = _expected(store, decision_reference, decision)
            _active_scope(store, receipt)
            _require(not any(a['run_id'] == receipt['current_run_id'] for a in store._attempts().values()),
                     'one-time target already has work', 'IDENTITY_CONFLICT')
            sha = store.publish(ADJUDICATION_ID, receipt)
            reference = {'manifest_id': ADJUDICATION_ID, 'sha256': sha}
            store._append(EVENT, reference, EVENT_ID)
        if store._attempts().get(receipt['retry_attempt_id']) is None:
            store._append('ATTEMPT', _attempt(receipt), 'attempt-' + receipt['retry_attempt_id'])
        resolve_preflight_recovery(store, reference)
        return reference


def forbid_extra_attempt(store, run_id=None, parent_attempt_id=None):
    """No ordinary creation/recovery path may retry the consumed one-time run."""
    for event in store._events():
        if event['event_kind'] != EVENT:
            continue
        receipt = _published(store, event['payload'], 'one-time preflight adjudication')
        if run_id == receipt['current_run_id'] or parent_attempt_id == receipt['retry_attempt_id']:
            raise ResearchError('RETRY_EXHAUSTED', 'one-time preflight recovery cannot be retried')


def is_recovery_attempt(store, attempt_id):
    for event in store._events():
        if event['event_kind'] == EVENT:
            receipt = _published(store, event['payload'], 'one-time preflight adjudication')
            if receipt.get('retry_attempt_id') == attempt_id:
                return True
    return False
