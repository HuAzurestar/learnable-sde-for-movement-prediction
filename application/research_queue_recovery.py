"""Audited cancellation of a PIRC-28 reservation that never claimed a slot.

Native owner-stop evidence is an explicit operator attestation plus a live-PID
veto. An absent worker cannot have an execution charge, but a claimed slot or
any launch evidence must use the ordinary measured-stop recovery path instead.
"""

from copy import deepcopy
from datetime import datetime
import re

from infrastructure.research_store import ResearchError, digest
from .research_budget import BudgetLedger


SCHEMA = 'pirc28-prelaunch-cancellation-v1'
STOP_SCHEMA = 'pirc28-prelaunch-owner-stop-v1'
ERROR = 'QUEUE_CANCELLED_BEFORE_CLAIM'


def _require(condition, message, code='RECOVERY_REQUIRED'):
    if not condition:
        raise ResearchError(code, message)


def _published(store, reference):
    _require(type(reference) is dict and set(reference) == {'manifest_id', 'sha256'}
             and type(reference['manifest_id']) is str
             and re.fullmatch('[0-9a-f]{64}', str(reference['sha256'])),
             'exact prelaunch evidence reference required')
    value = store._manifest(reference['manifest_id'])
    _require(digest(value) == reference['sha256'], 'prelaunch evidence hash differs')
    return value


def _stop_evidence(store, reference):
    value = _published(store, reference)
    fields = {'schema_version', 'task_id', 'session_id', 'pid', 'created_at',
              'stop_confirmed_at', 'native_child_count', 'native_identity_check',
              'reservation_id', 'attempt_id'}
    _require(set(value) == fields and value['schema_version'] == STOP_SCHEMA
             and type(value['pid']) is int and value['pid'] > 1
             and type(value['native_child_count']) is int and value['native_child_count'] == 0
             and value['native_identity_check'] == 'exact-command-parent-and-creation-time'
             and all(type(value[field]) is str and 0 < len(value[field]) <= 128
                     for field in ('task_id', 'session_id')),
             'bounded native prelaunch owner-stop evidence required')
    try:
        created = datetime.fromisoformat(value['created_at'])
        stopped = datetime.fromisoformat(value['stop_confirmed_at'])
        _require(created.tzinfo is not None and stopped.tzinfo is not None and created < stopped,
                 'prelaunch owner-stop timestamps differ')
    except (ValueError, TypeError) as error:
        if isinstance(error, ResearchError):
            raise
        raise ResearchError('RECOVERY_REQUIRED', 'invalid owner-stop timestamp') from error
    return value


def _unclaimed(store, reservation_id):
    events = store._events()
    reserves = [e for e in events if e['event_kind'] == 'RESERVE'
                and e['payload'].get('reservation_id') == reservation_id]
    queues = [e for e in events if e['event_kind'] == 'QUEUED'
              and e['payload'].get('reservation_id') == reservation_id]
    _require(len(reserves) == len(queues) == 1, 'one registered prelaunch queue reservation required')
    reserve, queued = reserves[0], queues[0]
    reservation = reserve['payload']
    attempt_id = reservation['attempt_id']
    _require(reservation['arm_id'] in {'pirc28-k1', 'pirc28-k2'}
             and reservation.get('budget_mode') == 'resource-guarded-no-time-cap'
             and reservation['reserved_ms'] is None
             and queued['payload'].get('attempt_id') == attempt_id
             and queued['payload'].get('arm_id') == reservation['arm_id']
             and reserve['sequence'] < queued['sequence'], 'PIRC-28 prelaunch reservation differs')
    _require(not any(e['event_kind'] in {'QUEUE_CLAIMED', 'WORKER_STARTED', 'WORKER_TREE_STOPPED',
                                       'WORKER_STOP_UNCONFIRMED', 'CHECKPOINT', 'CHECKPOINT_SAVED'}
                     and (e['payload'].get('reservation_id') == reservation_id
                          or e['payload'].get('attempt_id') == attempt_id) for e in events),
             'claimed or possibly launched work cannot be cancelled at zero cost')
    attempt = store._attempts().get(attempt_id)
    _require(attempt is not None and attempt['run_id'] == reservation['run_id']
             and attempt['started_at'] is None and attempt.get('artifact_id') is None
             and attempt.get('artifact_manifest_hash') is None
             and (attempt['state'] == 'REGISTERED' or
                  (attempt['state'] == 'PREFLIGHT_FAILED' and attempt['error_code'] == ERROR)),
             'attempt is not a result-free never-started queue entry')
    _require(not any(e['event_kind'] == 'ATTEMPT' and e['payload'].get('attempt_id') == attempt_id
                     and e['payload'].get('state') == 'RUNNING' for e in events),
             'running history cannot be cancelled as prelaunch')
    settlements = [e for e in events if e['event_kind'] == 'SETTLE'
                   and e['payload'].get('reservation_id') == reservation_id]
    _require(len(settlements) <= 1 and all(
        e['payload'].get('settled') is True
        and e['payload'].get('outcome') == 'PREFLIGHT_FAILED'
        and type(e['payload'].get('charged_ms')) is int
        and e['payload']['charged_ms'] == e['payload'].get('monotonic_elapsed_ms') == 0
        for e in settlements), 'existing execution cost cannot be replaced by prelaunch cancellation')
    return events, reserve, queued, attempt


def resolve_prelaunch_cancellation(store, reference):
    with store._read_transaction():
        receipt = _published(store, reference)
        _require(set(receipt) == {'schema_version', 'reservation_id', 'attempt_id',
            'run_id', 'stop_evidence', 'event_hashes', 'worker_charged_ms'}
            and receipt['schema_version'] == SCHEMA and type(receipt['worker_charged_ms']) is int
            and receipt['worker_charged_ms'] == 0, 'prelaunch cancellation schema differs')
        stopped = _stop_evidence(store, receipt['stop_evidence'])
        events, reserve, queued, attempt = _unclaimed(store, receipt['reservation_id'])
        _require(stopped['reservation_id'] == receipt['reservation_id']
                 and stopped['attempt_id'] == receipt['attempt_id']
                 and datetime.fromisoformat(stopped['stop_confirmed_at'])
                     >= datetime.fromisoformat(queued['created_at']),
                 'owner-stop evidence does not bind this queued reservation')
        selected = {}
        for name, kind in (('cancellation', 'QUEUE_PRELAUNCH_CANCELLED'), ('settlement', 'SETTLE'),
                           ('terminal', 'ATTEMPT')):
            matches = [e for e in events if e['hash'] == receipt['event_hashes'].get(name)]
            _require(len(matches) == 1 and matches[0]['event_kind'] == kind,
                     'prelaunch cancellation event binding differs')
            selected[name] = matches[0]
        cancellation, settlement, terminal = (selected[name] for name in
                                               ('cancellation', 'settlement', 'terminal'))
        expected = {'reservation_id': receipt['reservation_id'], 'attempt_id': receipt['attempt_id'],
                    'run_id': receipt['run_id'], 'stop_evidence': receipt['stop_evidence'],
                    'worker_charged_ms': 0}
        cost = settlement['payload']
        _require(set(receipt['event_hashes']) == {'reserve', 'queued', 'cancellation', 'settlement', 'terminal'}
                 and receipt['event_hashes']['reserve'] == reserve['hash']
                 and receipt['event_hashes']['queued'] == queued['hash']
                 and cancellation['payload'] == expected
                 and attempt['attempt_id'] == receipt['attempt_id'] and attempt['run_id'] == receipt['run_id']
                 and terminal['payload'] == attempt and attempt['state'] == 'PREFLIGHT_FAILED'
                 and attempt['error_code'] == ERROR
                 and cost['reservation_id'] == receipt['reservation_id']
                 and cost['attempt_id'] == receipt['attempt_id'] and cost['settled'] is True
                 and cost['outcome'] == 'PREFLIGHT_FAILED'
                 and all(cost.get(field) == reserve['payload'].get(field) for field in
                     ('run_id', 'study_id', 'arm_id', 'reserved_ms', 'worker_slot', 'execution_policy',
                      'source_code_hash', 'budget_mode', 'restart_adjudication'))
                 and type(cost['charged_ms']) is int and cost['charged_ms'] == cost['monotonic_elapsed_ms'] == 0
                 and type(cost['monotonic_elapsed_ms']) is int
                 and queued['sequence'] < cancellation['sequence'] < settlement['sequence'] < terminal['sequence'],
                 'prelaunch cancellation source/cost/order differs')
        return deepcopy(receipt)


def cancel_unclaimed(store, reservation_id, *, stop_evidence):
    """Recover only an explicitly stopped owner with no claim or worker history."""
    from infrastructure.process_tree import process_may_be_alive

    event_id = 'prelaunch-cancel-' + reservation_id
    receipt_id = 'prelaunch-receipt-' + reservation_id
    with store._read_transaction():
        _require(not (store.path / 'recovery-hold.json').exists(), 'shared recovery hold remains')
        stopped = _stop_evidence(store, stop_evidence)
        events, reserve, queued, attempt = _unclaimed(store, reservation_id)
        _require(stopped['reservation_id'] == reservation_id and stopped['attempt_id'] == attempt['attempt_id']
                 and datetime.fromisoformat(stopped['stop_confirmed_at'])
                     >= datetime.fromisoformat(queued['created_at']),
                 'owner-stop evidence does not bind this queued reservation')
        cancellations = [e for e in events if e['event_id'] == event_id]
        payload = {'reservation_id': reservation_id, 'attempt_id': attempt['attempt_id'],
                   'run_id': attempt['run_id'], 'stop_evidence': deepcopy(stop_evidence),
                   'worker_charged_ms': 0}
        _require(len(cancellations) <= 1 and all(e['payload'] == payload for e in cancellations),
                 'prelaunch cancellation replay changed stop evidence', 'IDENTITY_CONFLICT')
        if not cancellations:
            _require(not process_may_be_alive(stopped['pid']),
                     'prelaunch owner may still be alive', 'WORKER_ACTIVE')
        # The cancellation intent is durable before cost/terminal publication;
        # replay can finish the same ordered evidence after a publication fault.
        cancellation = store._append('QUEUE_PRELAUNCH_CANCELLED', payload, event_id)
        settled = BudgetLedger(store).settle(reservation_id, 0, outcome='PREFLIGHT_FAILED')
        store._transition(attempt['attempt_id'], 'PREFLIGHT_FAILED', error_code=ERROR)
        events = store._events()
        settlement = next(e for e in events if e['event_kind'] == 'SETTLE'
                          and e['payload'] == settled)
        terminal = next(e for e in events if e['event_kind'] == 'ATTEMPT'
                        and e['payload'].get('attempt_id') == attempt['attempt_id']
                        and e['payload'].get('state') == 'PREFLIGHT_FAILED')
        receipt = {'schema_version': SCHEMA, **payload, 'event_hashes': {
            'reserve': reserve['hash'], 'queued': queued['hash'], 'cancellation': cancellation['hash'],
            'settlement': settlement['hash'], 'terminal': terminal['hash']}}
        store.publish(receipt_id, receipt)
        reference = {'manifest_id': receipt_id, 'sha256': digest(receipt)}
        resolve_prelaunch_cancellation(store, reference)
        return reference
