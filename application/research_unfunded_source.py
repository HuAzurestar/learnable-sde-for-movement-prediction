"""Replace only an unfunded source registration; never recycle paid retries."""

from copy import deepcopy
import re

from infrastructure.research_store import ResearchError, digest
from .research_queue_recovery import _published


SCHEMA = 'pirc28-unfunded-source-supersession-v1'
ERROR = 'SOURCE_CORRECTED_BEFORE_FUNDING'


def _require(condition, message):
    if not condition:
        raise ResearchError('RECOVERY_REQUIRED', message)


def _evidence(store, restart):
    receipt = _published(store, restart)
    attempt = store._attempts().get(receipt.get('retry_attempt_id'))
    _require(attempt is not None and attempt['run_id'] == receipt.get('current_run_id')
             and attempt['parent_attempt_id'] == receipt.get('parent_attempt_id')
             and attempt.get('started_at') is None and attempt.get('artifact_id') is None
             and attempt.get('artifact_manifest_hash') is None
             and (attempt['state'] == 'REGISTERED' or
                  (attempt['state'] == 'CANCELLED' and attempt.get('error_code') == ERROR)),
             'source proposal is not an unfunded registration')
    events = store._events()
    registrations = [e for e in events if e['event_kind'] == 'ATTEMPT'
                     and e['payload'].get('attempt_id') == attempt['attempt_id']
                     and e['payload'].get('state') == 'REGISTERED']
    publications = [e for e in events if e['event_kind'] == 'RESTART_ADJUDICATED'
                    and e['payload'] == restart]
    _require(len(registrations) == len(publications) == 1
             and publications[0]['sequence'] < registrations[0]['sequence'],
             'one original source adjudication and registration required')
    forbidden = {'RESERVE', 'SETTLE', 'QUEUED', 'QUEUE_CLAIMED', 'ADMISSION', 'WORKER_STARTED',
                 'WORKER_TREE_STOPPED', 'WORKER_STOP_UNCONFIRMED', 'CHECKPOINT', 'READ_COMPLETED'}
    _require(not any(e['payload'].get('attempt_id') == attempt['attempt_id']
                     and (e['event_kind'] in forbidden or
                          (e['event_kind'] == 'ATTEMPT' and e['payload'].get('state') == 'RUNNING'))
                     for e in events), 'funded or admitted work cannot be replaced as a source proposal')
    return receipt, attempt, events, registrations[0]


def resolve_unfunded_supersession(store, reference, *, source_code_hash):
    with store._read_transaction():
        value = _published(store, reference)
        _require(set(value) == {'schema_version', 'restart_adjudication', 'attempt_id', 'run_id',
            'superseding_source_code_hash', 'decision_basis', 'registration_event_hash', 'terminal_event_hash'}
            and value['schema_version'] == SCHEMA and value['superseding_source_code_hash'] == source_code_hash,
            'unfunded source supersession scope differs')
        receipt, attempt, events, registration = _evidence(store, value['restart_adjudication'])
        _require(attempt['state'] == 'CANCELLED' and attempt['error_code'] == ERROR
                 and value['attempt_id'] == attempt['attempt_id'] and value['run_id'] == attempt['run_id']
                 and value['registration_event_hash'] == registration['hash']
                 and receipt['current_code_hash'] != source_code_hash
                 and any(e['hash'] == value['terminal_event_hash'] and e['event_kind'] == 'ATTEMPT'
                         and e['payload'] == attempt and e['sequence'] > registration['sequence'] for e in events),
                 'unfunded supersession evidence differs')
        basis = value['decision_basis']
        _require(type(basis) is dict and set(basis) == {'task_id', 'session_id', 'reason'}
                 and all(type(x) is str and 0 < len(x) <= 4096 for x in basis.values()),
                 'bounded source-correction decision required')
        return deepcopy(value)


def supersede_unfunded_restart(store, restart, *, source_code_hash, task_id, session_id, reason):
    """An atomic terminal transition makes competing funding fail, with no SETTLE.

    A registered source correction is metadata, not another funded execution.
    Every registration is retained. Any reservation, queue, admission or worker
    evidence forbids this path, even if the work later stopped or settled.
    """
    with store._read_transaction():
        _require(not (store.path / 'recovery-hold.json').exists(), 'shared recovery hold remains')
        receipt, attempt, _, registration = _evidence(store, restart)
        _require(re.fullmatch('[0-9a-f]{64}', str(source_code_hash))
                 and source_code_hash != receipt['current_code_hash'], 'distinct corrected source hash required')
        run = store._manifest('run-' + attempt['run_id'])
        _require(run['arm_id'] == 'pirc28-k1' and run['cell'].get('visibility') == 'synthetic',
                 'unfunded source correction is scoped to the PIRC-28 synthetic restart')
        basis = {'task_id': task_id, 'session_id': session_id, 'reason': reason}
        _require(all(type(x) is str and 0 < len(x) <= 4096 for x in basis.values()),
                 'bounded source-correction decision required')
        terminal = store._transition(attempt['attempt_id'], 'CANCELLED', error_code=ERROR)
        event = next(e for e in store._events() if e['event_kind'] == 'ATTEMPT' and e['payload'] == terminal)
        value = {'schema_version': SCHEMA, 'restart_adjudication': deepcopy(restart),
            'attempt_id': attempt['attempt_id'], 'run_id': attempt['run_id'],
            'superseding_source_code_hash': source_code_hash, 'decision_basis': basis,
            'registration_event_hash': registration['hash'], 'terminal_event_hash': event['hash']}
        object_id = 'pirc28-unfunded-source-' + attempt['attempt_id']
        store.publish(object_id, value)
        reference = {'manifest_id': object_id, 'sha256': digest(value)}
        resolve_unfunded_supersession(store, reference, source_code_hash=source_code_hash)
        return reference
