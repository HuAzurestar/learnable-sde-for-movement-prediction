"""The single approved cap reallocation, not a launcher or authority maker.

Metadata only: reconcile a fully closed predecessor with the supplied exact
human source. The application must independently retrieve that source, review
the candidate, validate real approval and observe native closure at admission.
No old record is changed and no models/arrays/data readers are invoked here.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path

from . import formal_budget as budget
from .formal_carryover import VERSION as COST_VERSION, MAX_ROOT_BYTES, _read_bound
from .protocol_core import canonical, digest, envelope, sha256, unpack

VERSION = 'pirc17-single-cap-reallocation-policy-v1'
RETRY_WORK_ID = '086bde36ac79bd38934492885cd3e92d7187f32829509a5675d5b318e9322937'
HUMAN_ID = '4faff134-b26e-4a8c-b249-0dacbc7d7a95'
HUMAN_CONTENT = '/goal resume 总预算仍为 48 小时，方法阶段 4 小时、地形阶段 31 小时，仅续跑缺项并补跑那 1 个预算截断项，同时恢复 GOAL'
HUMAN_FIELDS = ('id', 'author_type', 'type', 'content', 'created_at', 'task_id', 'session_id')
PHASE_SECONDS = dict(input_qualification_and_binding=3600, method_training=1800,
    terrain_training=1800, method_forecasts=10800, terrain_forecasts=115200,
    offline_common_scores=7200, mechanism_and_paired_inference=7200,
    runtime_and_forecast_replay=7200, independent_saved_output_reanalysis=10800,
    aggregate_export_and_integrity=7200)


def _load_reference(reference):
    budget._fields(reference, ('path', 'content_sha256', 'file_sha256'))
    path = Path(reference['path'])
    if not path.is_absolute() or path.is_symlink() or str(path.resolve()) != str(path):
        raise ValueError('canonical absolute policy source required')
    record, checksum = _read_bound(path, MAX_ROOT_BYTES)
    if checksum != sha256(reference['file_sha256']):
        raise ValueError('policy source bytes changed')
    return record, unpack(record, expected_sha256=sha256(reference['content_sha256']))


def _human_source(message):
    if not isinstance(message, dict) or not set(HUMAN_FIELDS) <= message.keys():
        raise ValueError('complete actual user-source metadata required')
    if (message['id'] != HUMAN_ID or message['author_type'] != 'user'
            or message['type'] != 'message' or message['content'] != HUMAN_CONTENT
            or message['created_at'] != '2026-10-03T03:01:19.0963983Z'
            or message['task_id'] != '1fa14b29-0a30-4c40-896e-8a6c21464419'
            or message['session_id'] != '45b5bb5b-dc13-47ca-8f3c-877e1305a12a'):
        raise ValueError('exact new user resource-and-single-retry direction required')
    return {key: message[key] for key in HUMAN_FIELDS}


def build_policy(*, cost_reference, bundle_reference, terminal_reference,
                 timeout_reference, human_message):
    """Build a compact non-runnable policy from complete pinned metadata.

    The historical failed generation remains debited; the single approved
    additional attempt costs a NEW token. The unique scientific matrix is
    unchanged. This cannot authorize another failure or repeated success.
    """
    human = _human_source(human_message)
    _, cost = _load_reference(cost_reference)
    _, bundle = _load_reference(bundle_reference)
    terminal_record, terminal = _load_reference(terminal_reference)
    _, observed = _load_reference(timeout_reference)
    protocol, execution, matrix = (bundle[key] for key in ('protocol', 'execution', 'matrix'))
    m, e, runtime = unpack(matrix), unpack(execution), unpack(bundle['runtime'])
    original = budget.contract_for_matrix(matrix, protocol_sha256=protocol['sha256'],
        execution_sha256=execution['sha256'], runtime_manifest_sha256=bundle['runtime']['sha256'],
        approval_sha256=cost['approval_sha256'], ledger_directory=cost['ledger_directory'])
    phases = {key: seconds * budget.NANOSECONDS for key, seconds in PHASE_SECONDS.items()}
    if (cost['schema_version'] != COST_VERSION or cost['read_only'] is not True
            or cost['authorizes_execution'] is not False or cost['final_eval_reads'] != 0
            or cost['phase_caps_ns'] != phases or cost['total_cap_ns'] != 172800 * budget.NANOSECONDS
            or original['phase_caps_ns'] != phases or original['total_cap_ns'] != cost['total_cap_ns']
            or digest(original) != cost['ledger_root_sha256']
            or any(cost[key + '_sha256'] != value['sha256'] for key, value in
                (('protocol', protocol), ('execution', execution), ('matrix', matrix), ('runtime_manifest', bundle['runtime'])))
            or e['protocol_sha256'] != protocol['sha256'] or e['matrix_sha256'] != matrix['sha256']
            or runtime['protocol_sha256'] != protocol['sha256'] or runtime['matrix_sha256'] != matrix['sha256']
            or runtime['ledger_directory'] != cost['ledger_directory']
            or original['max_generated_forecasts'] != 11513 or cost['halted_reason'] != 'phase_cap_reached'):
        raise ValueError('closed original48h cap-halted source scope required')
    terminal_identity = digest(dict(content_sha256=terminal_record['sha256'],
        file_sha256=terminal_reference['file_sha256']))
    if (terminal['ledger_root_sha256'] != cost['ledger_root_sha256']
            or terminal['bundle_sha256'] != bundle_reference['content_sha256']
            or terminal['candidate_complete'] is not False or terminal['error_type'] != 'ControllerStopped'
            or terminal['process_tree_closed'] is not True or terminal['approval_verified'] is not True
            or terminal_identity != cost['terminal_proof_sha256']):
        raise ValueError('fully reconciled original cap terminal required')
    works = {row['work_id']: row for row in m['workloads']}
    states = cost['work_dispositions']
    if (len(works) != 11659 or len(works) != len(m['workloads']) or set(states) - works.keys()
            or dict(Counter(states.values())) != {'success': 6298, 'timeout': 1}
            or states.get(RETRY_WORK_ID) != 'timeout'):
        raise ValueError('all original successes and ONLY the approved timeout required')
    successful = {key for key, state in states.items() if state == 'success'}
    if dict(Counter(works[key]['kind'] for key in successful)) != dict(
            input_qualification_and_population=1, method_fit=16, terrain_fit=10,
            scientific_forecast=6056, same_grid_reference=215):
        raise ValueError('completed original scientific inventory differs')
    retry = works[RETRY_WORK_ID]
    native = unpack(observed['native_observation'])
    closure = native['closure']
    if (retry['kind'] != 'scientific_forecast' or retry['phase'] != 'method_forecasts'
            or retry['max_active_seconds'] != 30 or retry['generated_forecasts'] != 1
            or observed['work_id'] != RETRY_WORK_ID or native['work_id'] != RETRY_WORK_ID
            or observed['ledger_root_sha256'] != cost['ledger_root_sha256']
            or native['ledger_root_sha256'] != cost['ledger_root_sha256']
            or observed['reservation_sha256'] != native['reservation_sha256']
            or observed['stop_reason'] != 'work_deadline' or native['status'] != 'timeout'
            or native['result_sha256'] is not None or native['barrier_sha256'] is not None
            or closure['process_tree_closed'] is not True or closure['accounting']['active_processes'] != 0
            or observed['scientific_manifest_validation'] is not None):
        raise ValueError('exact incomplete budget-truncated timeout and closed tree required')
    allocated = budget._integer(native['deadline_ns']) - budget._integer(native['started_ns'])
    if not 0 < allocated < 30 * budget.NANOSECONDS or native['elapsed_ns'] < allocated:
        raise ValueError('the approved failure must be truncated by phase balance')
    new_phases = dict(phases, method_forecasts=14400 * budget.NANOSECONDS,
        terrain_forecasts=111600 * budget.NANOSECONDS)
    for key in ('charged_ns_by_phase', 'measured_ns_by_phase', 'conservatively_charged_ns_by_phase',
                'control_charged_ns_by_phase', 'control_observed_ns_by_phase'):
        budget._fields(cost[key], phases)
        for value in cost[key].values():
            budget._integer(value)
    for phase in phases:
        charge = cost['charged_ns_by_phase'][phase]
        if (charge != cost['measured_ns_by_phase'][phase] + cost['conservatively_charged_ns_by_phase'][phase]
                or charge >= new_phases[phase]
                or cost['control_observed_ns_by_phase'][phase] > cost['control_charged_ns_by_phase'][phase]
                or cost['control_charged_ns_by_phase'][phase] > charge):
            raise ValueError('all retained cost categories must reconcile within revised caps')
    total = sum(cost['charged_ns_by_phase'].values())
    if (total != cost['charged_total_ns'] or total < terminal['cumulative_charge_lower_bound_ns']
            or cost['charged_ns_by_phase']['method_forecasts'] < phases['method_forecasts']):
        raise ValueError('cap reach and terminal cost cannot be refunded')
    generated = sum(works[key]['generated_forecasts'] for key in states)
    if generated != cost['generated_forecasts_reserved'] or generated != 6272:
        raise ValueError('failed and successful generation reservations must all be retained')
    return envelope(dict(schema_version=VERSION, human_source=human,
        cost_reference=deepcopy(cost_reference), bundle_reference=deepcopy(bundle_reference),
        terminal_reference=deepcopy(terminal_reference), timeout_reference=deepcopy(timeout_reference),
        protocol_sha256=protocol['sha256'], matrix_sha256=matrix['sha256'],
        predecessor_root_sha256=cost['ledger_root_sha256'], predecessor_head_sha256=cost['head_sha256'],
        predecessor_execution_sha256=cost['execution_sha256'], predecessor_approval_sha256=cost['approval_sha256'],
        predecessor_runtime_manifest_sha256=cost['runtime_manifest_sha256'],
        predecessor_ledger_directory=cost['ledger_directory'], original_workloads_sha256=digest(original['workloads']),
        original_phase_caps_ns=phases, phase_caps_ns=new_phases, total_cap_ns=sum(new_phases.values()),
        retry_work_id=RETRY_WORK_ID, retry_old_reservation_sha256=native['reservation_sha256'],
        additional_generation_allowance=1, original_generation_limit=11513, effective_generation_limit=11514,
        retained_generation_reservations=generated, retained_success_count=len(successful),
        retained_success_ids_sha256=digest(sorted(successful)), retained_charged_ns_by_phase=deepcopy(cost['charged_ns_by_phase']),
        retained_total_charge_ns=total, within_successor_attempts_per_item=1,
        authorizes_execution=False, authorizes_successful_reruns=False, old_ledger_modified=False))


def validate_policy(record):
    policy = unpack(record)
    from . import formal_recovery_policy as recovery
    from . import formal_continuation_policy as continuation
    from . import formal_expansion_policy as expansion
    if policy.get('schema_version') == expansion.VERSION:
        return expansion.validate_policy(record)
    if policy.get('schema_version') == continuation.VERSION:
        return continuation.validate_policy(record)
    if policy.get('schema_version') == recovery.VERSION:
        return recovery.validate_policy(record)
    if policy.get('schema_version') != VERSION:
        raise ValueError('explicit resource-only policy version required')
    expected = build_policy(**{key: policy[key] for key in
        ('cost_reference', 'bundle_reference', 'terminal_reference', 'timeout_reference')},
        human_message=policy['human_source'])
    if canonical(expected) != canonical(record):
        raise ValueError('policy changed approved caps,retained costs or single-retry semantics')
    return policy
