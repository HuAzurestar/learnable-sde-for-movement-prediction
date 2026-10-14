"""Exact second resource grant: preserve failed recovery costs, not new science.

The scientific source remains the earlier qualified saved-output scope. The
latest closed attempt contributes accounting only: it never admitted imports
or reserved a new forecast. This metadata policy grants no execution authority.
"""
from collections import Counter
from copy import deepcopy

from . import formal_budget as budget, formal_resource_policy as prior
from .formal_carryover import VERSION as COST_VERSION
from .protocol_core import canonical, digest, envelope, sha256, unpack

VERSION = 'pirc17-input-export-reallocation-policy-v2'
HUMAN_ID = '3966648d-18b4-45a0-a9fa-565269209b1c'
HUMAN_CONTENT = '/goal resume 输入/恢复 1→2 小时，导出检查 2→1 小时，总预算 48 小时、方法 4 小时、地形 31 小时不变，并恢复 GOAL'
COST_COLUMNS = ('charged_ns_by_phase', 'measured_ns_by_phase',
    'conservatively_charged_ns_by_phase', 'control_charged_ns_by_phase',
    'control_observed_ns_by_phase')
INPUT = 'input_qualification_and_binding'
EXPORT = 'aggregate_export_and_integrity'


def _human_source(message):
    if not isinstance(message, dict) or not set(prior.HUMAN_FIELDS) <= message.keys():
        raise ValueError('complete actual recovery user-source metadata required')
    expected = dict(id=HUMAN_ID, author_type='user', type='message', content=HUMAN_CONTENT,
        created_at='2026-10-03T08:33:17.0251761Z',
        task_id='1fa14b29-0a30-4c40-896e-8a6c21464419',
        session_id='45b5bb5b-dc13-47ca-8f3c-877e1305a12a')
    if any(type(message[key]) is not type(value) or message[key] != value
           for key, value in expected.items()):
        raise ValueError('exact input2h/export1h user direction required')
    return {key: message[key] for key in prior.HUMAN_FIELDS}


def _build_policy_and_history(*, previous_binding_reference, cost_reference,
                              bundle_reference, terminal_reference, human_message):
    """Bind latest debits and earlier science independently; no cost refund."""
    human = _human_source(human_message)
    from . import formal_resource_predecessor as resource
    previous, _ = prior._load_reference(previous_binding_reference)
    old_binding = unpack(previous)
    if (old_binding.get('schema_version') != resource.METADATA_VERSION
            or unpack(old_binding['resource_policy']).get('schema_version') != prior.VERSION):
        raise ValueError('only the prior metadata recovery source is eligible')
    history = resource._history(previous)
    _, old_cost, _, original, _ = history
    old_policy = old_binding['resource_policy']
    # The complete same private binding/history just validated this policy.
    # No caller-provided PASS, memo or separate cache crosses this call.
    old = unpack(old_policy)
    _, cost = prior._load_reference(cost_reference)
    _, bundle = prior._load_reference(bundle_reference)
    terminal_record, terminal = prior._load_reference(terminal_reference)
    runtime = unpack(bundle['runtime'])
    execution = unpack(bundle['execution'])
    if (bundle.get('schema_version') != 'pirc17-concrete-formal-entrypoint-v5-bundle'
            or runtime.get('schema_version') != 'pirc17-concrete-formal-entrypoint-v5-runtime'
            or canonical(runtime['predecessor']) != canonical(previous)
            or runtime['predecessor_reference'] != previous_binding_reference
            or bundle['protocol'] != original['protocol'] or bundle['matrix'] != original['matrix']
            or runtime['input_paths'] != unpack(original['runtime'])['input_paths']
            or runtime['protocol_sha256'] != old['protocol_sha256']
            or runtime['matrix_sha256'] != old['matrix_sha256']
            or execution['protocol_sha256'] != old['protocol_sha256']
            or execution['matrix_sha256'] != old['matrix_sha256']):
        raise ValueError('latest attempt must bind the exact earlier scientific source')
    contract = budget.contract_for_matrix(bundle['matrix'],
        protocol_sha256=bundle['protocol']['sha256'], execution_sha256=bundle['execution']['sha256'],
        runtime_manifest_sha256=bundle['runtime']['sha256'], approval_sha256=cost['approval_sha256'],
        ledger_directory=runtime['ledger_directory'], resource_policy=old_policy)
    if (cost['schema_version'] != COST_VERSION or cost['read_only'] is not True
            or cost['authorizes_execution'] is not False or cost['final_eval_reads'] != 0
            or cost['halted_reason'] != 'phase_cap_reached'
            or digest(contract) != cost['ledger_root_sha256']
            or cost['ledger_directory'] != runtime['ledger_directory']
            or cost['phase_caps_ns'] != old['phase_caps_ns']
            or cost['total_cap_ns'] != old['total_cap_ns']
            or cost['work_inventory_count'] != 11659
            or any(cost[key+'_sha256'] != bundle[name]['sha256'] for key, name in
                (('protocol','protocol'), ('matrix','matrix'), ('execution','execution'),
                 ('runtime_manifest','runtime')))):
        raise ValueError('complete closed latest accounting contract required')
    tip = cost['ledger_tip']
    budget._fields(tip, ('root_sha256', 'event_count', 'last_event_sha256'))
    count = budget._integer(tip['event_count'], positive=True)
    if count > 100000 or tip['root_sha256'] != cost['ledger_root_sha256'] or digest(tip) != cost['head_sha256']:
        raise ValueError('latest full closed accounting head required')
    sha256(tip['last_event_sha256'])
    files = cost['source_file_sha256']
    if set(files) != {'ledger.json', 'head.json', *(f'events/{index:06d}.json' for index in range(count))}:
        raise ValueError('every latest closed accounting file must be pinned')
    for checksum in files.values():
        sha256(checksum)
    kinds = Counter(cost['event_types'])
    if (len(cost['event_types']) != count or set(kinds) != {'resource_predecessor',
            'control_open', 'control_credit', 'control_close', 'control_terminal_tail'}
            or any(kinds[key] != 1 for key in kinds if key != 'control_credit')
            or cost['event_types'][0] != 'resource_predecessor'
            or cost['event_types'][-1] != 'control_terminal_tail'):
        raise ValueError('latest attempt must contain only failed recovery, no admission or dispatch')
    successes = {key: state for key, state in old_cost['work_dispositions'].items() if state == 'success'}
    if (cost['work_dispositions'] != successes
            or cost['generated_forecasts_reserved'] != old['retained_generation_reservations']
            or len(successes) != old['retained_success_count']):
        raise ValueError('all earlier science and the unused single retry must be retained')
    proof = digest(dict(content_sha256=terminal_record['sha256'], file_sha256=terminal_reference['file_sha256']))
    if (terminal['ledger_root_sha256'] != cost['ledger_root_sha256']
            or terminal['bundle_sha256'] != bundle_reference['content_sha256']
            or terminal['candidate_complete'] is not False or terminal['error_type'] != 'ControllerStopped'
            or terminal['process_tree_closed'] is not True or terminal['approval_verified'] is not True
            or cost['terminal_proof_sha256'] != proof):
        raise ValueError('latest closed recovery terminal must match the full cost tail')
    phases = old['phase_caps_ns']
    for name in COST_COLUMNS:
        budget._fields(cost[name], phases)
        for phase, value in cost[name].items():
            budget._integer(value)
            if value < old_cost[name][phase] or (phase != INPUT and value != old_cost[name][phase]):
                raise ValueError('no prior cost may be refunded and only recovery cost may increase')
    for phase in phases:
        if (cost['charged_ns_by_phase'][phase] != cost['measured_ns_by_phase'][phase]
                + cost['conservatively_charged_ns_by_phase'][phase]
                or cost['control_observed_ns_by_phase'][phase] > cost['control_charged_ns_by_phase'][phase]
                or cost['control_charged_ns_by_phase'][phase] > cost['charged_ns_by_phase'][phase]):
            raise ValueError('latest cumulative cost categories do not reconcile')
    total = sum(cost['charged_ns_by_phase'].values())
    if (cost['charged_ns_by_phase'][INPUT] != phases[INPUT]
            or total != cost['charged_total_ns'] or total < terminal['cumulative_charge_lower_bound_ns']):
        raise ValueError('actual recovery cap reach and all terminal costs must be retained')
    revised = dict(phases, **{INPUT:7200*budget.NANOSECONDS, EXPORT:3600*budget.NANOSECONDS})
    if sum(revised.values()) != old['total_cap_ns']:
        raise ValueError('recovery/export reallocation cannot grow the total budget')
    result = deepcopy(old)
    result.update(schema_version=VERSION, human_source=human,
        previous_binding_reference=deepcopy(previous_binding_reference),
        previous_resource_policy_sha256=old_policy['sha256'],
        cost_reference=deepcopy(cost_reference), bundle_reference=deepcopy(bundle_reference),
        terminal_reference=deepcopy(terminal_reference),
        predecessor_root_sha256=cost['ledger_root_sha256'], predecessor_head_sha256=cost['head_sha256'],
        predecessor_execution_sha256=cost['execution_sha256'], predecessor_approval_sha256=cost['approval_sha256'],
        predecessor_runtime_manifest_sha256=cost['runtime_manifest_sha256'],
        predecessor_ledger_directory=cost['ledger_directory'], prior_phase_caps_ns=deepcopy(phases),
        phase_caps_ns=revised, retained_charged_ns_by_phase=deepcopy(cost['charged_ns_by_phase']),
        retained_total_charge_ns=total, scientific_source_root_sha256=old_cost['ledger_root_sha256'],
        scientific_source_execution_sha256=old_cost['execution_sha256'],
        scientific_source_approval_sha256=old_cost['approval_sha256'])
    return envelope(result), history


def build_policy(*, previous_binding_reference, cost_reference,
                 bundle_reference, terminal_reference, human_message):
    return _build_policy_and_history(previous_binding_reference=previous_binding_reference,
        cost_reference=cost_reference, bundle_reference=bundle_reference,
        terminal_reference=terminal_reference, human_message=human_message)[0]


def _validate_policy_and_history(record):
    """Return only the history independently checked DURING this validation."""
    p = unpack(record)
    if p.get('schema_version') != VERSION:
        raise ValueError('exact input/export reallocation version required')
    expected, history = _build_policy_and_history(**{key:p[key] for key in ('previous_binding_reference',
        'cost_reference', 'bundle_reference', 'terminal_reference')}, human_message=p['human_source'])
    if canonical(expected) != canonical(record):
        raise ValueError('recovery policy changed approved caps, retained costs or original science')
    return p, history


def validate_policy(record):
    return _validate_policy_and_history(record)[0]
