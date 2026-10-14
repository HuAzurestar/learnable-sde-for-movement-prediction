"""Exact B grant: add five input hours without refund or science expansion.

Read-only metadata policy, NOT launch authority or domain admission. Original
scientific ownership is separate from the latest complete accounting floor.
"""
from collections import Counter
from copy import deepcopy

from . import formal_budget as budget, formal_resource_policy as prior
from . import formal_recovery_policy as recovery, formal_continuation_policy as continuation
from .formal_carryover import VERSION as COST_VERSION
from .protocol_core import canonical, digest, envelope, sha256, unpack

VERSION = 'pirc17-explicit-input7h-total53h-policy-v4'
HUMAN_ID = '2d477572-9d2d-4a4b-a7e2-0774ddc75dcc'
HUMAN_CONTENT = '/goal resume B：额外增加 5H，恢复累计 7H，总预算 53H。'


def _human_source(message):
    if not isinstance(message, dict) or not set(prior.HUMAN_FIELDS) <= message.keys():
        raise ValueError('complete actual B user-source metadata required')
    expected = dict(id=HUMAN_ID, author_type='user', type='message', content=HUMAN_CONTENT,
        created_at='2026-10-04T00:35:45.1627621Z',
        task_id='1fa14b29-0a30-4c40-896e-8a6c21464419',
        session_id='45b5bb5b-dc13-47ca-8f3c-877e1305a12a')
    if any(type(message[key]) is not type(value) or message[key] != value
           for key, value in expected.items()):
        raise ValueError('exact additional5h/input7h/total53h B direction required')
    return {key:message[key] for key in prior.HUMAN_FIELDS}


def _build_policy_and_history(*, previous_binding_reference, cost_reference,
                              bundle_reference, terminal_reference, human_message):
    human = _human_source(human_message)
    from . import formal_resource_predecessor as resource
    previous, old_binding = prior._load_reference(previous_binding_reference)
    if (old_binding.get('schema_version') != resource.CONTINUATION_VERSION
            or unpack(old_binding['resource_policy']).get('schema_version') != continuation.VERSION):
        raise ValueError('B grant requires the closed existing same-cap continuation')
    history = resource._history(previous)
    _, _, _, original, _ = history
    old = unpack(old_binding['resource_policy'])  # Full validation above, same call.
    _, old_cost = prior._load_reference(old['cost_reference'])
    _, cost = prior._load_reference(cost_reference)
    _, bundle = prior._load_reference(bundle_reference)
    terminal_record, terminal = prior._load_reference(terminal_reference)
    runtime, execution = unpack(bundle['runtime']), unpack(bundle['execution'])
    if (bundle.get('schema_version') != 'pirc17-concrete-formal-entrypoint-v5-bundle'
            or runtime.get('schema_version') != 'pirc17-concrete-formal-entrypoint-v5-runtime'
            or canonical(runtime['predecessor']) != canonical(previous)
            or runtime['predecessor_reference'] != previous_binding_reference
            or bundle['protocol'] != original['protocol'] or bundle['matrix'] != original['matrix']
            or runtime['input_paths'] != unpack(original['runtime'])['input_paths']
            or any(row[key] != old[key] for row in (runtime, execution)
                   for key in ('protocol_sha256', 'matrix_sha256'))):
        raise ValueError('latest closed attempt must preserve exact original science')
    contract = budget.contract_for_matrix(bundle['matrix'],
        protocol_sha256=bundle['protocol']['sha256'], execution_sha256=bundle['execution']['sha256'],
        runtime_manifest_sha256=bundle['runtime']['sha256'], approval_sha256=cost['approval_sha256'],
        ledger_directory=runtime['ledger_directory'], resource_policy=old_binding['resource_policy'])
    phases = old['phase_caps_ns']
    if (old['total_cap_ns'] != 172800*budget.NANOSECONDS
            or phases[recovery.INPUT] != 7200*budget.NANOSECONDS
            or phases['method_forecasts'] != 14400*budget.NANOSECONDS
            or phases['terrain_forecasts'] != 111600*budget.NANOSECONDS
            or phases[recovery.EXPORT] != 3600*budget.NANOSECONDS
            or cost['schema_version'] != COST_VERSION or cost['read_only'] is not True
            or cost['authorizes_execution'] is not False or cost['final_eval_reads'] != 0
            or cost['halted_reason'] != 'phase_cap_reached'
            or digest(contract) != cost['ledger_root_sha256']
            or cost['ledger_directory'] != runtime['ledger_directory']
            or cost['phase_caps_ns'] != phases or cost['total_cap_ns'] != old['total_cap_ns']
            or cost['work_inventory_count'] != 11659
            or any(cost[key+'_sha256'] != bundle[name]['sha256'] for key, name in
                   (('protocol','protocol'), ('matrix','matrix'), ('execution','execution'),
                    ('runtime_manifest','runtime')))):
        raise ValueError('complete closed input2h/total48h accounting required')
    tip = cost['ledger_tip']
    budget._fields(tip, ('root_sha256', 'event_count', 'last_event_sha256'))
    count = budget._integer(tip['event_count'], positive=True)
    if (count > 100000 or tip['root_sha256'] != cost['ledger_root_sha256']
            or digest(tip) != cost['head_sha256']):
        raise ValueError('latest complete committed accounting head required')
    sha256(tip['last_event_sha256'])
    files = cost['source_file_sha256']
    if set(files) != {'ledger.json', 'head.json', *(f'events/{i:06d}.json' for i in range(count))}:
        raise ValueError('every latest accounting file must remain pinned')
    for checksum in files.values(): sha256(checksum)
    events = cost['event_types']
    kinds = Counter(events)
    if (len(events) != count or set(kinds) != {'resource_predecessor', 'control_open',
            'control_credit', 'control_close', 'control_terminal_tail'}
            or any(kinds[key] != 1 for key in kinds if key != 'control_credit')
            or events[:2] != ['resource_predecessor', 'control_open']
            or events[-2:] != ['control_close', 'control_terminal_tail']
            or any(key != 'control_credit' for key in events[2:-2])):
        raise ValueError('only closed recovery control; no admission or scientific dispatch')
    successes = {key:value for key,value in old_cost['work_dispositions'].items() if value == 'success'}
    if (cost['work_dispositions'] != successes or len(successes) != old['retained_success_count']
            or digest(sorted(successes)) != old['retained_success_ids_sha256']
            or cost['generated_forecasts_reserved'] != old['retained_generation_reservations']):
        raise ValueError('retain every success, generation debit and unused single retry')
    proof = digest(dict(content_sha256=terminal_record['sha256'], file_sha256=terminal_reference['file_sha256']))
    expected_terminal = dict(schema_version='pirc17-concrete-formal-entrypoint-v5-launch-terminal',
        ledger_root_sha256=cost['ledger_root_sha256'], bundle_sha256=bundle_reference['content_sha256'],
        candidate_complete=False, error_type='ControllerStopped', process_tree_closed=True,
        approval_verified=True, predecessor_floor_transferred_to_ledger=True,
        predecessor_charge_location='ledger', predecessor_charged_ns_by_phase=old['retained_charged_ns_by_phase'])
    if (any(type(terminal[key]) is not type(value) or terminal[key] != value
            for key,value in expected_terminal.items()) or cost['terminal_proof_sha256'] != proof):
        raise ValueError('actual closed terminal and transferred previous charges required')
    for name in recovery.COST_COLUMNS:
        budget._fields(cost[name], phases)
        for phase,value in cost[name].items():
            budget._integer(value)
            if value < old_cost[name][phase] or (phase != recovery.INPUT and value != old_cost[name][phase]):
                raise ValueError('no refund or rephase; only input recovery cost may increase')
    revised = dict(phases, **{recovery.INPUT:25200*budget.NANOSECONDS})
    for phase,cap in revised.items():
        charge = cost['charged_ns_by_phase'][phase]
        if (charge != cost['measured_ns_by_phase'][phase] + cost['conservatively_charged_ns_by_phase'][phase]
                or charge >= cap
                or cost['control_observed_ns_by_phase'][phase] > cost['control_charged_ns_by_phase'][phase]
                or cost['control_charged_ns_by_phase'][phase] > charge):
            raise ValueError('all cost columns must reconcile within exact revised caps')
    total = sum(cost['charged_ns_by_phase'].values())
    if (cost['charged_ns_by_phase'][recovery.INPUT] != phases[recovery.INPUT]
            or total != cost['charged_total_ns']
            or total < budget._integer(terminal['cumulative_charge_lower_bound_ns'])
            or sum(revised.values()) != 190800*budget.NANOSECONDS
            or sum(revised.values())-old['total_cap_ns'] != 18000*budget.NANOSECONDS):
        raise ValueError('actual inputcap reach, full cost tail and exactly five additional hours required')
    result = deepcopy(old)
    result.update(schema_version=VERSION, human_source=human,
        previous_binding_reference=deepcopy(previous_binding_reference),
        previous_resource_policy_sha256=old_binding['resource_policy']['sha256'],
        cost_reference=deepcopy(cost_reference), bundle_reference=deepcopy(bundle_reference),
        terminal_reference=deepcopy(terminal_reference),
        predecessor_root_sha256=cost['ledger_root_sha256'], predecessor_head_sha256=cost['head_sha256'],
        predecessor_execution_sha256=cost['execution_sha256'], predecessor_approval_sha256=cost['approval_sha256'],
        predecessor_runtime_manifest_sha256=cost['runtime_manifest_sha256'],
        predecessor_ledger_directory=cost['ledger_directory'], prior_phase_caps_ns=deepcopy(phases),
        phase_caps_ns=revised, total_cap_ns=sum(revised.values()),
        retained_charged_ns_by_phase=deepcopy(cost['charged_ns_by_phase']), retained_total_charge_ns=total,
        same_approved_caps=False, expansion_of_resource_policy_sha256=old_binding['resource_policy']['sha256'])
    return envelope(result), history


def build_policy(*, previous_binding_reference, cost_reference,
                 bundle_reference, terminal_reference, human_message):
    return _build_policy_and_history(previous_binding_reference=previous_binding_reference,
        cost_reference=cost_reference, bundle_reference=bundle_reference,
        terminal_reference=terminal_reference, human_message=human_message)[0]


def _validate_policy_and_history(record):
    p = unpack(record)
    if p.get('schema_version') != VERSION:
        raise ValueError('distinct explicit B expansion version required')
    expected, history = _build_policy_and_history(**{key:p[key] for key in
        ('previous_binding_reference', 'cost_reference', 'bundle_reference', 'terminal_reference')},
        human_message=p['human_source'])
    if canonical(expected) != canonical(record):
        raise ValueError('expansion changed genuine caps, costs, original science or retry')
    return p, history


def validate_policy(record):
    return _validate_policy_and_history(record)[0]
