"""Same-grant continuation after failed credit supervision; no new budget.

Keep the already approved input2h/export1h and all latest failed recovery
charges. The completed scientific owner stays separate. This metadata-only
policy cannot launch, admit saved science, refund costs or buy another retry.
"""
from collections import Counter
from copy import deepcopy

from . import formal_budget as budget, formal_resource_policy as prior
from . import formal_recovery_policy as recovery
from .formal_carryover import VERSION as COST_VERSION
from .protocol_core import canonical, digest, envelope, sha256, unpack

VERSION = 'pirc17-same-cap-credit-recovery-policy-v3'


def _build_policy_and_history(*, previous_binding_reference, cost_reference,
                              bundle_reference, terminal_reference):
    """Conserve a fully closed, unadmitted attempt under the EXISTING grant."""
    from . import formal_resource_predecessor as resource
    previous, old_binding = prior._load_reference(previous_binding_reference)
    if (old_binding.get('schema_version') != resource.RECOVERY_VERSION
            or unpack(old_binding['resource_policy']).get('schema_version') != recovery.VERSION):
        raise ValueError('same-cap continuation requires the existing input2h recovery binding')
    # History supplies original science; the immediately previous grant's
    # accounting is newer and MUST be the lower bound for every cost column.
    history = resource._history(previous)
    _, _, _, original, _ = history
    # That complete private history already validated the previous policy.
    old = unpack(old_binding['resource_policy'])
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
        raise ValueError('failed attempt must preserve the exact approved scientific scope')
    contract = budget.contract_for_matrix(bundle['matrix'],
        protocol_sha256=bundle['protocol']['sha256'], execution_sha256=bundle['execution']['sha256'],
        runtime_manifest_sha256=bundle['runtime']['sha256'], approval_sha256=cost['approval_sha256'],
        ledger_directory=runtime['ledger_directory'], resource_policy=old_binding['resource_policy'])
    phases = old['phase_caps_ns']
    if (cost['schema_version'] != COST_VERSION or cost['read_only'] is not True
            or cost['authorizes_execution'] is not False or cost['final_eval_reads'] != 0
            or cost['halted_reason'] != 'control_credit_exhausted'
            or digest(contract) != cost['ledger_root_sha256']
            or cost['ledger_directory'] != runtime['ledger_directory']
            or cost['phase_caps_ns'] != phases or cost['total_cap_ns'] != old['total_cap_ns']
            or cost['work_inventory_count'] != 11659
            or any(cost[key+'_sha256'] != bundle[name]['sha256'] for key, name in
                (('protocol','protocol'), ('matrix','matrix'), ('execution','execution'),
                 ('runtime_manifest','runtime')))):
        raise ValueError('fully closed same-cap credit-failure accounting required')
    tip = cost['ledger_tip']
    budget._fields(tip, ('root_sha256', 'event_count', 'last_event_sha256'))
    count = budget._integer(tip['event_count'], positive=True)
    if (count > 100000 or tip['root_sha256'] != cost['ledger_root_sha256']
            or digest(tip) != cost['head_sha256']):
        raise ValueError('latest complete committed accounting head required')
    sha256(tip['last_event_sha256'])
    files = cost['source_file_sha256']
    if set(files) != {'ledger.json', 'head.json', *(f'events/{index:06d}.json' for index in range(count))}:
        raise ValueError('every latest accounting file must remain pinned')
    for checksum in files.values(): sha256(checksum)
    events = cost['event_types']
    kinds = Counter(events)
    required = {'resource_predecessor', 'control_open', 'control_credit', 'halt',
                'control_close', 'control_terminal_tail'}
    suffix = ['halt', 'control_close', 'control_terminal_tail']
    if 'control_late_overrun' in kinds: suffix.append('control_late_overrun')
    if (len(events) != count or set(kinds) != required | set(suffix)
            or any(kinds[key] != 1 for key in kinds if key != 'control_credit')
            or events[:2] != ['resource_predecessor', 'control_open']
            or events[-len(suffix):] != suffix
            or any(key != 'control_credit' for key in events[2:-len(suffix)])):
        raise ValueError('only one failed recovery control; no domain admission or scientific dispatch')
    successes = {key:value for key,value in old_cost['work_dispositions'].items() if value == 'success'}
    if (cost['work_dispositions'] != successes or len(successes) != old['retained_success_count']
            or cost['generated_forecasts_reserved'] != old['retained_generation_reservations']
            or digest(sorted(successes)) != old['retained_success_ids_sha256']):
        raise ValueError('retain every success and all generation debits; approved retry still unused')
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
                raise ValueError('no cost refund; only input recovery cost may increase')
    for phase,cap in phases.items():
        charge = cost['charged_ns_by_phase'][phase]
        if (charge != cost['measured_ns_by_phase'][phase] + cost['conservatively_charged_ns_by_phase'][phase]
                or charge >= cap
                or cost['control_observed_ns_by_phase'][phase] > cost['control_charged_ns_by_phase'][phase]
                or cost['control_charged_ns_by_phase'][phase] > charge):
            raise ValueError('reconciled original caps and positive remaining input credit required')
    total = sum(cost['charged_ns_by_phase'].values())
    if (total != cost['charged_total_ns'] or total >= old['total_cap_ns']
            or total < budget._integer(terminal['cumulative_charge_lower_bound_ns'])
            or cost['charged_ns_by_phase'][recovery.INPUT] <= old_cost['charged_ns_by_phase'][recovery.INPUT]):
        raise ValueError('latest failed recovery and all terminal costs must be retained')
    result = deepcopy(old)
    result.update(schema_version=VERSION, previous_binding_reference=deepcopy(previous_binding_reference),
        previous_resource_policy_sha256=old_binding['resource_policy']['sha256'],
        cost_reference=deepcopy(cost_reference), bundle_reference=deepcopy(bundle_reference),
        terminal_reference=deepcopy(terminal_reference),
        predecessor_root_sha256=cost['ledger_root_sha256'], predecessor_head_sha256=cost['head_sha256'],
        predecessor_execution_sha256=cost['execution_sha256'], predecessor_approval_sha256=cost['approval_sha256'],
        predecessor_runtime_manifest_sha256=cost['runtime_manifest_sha256'],
        predecessor_ledger_directory=cost['ledger_directory'],
        retained_charged_ns_by_phase=deepcopy(cost['charged_ns_by_phase']), retained_total_charge_ns=total,
        same_approved_caps=True, continuation_of_resource_policy_sha256=old_binding['resource_policy']['sha256'])
    return envelope(result), history


def build_policy(*, previous_binding_reference, cost_reference,
                 bundle_reference, terminal_reference):
    return _build_policy_and_history(previous_binding_reference=previous_binding_reference,
        cost_reference=cost_reference, bundle_reference=bundle_reference,
        terminal_reference=terminal_reference)[0]


def _validate_policy_and_history(record):
    """Validate fully; expose its just-checked private history, not a cache."""
    p = unpack(record)
    if p.get('schema_version') != VERSION:
        raise ValueError('distinct same-cap continuation version required')
    expected, history = _build_policy_and_history(**{key:p[key] for key in
        ('previous_binding_reference', 'cost_reference', 'bundle_reference', 'terminal_reference')})
    if canonical(expected) != canonical(record):
        raise ValueError('continuation changed existing authority, caps, costs, science or retry')
    return p, history


def validate_policy(record):
    return _validate_policy_and_history(record)[0]
