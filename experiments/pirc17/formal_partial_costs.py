"""Pinned, read-only accounting for a pre-dispatch head-publication failure.

Distinct from the fully closed-ledger reader. Old events and the contradictory
terminal receipt are never repaired. Keep every charge, including the unused
reservation and the terminal's conservative total hold. This is NOT authority
to retry, transfer a generation token, or launch a successor.
"""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from . import formal_budget as budget, formal_controller as control, formal_session as native
from .formal_carryover import MAX_EVENTS, MAX_RECORD_BYTES, MAX_ROOT_BYTES, _read_bound
from .protocol_core import envelope, sha256, under, unpack

VERSION = 'pirc17-read-only-predispatch-partial-cost-v1'
ENTRY_VERSION = 'pirc17-concrete-formal-entrypoint-v3'


def inspect_predispatch_costs(directory, *, expected_root_sha256, expected_head_sha256,
                              expected_launch_sha256, expected_terminal_sha256):
    """Only the exact one-event reserve suffix, with actual old-tree closure.

    A terminal's erroneous startup-location flags cannot erase durably committed
    costs. Its conservative total is ALSO retained, never silently refunded.
    The excess is a separately labelled unknown-tail hold in the interrupted
    phase; no measured costs move between phases and no cap is enlarged.
    """
    supplied = Path(directory)
    if not supplied.is_absolute() or supplied.is_symlink():
        raise ValueError('absolute regular predecessor directory required')
    directory = supplied.resolve(strict=True)
    claim = Path(str(directory)+'.launch')
    bindings = {}

    def bound(path, limit=MAX_RECORD_BYTES, expected=None):
        record, checksum = _read_bound(path, limit)
        value = unpack(record, expected_sha256=expected)
        bindings[path] = (checksum, limit)
        return record, value

    root, contract = bound(under(directory,'ledger.json'), MAX_ROOT_BYTES, expected_root_sha256)
    budget.validate_contract(contract)
    if contract['ledger_directory'] != str(directory):
        raise ValueError('partial source directory differs from contract')
    head_record, head = bound(under(directory,'head.json'), expected=expected_head_sha256)
    budget._fields(head, ('root_sha256','event_count','last_event_sha256'))
    count = budget._integer(head['event_count'], positive=True)
    if count >= MAX_EVENTS or head['root_sha256'] != root['sha256']:
        raise ValueError('bounded pinned partial source head required')
    sha256(head['last_event_sha256'])
    launch_record, launch = bound(under(claim,'start.json'), expected=expected_launch_sha256)
    terminal_record, terminal = bound(under(claim,'terminal.json'), expected=expected_terminal_sha256)
    if (launch.get('schema_version') != ENTRY_VERSION+'-launch'
            or terminal.get('schema_version') != ENTRY_VERSION+'-launch-terminal'
            or launch.get('ledger_directory') != str(directory)
            or terminal.get('launch_sha256') != launch_record['sha256']
            or terminal.get('bundle_sha256') != launch.get('bundle_sha256')
            or terminal.get('ledger_root_sha256') != root['sha256']
            or terminal.get('error_type') != 'PermissionError'
            or terminal.get('process_tree_closed') is not True
            or terminal.get('approval_verified') is not True
            or terminal.get('candidate_complete') is not False
            or terminal.get('accounting_complete') is not False):
        raise ValueError('pinned failed head-publication launch/terminal required')
    auth = launch.get('authority')
    if (not isinstance(auth,dict) or auth.get('approval_sha256') != contract['approval_sha256']
            or auth.get('journal_directory') != str(under(directory,'access'))):
        raise ValueError('original launch authority differs from source ledger')
    inventory = [f'{i:06d}.json' for i in range(count+1)]
    events = under(directory,'events')
    if _inventory(events) != inventory:
        raise ValueError('exact one-event contiguous reserve suffix required')
    state = budget._State(contract,root['sha256'])
    last, phase_work = None, {}
    predecessor_phases = dict.fromkeys(contract['phase_caps_ns'],0)
    for index in range(count+1):
        record, event = bound(under(events,f'{index:06d}.json'))
        if event.get('root_sha256') != root['sha256']:
            raise ValueError('partial event root differs')
        if event['type'] == 'settle':
            elapsed = event['row']['elapsed_ns']
            if event['row']['status'] != 'success' or elapsed is None:
                raise ValueError('partial recovery requires honest completed-success prefix')
            key = state.active_control
            phase_work[key] = phase_work.get(key,0)+budget._integer(elapsed)
        state.apply(event,record['sha256'])
        if index == 0 and event['type'] == 'startup_predecessor':
            predecessor_phases = dict(state.charged)
        if state.count == count and state.tip != head['last_event_sha256']:
            raise ValueError('committed head does not anchor its source prefix')
        last = event
    durable_tip = dict(root_sha256=root['sha256'],event_count=state.count,last_event_sha256=state.tip)
    if (last['type'] != 'reserve' or state.pending is None
            or terminal.get('ledger_tip_before_terminal') != durable_tip
            or state.active_control is None or state.terminal_control is not None
            or state.halted is not None):
        raise ValueError('terminal must pin the sole undispatched reserve suffix')
    if (launch.get('predecessor_charged_ns_by_phase') != predecessor_phases
            or terminal.get('predecessor_charged_ns_by_phase') != predecessor_phases):
        raise ValueError('predecessor floor was dropped or counted in a different phase')
    pending = state.pending
    dispatch = under(directory,'dispatches/'+pending['reservation_sha256']+'.json')
    if dispatch.exists() or dispatch.is_symlink():
        raise ValueError('pending work already has a dispatch claim; not pre-dispatch recovery')
    requests = _requests(directory)
    for path in requests:
        _, value = bound(path)
        if value.get('reservation_sha256') == pending['reservation_sha256']:
            raise ValueError('pending work reached a worker request; no transferable pre-dispatch token')
    capsule = SimpleNamespace(directory=directory,root_sha256=root['sha256'],_state=state)
    starts, jobs = {}, {}
    for key, span in state.controls.items():
        path = under(directory,'controls/'+key+'.json')
        bound(path,expected=key)
        start = control._control_record(capsule,key)
        starts[key] = start
        session_dir = Path(start['session_directory'])
        session_record, session = bound(under(session_dir,'session.json'))
        native._load_session(session_dir,session_record['sha256'])
        if (session['job_name'] != start['worker_job_name']
                or session['worker_command'] != start['worker_command']):
            raise ValueError('control and original session job/command differ')
        if key != state.active_control and (not span['closed'] or span.get('reason') != 'phase_complete'):
            raise ValueError('earlier control span did not close normally')
        if span['observed_ns'] > span['credit_ns']:
            raise ValueError('recorded control overrun cannot be hidden')
        jobs[session['job_name']] = _closed_job(session['job_name'])
    active = state.controls[state.active_control]
    if active['closed'] or active['phase'] != pending['phase']:
        raise ValueError('pending work is outside its actual open phase')
    end = budget._integer(launch['started_ns'])+budget._integer(terminal['elapsed_through_receipt_ns'])
    start = budget._integer(starts[state.active_control]['started_ns'])
    observed_work = phase_work.get(state.active_control,0)
    overhead = end-start-observed_work
    if overhead < active['observed_ns']:
        raise ValueError('terminal clock loses recorded active work/control time')
    charged, measured, conservative = dict(state.charged), dict(state.measured), dict(state.conservative)
    # A reserve contributes to charged, but not yet measured/conservative in
    # _State. Classify it honestly as unknown, without writing a settlement.
    phase = pending['phase']
    conservative[phase] += pending['reserved_ns']
    control_extra = max(0,overhead-active['credit_ns'])
    charged[phase] += control_extra
    measured[phase] += control_extra
    old_claim_hold = budget._integer(terminal['cumulative_charge_lower_bound_ns'])
    extra_hold = max(0,old_claim_hold-sum(charged.values()))
    charged[phase] += extra_hold
    conservative[phase] += extra_hold
    control_charged, control_observed = dict(state.control_charged), dict(state.control_observed)
    control_charged[phase] += control_extra
    control_observed[phase] += overhead-active['observed_ns']
    if any(charged[k] != measured[k]+conservative[k] for k in charged):
        raise ValueError('partial charge categories do not reconcile')
    if any(charged[k] > contract['phase_caps_ns'][k] for k in charged):
        raise TimeoutError('retained partial charges exhaust original phase capacity')
    if sum(charged.values()) > contract['total_cap_ns']:
        raise TimeoutError('retained partial charges exhaust original total capacity')
    # No mutation, invented settlement, reservation cancellation or refund.
    for path,(checksum,limit) in bindings.items():
        _, actual = _read_bound(path,limit)
        if actual != checksum:
            raise ValueError('partial source bytes changed during inspection')
    if _inventory(events) != inventory or _requests(directory) != requests or dispatch.exists():
        raise ValueError('source inventory/dispatch changed during inspection')
    for name in jobs:
        _closed_job(name)
    return envelope(dict(schema_version=VERSION,ledger_directory=str(directory),
        ledger_root_sha256=root['sha256'],head_sha256=head_record['sha256'],committed_tip=head,
        durable_tip=durable_tip,launch_sha256=launch_record['sha256'],terminal_sha256=terminal_record['sha256'],
        source_file_sha256={str(p):checksum for p,(checksum,_) in bindings.items()},
        protocol_sha256=contract['protocol_sha256'],execution_sha256=contract['execution_sha256'],
        matrix_sha256=contract['matrix_sha256'],runtime_manifest_sha256=contract['runtime_manifest_sha256'],
        approval_sha256=contract['approval_sha256'],phase_caps_ns=contract['phase_caps_ns'],
        total_cap_ns=contract['total_cap_ns'],charged_ns_by_phase=charged,measured_ns_by_phase=measured,
        conservatively_charged_ns_by_phase=conservative,charged_total_ns=sum(charged.values()),
        control_charged_ns_by_phase=control_charged,control_observed_ns_by_phase=control_observed,
        original_durable_charged_ns_by_phase=state.charged,
        original_control_charged_ns_by_phase=state.control_charged,
        original_control_observed_ns_by_phase=state.control_observed,
        predecessor_floor=state.predecessor_floor,predecessor_phase_charges=predecessor_phases,
        startup_floor_durably_transferred=state.predecessor_floor is not None,
        startup_control_durably_committed=bool(state.controls),
        terminal_transfer_flags=dict(predecessor=terminal['predecessor_floor_transferred_to_ledger'],
                                     startup=terminal['startup_transferred_to_ledger']),
        active_phase_overhead_through_receipt_ns=overhead,additional_observed_control_charge_ns=control_extra,
        retained_terminal_total_hold_ns=old_claim_hold,unknown_tail_hold_ns=extra_hold,unknown_tail_phase=phase,
        pending_reservation=pending,pending_dispatched=False,pending_requested=False,
        generation_reservations_retained=state.generated,generated_calls_released=0,
        work_dispositions=state.status,work_disposition_counts=dict(Counter(state.status.values())),
        native_job_observations=jobs,process_tree_closed=True,read_only=True,
        accounting_categories_reconciled=True,unknown_tail_is_measured=False,
        authorizes_execution=False,authorizes_generation_token_transfer=False,
        old_ledger_modified=False,scope='retained partial cost and pre-dispatch evidence;not successor admission'))


def _inventory(directory):
    names = []
    for path in directory.iterdir():
        if len(names) >= MAX_EVENTS:
            raise ValueError('partial event inventory exceeds bound')
        names.append(path.name)
    return sorted(names)


def _requests(directory):
    result = []
    for path in directory.glob('session-*/requests/*.json'):
        if len(result) >= MAX_EVENTS:
            raise ValueError('partial request inventory exceeds bound')
        # Force canonical, inside-ledger paths (including intermediate links).
        result.append(under(directory,path.relative_to(directory).as_posix()))
    return sorted(result)


def _closed_job(name):
    observation = native._query_job(name)
    if observation['exists'] and observation['accounting']['active_processes'] != 0:
        raise native.UnclosedTree('old native tree is still active; no partial continuation')
    return observation
