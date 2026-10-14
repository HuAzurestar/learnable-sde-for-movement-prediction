"""Read-only pinned cost snapshots, not replacement-run authorization.

Never open Ledger's writer/recovery API: it can advance head.json. A predecessor
must have a fully anchored terminal tail. This reader preserves all accounting
categories and refuses incomplete/changed records instead of repairing them.
It does NOT prove native process closure, startup-only eligibility, new human
approval, or enforce a debit in a replacement ledger. Those are separate layers.
"""
import hashlib
from pathlib import Path

from . import formal_budget as budget
from .protocol_core import decode, envelope, sha256, under, unpack

VERSION = 'pirc17-read-only-closed-budget-v1'
MAX_EVENTS = 100_000
MAX_ROOT_BYTES = 32*1024*1024
MAX_RECORD_BYTES = 64*1024


def _read_bound(path, max_bytes):
    # Meaning and byte checksum must come from the SAME bounded read. Reading
    # JSON and only then hashing a possibly replaced file could bind old meaning
    # to new bytes even when a later hash check is stable.
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError('bounded regular JSON file required')
    with path.open('rb') as stream:
        raw = stream.read(max_bytes+1)
    if len(raw) > max_bytes:
        raise ValueError('JSON file grew beyond its bound')
    return decode(raw), hashlib.sha256(raw).hexdigest()


def inspect_closed_ledger(directory, *, expected_root_sha256, expected_head_sha256,
                          expected_terminal_proof_sha256):
    """Replay exact pinned bytes, without locks, mutation, recovery or admission.

    The caller must independently pin the identities. A complete budget journal
    is not a physical process-tree certificate or authority for another attempt.
    Control charges overlap measured/conservative totals; do not sum all columns.
    """
    supplied = Path(directory)
    if not supplied.is_absolute():
        raise ValueError('canonical absolute predecessor ledger required')
    directory = supplied.resolve(strict=True)
    root_path, head_path = under(directory, 'ledger.json'), under(directory, 'head.json')
    root, root_file_sha = _read_bound(root_path, MAX_ROOT_BYTES)
    contract = budget.validate_contract(unpack(root, expected_sha256=expected_root_sha256))
    if contract['ledger_directory'] != str(directory):
        raise ValueError('predecessor ledger location differs from its contract')
    head_record, head_file_sha = _read_bound(head_path, MAX_RECORD_BYTES)
    head = unpack(head_record, expected_sha256=expected_head_sha256)
    budget._fields(head, ('root_sha256', 'event_count', 'last_event_sha256'))
    count = budget._integer(head['event_count'])
    if head['root_sha256'] != root['sha256'] or not 0 < count <= MAX_EVENTS:
        raise ValueError('bounded nonempty pinned predecessor head required')
    sha256(head['last_event_sha256'])
    terminal_proof = sha256(expected_terminal_proof_sha256)
    events_path = under(directory, 'events')
    files = []
    for path in events_path.iterdir():
        if len(files) >= MAX_EVENTS:
            raise ValueError('predecessor event inventory exceeds its bound')
        files.append(path.name)
    # No unanchored suffix or pending temp is silently recovered/adopted.
    if sorted(files) != [f'{index:06d}.json' for index in range(count)]:
        raise ValueError('predecessor events missing/extra/unanchored/partial')
    state = budget._State(contract, root['sha256'])
    byte_bindings = {'ledger.json': root_file_sha, 'head.json': head_file_sha}
    event_types = []
    for index in range(count):
        relative = f'events/{index:06d}.json'
        path = under(directory, relative)
        record, file_sha = _read_bound(path, MAX_RECORD_BYTES)
        payload = unpack(record)
        if payload.get('root_sha256') != root['sha256']:
            raise ValueError('predecessor event root changed')
        state.apply(payload, record['sha256'])
        event_types.append(payload['type'])
        byte_bindings[relative] = file_sha
    if state.count != count or state.tip != head['last_event_sha256']:
        raise ValueError('predecessor head does not exactly anchor its full chain')
    if state.pending is not None or state.active_control is not None or state.terminal_control is None:
        raise ValueError('predecessor pending/control/terminal accounting is incomplete')
    terminal = state.controls[state.terminal_control]
    if (not terminal['closed'] or terminal.get('reason') == 'recovered_unknown'
            or terminal.get('terminal_evidence_sha256') != terminal_proof
            or any(not span['closed'] or span.get('reason') == 'recovered_unknown'
                   for span in state.controls.values())):
        raise ValueError('pinned measured terminal-tail evidence required')
    if any(state.charged[phase] != state.measured[phase]+state.conservative[phase]
           for phase in state.charged):
        raise ValueError('predecessor cost categories do not reconcile')
    # Reject observed changes; do not claim an OS snapshot against malicious
    # concurrent replacement. No writer lease or process-liveness claim is made.
    for relative, checksum in byte_bindings.items():
        _, actual_checksum = _read_bound(under(directory, relative),
            MAX_ROOT_BYTES if relative == 'ledger.json' else MAX_RECORD_BYTES)
        if actual_checksum != checksum:
            raise ValueError('predecessor bytes changed during read-only inspection')
    if sorted(p.name for p in events_path.iterdir()) != sorted(files):
        raise ValueError('predecessor event inventory changed during inspection')
    return envelope(dict(schema_version=VERSION, ledger_directory=str(directory),
        ledger_root_sha256=root['sha256'], head_sha256=head_record['sha256'],
        ledger_tip=head, source_file_sha256=byte_bindings, event_types=event_types,
        protocol_sha256=contract['protocol_sha256'], execution_sha256=contract['execution_sha256'],
        matrix_sha256=contract['matrix_sha256'], runtime_manifest_sha256=contract['runtime_manifest_sha256'],
        approval_sha256=contract['approval_sha256'], phase_caps_ns=contract['phase_caps_ns'],
        total_cap_ns=contract['total_cap_ns'], charged_ns_by_phase=dict(state.charged),
        measured_ns_by_phase=dict(state.measured), conservatively_charged_ns_by_phase=dict(state.conservative),
        control_charged_ns_by_phase=dict(state.control_charged),
        control_observed_ns_by_phase=dict(state.control_observed),
        charged_total_ns=sum(state.charged.values()), generated_forecasts_reserved=state.generated,
        work_dispositions=dict(state.status), work_inventory_count=len(state.work),
        halted_reason=state.halted, terminal_proof_sha256=terminal_proof,
        read_only=True, final_eval_reads=0, authorizes_execution=False,
        scope='pinned recorded cost only;not process closure,startup eligibility or replacement authorization'))
