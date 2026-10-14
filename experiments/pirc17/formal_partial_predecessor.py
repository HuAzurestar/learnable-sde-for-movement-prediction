"""Explicit partial-source handoff, not a new launch or human authorization.

Keep large source evidence in independently byte/content-pinned immutable
records. The runtime's compact binding fits the existing bounded mailbox and
journal. Replay reads pinned records but never queries a Job, trains, predicts,
or turns a terminal failure into a success. Live admission independently checks
the old source again; the executable entrypoint must still validate authority.
"""
from collections import Counter
from pathlib import Path

from . import formal_budget as budget, formal_partial_costs as costs, formal_partial_science as science
from .formal_carryover import MAX_ROOT_BYTES, _read_bound
from .protocol_core import canonical, digest, envelope, relative_path, sha256, unpack

VERSION = 'pirc17-verified-partial-predecessor-v1'
REFERENCE_FIELDS = ('path','content_sha256','file_sha256')


def reference(path, *, expected_sha256=None):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or str(path.resolve()) != str(path):
        raise ValueError('canonical absolute immutable partial record required')
    record, checksum = _read_bound(path,MAX_ROOT_BYTES)
    unpack(record,expected_sha256=expected_sha256)
    return dict(path=str(path),content_sha256=record['sha256'],file_sha256=checksum)


def _load(ref):
    budget._fields(ref,REFERENCE_FIELDS)
    path = Path(ref['path'])
    if not path.is_absolute() or path.is_symlink() or str(path.resolve()) != str(path):
        raise ValueError('canonical absolute partial source reference required')
    record, checksum = _read_bound(path,MAX_ROOT_BYTES)
    if checksum != sha256(ref['file_sha256']):
        raise ValueError('partial source byte identity changed')
    return record, unpack(record,expected_sha256=sha256(ref['content_sha256']))


def _history(binding):
    b = unpack(binding)
    from . import formal_resource_predecessor as resource
    if b.get('schema_version') in {resource.VERSION, resource.METADATA_VERSION, resource.RECOVERY_VERSION,
                                 resource.CONTINUATION_VERSION}:
        return resource._history(binding)
    budget._fields(b,('schema_version','ledger_directory','cost_snapshot','science_snapshot',
                     'original_bundle','read_only','authorizes_execution'))
    if b['schema_version'] != VERSION or b['read_only'] is not True or b['authorizes_execution'] is not False:
        raise ValueError('distinct read-only partial predecessor required')
    _, c = _load(b['cost_snapshot'])
    _, s = _load(b['science_snapshot'])
    _, original = _load(b['original_bundle'])
    p, e, m = (original[k] for k in ('protocol','execution','matrix'))
    ep, mp = unpack(e), unpack(m)
    old_runtime = unpack(original['runtime'],expected_sha256=c['runtime_manifest_sha256'])
    if (c['schema_version'] != costs.VERSION or s['schema_version'] != science.VERSION
            or c['read_only'] is not True or s['read_only'] is not True
            or c['authorizes_execution'] is not False or s['authorizes_execution'] is not False
            or c['authorizes_generation_token_transfer'] is not False
            or c['process_tree_closed'] is not True or c['pending_dispatched'] is not False
            or c['pending_requested'] is not False or c['old_ledger_modified'] is not False
            or c['accounting_categories_reconciled'] is not True
            or s['cross_execution_admitted'] is not False or s['new_fits'] != 0 or s['new_forecasts'] != 0
            or b['ledger_directory'] != c['ledger_directory']
            or s['ledger_root_sha256'] != c['ledger_root_sha256'] or s['committed_tip'] != c['committed_tip']
            or ep['protocol_sha256'] != p['sha256'] or ep['matrix_sha256'] != m['sha256']
            or mp['protocol_sha256'] != p['sha256']
            or old_runtime['protocol_sha256'] != p['sha256'] or old_runtime['matrix_sha256'] != m['sha256']
            or old_runtime['ledger_directory'] != b['ledger_directory']
            or any(c[k+'_sha256'] != record['sha256'] or s['original_'+k+'_sha256'] != record['sha256']
                   for k,record in (('protocol',p),('execution',e),('matrix',m)))
            or s['original_approval_sha256'] != c['approval_sha256']):
        raise ValueError('partial accounting and scientific original scopes differ')
    works = {w['work_id']:w for w in mp['workloads']}
    if len(works) != len(mp['workloads']):
        raise ValueError('partial source work IDs are not unique')
    statuses = c['work_dispositions']
    completed = {k for k,v in statuses.items() if v == 'success'}
    pending = c['pending_reservation']
    budget._fields(pending,('work_id','phase','reserved_ns','generated_forecasts','reservation_name','reservation_sha256'))
    wid = pending['work_id']
    if (set(statuses)-works.keys() or statuses.get(wid) != 'reserved'
            or set(statuses) != completed | {wid} or wid in s['completed_sources']
            or set(s['completed_sources']) != completed
            or c['work_disposition_counts'] != dict(Counter(statuses.values()))):
        raise ValueError('completed work/pending scope was dropped or changed')
    counts = dict(Counter(works[k]['kind'] for k in completed))
    if (counts != s['successful_work_counts']
            or counts.get('input_qualification_and_population') != 1
            or counts.get('method_fit') != 16 or counts.get('terrain_fit') != 10
            or set(counts)-({'input_qualification_and_population','method_fit','terrain_fit'} | science.KINDS)
            or works[wid]['kind'] not in {'scientific_forecast','same_grid_reference'}
            or pending['phase'] != works[wid]['phase']
            or pending['generated_forecasts'] != works[wid]['generated_forecasts']
            or type(pending['generated_forecasts']) is not int or pending['generated_forecasts'] != 1):
        raise ValueError('all26 fit owners and exact first-undispatched forecast required')
    forecast_ids = {k for k in completed if works[k]['kind'] in science.KINDS}
    fit_ids = {works[k]['fit_identity'] for k in completed if works[k]['kind'] in {'method_fit','terrain_fit'}}
    if (set(s['forecast_index']) != forecast_ids or set(s['fit_bindings']) != fit_ids
            or s['forecast_kind_counts'] != dict(Counter(works[k]['kind'] for k in forecast_ids))
            or s['forecast_status_counts'] != {'success':len(forecast_ids)} and forecast_ids):
        raise ValueError('complete successful partial model/forecast inventory required')
    for key, source in s['completed_sources'].items():
        budget._fields(source,('result_sha256','settlement_sha256','observation_sha256','controller_elapsed_ns','artifact'))
        for name in ('result_sha256','settlement_sha256','observation_sha256'):
            sha256(source[name])
        budget._integer(source['controller_elapsed_ns'])
        work = works[key]
        expected = (s['fit_bindings'][work['fit_identity']] if work['kind'] in {'method_fit','terrain_fit'}
                    else s['forecast_index'][key] if key in forecast_ids else None)
        if source['artifact'] != expected:
            raise ValueError('partial artifact index differs from its actual completed owner')
        if expected is not None:
            budget._fields(expected,('path','file_sha256','content_sha256'))
            relative_path(expected['path'])
            sha256(expected['file_sha256']); sha256(expected['content_sha256'])
    generated = sum(works[k]['generated_forecasts'] for k in completed)+pending['generated_forecasts']
    if (type(c['generation_reservations_retained']) is not int
            or c['generation_reservations_retained'] != generated or c['generated_calls_released'] != 0):
        raise ValueError('partial generation reservations cannot be refunded or expanded')
    original_contract = budget.contract_for_matrix(m,protocol_sha256=p['sha256'],execution_sha256=e['sha256'],
        runtime_manifest_sha256=c['runtime_manifest_sha256'],approval_sha256=c['approval_sha256'],
        ledger_directory=c['ledger_directory'])
    if (digest(original_contract) != c['ledger_root_sha256']
            or c['phase_caps_ns'] != original_contract['phase_caps_ns'] or c['total_cap_ns'] != original_contract['total_cap_ns']
            or generated > original_contract['max_generated_forecasts']):
        raise ValueError('original partial phase/matrix/generation ceiling differs')
    for name in ('charged_ns_by_phase','measured_ns_by_phase','conservatively_charged_ns_by_phase',
                 'control_charged_ns_by_phase','control_observed_ns_by_phase'):
        budget._fields(c[name],original_contract['phase_caps_ns'])
        for amount in c[name].values(): budget._integer(amount)
    for phase,cap in original_contract['phase_caps_ns'].items():
        charge = c['charged_ns_by_phase'][phase]
        if (charge != c['measured_ns_by_phase'][phase]+c['conservatively_charged_ns_by_phase'][phase]
                or charge > cap or c['control_observed_ns_by_phase'][phase] > c['control_charged_ns_by_phase'][phase]
                or c['control_charged_ns_by_phase'][phase] > charge):
            raise ValueError('partial categories/caps must reconcile without credit transfer')
    if (budget._integer(c['charged_total_ns']) != sum(c['charged_ns_by_phase'].values())
            or c['charged_total_ns'] < budget._integer(c['retained_terminal_total_hold_ns'])):
        raise ValueError('partial cumulative/terminal hold was refunded')
    sha256(pending['reservation_sha256'])
    budget._integer(pending['reserved_ns'],positive=True)
    if pending['reserved_ns'] > works[wid]['max_active_seconds']*budget.NANOSECONDS:
        raise ValueError('carried reservation exceeds original work cap')
    return b,c,s,original,original_contract


def build_binding(*,cost_reference,science_reference,bundle_reference):
    _, c = _load(cost_reference)
    record = envelope(dict(schema_version=VERSION,ledger_directory=c['ledger_directory'],
        cost_snapshot=cost_reference,science_snapshot=science_reference,original_bundle=bundle_reference,
        read_only=True,authorizes_execution=False))
    _history(record)
    return record


def partial_floor(runtime_manifest,contract):
    """Replay-only scope/cost validation, with no Job or scientific operation."""
    runtime = unpack(runtime_manifest,expected_sha256=contract['runtime_manifest_sha256'])
    if (runtime.get('protocol_sha256') != contract['protocol_sha256']
            or runtime.get('matrix_sha256') != contract['matrix_sha256']
            or runtime.get('ledger_directory') != contract['ledger_directory']):
        raise ValueError('partial predecessor belongs to a different successor runtime')
    binding = runtime['predecessor']
    b,c,s,original,old_contract = _history(binding)
    if (contract['ledger_directory'] == b['ledger_directory']
            or contract['execution_sha256'] == c['execution_sha256']
            or contract['approval_sha256'] == c['approval_sha256']
            or contract['protocol_sha256'] != c['protocol_sha256']
            or contract['matrix_sha256'] != c['matrix_sha256']
            or contract['phase_caps_ns'] != old_contract['phase_caps_ns']
            or contract['total_cap_ns'] != old_contract['total_cap_ns']
            or contract['workloads'] != old_contract['workloads']
            or contract['max_generated_forecasts'] != old_contract['max_generated_forecasts']):
        raise ValueError('partial successor must retain all original scientific work/caps with distinct authority')
    return binding,c,s


def verify_partial_predecessor(binding):
    """Actual readonly closure and complete scientific source replay at import.

    This grants no approval and creates no ledger. All time must be charged by
    the real entrypoint's original startup/phase controller before launch.
    """
    b,c,s,original,old_contract = _history(binding)
    actual = costs.inspect_predispatch_costs(b['ledger_directory'],
        expected_root_sha256=c['ledger_root_sha256'],expected_head_sha256=c['head_sha256'],
        expected_launch_sha256=c['launch_sha256'],expected_terminal_sha256=c['terminal_sha256'])
    if canonical(unpack(actual)) != canonical(c):
        raise ValueError('live partial costs/closure differ from the sealed handoff')
    restored = science.inspect_partial_science(b['ledger_directory'],contract=old_contract,tip=s['committed_tip'],
        protocol=original['protocol'],execution=original['execution'],matrix=original['matrix'],
        access_journal=Path(b['ledger_directory'])/'access')
    if canonical(unpack(restored)) != canonical(s):
        raise ValueError('complete saved partial science differs from the sealed handoff')
    _history(binding)  # Source-reference bytes must still match after replay.
    return binding
