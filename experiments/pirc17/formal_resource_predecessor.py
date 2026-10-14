"""Closed cap continuation: retain actual successes and debit the failed call.

Unlike a pre-dispatch import, no generation token is transferred. Only the
specific approved timeout becomes outstanding; its original cost and attempt
remain in pinned provenance, and its successor dispatch buys a NEW token.
This binding cannot launch, approve, or create a scientific settlement.
"""
from collections import Counter
from pathlib import Path

from . import formal_budget as budget, formal_carryover as costs
from . import formal_partial_science as science, formal_resource_policy as policy
from .protocol_core import canonical, digest, envelope, read_json, relative_path, sha256, under, unpack

VERSION = 'pirc17-closed-cap-resource-predecessor-v1'
METADATA_VERSION = 'pirc17-closed-cap-resource-predecessor-v2'
RECOVERY_VERSION = 'pirc17-closed-cap-resource-predecessor-v3'
CONTINUATION_VERSION = 'pirc17-closed-cap-resource-predecessor-v4'


def _recovery_history(binding):
    """Return ORIGINAL science scope; the accounting floor is separate."""
    b = unpack(binding)
    budget._fields(b, ('schema_version', 'ledger_directory', 'resource_policy',
        'science_snapshot', 'science_predecessor', 'read_only', 'authorizes_execution'))
    from . import formal_recovery_policy as recovery, formal_continuation_policy as continuation
    from . import formal_expansion_policy as expansion
    version = unpack(b['resource_policy'])['schema_version']
    validator = {recovery.VERSION: recovery._validate_policy_and_history,
                 continuation.VERSION: continuation._validate_policy_and_history,
                 expansion.VERSION: expansion._validate_policy_and_history}.get(version)
    if validator is None:
        raise ValueError('exact separate recovery accounting and scientific predecessor required')
    # Reuse ONLY the full history produced by this very policy validation.
    # There is no global memo, caller-supplied proof or cold-reader fast path.
    p, history = validator(b['resource_policy'])
    expected_version = {recovery.VERSION: RECOVERY_VERSION, continuation.VERSION: CONTINUATION_VERSION,
                        expansion.VERSION: CONTINUATION_VERSION}.get(p['schema_version'])
    if (expected_version is None or b['schema_version'] != expected_version or b['read_only'] is not True
            or b['authorizes_execution'] is not False
            or b['science_predecessor'] != p['previous_binding_reference']):
        raise ValueError('exact separate recovery accounting and scientific predecessor required')
    previous, _ = policy._load_reference(b['science_predecessor'])
    old_b, c, s, original, old_contract = history
    # Preserve the original post-validation physical predecessor byte check;
    # compare its complete meaning before reusing the privately checked tuple.
    if (canonical(unpack(previous)) != canonical(old_b)
            or b['ledger_directory'] != old_b['ledger_directory']
            or b['science_snapshot'] != old_b['science_snapshot']
            or c['ledger_root_sha256'] != p['scientific_source_root_sha256']
            or c['execution_sha256'] != p['scientific_source_execution_sha256']
            or c['approval_sha256'] != p['scientific_source_approval_sha256']):
        raise ValueError('recovery cannot relabel old models/arrays under the latest failed scope')
    return b, c, s, original, old_contract


def _history(binding):
    b = unpack(binding)
    if b.get('schema_version') in {RECOVERY_VERSION, CONTINUATION_VERSION}:
        return _recovery_history(binding)
    budget._fields(b, ('schema_version', 'ledger_directory', 'resource_policy',
        'science_snapshot', 'read_only', 'authorizes_execution'))
    if (b['schema_version'] not in {VERSION, METADATA_VERSION} or b['read_only'] is not True
            or b['authorizes_execution'] is not False):
        raise ValueError('distinct read-only closed-cap predecessor required')
    p = policy.validate_policy(b['resource_policy'])
    _, c = policy._load_reference(p['cost_reference'])
    _, s = policy._load_reference(b['science_snapshot'])
    _, original = policy._load_reference(p['bundle_reference'])
    metadata_only = b['schema_version'] == METADATA_VERSION
    if metadata_only:
        from .formal_source_metadata import validate_metadata
        validate_metadata(s, resource_policy=b['resource_policy'])
    if (b['ledger_directory'] != p['predecessor_ledger_directory']
            or (not metadata_only and s['schema_version'] != science.VERSION) or s['read_only'] is not True
            or s['authorizes_execution'] is not False or s['cross_execution_admitted'] is not False
            or s['new_fits'] != 0 or s['new_forecasts'] != 0
            or s['ledger_root_sha256'] != c['ledger_root_sha256']
            or s['committed_tip'] != c['ledger_tip']
            or any(s['original_'+key+'_sha256'] != c[key+'_sha256']
                   for key in ('protocol', 'execution', 'matrix', 'approval'))):
        raise ValueError('closed cost and scientific source scopes differ')
    budget._fields(c['ledger_tip'], ('root_sha256', 'event_count', 'last_event_sha256'))
    if (c['ledger_tip']['root_sha256'] != c['ledger_root_sha256']
            or digest(c['ledger_tip']) != c['head_sha256']):
        raise ValueError('scientific prefix must be the full closed head')
    budget._integer(c['ledger_tip']['event_count'], positive=True)
    sha256(c['ledger_tip']['last_event_sha256'])
    works = {w['work_id']: w for w in unpack(original['matrix'])['workloads']}
    completed = {key for key, value in c['work_dispositions'].items() if value == 'success'}
    forecasts = {key for key in completed if works[key]['kind'] in science.KINDS}
    fits = {works[key]['fit_identity'] for key in completed
            if works[key]['kind'] in {'method_fit', 'terrain_fit'}}
    if (set(s['completed_sources']) != completed
            or set(s['forecast_index']) != forecasts or set(s['fit_bindings']) != fits
            or s['successful_work_counts'] != dict(Counter(works[key]['kind'] for key in completed))
            or s['forecast_kind_counts'] != dict(Counter(works[key]['kind'] for key in forecasts))
            or s['forecast_status_counts'] != {('not_replayed' if metadata_only else 'success'): len(forecasts)}):
        raise ValueError('every successful source and its complete typed index required')
    for key, source in s['completed_sources'].items():
        budget._fields(source, ('result_sha256', 'settlement_sha256', 'observation_sha256',
            'controller_elapsed_ns', 'artifact'))
        for name in ('result_sha256', 'settlement_sha256', 'observation_sha256'):
            sha256(source[name])
        budget._integer(source['controller_elapsed_ns'])
        work = works[key]
        artifact = (s['fit_bindings'][work['fit_identity']]
                    if work['kind'] in {'method_fit', 'terrain_fit'}
                    else s['forecast_index'][key] if key in forecasts else None)
        if source['artifact'] != artifact:
            raise ValueError('successful artifact differs from its actual source owner')
        if artifact is not None:
            budget._fields(artifact, ('path', 'file_sha256', 'content_sha256'))
            relative_path(artifact['path'])
            sha256(artifact['file_sha256']); sha256(artifact['content_sha256'])
    old_contract = budget.contract_for_matrix(original['matrix'],
        protocol_sha256=c['protocol_sha256'], execution_sha256=c['execution_sha256'],
        runtime_manifest_sha256=c['runtime_manifest_sha256'], approval_sha256=c['approval_sha256'],
        ledger_directory=c['ledger_directory'])
    return b, c, s, original, old_contract


def build_binding(*, resource_policy, science_reference):
    p = policy.validate_policy(resource_policy)
    from . import formal_recovery_policy as recovery, formal_continuation_policy as continuation
    from . import formal_expansion_policy as expansion
    if p['schema_version'] in {recovery.VERSION, continuation.VERSION, expansion.VERSION}:
        previous, _ = policy._load_reference(p['previous_binding_reference'])
        old_b, _, _, _, _ = _history(previous)
        if science_reference != old_b['science_snapshot']:
            raise ValueError('recovery must reuse the exact earlier scientific inventory')
        version = RECOVERY_VERSION if p['schema_version'] == recovery.VERSION else CONTINUATION_VERSION
        binding = envelope(dict(schema_version=version,
            ledger_directory=old_b['ledger_directory'], resource_policy=resource_policy,
            science_snapshot=science_reference, science_predecessor=p['previous_binding_reference'],
            read_only=True, authorizes_execution=False))
        _history(binding)
        return binding
    from .formal_source_metadata import VERSION as SOURCE_METADATA_VERSION
    _, saved = policy._load_reference(science_reference)
    version = METADATA_VERSION if saved['schema_version'] == SOURCE_METADATA_VERSION else VERSION
    binding = envelope(dict(schema_version=version,
        ledger_directory=p['predecessor_ledger_directory'], resource_policy=resource_policy,
        science_snapshot=science_reference, read_only=True, authorizes_execution=False))
    _history(binding)
    return binding


def _resource_floor_and_history(runtime_manifest, contract):
    """Full floor and its privately validated history from this same call.

    No caller-provided proof or cross-call cache. The original scientific
    owner remains separate from the latest accounting floor, including v3/v4.
    """
    runtime = unpack(runtime_manifest, expected_sha256=contract['runtime_manifest_sha256'])
    binding = runtime['predecessor']
    b, c, s, original, old_contract = _history(binding)
    p = unpack(b['resource_policy'])
    if b['schema_version'] in {RECOVERY_VERSION, CONTINUATION_VERSION}:
        # All callers of _history restore under the original scientific owner.
        # ONLY the resource budget floor adopts the later accounting scope.
        _, c = policy._load_reference(p['cost_reference'])
    if (runtime.get('protocol_sha256') != contract['protocol_sha256']
            or runtime.get('matrix_sha256') != contract['matrix_sha256']
            or runtime.get('ledger_directory') != contract['ledger_directory']
            or contract.get('resource_policy') != b['resource_policy']
            or contract['protocol_sha256'] != c['protocol_sha256']
            or contract['matrix_sha256'] != c['matrix_sha256']
            or contract['ledger_directory'] == c['ledger_directory']
            or contract['execution_sha256'] == c['execution_sha256']
            or contract['approval_sha256'] == c['approval_sha256']
            or contract['phase_caps_ns'] != p['phase_caps_ns']
            or contract['total_cap_ns'] != p['total_cap_ns']
            or contract['workloads'] != old_contract['workloads']
            or contract['max_generated_forecasts'] != p['effective_generation_limit']
            or contract['max_attempts_per_item'] != 1):
        raise ValueError('exact approved resource successor with unchanged science required')
    return binding, c, s, original, old_contract


def resource_floor(runtime_manifest, contract):
    """Pure pinned replay, not a current native observation or admission."""
    return _resource_floor_and_history(runtime_manifest, contract)[:3]


def verify_resource_predecessor(binding):
    """Actual closed source replay; caller MUST meter and supervise this work.

    Native absence/closure is observed now, not inferred from historical JSON.
    Full saved scientific replay remains required before new work: legacy v1
    checks it here, whereas v2 can import only accounting/metadata and its
    reservation gate requires paid bootstrap admission. No old writer/recovery
    API, model fit or prediction is invoked.
    """
    from .formal_partial_costs import _closed_job
    b, c, s, original, old_contract = _history(binding)
    p = unpack(b['resource_policy'])
    _, timeout = policy._load_reference(p['timeout_reference'])
    job = unpack(timeout['native_observation'])['closure']['job_name']
    _closed_job(job)
    if b['schema_version'] in {METADATA_VERSION, RECOVERY_VERSION, CONTINUATION_VERSION}:
        # Metadata-only import cannot dispense science. Full original chains,
        # owned bytes/models/arrays and current guards are independently checked
        # by BOTH restoration sides inside the paid owned input control before
        # bind_partial_imports can remove the reservation gate.
        head = read_json(under(b['ledger_directory'], 'head.json'))
        root = read_json(under(b['ledger_directory'], 'ledger.json'))
        if (head['sha256'] != c['head_sha256'] or unpack(head) != c['ledger_tip']
                or root['sha256'] != c['ledger_root_sha256'] or canonical(unpack(root)) != canonical(old_contract)):
            raise ValueError('current closed metadata head/root differs from the pinned source')
        if b['schema_version'] in {RECOVERY_VERSION, CONTINUATION_VERSION}:
            _, latest = policy._load_reference(p['cost_reference'])
            directory = Path(latest['ledger_directory'])
            # The control is anchored by the byte-pinned, fully replayed latest
            # journal. Observe BOTH old-science and latest-accounting Jobs now.
            index = latest['event_types'].index('control_open')
            event, checksum = costs._read_bound(under(directory, f'events/{index:06d}.json'), costs.MAX_RECORD_BYTES)
            if checksum != latest['source_file_sha256'][f'events/{index:06d}.json']:
                raise ValueError('latest accounting control event bytes changed')
            key = unpack(event)['row']['control_sha256']
            control, _ = costs._read_bound(under(directory, 'controls/'+key+'.json'), costs.MAX_RECORD_BYTES)
            current = unpack(control, expected_sha256=key)
            if current['ledger_root_sha256'] != latest['ledger_root_sha256']:
                raise ValueError('latest accounting control belongs to a different ledger')
            _closed_job(current['worker_job_name'])
            actual = costs.inspect_closed_ledger(directory,
                expected_root_sha256=latest['ledger_root_sha256'], expected_head_sha256=latest['head_sha256'],
                expected_terminal_proof_sha256=latest['terminal_proof_sha256'])
            if canonical(unpack(actual)) != canonical(latest):
                raise ValueError('latest accounting floor differs from its complete pinned journal')
            _closed_job(current['worker_job_name'])
        _closed_job(job)
        _history(binding)
        return binding
    actual = costs.inspect_closed_ledger(b['ledger_directory'],
        expected_root_sha256=c['ledger_root_sha256'], expected_head_sha256=c['head_sha256'],
        expected_terminal_proof_sha256=c['terminal_proof_sha256'])
    if canonical(unpack(actual)) != canonical(c):
        raise ValueError('current closed accounting differs from its pinned source')
    restored = science.inspect_partial_science(b['ledger_directory'], contract=old_contract,
        tip=s['committed_tip'], protocol=original['protocol'], execution=original['execution'],
        matrix=original['matrix'], access_journal=Path(b['ledger_directory'])/'access')
    if canonical(unpack(restored)) != canonical(s):
        raise ValueError('complete saved science differs from the pinned closed source')
    _closed_job(job)
    _history(binding)
    return binding
