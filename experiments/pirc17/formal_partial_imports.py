"""Measured, owned saved-science admission, not new scientific settlements.

The worker manifest is independently domain checked by the controller before
its compact reference is journalled. Cold readers check historical native
evidence and original completions; they never remeasure a Job, launch, fit,
predict, acquire raw data, or refund the predecessor's consumed budgets.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from . import formal_budget as budget, formal_controller as control, formal_session as native
from . import formal_partial_predecessor as predecessor
from .protocol_core import canonical, digest, sha256, under, unpack

VERSION = 'pirc17-partial-import-admission-v1'
MANIFEST_VERSION = 'pirc17-partial-worker-import-manifest-v1'


def _owned_reference(reference, root):
    path = Path(reference['path'])
    if not path.is_absolute() or not path.is_relative_to(root):
        raise ValueError('partial import reference must be owned by this bootstrap')
    if under(root, path.relative_to(root).as_posix()) != path:
        raise ValueError('partial import reference escapes its owned bootstrap')
    return predecessor._load(reference)


def _manifest(state, reference, output):
    record, value = _owned_reference(reference, output)
    budget._fields(value, ('schema_version', 'ledger_root_sha256', 'protocol_sha256',
        'execution_sha256', 'matrix_sha256', 'approval_sha256', 'predecessor_reference',
        'import_scope', 'imported_entries', 'new_fits', 'new_forecasts', 'raw_sources_independently_reloaded'))
    expected = dict(schema_version=MANIFEST_VERSION, ledger_root_sha256=digest(state.contract),
        **{k: state.contract[k] for k in ('protocol_sha256', 'execution_sha256', 'matrix_sha256', 'approval_sha256')},
        new_fits=0, new_forecasts=0, raw_sources_independently_reloaded=False)
    if (any(type(value[k]) is not type(v) or value[k] != v for k, v in expected.items())
            or state.predecessor_floor is None or not state.imported_success
            or value['predecessor_reference']['content_sha256'] != state.predecessor_floor['binding_sha256']
            or set(value['imported_entries']) != set(state.imported_success)):
        raise ValueError('partial import manifest changed scope or completed population')
    _, scope = _owned_reference(value['import_scope'], output)
    from .formal_import_scope import VERSION as SCOPE_VERSION
    if (scope['schema_version'] != SCOPE_VERSION or scope['predecessor'] != value['predecessor_reference']
            or scope['predecessor_sha256'] != state.predecessor_floor['binding_sha256']
            or scope['protocol_sha256'] != state.contract['protocol_sha256']
            or scope['matrix_sha256'] != state.contract['matrix_sha256']
            or scope['successor_execution_sha256'] != state.contract['execution_sha256']):
        raise ValueError('partial import manifest and scientific scope bridge differ')
    _owned_reference(scope['successor_context'], output/'input')
    expected_files = {Path(reference['path']), Path(value['import_scope']['path'])}
    for wid, entry in value['imported_entries'].items():
        budget._fields(entry, ('directory', 'manifest', 'artifacts', 'source_completion'))
        directory = output/('input' if state.imported_success[wid]['artifact'] is None else 'imports/'+wid)
        if entry['directory'] != str(directory) or entry['source_completion'] != state.imported_success[wid]:
            raise ValueError('partial import changed owned directory or original completion')
        if not isinstance(entry['artifacts'], dict) or not entry['artifacts']:
            raise ValueError('complete owned partial import inventory required')
        expected_files.update(under(directory, relative) for relative in entry['artifacts'])
    actual_files = set()
    for path in output.rglob('*'):
        checked = under(output, path.relative_to(output).as_posix())
        if checked.is_file():
            actual_files.add(checked)
    if actual_files != expected_files:
        raise ValueError('missing/extra owned bootstrap import file')
    return record, value, scope


def _files(entry, work):
    if under(entry['directory'], 'result.json').exists():
        raise ValueError('partial imports have no new native scientific result.json')
    value = dict(schema_version=control.VERSION+'-artifact-verification', work_id=work['work_id'],
        verified=True, artifacts=entry['artifacts'], details=dict(partial_owned_import=True))
    control.verify_artifacts(value, work, Path(entry['directory']))


def validate_binding(state, row):
    """Pure historical replay of ONE admission under its actual input control.

    No model construction or current OS/clock query here. Domain validation
    already happened inside this same measured controller span. Source/domain
    readers independently verify their own imported bytes when consumed.
    """
    budget._fields(row, ('manifest_reference', 'bootstrap_observation_sha256'))
    if (state.partial_imports is not None or state.pending is not None or state.halted is not None
            or len(state.controls) != 1 or state.active_control is None
            or state.status != dict.fromkeys(state.imported_success, 'success')
            or not state.imported_success):
        raise ValueError('once-only initial idle input import admission required')
    root = Path(state.contract['ledger_directory'])
    key = state.active_control
    ledger = SimpleNamespace(directory=root, root_sha256=digest(state.contract), _state=state)
    start = control._control_record(ledger, key)
    if start['phase'] != state.contract['workloads'][0]['phase']:
        raise ValueError('partial imports must be metered in the original input phase')
    checksum = sha256(row['bootstrap_observation_sha256'])
    proof = unpack(native._read(under(root, 'controls/'+checksum+'.json'), checksum))
    budget._fields(proof, ('schema_version', 'ledger_root_sha256', 'control_sha256',
        'native_observation', 'restoration_verification', 'checked_through_ns'))
    if (proof['schema_version'] != control.VERSION+'-bootstrap-observation'
            or proof['ledger_root_sha256'] != ledger.root_sha256 or proof['control_sha256'] != key):
        raise ValueError('partial import lacks its actual controller bootstrap proof')
    observation = unpack(proof['native_observation'])
    budget._fields(observation, ('schema_version', 'ledger_root_sha256', 'session_sha256', 'control_sha256',
        'request_sha256', 'barrier', 'started_ns', 'ended_ns', 'elapsed_ns', 'deadline_ns', 'worker_pid', 'accounting'))
    directory = Path(start['session_directory'])
    session, contract = native._load_session(directory, observation['session_sha256'])
    if (contract != state.contract or session['job_name'] != start['worker_job_name']
            or session['worker_command'] != start['worker_command']):
        raise ValueError('partial import bootstrap switched its native session')
    ready = unpack(native._read(under(directory, 'ready.json')))
    budget._fields(ready, ('schema_version', 'session_sha256', 'worker_pid'))
    pid = budget._integer(ready['worker_pid'], positive=True)
    expected = dict(schema_version=native.VERSION+'-bootstrap-observation', ledger_root_sha256=ledger.root_sha256,
        session_sha256=observation['session_sha256'], control_sha256=key, worker_pid=pid,
        started_ns=start['started_ns'], deadline_ns=start['phase_deadline_ns'])
    if (ready['schema_version'] != native.VERSION+'-ready' or ready['session_sha256'] != observation['session_sha256']
            or any(type(observation[k]) is not type(v) or observation[k] != v for k, v in expected.items())):
        raise ValueError('partial import native ready/control/observation differs')
    begin, end, checked = (budget._integer(x) for x in
        (observation['started_ns'], observation['ended_ns'], proof['checked_through_ns']))
    span = state.controls[key]
    if (not begin <= end <= checked < observation['deadline_ns'] or observation['elapsed_ns'] != end-begin
            or checked-begin >= span['credit_ns']):
        raise ValueError('partial import validation exceeded original input credit/deadline')
    request = native._read(under(directory, 'bootstrap-request.json'), observation['request_sha256'])
    expected_request = dict(schema_version=native.VERSION+'-bootstrap-request', session_sha256=observation['session_sha256'],
        control_sha256=key, started_ns=begin, deadline_ns=observation['deadline_ns'])
    if canonical(unpack(request)) != canonical(expected_request):
        raise ValueError('partial import lacks the original native bootstrap request')
    barrier = native._read(under(directory, 'bootstrap-barrier.json'), observation['barrier']['sha256'])
    if barrier != observation['barrier']:
        raise ValueError('partial import bootstrap barrier bytes differ')
    b = unpack(barrier)
    budget._fields(b, ('schema_version', 'session_sha256', 'request_sha256', 'control_sha256',
        'worker_pid', 'status', 'value', 'error_type'))
    expected_barrier = dict(schema_version=native.VERSION+'-bootstrap-barrier', session_sha256=observation['session_sha256'],
        request_sha256=request['sha256'], control_sha256=key, worker_pid=pid, status='success', error_type=None)
    if (any(type(b[k]) is not type(v) or b[k] != v for k, v in expected_barrier.items())
            or b['value'] != row['manifest_reference']):
        raise ValueError('partial import is not the worker actual successful bootstrap value')
    accounting = observation['accounting']
    if (not isinstance(accounting, dict) or budget._integer(accounting['active_processes']) < 2
            or budget._integer(accounting['total_processes']) < accounting['active_processes']):
        raise ValueError('partial import historical process accounting unavailable')
    _, manifest, scope = _manifest(state, row['manifest_reference'], under(directory, 'bootstrap'))
    verification = proof['restoration_verification']
    expected_verification = verification_for(row['manifest_reference'], manifest, scope)
    if canonical(verification) != canonical(expected_verification):
        raise ValueError('partial import controller domain verification differs')
    from .formal_entrypoint import MODULE, read_runtime_evidence
    if session['worker_command'][1:5] == ['-u','-m',MODULE,'worker']:
        read_runtime_evidence(directory,contract=state.contract,session_sha256=observation['session_sha256'],
            ledger_root_sha256=ledger.root_sha256,worker_pid=pid,phase=None,first_work_id=None)
    return deepcopy(row)


def verification_for(reference, manifest, scope):
    return dict(schema_version=VERSION, manifest_reference=deepcopy(reference),
        import_scope_sha256=manifest['import_scope']['content_sha256'],
        context_sha256=scope['successor_context']['content_sha256'],
        imported_success_count=len(manifest['imported_entries']))


def verify_manifest(observation, output, tick, *, ledger, protocol, execution, matrix, access_journal,on_verified=None):
    """Real controller callback: original chains, current guard and all domains.

    Every loop remains inside the original input control with owner-thread
    credit callbacks. No worker-produced model objects or indexes are trusted.
    """
    from .formal_restoration_costs import RestorationCosts
    costs = RestorationCosts(Path(output).parent, 'controller')
    try:
        return _verify_manifest(observation, output, tick, ledger=ledger, protocol=protocol,
            execution=execution, matrix=matrix, access_journal=access_journal, on_verified=on_verified, costs=costs)
    except BaseException:
        costs.close('failed')
        raise
    finally:
        costs.close('returned')


def _verify_manifest(observation, output, tick, *, ledger, protocol, execution, matrix, access_journal, on_verified, costs):
    from .formal_closed import ClosedOutputs
    from .formal_input_work import read_input_work, _owned_record
    from .formal_import_scope import ScopeBridge
    from .formal_metadata_batch import metadata_batch
    costs.mark('manifest-and-original-chain')
    tick()
    reference = unpack(observation['barrier'])['value']
    _, manifest, scope = _manifest(ledger._state, reference, output)
    binding, _ = predecessor._load(manifest['predecessor_reference'])
    _, _, science, _, contract = predecessor._history(binding)
    closed = ClosedOutputs(contract['ledger_directory'], contract=contract, tip=science['committed_tip'])
    works = {w['work_id']: w for w in unpack(matrix)['workloads']}
    input_id = next(wid for wid in manifest['imported_entries'] if works[wid]['kind'] == 'input_qualification_and_population')
    entry = manifest['imported_entries'][input_id]
    tick()
    costs.mark('independent-input-restoration')
    args, paths, _ = read_input_work(entry['manifest'], directory=entry['directory'], protocol=protocol,
        execution=execution, matrix=matrix, approval_sha256=ledger._state.contract['approval_sha256'], access_journal=access_journal)
    costs.mark('independent-scope-restoration')
    bridge = ScopeBridge(manifest['import_scope'], protocol=protocol, execution=execution,
        matrix=matrix, input_identity=args['input_identity'])
    with metadata_batch(bridge, closed):
        receipts = {}
        costs.mark('owned-artifacts-and-completions', total=len(manifest['imported_entries']))
        count = 0
        for wid, entry in manifest['imported_entries'].items():
            tick()
            work = works[wid]
            _files(entry, work)
            item = closed.read(work)
            _completion(item, entry['source_completion'])
            if work['kind'] in {'method_fit', 'terrain_fit'}:
                _, record = _owned_record(entry['manifest'], entry['directory'], 'artifact_path', 'artifact_sha256')
                if record != bridge.fit_record(work):
                    raise ValueError('partial import model differs from original scientific fit')
                receipts[work['fit_identity']] = record
            count += 1
            costs.progress(count)
        forecast_count = sum(works[wid]['kind'] not in {'input_qualification_and_population', 'method_fit', 'terrain_fit'}
                             for wid in manifest['imported_entries'])
        costs.mark('independent-forecast-domains', total=forecast_count)
        count = 0
        for wid, entry in manifest['imported_entries'].items():
            work = works[wid]
            if work['kind'] in {'input_qualification_and_population', 'method_fit', 'terrain_fit'}:
                continue
            tick()
            _, record = _owned_record(entry['manifest'], entry['directory'], 'artifact_path', 'artifact_sha256')
            cases = bridge.new_cases[work['origin_mode']]
            case = cases[work['origin_rank']] if work['origin_rank'] < len(cases) else None
            expected = bridge.forecast_record(work, case=case, fit_receipt=receipts[work['fit_identity']])
            if record != expected:
                raise ValueError('partial import prediction differs from original saved science')
            count += 1
            costs.progress(count)
        costs.mark('closing-metadata-batch')
    # Full closing byte checks precede both the admission receipt and any
    # parent reader/predictor handoff. A failed batch cannot release either.
    tick()
    costs.mark('closing-source-and-inventory')
    bridge.check_references()
    _manifest(ledger._state, reference, output)
    if on_verified is not None:
        costs.mark('independent-reader-handoff')
        index = {}
        for wid, entry in manifest['imported_entries'].items():
            if works[wid]['kind'] not in {'input_qualification_and_population','method_fit','terrain_fit'}:
                path = Path(entry['manifest']['artifact_path'])
                relative = path.relative_to(Path(entry['directory'])).as_posix()
                index[wid] = dict(path=path.relative_to(ledger.directory).as_posix(),
                    content_sha256=entry['manifest']['artifact_sha256'],file_sha256=entry['artifacts'][relative]['file_sha256'])
        on_verified(reference=reference,context=args,population_paths=paths,receipts=receipts,index=index,bridge=bridge)
        tick()
    return verification_for(reference, manifest, scope)


def _completion(item, source):
    if item is None or any(item[k] != source[k] for k in
        ('result_sha256', 'settlement_sha256', 'observation_sha256', 'controller_elapsed_ns')):
        raise ValueError('partial import original closed completion differs')


class ImportedOutputs:
    """New owned files PLUS original actual completion, never a new settlement."""
    def __init__(self, closed):
        from .formal_closed import ClosedOutputs
        self.closed = closed
        row = closed._state.partial_imports
        if row is None:
            raise ValueError('saved science lacks committed partial import admission')
        self.reference = deepcopy(row['manifest_reference'])
        proof = unpack(native._read(under(closed.directory,
            'controls/'+row['bootstrap_observation_sha256']+'.json'), row['bootstrap_observation_sha256']))
        observation = unpack(proof['native_observation'])
        start = control._control_record(closed, proof['control_sha256'])
        session, _ = native._load_session(Path(start['session_directory']), observation['session_sha256'])
        self.output = under(session['directory'], 'bootstrap')
        self.record, self.manifest, self.scope = _manifest(closed._state, self.reference, self.output)
        from .formal_pinned_metadata import PinnedMetadata
        self._manifest_reader = PinnedMetadata(self.reference, root=self.output)
        record, manifest = self._manifest_reader.read(self.reference)
        if record != self.record:
            raise ValueError('committed partial import manifest changed during construction')
        self.record, self.manifest = record, manifest
        binding, _ = predecessor._load(self.manifest['predecessor_reference'])
        _, _, science, _, contract = predecessor._history(binding)
        self.original = ClosedOutputs(contract['ledger_directory'], contract=contract, tip=science['committed_tip'])

    def read(self, work):
        return self._read(work, verify_owned_bytes=True)

    def read_metadata(self, work):
        return self._read(work, verify_owned_bytes=False)

    def _read(self, work, *, verify_owned_bytes):
        from .formal_metadata_batch import read_metadata, share_batch
        record, _ = read_metadata(self, self._manifest_reader, self.reference)
        if record != self.record:
            raise ValueError('committed partial import manifest changed')
        entry = self.manifest['imported_entries'][work['work_id']]
        if verify_owned_bytes:
            _files(entry, work)
        share_batch(self, self.original)
        original = (self.original.read(work) if verify_owned_bytes else self.original.read_metadata(work))
        _completion(original, entry['source_completion'])
        return dict(directory=Path(entry['directory']), manifest=deepcopy(entry['manifest']),
            artifacts=deepcopy(entry['artifacts']), **{k: original[k] for k in
            ('settlement_sha256', 'observation_sha256', 'result_sha256', 'controller_elapsed_ns')},
            imported=True, import_admission=deepcopy(self.closed._state.partial_imports),
            original_directory=str(original['directory']),
            **({} if verify_owned_bytes else dict(metadata_only=True, owned_source_bytes_rechecked=False)))
