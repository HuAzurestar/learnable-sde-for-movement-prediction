"""Read-only domain replay of an interrupted, pre-score scientific prefix.

This is a source inventory for recovery, NOT execution authority, a corrected
budget, a new settlement, or cross-execution admission. Every saved model and
forecast is validated under its ORIGINAL scope and actual completion chain.
Unattempted work remains unattempted. No source acquisition or generation.
"""
from collections import Counter

from . import formal_budget as budget
from .formal_closed import ClosedOutputs
from .formal_fit_records import restore_registered_fit
from .formal_forecast_records import KINDS
from .formal_input_work import _owned_record, input_verification
from .formal_origins import build_origin_cases
from .formal_saved import SavedForecasts
from .protocol_core import canonical, envelope, file_hash, under, unpack

VERSION = 'pirc17-original-scope-partial-science-v1'


def inspect_partial_science(directory, *, contract, tip, protocol, execution,
                            matrix, access_journal):
    """Replay all successful pre-score work, never a convenient subset.

    The independently pinned committed tip excludes unanchored events. Those
    events and terminal costs MUST be handled separately before continuation.
    This function cannot clear pending work or import old results into a new
    execution. It does not revalidate current source against an old approval.
    """
    expected = budget.contract_for_matrix(matrix, protocol_sha256=protocol['sha256'],
        execution_sha256=execution['sha256'], runtime_manifest_sha256=contract['runtime_manifest_sha256'],
        approval_sha256=contract['approval_sha256'], ledger_directory=directory)
    if canonical(expected) != canonical(contract):
        raise ValueError('partial science differs from original complete contract')
    closed = ClosedOutputs(directory, contract=contract, tip=tip)
    head_path = under(closed.directory, 'head.json')
    head_bytes = file_hash(head_path)
    works = unpack(matrix)['workloads']
    inputs = [w for w in works if w['kind'] == 'input_qualification_and_population']
    fits = [w for w in works if w['kind'] in {'method_fit', 'terrain_fit'}]
    if len(inputs) != 1 or len(fits) != 26:
        raise ValueError('original single input and26 fit owners required')
    allowed = {'input_qualification_and_population', 'method_fit', 'terrain_fit'} | KINDS
    successful = [w for w in works if closed.disposition(w['work_id']) == 'success']
    if any(w['kind'] not in allowed for w in successful):
        raise ValueError('pre-score partial recovery cannot silently omit completed analysis')
    if closed.disposition(inputs[0]['work_id']) != 'success':
        raise ValueError('successfully settled guarded input required')
    item = closed.read(inputs[0])
    validation, context, _ = input_verification(inputs[0], item['manifest'], item['directory'],
        protocol=protocol, execution=execution, matrix=matrix,
        approval_sha256=contract['approval_sha256'], access_journal=access_journal)
    _, context_record = _owned_record(item['manifest'], item['directory'], 'context_path', 'context_sha256')
    import_bridge = None
    if item.get('imported'):
        from .formal_import_scope import ScopeBridge
        import_bridge = ScopeBridge(closed.imports.manifest['import_scope'],
            **{k: context[k] for k in ('protocol', 'execution', 'matrix', 'input_identity')})
    sources = {inputs[0]['work_id']: _source(item, artifact=None)}
    receipts, fit_bindings = {}, {}
    for work in fits:
        if closed.disposition(work['work_id']) != 'success':
            continue
        item = closed.read(work)
        path, record = _owned_record(item['manifest'], item['directory'], 'artifact_path', 'artifact_sha256')
        restore_registered_fit(record, work=work, protocol=protocol, execution=execution,
            matrix=matrix, input_identity=context['input_identity'], import_bridge=import_bridge)
        receipts[work['fit_identity']] = record
        binding = _binding(path, record, closed.directory)
        fit_bindings[work['fit_identity']] = binding
        sources[work['work_id']] = _source(item, artifact=binding)
    forecasts = [w for w in successful if w['kind'] in KINDS]
    if forecasts and len(receipts) != len(fits):
        raise ValueError('completed predictions require all26 settled fit owners')
    cases = build_origin_cases(context['positions'].prefixes, context['population'], prior=context['prior'])
    saved = SavedForecasts(**{k: context[k] for k in
        ('protocol', 'execution', 'matrix', 'input_identity', 'population', 'map_catalog')},
        cases=cases, fit_receipts=receipts, root=closed.directory, index={}, import_bridge=import_bridge)
    statuses, forecast_kinds = Counter(), Counter()
    for work in forecasts:
        item = closed.read(work)
        path, record = _owned_record(item['manifest'], item['directory'], 'artifact_path', 'artifact_sha256')
        binding = _binding(path, record, closed.directory)
        saved.admit(work, binding)
        prediction = saved.read(work)  # Actual array/domain/stream validation.
        statuses[prediction.status] += 1
        forecast_kinds[work['kind']] += 1
        sources[work['work_id']] = _source(item, artifact=binding)
    if set(sources) != {w['work_id'] for w in successful}:
        raise ValueError('partial science lost a successful settled work item')
    if file_hash(head_path) != head_bytes:
        raise ValueError('partial source head changed during restoration')
    return envelope(dict(schema_version=VERSION, ledger_root_sha256=closed.root_sha256,
        committed_tip=tip, original_protocol_sha256=protocol['sha256'],
        original_execution_sha256=execution['sha256'], original_matrix_sha256=matrix['sha256'],
        original_approval_sha256=contract['approval_sha256'], context_sha256=context_record['sha256'],
        input_validation=validation['details'], fit_bindings=fit_bindings,
        forecast_index=saved.index, completed_sources=sources,
        successful_work_counts=dict(Counter(w['kind'] for w in successful)),
        forecast_kind_counts=dict(forecast_kinds), forecast_status_counts=dict(statuses),
        work_disposition_counts=dict(Counter(closed.disposition(w['work_id']) for w in works)),
        saved_context_sha256=saved.identity['sha256'], new_fits=0, new_forecasts=0,
        raw_sources_independently_reloaded=False, historical_process_observations_remeasured=False,
        read_only=True, cost_allocation_complete=False, cross_execution_admitted=False,
        authorizes_execution=False, scientific_claim_authorized=False))


def _binding(path, record, root):
    return dict(path=path.relative_to(root).as_posix(), file_sha256=file_hash(path),
                content_sha256=record['sha256'])


def _source(item, *, artifact):
    return dict(result_sha256=item['result_sha256'], settlement_sha256=item['settlement_sha256'],
        observation_sha256=item['observation_sha256'], controller_elapsed_ns=item['controller_elapsed_ns'],
        artifact=artifact)
