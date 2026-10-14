"""Pinned closed output metadata; explicitly NOT saved-science qualification.

No model, context values, NPZ array, map backend or scientific kernel is opened.
The original completion chain and its recorded artifact hashes are indexed;
current owned bytes and typed domains MUST be replayed in paid bootstrap.
"""
from collections import Counter
from pathlib import Path

from . import formal_budget as budget, formal_resource_policy as policy
from .formal_closed import ClosedOutputs
from .formal_forecast_records import KINDS
from .protocol_core import envelope, read_json, sha256, under, unpack

VERSION = 'pirc17-original-scope-source-metadata-v1'
FLAGS = dict(read_only=True, authorizes_execution=False, cross_execution_admitted=False,
    new_fits=0, new_forecasts=0, metadata_only=True, owned_source_bytes_rechecked=False,
    forecast_domains_verified=False, new_array_reads=0, requires_metered_domain_admission=True)
FIELDS = ('schema_version', 'resource_policy_sha256', 'ledger_root_sha256', 'committed_tip',
    'original_protocol_sha256', 'original_execution_sha256', 'original_matrix_sha256',
    'original_approval_sha256', 'context_sha256', 'completed_sources', 'fit_bindings',
    'forecast_index', 'successful_work_counts', 'forecast_kind_counts', 'forecast_status_counts', *FLAGS)


def validate_metadata(value, *, resource_policy):
    budget._fields(value, FIELDS)
    if (value['schema_version'] != VERSION or value['resource_policy_sha256'] != resource_policy['sha256']
            or any(type(value[k]) is not type(v) or value[k] != v for k, v in FLAGS.items())
            or value['forecast_status_counts'] != {'not_replayed': len(value['forecast_index'])}):
        raise ValueError('explicit unqualified original-source metadata required')
    sha256(value['context_sha256'])
    return value


def _artifact(item, *, root):
    manifest, directory = item['manifest'], Path(item['directory'])
    path = Path(manifest['artifact_path'])
    if (not path.is_absolute() or not path.is_relative_to(directory) or not path.is_relative_to(root)
            or under(directory, path.relative_to(directory).as_posix()) != path):
        raise ValueError('recorded source artifact escapes its actual closed owner')
    recorded = item['artifacts'][path.relative_to(directory).as_posix()]
    return dict(path=path.relative_to(root).as_posix(),
        file_sha256=sha256(recorded['file_sha256']), content_sha256=sha256(manifest['artifact_sha256']))


def inspect_source_metadata(resource_policy, *, progress=None):
    """Original closed-chain metadata only; no current scientific PASS flag.

    The complete original matrix and committed head remain fixed. Metadata
    reader preserves every original completion and recorded artifact binding;
    it deliberately does NOT call ClosedOutputs.read or artifact/domain loaders.
    """
    p = policy.validate_policy(resource_policy)
    if progress is not None and not callable(progress):
        raise ValueError('metadata progress callback must be callable')
    from .formal_partial_costs import _closed_job
    _, timeout = policy._load_reference(p['timeout_reference'])
    job = unpack(timeout['native_observation'])['closure']['job_name']
    _closed_job(job)
    _, cost = policy._load_reference(p['cost_reference'])
    _, original = policy._load_reference(p['bundle_reference'])
    protocol, execution, matrix = (original[k] for k in ('protocol', 'execution', 'matrix'))
    root = Path(cost['ledger_directory'])
    contract = budget.contract_for_matrix(matrix, protocol_sha256=protocol['sha256'],
        execution_sha256=execution['sha256'], runtime_manifest_sha256=cost['runtime_manifest_sha256'],
        approval_sha256=cost['approval_sha256'], ledger_directory=root)
    head = read_json(under(root, 'head.json'))
    if head['sha256'] != cost['head_sha256'] or unpack(head) != cost['ledger_tip']:
        raise ValueError('source metadata must name the full pinned closed head')
    closed = ClosedOutputs(root, contract=contract, tip=cost['ledger_tip'])
    sources, fits, forecasts, counts, kinds = {}, {}, {}, Counter(), Counter()
    context = None
    works = unpack(matrix)['workloads']
    for work in works:
        wid, kind = work['work_id'], work['kind']
        if closed.disposition(wid) != cost['work_dispositions'].get(wid, 'unattempted'):
            raise ValueError('closed metadata changed an original work disposition')
        if closed.disposition(wid) != 'success':
            continue
        if kind not in {'input_qualification_and_population', 'method_fit', 'terrain_fit'} | KINDS:
            raise ValueError('source metadata cannot omit already completed analysis')
        item = closed.read_metadata(work)
        if item.get('metadata_only') is not True or item.get('owned_source_bytes_rechecked') is not False:
            raise ValueError('source inventory must not pretend to revalidate scientific values')
        artifact = None if kind == 'input_qualification_and_population' else _artifact(item, root=root)
        sources[wid] = dict(**{k: item[k] for k in
            ('result_sha256','settlement_sha256','observation_sha256','controller_elapsed_ns')}, artifact=artifact)
        counts[kind] += 1
        if kind == 'input_qualification_and_population':
            if context is not None: raise ValueError('one original context owner required')
            context = sha256(item['manifest']['context_sha256'])
        elif kind in {'method_fit', 'terrain_fit'}:
            fits[work['fit_identity']] = artifact
        else:
            forecasts[wid] = artifact; kinds[kind] += 1
        if progress is not None and len(sources) % 250 == 0:
            progress(len(sources), 6298)
    if (counts != Counter(input_qualification_and_population=1, method_fit=16, terrain_fit=10,
                          scientific_forecast=6056, same_grid_reference=215)
            or set(sources) != {k for k, v in cost['work_dispositions'].items() if v == 'success'}
            or read_json(under(root, 'head.json')) != head):
        raise ValueError('complete closed success inventory and unchanged source head required')
    result = dict(schema_version=VERSION, resource_policy_sha256=resource_policy['sha256'],
        ledger_root_sha256=cost['ledger_root_sha256'], committed_tip=cost['ledger_tip'],
        **{'original_'+k+'_sha256': cost[k+'_sha256'] for k in ('protocol','execution','matrix','approval')},
        context_sha256=context, completed_sources=sources, fit_bindings=fits, forecast_index=forecasts,
        successful_work_counts=dict(counts), forecast_kind_counts=dict(kinds),
        forecast_status_counts={'not_replayed': len(forecasts)}, **FLAGS)
    validate_metadata(result, resource_policy=resource_policy)
    _closed_job(job)
    if progress is not None: progress(len(sources), 6298)
    return envelope(result)
