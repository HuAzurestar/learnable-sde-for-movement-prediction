"""Exact partial formal continuation; no authority generator or budget reset.

Preparation reads only pinned metadata and the local execution environment.
The legacy floor inspects complete old science. The distinct resource v2 floor
imports only closed accounting/source metadata; complete guarded restoration
and independent domain admission run inside the owned paid input bootstrap
before its compulsory reservation gate can admit any unfinished science.
"""
from copy import deepcopy
from pathlib import Path

from . import formal_budget as budget, formal_entrypoint as entry
from . import formal_partial_predecessor as predecessor
from .protocol_core import envelope, publish, unpack, under


def _floor(record, contract):
    if unpack(record)['schema_version'] == entry.RESOURCE_VERSION+'-runtime':
        from .formal_resource_predecessor import resource_floor
        return resource_floor(record, contract)
    return predecessor.partial_floor(record, contract)


def validate_partial_runtime(record, contract, *, protocol, matrix):
    runtime = unpack(record)
    if runtime['schema_version'] == entry.RESOURCE_VERSION+'-runtime':
        from .formal_resource_predecessor import _resource_floor_and_history
        # Reuse ONLY the complete history validated by this very floor call.
        # Physical binding bytes and all runtime scope checks still follow;
        # neither a producer PASS nor a prior invocation can supply history.
        binding, _, _, original, _ = _resource_floor_and_history(record, contract)
    else:
        binding, _, _ = _floor(record,contract)
        original = None
    actual, _ = predecessor._load(runtime['predecessor_reference'])
    if original is None:
        _, _, _, original, _ = predecessor._history(binding)
    if (actual != binding or protocol != original['protocol'] or matrix != original['matrix']
            or runtime['input_paths'] != unpack(original['runtime'])['input_paths']):
        raise ValueError('partial runtime changed pinned predecessor or original scientific input paths')
    return runtime


def launch_floor(bundle, *, approval_sha256):
    """Original charges from the bound complete history, not caller amounts."""
    p,m,r,e = (bundle[k] for k in ('protocol','matrix','runtime','execution'))
    contract = entry.budget_contract(bundle, approval_sha256=approval_sha256)
    validate_partial_runtime(r,contract,protocol=p,matrix=m)
    _, costs, _ = _floor(r,contract)
    return deepcopy(costs['charged_ns_by_phase']),costs['ledger_root_sha256']


def prepare_partial(*, output_directory, ledger_directory, predecessor_reference_path, resource=False):
    """Seal the actual successor while keeping old protocol/matrix/inputs."""
    from .formal_environment import ConfiguredRuntime
    from .protocol import EXECUTION_VERSION, source_catalog
    reference = predecessor.reference(Path(predecessor_reference_path).resolve())
    binding,_ = predecessor._load(reference)
    from .formal_resource_predecessor import VERSION as RESOURCE_PREDECESSOR_VERSION, METADATA_VERSION, RECOVERY_VERSION, CONTINUATION_VERSION
    binding_version = unpack(binding)['schema_version']
    allowed = {RESOURCE_PREDECESSOR_VERSION, METADATA_VERSION, RECOVERY_VERSION, CONTINUATION_VERSION} if resource else {predecessor.VERSION}
    if type(resource) is not bool or binding_version not in allowed:
        raise ValueError('preparation command must select the exact distinct predecessor type')
    _,_,_,original,_ = predecessor._history(binding)
    protocol,matrix = original['protocol'],original['matrix']
    ledger,output = Path(ledger_directory).resolve(),Path(output_directory).resolve()
    claim = entry.launch_directory(ledger)
    if ledger.exists() or claim.exists() or output.is_relative_to(ledger) or output.is_relative_to(claim):
        raise ValueError('partial preparation requires an unused distinct ledger and launch location')
    environment = ConfiguredRuntime(protocol).identity()
    version = entry.RESOURCE_VERSION if resource else entry.PARTIAL_VERSION
    runtime = envelope(dict(schema_version=version+'-runtime',protocol_sha256=protocol['sha256'],
        matrix_sha256=matrix['sha256'],environment=environment,ledger_directory=str(ledger),
        input_paths=deepcopy(unpack(original['runtime'])['input_paths']),
        working_directory=str(Path(entry.__file__).resolve().parents[2]),worker_module=entry.MODULE,
        phase_order=list(entry.PHASE_ORDER),predecessor=binding,predecessor_reference=reference,final_eval_authorized=False))
    entry.validate_runtime_binding(runtime,protocol,matrix)
    p = unpack(protocol)
    execution = envelope(dict(schema_version=EXECUTION_VERSION,protocol_sha256=protocol['sha256'],
        source_sha256=source_catalog(),matrix_sha256=matrix['sha256'],runtime_manifest_sha256=runtime['sha256'],
        budget_ledger_schema_version=budget.VERSION,entrypoint=entry.ENTRYPOINT,
        phase_caps_seconds=p['resource_contract']['phase_caps_seconds'],max_generated_forecasts=11513,
        scientific_forecasts=11020,method_required_slots=sorted(k for k,v in
            p['components']['method_mechanisms']['slots'].items() if v['disposition']=='REQUIRED'),
        terrain_configurations=sorted(p['components']['terrain_configurations']),final_eval_authorized=False))
    result = envelope(dict(schema_version=version+'-bundle',protocol=protocol,
        execution=execution,matrix=matrix,runtime=runtime))
    entry.validate_bundle(result)
    return publish(output,unpack(result))


def bootstrap_callbacks(callbacks, observation, output, tick):
    """Initialize the real parent readers in the metered bootstrap span."""
    from . import formal_session as native
    from .formal_environment import ConfiguredRuntime
    from .formal_results import ScientificResults
    from .formal_partial_imports import verify_manifest
    if callbacks.failed or callbacks.results is not None:
        raise ValueError('once-only fresh partial callbacks required')
    tick()
    bundle,runtime = entry.validate_bundle(callbacks.bundle)
    ledger = callbacks.ledger
    if (not entry.is_restoration(runtime)
            or not isinstance(callbacks.session,native.OwnedSession) or callbacks.session.ledger is not ledger
            or runtime['ledger_directory'] != str(ledger.directory) or not ledger._state.imported_success
            or ledger._state.pending is not None or ledger._state.partial_imports is not None
            or ledger._state.active_control is None
            or ledger._state.controls[ledger._state.active_control]['phase'] != entry.PHASE_ORDER[0]
            or observation['control_sha256'] != ledger._state.active_control
            or observation['ledger_root_sha256'] != ledger.root_sha256
            or output != under(callbacks.session.directory,'bootstrap')):
        raise ValueError('exact owned initial partial input control required')
    entry._approval(callbacks.authority,bundle['protocol'],bundle['execution'])
    tick()
    callbacks.environment = ConfiguredRuntime(bundle['protocol'])
    callbacks.environment.verify(runtime['environment'])
    callbacks.bound = bundle
    callbacks.first_work = {}
    for work in unpack(bundle['matrix'])['workloads']:
        if work['work_id'] not in ledger._state.imported_success:
            callbacks.first_work.setdefault(work['phase'],work['work_id'])
    tick()
    entry.read_runtime_evidence(callbacks.session.directory,contract=ledger._state.contract,
        session_sha256=callbacks.session.session['sha256'],ledger_root_sha256=ledger.root_sha256,
        worker_pid=callbacks.session.worker_pid,phase=None,first_work_id=None)
    results = ScientificResults(ledger,**{k:bundle[k] for k in ('protocol','execution','matrix')},authority=callbacks.authority)
    results._start()
    results._check_authority(entry.PHASE_ORDER[0])
    verification = verify_manifest(observation,output,tick,ledger=ledger,
        **{k:bundle[k] for k in ('protocol','execution','matrix')},access_journal=callbacks.authority['journal_directory'],
        on_verified=results.restore_imports)
    callbacks.environment.check()
    callbacks.results = results
    return verification
