"""Read-only eligibility of the single closed, input-bearing interruption.

This is not a general partial-run recovery or a retry/approval API. Production
pins include the complete actual byte inventory, not only selected receipts.
Saved input semantics are independently replayed; raw sources, lost worker
training state and current launch authority are NOT reconstructed here.
"""
import hashlib
from collections import Counter
from pathlib import Path

from . import formal_budget as budget, formal_carryover as costs
from . import formal_predecessor as startup, formal_session as native
from .protocol_core import canonical, digest, envelope, sha256, under, unpack

VERSION = 'pirc17-closed-input-interruption-v1'
HISTORICAL_ENTRY = 'pirc17-concrete-formal-entrypoint-v2'
REGISTERED_CHARGED_NS = 570_736_000_000
REGISTERED_COST_SNAPSHOT = '74a0e6aa32226c65ac6d147c6f74cec2233cf6d88c20a034bd269e3daf1e073d'
REGISTERED = dict(
    expected_root_sha256='b784bcdd5b00ef603eed0b486151bffa5d90aba92e814acfc06c9d891b337d0b',
    expected_head_sha256='78c57bedbeafb42562a894aad5f95e3ee68199b21aec763ceb4ac094d2912adf',
    expected_terminal_proof_sha256='81169f059d075a1e22cf7956b9e86df369d3ec2ec9068505c15254b33fa851dd')
REGISTERED_INVENTORY = '5cbdbc008cbaf77d64733842ed815e0b0e8066100039afdd4801fbf20352bacc'
REGISTERED_CLAIM_INVENTORY = 'c71d81de5387e5b85ad6018488f792c1d1f134c15762202c8e33d8b9201505ce'
REGISTERED_BUNDLE = '698d55b5c2675240c883e0df4adc310061d51625404020b5647bb2ccc7288e12'
REGISTERED_ENVIRONMENT = '2b9b6825148bad0a78e12bed7cb2fb69d9604d15ffd30926f2854fc5031ef2b1'
REGISTERED_PREDECESSOR = '9891cfec874d82eb37b7803fc98cc0a71f4883fb2555ce16c461ae4bb97b2ac7'
REGISTERED_CONTEXT = '36c404aef472c96e376abd66cb5b4ae29313c4dd998201c02a5aa726e47c5162'
REGISTERED_QUALIFICATION = '28a136387919596bcb67cc773d590fc35d29cba92ec2e10d5f2142480505c93b'
REGISTERED_CONTEXT_SEMANTICS = '5da7ade16367dda349f035ec2737132224eef0e86e07fd89246977c3dedd3678'
REGISTERED_ELIGIBILITY_SEMANTICS = '751d1fc605afbc33f305b13e1847f88e68e463dd5c2cbd9cca35d9dee4377b66'
MAX_NODES = 192
MAX_FILE_BYTES = 8*1024*1024
MAX_TOTAL_BYTES = 24*1024*1024
PHASE = 'input_qualification_and_binding'
EVENT_TYPES = ['startup_predecessor', 'control_open', 'reserve', 'settle',
               'halt', 'control_close', 'control_terminal_tail']
REGISTERED_ACCESS = dict(event_count=8, started_reads=4, returned_reads=4,
    kinds=dict(final_eval_eligibility=1, final_eval_positions=1, final_eval_features=2),
    events_sha256='067e8b99e88c462f2ae48aa9c88f9cb25424e3c9603dc6dd94982a9e258200bc',
    possible_reads_retained=True, historically_exposed=True)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_binding_scope(value):
    """Pure budget replay validation; does not query old files/processes.

    Pin the independently observed complete accounting, not just its total.
    Live reinspection remains mandatory at admission. This preserves historical
    v1 startup replay while giving post-read history a distinct meaning.
    """
    unpack(value['cost_snapshot'], expected_sha256=REGISTERED_COST_SNAPSHOT)
    _require(value['schema_version'] == VERSION and value['eligible_closed_input_interruption'] is True
        and value['process_tree_closed'] is True and value['read_only'] is True
        and value['authorizes_execution'] is False
        and type(value['generated_forecasts']) is int and value['generated_forecasts'] == 0
        and type(value['predictive_model_fits']) is int and value['predictive_model_fits'] == 0
        and value['raw_sources_independently_reloaded'] is False and value['worker_training_state_retained'] is False
        and value['inventory_sha256'] == REGISTERED_INVENTORY
        and value['claim_inventory_sha256'] == REGISTERED_CLAIM_INVENTORY
        and value['bundle_sha256'] == REGISTERED_BUNDLE
        and value['original_predecessor_sha256'] == REGISTERED_PREDECESSOR
        and value['input_context_sha256'] == REGISTERED_CONTEXT
        and value['input_qualification_sha256'] == REGISTERED_QUALIFICATION
        and type(value['selected_samples']) is int and value['selected_samples'] == 46
        and canonical(value['historical_access']) == canonical(REGISTERED_ACCESS),
        'pinned post-read eligibility, exposure and unchanged saved input scope required')
    return value


def _without(value, keys):
    # Explicit locations only: never recursively remove every hash/identity.
    return {k:v for k,v in value.items() if k not in keys}


def context_semantic_identity(record):
    """Retain all science; remove ONLY approval/execution transport identities.

    Call after exact new-execution typed restoration. Dataset/window/source,
    prior, development population, map assets/policy, ordered prefixes, UTC
    clocks, coordinates and targets remain bound, including their source hashes.
    Population seal changes when its execution/report hash changes; that is not
    permission to change the population's full identities or selection itself.
    """
    value = _without(unpack(record), {'execution_sha256'})
    for key, omitted in (
        ('input_identity', {'execution_sha256','approval_sha256','population_sha256','access_started_sha256'}),
        ('map_catalog', {'execution_sha256','approval_sha256','population_sha256','access_started_sha256'}),
        ('population', {'execution_sha256','eligibility_sha256'}),
        ('scoring_inputs', {'execution_sha256','population_sha256','positions_access_started_sha256'})):
        value[key] = _without(unpack(value[key]), omitted)
    value['causal_prefixes'] = [_without(p, {'population_sha256'}) for p in value['causal_prefixes']]
    return digest(value)


def eligibility_semantic_identity(report, population):
    return digest(dict(eligibility=_without(unpack(report), {'execution_sha256'}),
        population=_without(unpack(population), {'execution_sha256','eligibility_sha256'})))


def verify_continued_context(record):
    _require(context_semantic_identity(record) == REGISTERED_CONTEXT_SEMANTICS,
             'continued input changed original source, training, prefix, map or scoring semantics')


def verify_continued_population(report, population):
    _require(eligibility_semantic_identity(report, population) == REGISTERED_ELIGIBILITY_SEMANTICS,
             'continued eligibility or original ordered population changed')


def verify_continued_manifest(manifest, directory):
    """Independent controller check after its normal exact-execution replay."""
    from .formal_input_work import _owned_record
    _, context = _owned_record(manifest, directory, 'context_path', 'context_sha256')
    qpath, qualified = _owned_record(manifest, directory, 'qualification_path', 'qualification_sha256')
    q = unpack(qualified)
    from .protocol_core import read_json
    verify_continued_population(read_json(under(qpath.parent,q['eligibility_path'])),
                               read_json(under(qpath.parent,q['population_path'])))
    verify_continued_context(context)
    return dict(original_context_sha256=REGISTERED_CONTEXT, original_qualification_sha256=REGISTERED_QUALIFICATION,
        context_semantics_sha256=REGISTERED_CONTEXT_SEMANTICS, eligibility_semantics_sha256=REGISTERED_ELIGIBILITY_SEMANTICS,
        ordered_original_population_verified=True, new_raw_source_reload_claimed=False)


def _catalog(directory):
    """Bound both traversal and same-read hashes, including empty directories."""
    directory = Path(directory)
    _require(directory.is_absolute() and directory.is_dir() and not directory.is_symlink(),
             'absolute regular interruption directory required')
    pending, found, total = [directory], {}, 0
    while pending:
        for path in pending.pop().iterdir():
            _require(len(found) < MAX_NODES, 'bounded interruption inventory required')
            _require(not path.is_symlink() and path.resolve().is_relative_to(directory),
                     'interruption inventory links/escapes forbidden')
            relative = path.relative_to(directory).as_posix()
            if path.is_dir():
                found[relative] = dict(kind='directory')
                pending.append(path)
                continue
            _require(path.is_file() and path.stat().st_size <= MAX_FILE_BYTES,
                     'bounded regular interruption bytes required')
            with path.open('rb') as stream:
                raw = stream.read(min(MAX_FILE_BYTES, MAX_TOTAL_BYTES-total)+1)
            total += len(raw)
            _require(len(raw) <= MAX_FILE_BYTES and total <= MAX_TOTAL_BYTES,
                     'interruption bytes exceed fixed inventory budget')
            found[relative] = dict(kind='file', bytes=len(raw), file_sha256=hashlib.sha256(raw).hexdigest())
    return found


def _cost_scope(c, contract):
    _require(type(c['charged_total_ns']) is int and c['charged_total_ns'] == REGISTERED_CHARGED_NS
             and c['event_types'] == EVENT_TYPES
             and type(c['generated_forecasts_reserved']) is int and c['generated_forecasts_reserved'] == 0
             and c['halted_reason'] == 'supervision_error'
             and c['work_inventory_count'] == 11659
             and c['work_dispositions'] == {contract['workloads'][0]['work_id']: 'failure'}
             and contract['workloads'][0]['phase'] == PHASE
             and all(c['charged_ns_by_phase'][k] == 0 for k in c['phase_caps_ns'] if k != PHASE),
             'only the pinned failed input work with all cumulative costs is eligible')


def _access_evidence(directory, contract):
    """Four real starts/returns, not eight independent reads or zero exposure."""
    records = {}
    for path in directory.iterdir():
        _require(path.is_file() and path.suffix == '.json' and len(records) < 8,
                 'exact bounded historical access journal required')
        record, _ = costs._read_bound(path, costs.MAX_RECORD_BYTES)
        v = unpack(record)
        _require(path.name == record['sha256']+'.json'
                 and v['schema_version'] == 'pirc17-final-access-event-v1'
                 and all(v[k] == contract[k] for k in ('protocol_sha256','execution_sha256','approval_sha256'))
                 and v['event'] in {'started','returned'}, 'historical access authority/event linkage changed')
        records[record['sha256']] = v
    counts = Counter((v['access_kind'], v['event']) for v in records.values())
    _require(counts == Counter({(kind, event): n for kind,n in
        [('final_eval_eligibility',1),('final_eval_positions',1),('final_eval_features',2)]
        for event in ('started','returned')}), 'original four guarded reads and eight events required')
    returned = set()
    for v in records.values():
        if v['event'] != 'returned':
            continue
        key = sha256(v['started_sha256'])
        start = records.get(key)
        _require(start is not None and start['event'] == 'started' and key not in returned
                 and all(v[k] == start[k] for k in ('access_kind','attempt_id','population_sha256'))
                 and v['possible_reads_retained'] is True, 'historical read return/start differs')
        returned.add(key)
    return dict(event_count=8, started_reads=4, returned_reads=4,
        kinds=dict(final_eval_eligibility=1, final_eval_positions=1, final_eval_features=2),
        events_sha256=digest(records), possible_reads_retained=True, historically_exposed=True)


def inspect_input_interruption(directory):
    """Bind exact actual closed history and saved inputs without mutating it."""
    supplied = Path(directory)
    _require(supplied.is_absolute(), 'absolute input interruption ledger required')
    directory = supplied.resolve(strict=True)
    claim = under(directory.parent, directory.name+'.launch')
    before, claim_before = _catalog(directory), _catalog(claim)
    _require(digest(before) == REGISTERED_INVENTORY and digest(claim_before) == REGISTERED_CLAIM_INVENTORY,
             'complete registered input interruption byte inventory changed')
    reads = startup._Reads()
    snapshot = costs.inspect_closed_ledger(directory, **REGISTERED)
    c = unpack(snapshot)
    root = reads.read(under(directory,'ledger.json'), c['ledger_root_sha256'], large=True)
    contract = unpack(root)
    _cost_scope(c, contract)
    events = [reads.read(under(directory,f'events/{i:06d}.json')) for i in range(7)]
    floor, opened, reservation, settlement, halted, closed, tail = [unpack(r)['row'] for r in events]
    runtime_record = floor['runtime_manifest']
    runtime = unpack(runtime_record, expected_sha256=contract['runtime_manifest_sha256'])
    predecessor = runtime['predecessor']
    unpack(predecessor, expected_sha256=REGISTERED_PREDECESSOR)
    startup.verify_startup_predecessor(predecessor)
    _require(unpack(runtime['environment'], expected_sha256=REGISTERED_ENVIRONMENT)
             and runtime['ledger_directory'] == str(directory), 'original full environment/runtime changed')
    terminal_record = reads.read(under(claim,'terminal.json'))
    terminal = unpack(terminal_record)
    _require(digest(dict(content_sha256=terminal_record['sha256'],
        file_sha256=reads.files[str(under(claim,'terminal.json'))][0])) == c['terminal_proof_sha256']
        and terminal['schema_version'] == HISTORICAL_ENTRY+'-launch-terminal'
        and terminal['ledger_root_sha256'] == root['sha256'] and terminal['process_tree_closed'] is True
        and terminal['candidate_complete'] is False and terminal['approval_verified'] is True
        and terminal['predecessor_floor_transferred_to_ledger'] is True
        and terminal['cumulative_charge_lower_bound_ns'] == REGISTERED_CHARGED_NS,
        'closed input launch terminal or cumulative debit changed')
    launch = unpack(reads.read(under(claim,'start.json'), terminal['launch_sha256']))
    bundle_record = reads.read(Path(launch['bundle_path']), REGISTERED_BUNDLE, large=True)
    _require(reads.files[launch['bundle_path']][0] == launch['bundle_file_sha256'], 'historical bundle bytes changed')
    bundle = unpack(bundle_record)
    _require(bundle['schema_version'] == HISTORICAL_ENTRY+'-bundle' and bundle['runtime'] == runtime_record,
             'original consumed bundle/runtime differs')
    for key, identity in [('protocol','protocol_sha256'),('execution','execution_sha256'),('matrix','matrix_sha256')]:
        unpack(bundle[key], expected_sha256=contract[identity])
    authority = launch['authority']
    approval = unpack(reads.read(Path(authority['approval_path']), contract['approval_sha256']))
    _require(approval['schema_version'] == 'pirc17-human-accept01-v1' and approval['task'] == 'ACCEPT-01'
        and approval['decision'] == 'CONFIRMED' and approval['human_confirmation']['kind'] == 'user-message'
        and approval['protocol_sha256'] == contract['protocol_sha256']
        and approval['execution_sha256'] == contract['execution_sha256'], 'original human scope differs')
    for name, task in [('test','TEST-01'),('review','REVIEW-01')]:
        check = unpack(reads.read(Path(authority[name+'_path']), approval[name+'_sha256']))
        _require(check['schema_version'] == 'pirc17-pre-eval-check-v1' and check['task'] == task and check['result'] == 'PASS'
            and check['protocol_sha256'] == contract['protocol_sha256']
            and check['execution_sha256'] == contract['execution_sha256'], 'original check linkage differs')
    session_dir = under(directory,'session-000001')
    session_record = reads.read(under(session_dir,'session.json'))
    session = unpack(session_record)
    native._job_name(session['job_name'])
    startup._closed_job(session['job_name'])
    request_record = reads.read(under(session_dir,'requests/000000.json'))
    barrier_record = reads.read(under(session_dir,'barriers/000000.json'))
    ready = unpack(reads.read(under(session_dir,'ready.json')))
    result = native._barrier_result(barrier_record, request_record, directory=session_dir,
        session_sha256=session_record['sha256'], sequence=0, previous_sha256=session_record['sha256'],
        worker_pid=ready['worker_pid'])
    _require(result is not None and unpack(barrier_record)['status'] == 'success'
        and result['reservation_sha256'] == events[2]['sha256']
        and result['work_id'] == reservation['work_id'] and settlement['status'] == 'failure'
        and settlement['result_sha256'] is None, 'worker success must not erase the actual failed settlement')
    observed = unpack(reads.read(under(directory,'controls/'+settlement['completion_evidence_sha256']+'.json'),
        settlement['completion_evidence_sha256']))
    n = unpack(observed['native_observation'])
    _require(observed['candidate_status'] == 'failure' and observed['scientific_manifest_validation'] is None
        and n['status'] == 'failure' and n['error_type'] == 'ValueError'
        and n['stop_observation']['reason'] == 'transport_failure'
        and n['barrier_sha256'] == barrier_record['sha256']
        and n['closure']['process_tree_closed'] is True and n['closure']['job_name'] == session['job_name']
        and type(n['closure']['accounting']['active_processes']) is int
        and n['closure']['accounting']['active_processes'] == 0, 'actual transport failure/native closure differs')
    manifest = result['value']
    _require(manifest['context_sha256'] == REGISTERED_CONTEXT
        and manifest['qualification_sha256'] == REGISTERED_QUALIFICATION, 'original saved input identities changed')
    from .formal_input_work import input_verification
    validation, args, _ = input_verification(contract['workloads'][0], manifest,
        under(session_dir,'outputs/000000'), protocol=bundle['protocol'], execution=bundle['execution'],
        matrix=bundle['matrix'], approval_sha256=contract['approval_sha256'], access_journal=under(directory,'access'))
    access = _access_evidence(under(directory,'access'), contract)
    details = validation['details']
    _require(details['qualification']['denominator_samples'] == 12370
        and details['qualification']['eligible_samples'] == 106
        and details['qualification']['selected_samples'] == 46, 'original complete selected population required')
    runtime_start = unpack(reads.read(under(session_dir,'runtime/start.json')))
    _require(runtime_start['environment'] == runtime['environment']
        and runtime_start['predecessor_floor_event_sha256'] == events[0]['sha256'], 'actual runtime floor/environment differs')
    reads.verify_unchanged()
    _require(_catalog(directory) == before and _catalog(claim) == claim_before
        and costs.inspect_closed_ledger(directory, **REGISTERED) == snapshot,
        'closed interruption changed during semantic inspection')
    closure_now = startup._closed_job(session['job_name'])
    selection = unpack(args['population'])['selection']['selected']
    binding = envelope(dict(schema_version=VERSION, ledger_directory=str(directory), cost_snapshot=snapshot,
        inventory_sha256=REGISTERED_INVENTORY, claim_inventory_sha256=REGISTERED_CLAIM_INVENTORY,
        external_file_sha256={name:checksum for name,(checksum,_,_) in reads.files.items()
                              if not Path(name).is_relative_to(directory) and not Path(name).is_relative_to(claim)},
        bundle_sha256=bundle_record['sha256'], launch_sha256=terminal['launch_sha256'], terminal_sha256=terminal_record['sha256'],
        original_predecessor_sha256=predecessor['sha256'], original_predecessor_directory=unpack(predecessor)['ledger_directory'],
        input_context_sha256=REGISTERED_CONTEXT, input_qualification_sha256=REGISTERED_QUALIFICATION,
        saved_input_verification_sha256=digest(validation), input_details=details,
        selected_population_sha256=args['population']['sha256'], selected_samples=46,
        ordered_selection_sha256=digest(selection), historical_access=access,
        native_job_name=session['job_name'], native_job_observation=closure_now, process_tree_closed=True,
        eligible_closed_input_interruption=True, generated_forecasts=0, predictive_model_fits=0,
        read_only=True, raw_sources_independently_reloaded=False, worker_training_state_retained=False,
        authorizes_execution=False))
    _require(len(canonical(binding)) <= costs.MAX_RECORD_BYTES, 'constant-size interruption binding required')
    validate_binding_scope(unpack(binding))
    return binding


def verify_input_interruption(binding):
    value = unpack(binding)
    _require(value.get('schema_version') == VERSION, 'exact input interruption binding required')
    fresh = inspect_input_interruption(value['ledger_directory'])
    _require(canonical(fresh) == canonical(binding), 'sealed input interruption binding changed since registration')
    return unpack(fresh)
