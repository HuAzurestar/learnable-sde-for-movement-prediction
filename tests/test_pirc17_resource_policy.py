"""Synthetic full-size metadata only; no actual approval, Job, fits or forecasts."""
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.pirc17 import formal_budget as budget, formal_resource_policy as module
from experiments.pirc17.formal_carryover import VERSION as COST_VERSION
from experiments.pirc17.protocol_core import digest, envelope, file_hash, publish, unpack

NS = budget.NANOSECONDS


@pytest.fixture
def fixture(tmp_path):
    return make_policy_fixture(tmp_path)


def make_policy_fixture(tmp_path, *, protocol=None, matrix=None):
    """Same closed synthetic history, optionally using the real frozen plan."""
    phases = module.PHASE_SECONDS
    rows = []

    def add(kind, phase, count, *, calls=0, cap=30):
        for index in range(count):
            wid = (module.RETRY_WORK_ID if kind == 'scientific_forecast' and phase == 'method_forecasts'
                   and index == 6056 else digest([kind, phase, index]))
            row = dict(work_id=wid, kind=kind, phase=phase, max_active_seconds=cap,
                       generated_forecasts=calls)
            if kind in {'method_fit', 'terrain_fit'}:
                row['fit_identity'] = digest(['fixture-fit', kind, index])
            rows.append(row)

    add('input_qualification_and_population', 'input_qualification_and_binding', 1, cap=3600)
    add('method_fit', 'method_training', 16)
    add('terrain_fit', 'terrain_training', 10)
    add('scientific_forecast', 'method_forecasts', 8120, calls=1)
    add('scientific_forecast', 'terrain_forecasts', 2900, calls=1, cap=180)
    add('same_grid_reference', 'method_forecasts', 290, calls=1)
    add('forecast_replay', 'runtime_and_forecast_replay', 203, calls=1)
    add('common_scores', 'offline_common_scores', 119, cap=7200)
    assert len(rows) == 11659
    if protocol is None and matrix is None:
        protocol = envelope(dict(software_fixture_only=True, not_actual_science=True))
        matrix = envelope(dict(protocol_sha256=protocol['sha256'], workloads=rows,
            max_generated_forecasts=11513, phase_caps_seconds=phases))
    elif protocol is None or matrix is None:
        raise ValueError('both complete frozen protocol and matrix required')
    else:
        rows = unpack(matrix)['workloads']
    old = (tmp_path / 'software-source-not-an-empirical-ledger').resolve()
    runtime = envelope(dict(protocol_sha256=protocol['sha256'], matrix_sha256=matrix['sha256'],
        ledger_directory=str(old), software_fixture_only=True,
        input_paths={k: str((tmp_path/('ABSENT-'+k)).resolve()) for k in
                     ('release', 'snapshot', 'data_root', 'trajectory_path', 'development_eligibility_path')}))
    execution = envelope(dict(protocol_sha256=protocol['sha256'], matrix_sha256=matrix['sha256']))
    bundle = dict(protocol=protocol, matrix=matrix, execution=execution, runtime=runtime)
    approval = digest('SOFTWARE-NOT-REAL-HUMAN-APPROVAL')
    contract = budget.contract_for_matrix(matrix, protocol_sha256=protocol['sha256'],
        execution_sha256=execution['sha256'], runtime_manifest_sha256=runtime['sha256'],
        approval_sha256=approval, ledger_directory=old)
    states, seen = {}, Counter()
    for work in rows:
        if work['work_id'] == module.RETRY_WORK_ID:
            continue  # Never overwrite an actual successful item with timeout.
        kind, phase = work['kind'], work['phase']
        seen[kind, phase] += 1
        if (kind in {'input_qualification_and_population', 'method_fit', 'terrain_fit'}
                or kind == 'scientific_forecast' and phase == 'method_forecasts' and seen[kind, phase] <= 6056
                or kind == 'same_grid_reference' and seen[kind, phase] <= 215):
            states[work['work_id']] = 'success'
    states[module.RETRY_WORK_ID] = 'timeout'
    charges = dict.fromkeys(phases, 0)
    charges.update(input_qualification_and_binding=2139205_000000, method_training=82765_000000,
        terrain_training=57438_000000, method_forecasts=10800027_000000)
    measured = dict.fromkeys(phases, 0)
    measured.update(input_qualification_and_binding=1101751_000000, method_training=77765_000000,
        terrain_training=52438_000000, method_forecasts=9743559_000000)
    control = dict.fromkeys(phases, 0)
    control.update(input_qualification_and_binding=1037454_000000, method_training=5*NS,
        terrain_training=5*NS, method_forecasts=50*NS)
    tip = dict(root_sha256=digest(contract), event_count=11635, last_event_sha256=digest('fixture-tip'))
    cost = dict(schema_version=COST_VERSION, read_only=True, authorizes_execution=False, final_eval_reads=0,
        ledger_directory=str(old), ledger_root_sha256=digest(contract), head_sha256=digest(tip), ledger_tip=tip,
        protocol_sha256=protocol['sha256'], execution_sha256=execution['sha256'], matrix_sha256=matrix['sha256'],
        runtime_manifest_sha256=runtime['sha256'], approval_sha256=approval,
        phase_caps_ns=contract['phase_caps_ns'], total_cap_ns=contract['total_cap_ns'],
        charged_ns_by_phase=charges, measured_ns_by_phase=measured,
        conservatively_charged_ns_by_phase={key: charges[key]-measured[key] for key in phases},
        control_charged_ns_by_phase=control, control_observed_ns_by_phase=dict.fromkeys(phases, 0),
        charged_total_ns=sum(charges.values()), generated_forecasts_reserved=6272,
        work_dispositions=states, halted_reason='phase_cap_reached')
    native = dict(work_id=module.RETRY_WORK_ID, ledger_root_sha256=cost['ledger_root_sha256'],
        reservation_sha256=digest('fixture-old-failed-reservation'), started_ns=NS,
        deadline_ns=NS+645_000000, elapsed_ns=672_000000, status='timeout', result_sha256=None,
        barrier_sha256=None, closure=dict(process_tree_closed=True, accounting=dict(active_processes=0)))
    timeout = dict(work_id=module.RETRY_WORK_ID, ledger_root_sha256=cost['ledger_root_sha256'],
        reservation_sha256=native['reservation_sha256'], stop_reason='work_deadline',
        native_observation=envelope(native), scientific_manifest_validation=None)
    terminal = dict(ledger_root_sha256=cost['ledger_root_sha256'], bundle_sha256=digest(bundle),
        candidate_complete=False, error_type='ControllerStopped', process_tree_closed=True,
        approval_verified=True, cumulative_charge_lower_bound_ns=cost['charged_total_ns'])

    def ref(name, payload):
        path, record = publish(tmp_path / name, payload)
        return dict(path=str(path), content_sha256=record['sha256'], file_sha256=file_hash(path))

    terminal_ref = ref('terminal', terminal)
    cost['terminal_proof_sha256'] = digest(dict(content_sha256=terminal_ref['content_sha256'],
        file_sha256=terminal_ref['file_sha256']))
    # Copying the source text in a software fixture does NOT authenticate it;
    # the module deliberately never grants execution authority.
    human = dict(id=module.HUMAN_ID, author_type='user', type='message', content=module.HUMAN_CONTENT,
        created_at='2026-10-03T03:01:19.0963983Z', task_id='1fa14b29-0a30-4c40-896e-8a6c21464419',
        session_id='45b5bb5b-dc13-47ca-8f3c-877e1305a12a')
    args = dict(cost_reference=ref('cost', cost), bundle_reference=ref('bundle', bundle),
        terminal_reference=terminal_ref, timeout_reference=ref('timeout', timeout), human_message=human)
    return dict(args=args, cost=cost, timeout=timeout, ref=ref, matrix=matrix, contract=contract,
                protocol=protocol, bundle=bundle)


def test_fixed_total_one_retry_policy_preserves_all_costs_and_original_scope(fixture):
    record = module.build_policy(**fixture['args'])
    result = module.validate_policy(record)
    assert result['total_cap_ns'] == 48*3600*NS == sum(result['phase_caps_ns'].values())
    assert result['phase_caps_ns']['method_forecasts'] == 4*3600*NS
    assert result['phase_caps_ns']['terrain_forecasts'] == 31*3600*NS
    assert result['retained_charged_ns_by_phase'] == fixture['cost']['charged_ns_by_phase']
    assert result['retained_total_charge_ns'] == 13079435_000000
    assert result['retained_success_count'] == 6298 and result['retained_generation_reservations'] == 6272
    assert result['effective_generation_limit'] == 11514 and result['additional_generation_allowance'] == 1
    assert result['retry_work_id'] == module.RETRY_WORK_ID
    assert result['within_successor_attempts_per_item'] == 1
    assert result['authorizes_execution'] is False and result['authorizes_successful_reruns'] is False
    assert result['old_ledger_modified'] is False


@pytest.mark.parametrize('change', ['successful_retry', 'second_timeout', 'dropped_success',
    'refunded_cost', 'other_phase_change', 'released_generation'])
def test_dropped_refunded_or_unauthorized_source_is_rejected(fixture, change):
    cost = deepcopy(fixture['cost'])
    if change == 'successful_retry':
        cost['work_dispositions'][module.RETRY_WORK_ID] = 'success'
    elif change == 'second_timeout':
        key = next(k for k, v in cost['work_dispositions'].items() if v == 'success')
        cost['work_dispositions'][key] = 'timeout'
    elif change == 'dropped_success':
        del cost['work_dispositions'][next(iter(cost['work_dispositions']))]
    elif change == 'refunded_cost':
        cost['charged_ns_by_phase']['method_forecasts'] -= NS
    elif change == 'other_phase_change':
        cost['phase_caps_ns']['input_qualification_and_binding'] += NS
    elif change == 'released_generation':
        cost['generated_forecasts_reserved'] -= 1
    args = dict(fixture['args'], cost_reference=fixture['ref']('altered-cost', cost))
    with pytest.raises(ValueError):
        module.build_policy(**args)


@pytest.mark.parametrize('change', ['not_truncated', 'active_tree', 'wrong_work', 'success'])
def test_only_exact_budget_truncated_closed_timeout_is_eligible(fixture, change):
    observed = deepcopy(fixture['timeout'])
    native = unpack(observed['native_observation'])
    if change == 'not_truncated':
        native['deadline_ns'] = native['started_ns'] + 30*NS
        native['elapsed_ns'] = 31*NS
    elif change == 'active_tree':
        native['closure']['accounting']['active_processes'] = 1
    elif change == 'wrong_work':
        native['work_id'] = digest('another-failed-item')
    elif change == 'success':
        native['status'] = 'success'
    observed['native_observation'] = envelope(native)
    args = dict(fixture['args'], timeout_reference=fixture['ref']('altered-timeout', observed))
    with pytest.raises(ValueError):
        module.build_policy(**args)


@pytest.mark.parametrize('field,value', [('author_type', 'agent'), ('id', 'automatic-continuation'),
    ('content', module.HUMAN_CONTENT.replace('48', '49'))])
def test_automatic_or_different_direction_does_not_allow_policy(fixture, field, value):
    args = deepcopy(fixture['args'])
    args['human_message'][field] = value
    with pytest.raises(ValueError, match='user resource'):
        module.build_policy(**args)


@pytest.mark.parametrize('field,value', [('total_cap_ns', 49*3600*NS),
    ('additional_generation_allowance', 2), ('authorizes_execution', True),
    ('retained_total_charge_ns', 0)])
def test_self_resealed_policy_cannot_expand_or_erase_scope(fixture, field, value):
    policy = unpack(module.build_policy(**fixture['args']))
    policy[field] = value
    with pytest.raises(ValueError, match='policy changed'):
        module.validate_policy(envelope(policy))


def test_byte_reference_mismatch_is_rejected_even_with_same_semantics(fixture):
    args = deepcopy(fixture['args'])
    args['cost_reference']['file_sha256'] = digest('wrong-bytes')
    with pytest.raises(ValueError, match='bytes changed'):
        module.build_policy(**args)


def resource_contract(fixture):
    source = fixture['contract']
    return budget.contract_for_matrix(fixture['matrix'], protocol_sha256=source['protocol_sha256'],
        execution_sha256=digest('software-new-execution'), runtime_manifest_sha256=digest('software-new-runtime'),
        approval_sha256=digest('software-new-NOT-HUMAN-approval'),
        ledger_directory=str(Path(source['ledger_directory']).parent / 'unused-software-successor'),
        resource_policy=module.build_policy(**fixture['args']))


def test_resource_contract_preserves_original_matrix_work_ids_and_one_attempt(fixture):
    old = deepcopy(fixture['contract'])
    contract = resource_contract(fixture)
    assert contract['workloads'] == old['workloads'] and fixture['contract'] == old
    assert contract['matrix_sha256'] == old['matrix_sha256']
    assert contract['protocol_sha256'] == old['protocol_sha256']
    assert contract['max_generated_forecasts'] == 11514 and contract['max_attempts_per_item'] == 1
    assert contract['total_cap_ns'] == old['total_cap_ns']
    assert contract['phase_caps_ns']['method_forecasts'] == 14400*NS
    assert contract['phase_caps_ns']['terrain_forecasts'] == 111600*NS


@pytest.mark.parametrize('change', ['work_cap', 'old_directory', 'old_approval', 'old_runtime', 'old_execution', 'extra_call'])
def test_contract_rejects_changed_work_or_reused_authority_and_extra_retry(fixture, change):
    contract = resource_contract(fixture)
    if change == 'work_cap':
        contract['workloads'][0]['max_active_ns'] -= NS
    elif change == 'old_directory':
        contract['ledger_directory'] = fixture['contract']['ledger_directory']
    elif change == 'extra_call':
        contract['max_generated_forecasts'] += 1
    else:
        key = {'old_approval': 'approval_sha256', 'old_runtime': 'runtime_manifest_sha256',
               'old_execution': 'execution_sha256'}[change]
        contract[key] = fixture['contract'][key]
    with pytest.raises(ValueError):
        budget.validate_contract(contract)


def test_without_policy_the_old_generation_ceiling_is_still_strict(fixture):
    contract = deepcopy(fixture['contract'])
    contract['max_generated_forecasts'] += 1
    with pytest.raises(ValueError, match='denominator'):
        budget.validate_contract(contract)
