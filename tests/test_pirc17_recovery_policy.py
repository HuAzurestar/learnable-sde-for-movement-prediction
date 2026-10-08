"""Full-size synthetic metadata only; no real data, Job or launch authority."""
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.pirc17 import formal_budget as budget, formal_resource_policy as prior
from experiments.pirc17 import formal_resource_predecessor as resource, formal_recovery_policy as module
from experiments.pirc17.protocol_core import digest, envelope, unpack
from tests.test_pirc17_resource_policy import make_policy_fixture
from tests.test_pirc17_source_metadata import metadata_payload

NS = budget.NANOSECONDS


@pytest.fixture(scope='module')
def fixture(tmp_path_factory):
    return make_recovery_fixture(tmp_path_factory.mktemp('recovery-policy-metadata'))


def make_recovery_fixture(root, *, protocol=None, matrix=None):
    f = make_policy_fixture(root, protocol=protocol, matrix=matrix)
    timeout = deepcopy(f['timeout'])
    timeout['native_observation']['payload']['closure']['job_name'] = 'PIRC17-FORMAL-'+'a'*32
    timeout['native_observation'] = envelope(timeout['native_observation']['payload'])
    f['args']['timeout_reference'] = f['ref']('named-timeout', timeout)
    old_policy = prior.build_policy(**f['args'])
    saved = metadata_payload(f, old_policy)
    source_ref = f['ref']('original-science', saved)
    previous = resource.build_binding(resource_policy=old_policy, science_reference=source_ref)
    previous_ref = f['ref']('previous-binding', unpack(previous))
    directory = Path(f['cost']['ledger_directory']).parent/'closed-failed-recovery'
    runtime = envelope(dict(schema_version='pirc17-concrete-formal-entrypoint-v5-runtime',
        protocol_sha256=f['protocol']['sha256'], matrix_sha256=f['matrix']['sha256'],
        predecessor=previous, predecessor_reference=previous_ref,
        ledger_directory=str(directory), input_paths=unpack(f['bundle']['runtime'])['input_paths']))
    execution = envelope(dict(protocol_sha256=f['protocol']['sha256'], matrix_sha256=f['matrix']['sha256'],
        runtime_manifest_sha256=runtime['sha256'], software_fixture_only=True))
    bundle = dict(schema_version='pirc17-concrete-formal-entrypoint-v5-bundle',
        protocol=f['protocol'], matrix=f['matrix'], execution=execution, runtime=runtime)
    approval = digest('synthetic recovery attempt, not human acceptance')
    contract = budget.contract_for_matrix(f['matrix'], protocol_sha256=f['protocol']['sha256'],
        execution_sha256=execution['sha256'], runtime_manifest_sha256=runtime['sha256'],
        approval_sha256=approval, ledger_directory=directory, resource_policy=old_policy)
    cost = deepcopy(f['cost'])
    tip = dict(root_sha256=digest(contract), event_count=5, last_event_sha256=digest('synthetic final recovery tip'))
    cost.update(ledger_directory=str(directory), ledger_root_sha256=digest(contract), head_sha256=digest(tip),
        ledger_tip=tip, execution_sha256=execution['sha256'], runtime_manifest_sha256=runtime['sha256'],
        approval_sha256=approval, phase_caps_ns=contract['phase_caps_ns'], work_inventory_count=11659,
        source_file_sha256={name:digest(['synthetic bytes', name]) for name in
            ['ledger.json', 'head.json', *(f'events/{i:06d}.json' for i in range(5))]},
        event_types=['resource_predecessor', 'control_open', 'control_credit', 'control_close', 'control_terminal_tail'],
        work_dispositions={key:value for key,value in f['cost']['work_dispositions'].items() if value == 'success'})
    delta = 3600*NS - cost['charged_ns_by_phase'][module.INPUT]
    cost['charged_ns_by_phase'][module.INPUT] += delta
    cost['conservatively_charged_ns_by_phase'][module.INPUT] += delta
    cost['control_charged_ns_by_phase'][module.INPUT] += delta
    cost['control_observed_ns_by_phase'][module.INPUT] += delta
    cost['charged_total_ns'] = sum(cost['charged_ns_by_phase'].values())
    bundle_ref = f['ref']('latest-bundle', bundle)
    terminal = dict(ledger_root_sha256=cost['ledger_root_sha256'], bundle_sha256=bundle_ref['content_sha256'],
        candidate_complete=False, error_type='ControllerStopped', process_tree_closed=True,
        approval_verified=True, cumulative_charge_lower_bound_ns=cost['charged_total_ns'])
    terminal_ref = f['ref']('latest-terminal', terminal)
    cost['terminal_proof_sha256'] = digest(dict(content_sha256=terminal_ref['content_sha256'],
        file_sha256=terminal_ref['file_sha256']))
    human = dict(id=module.HUMAN_ID, author_type='user', type='message', content=module.HUMAN_CONTENT,
        created_at='2026-10-03T08:33:17.0251761Z', task_id='1fa14b29-0a30-4c40-896e-8a6c21464419',
        session_id='45b5bb5b-dc13-47ca-8f3c-877e1305a12a')
    args = dict(previous_binding_reference=previous_ref, cost_reference=f['ref']('latest-cost', cost),
        bundle_reference=bundle_ref, terminal_reference=terminal_ref, human_message=human)
    return dict(f=f, args=args, old_policy=old_policy, previous=previous,
        cost=cost, bundle=bundle, terminal=terminal, contract=contract)


def test_recovery_grant_retains_all_latest_costs_and_original_science_separately(fixture):
    q = fixture
    record = module.build_policy(**q['args'])
    p = prior.validate_policy(record)
    old = unpack(q['old_policy'])
    assert p['phase_caps_ns'][module.INPUT] == 7200*NS
    assert p['phase_caps_ns'][module.EXPORT] == 3600*NS
    assert p['phase_caps_ns']['method_forecasts'] == 14400*NS
    assert p['phase_caps_ns']['terrain_forecasts'] == 111600*NS
    assert p['total_cap_ns'] == 172800*NS == sum(p['phase_caps_ns'].values())
    assert p['retained_total_charge_ns'] == 14540230_000000
    assert p['retained_charged_ns_by_phase'] == q['cost']['charged_ns_by_phase']
    assert p['phase_caps_ns'][module.INPUT]-p['retained_charged_ns_by_phase'][module.INPUT] == 3600*NS
    assert p['predecessor_root_sha256'] == q['cost']['ledger_root_sha256']
    assert p['scientific_source_root_sha256'] == q['f']['cost']['ledger_root_sha256']
    assert p['scientific_source_approval_sha256'] == q['f']['cost']['approval_sha256']
    assert p['predecessor_approval_sha256'] != p['scientific_source_approval_sha256']
    for key in ('retry_work_id', 'retry_old_reservation_sha256', 'additional_generation_allowance',
                'original_generation_limit', 'effective_generation_limit', 'retained_generation_reservations',
                'retained_success_count', 'retained_success_ids_sha256', 'original_workloads_sha256'):
        assert p[key] == old[key]
    assert p['authorizes_execution'] is False and p['authorizes_successful_reruns'] is False
    contract = budget.contract_for_matrix(q['f']['matrix'], protocol_sha256=p['protocol_sha256'],
        execution_sha256=digest('future synthetic execution'), runtime_manifest_sha256=digest('future runtime'),
        approval_sha256=digest('not real authority'), ledger_directory=Path(q['cost']['ledger_directory']).parent/'unused',
        resource_policy=record)
    assert contract['phase_caps_ns'] == p['phase_caps_ns'] and contract['max_generated_forecasts'] == 11514
    assert contract['workloads'] == q['contract']['workloads']


@pytest.mark.parametrize('change', ['refund_input', 'refund_method', 'new_forecast', 'new_admission',
    'spent_retry', 'missing_success', 'phase_change', 'wrong_head', 'missing_file', 'new_result', 'lower_bound'])
def test_failed_recovery_cannot_refund_relabel_or_add_science(fixture, change):
    q = fixture; cost = deepcopy(q['cost']); args = deepcopy(q['args'])
    if change == 'refund_input':
        cost['charged_ns_by_phase'][module.INPUT] -= NS
        cost['conservatively_charged_ns_by_phase'][module.INPUT] -= NS
        cost['charged_total_ns'] -= NS
    elif change == 'refund_method': cost['charged_ns_by_phase']['method_forecasts'] -= NS
    elif change == 'new_forecast': cost['event_types'][2] = 'reserve'
    elif change == 'new_admission': cost['event_types'][2] = 'bind_partial_imports'
    elif change == 'spent_retry': cost['generated_forecasts_reserved'] += 1
    elif change == 'missing_success': cost['work_dispositions'].pop(next(iter(cost['work_dispositions'])))
    elif change == 'phase_change': cost['phase_caps_ns']['terrain_forecasts'] += NS
    elif change == 'wrong_head': cost['ledger_tip']['last_event_sha256'] = digest('wrong closed tip')
    elif change == 'missing_file': cost['source_file_sha256'].pop('events/000001.json')
    elif change == 'new_result': cost['work_dispositions'][prior.RETRY_WORK_ID] = 'success'
    else:
        terminal = dict(q['terminal'], cumulative_charge_lower_bound_ns=cost['charged_total_ns']+NS)
        args['terminal_reference'] = q['f']['ref']('altered-terminal', terminal)
        cost['terminal_proof_sha256'] = digest({key:args['terminal_reference'][key]
            for key in ('content_sha256', 'file_sha256')})
    args['cost_reference'] = q['f']['ref']('altered-'+change, cost)
    with pytest.raises(ValueError): module.build_policy(**args)


@pytest.mark.parametrize('field,value', [('id',prior.HUMAN_ID), ('author_type','agent'),
    ('content','resume with a fresh two hours'), ('created_at','2026-10-03T08:33:17Z')])
def test_no_new_source_or_broader_human_grant_can_be_inferred(fixture, field, value):
    args = deepcopy(fixture['args']); args['human_message'][field] = value
    with pytest.raises(ValueError): module.build_policy(**args)


def test_rehashed_policy_cannot_buy_another_retry_or_change_a_third_phase(fixture):
    p = unpack(module.build_policy(**fixture['args']))
    for key, value in [('additional_generation_allowance',2), ('retained_total_charge_ns',0),
                       ('scientific_source_approval_sha256',fixture['cost']['approval_sha256'])]:
        with pytest.raises(ValueError): prior.validate_policy(envelope(dict(p, **{key:value})))
    changed = deepcopy(p); changed['phase_caps_ns']['method_forecasts'] += NS
    with pytest.raises(ValueError): prior.validate_policy(envelope(changed))
