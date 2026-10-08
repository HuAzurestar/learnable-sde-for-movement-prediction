"""Synthetic metadata/accounting only; NOT actual B authority or admission."""
from copy import deepcopy
from pathlib import Path

import pytest

from tests.test_pirc17_same_cap_continuation import fixture as continuation_fixture
from experiments.pirc17 import formal_budget as budget, formal_resource_policy as prior
from experiments.pirc17 import formal_resource_predecessor as resource, formal_partial_predecessor as partial
from experiments.pirc17 import formal_recovery_policy as recovery, formal_expansion_policy as module
from experiments.pirc17.protocol_core import digest, envelope, unpack

NS = budget.NANOSECONDS


def human():
    return dict(id=module.HUMAN_ID, author_type='user', type='message', content=module.HUMAN_CONTENT,
        created_at='2026-10-04T00:35:45.1627621Z', task_id='1fa14b29-0a30-4c40-896e-8a6c21464419',
        session_id='45b5bb5b-dc13-47ca-8f3c-877e1305a12a')


@pytest.fixture(scope='module')
def expansion_fixture(continuation_fixture):
    f = continuation_fixture
    ref = f['ref']
    original = f['q']['f']['bundle']
    previous = f['binding']
    previous_ref = ref('expansion-old-binding', unpack(previous))
    directory = Path(f['cost']['ledger_directory']).parent/'closed-input2h'
    runtime = envelope(dict(schema_version='pirc17-concrete-formal-entrypoint-v5-runtime',
        protocol_sha256=original['protocol']['sha256'], matrix_sha256=original['matrix']['sha256'],
        predecessor=previous, predecessor_reference=previous_ref, ledger_directory=str(directory),
        input_paths=unpack(original['runtime'])['input_paths']))
    execution = envelope(dict(protocol_sha256=original['protocol']['sha256'],
        matrix_sha256=original['matrix']['sha256'], runtime_manifest_sha256=runtime['sha256'], software_only=True))
    bundle = dict(schema_version='pirc17-concrete-formal-entrypoint-v5-bundle',
        protocol=original['protocol'], matrix=original['matrix'], runtime=runtime, execution=execution)
    contract = budget.contract_for_matrix(original['matrix'], protocol_sha256=original['protocol']['sha256'],
        execution_sha256=execution['sha256'], runtime_manifest_sha256=runtime['sha256'],
        approval_sha256=digest('synthetic final inputcap approval'), ledger_directory=directory,
        resource_policy=f['record'])
    cost = deepcopy(f['cost'])
    events = ['resource_predecessor', 'control_open', 'control_credit', 'control_close', 'control_terminal_tail']
    tip = dict(root_sha256=digest(contract), event_count=len(events), last_event_sha256=digest('synthetic inputcap tip'))
    cost.update(ledger_directory=str(directory), ledger_root_sha256=digest(contract), ledger_tip=tip,
        head_sha256=digest(tip), execution_sha256=execution['sha256'], runtime_manifest_sha256=runtime['sha256'],
        approval_sha256=contract['approval_sha256'], halted_reason='phase_cap_reached', event_types=events,
        source_file_sha256={name:digest(['synthetic bytes', name]) for name in
            ['ledger.json','head.json',*(f'events/{i:06d}.json' for i in range(len(events)))]})
    delta = 7200*NS-cost['charged_ns_by_phase'][recovery.INPUT]
    for column in ('charged_ns_by_phase', 'conservatively_charged_ns_by_phase',
                   'control_charged_ns_by_phase', 'control_observed_ns_by_phase'):
        cost[column][recovery.INPUT] += delta
    cost['charged_total_ns'] = sum(cost['charged_ns_by_phase'].values())
    bundle_ref = ref('expansion-closed-bundle', bundle)
    terminal = dict(schema_version='pirc17-concrete-formal-entrypoint-v5-launch-terminal',
        ledger_root_sha256=cost['ledger_root_sha256'], bundle_sha256=bundle_ref['content_sha256'],
        candidate_complete=False, error_type='ControllerStopped', process_tree_closed=True,
        approval_verified=True, predecessor_floor_transferred_to_ledger=True,
        predecessor_charge_location='ledger',
        predecessor_charged_ns_by_phase=unpack(f['record'])['retained_charged_ns_by_phase'],
        cumulative_charge_lower_bound_ns=cost['charged_total_ns'])
    terminal_ref = ref('expansion-closed-terminal', terminal)
    cost['terminal_proof_sha256'] = digest({key:terminal_ref[key] for key in ('content_sha256','file_sha256')})
    args = dict(previous_binding_reference=previous_ref, cost_reference=ref('expansion-closed-cost', cost),
        bundle_reference=bundle_ref, terminal_reference=terminal_ref, human_message=human())
    record = module.build_policy(**args)
    return dict(f=f, ref=ref, args=args, cost=cost, record=record, previous=previous, terminal=terminal)


def test_exact_five_additional_hours_preserve_every_other_cap_cost_and_retry(expansion_fixture):
    f = expansion_fixture
    p = prior.validate_policy(f['record'])
    old = unpack(f['f']['record'])
    assert p['phase_caps_ns'] == dict(old['phase_caps_ns'], **{recovery.INPUT:25200*NS})
    assert p['total_cap_ns'] == 190800*NS == sum(p['phase_caps_ns'].values())
    assert p['retained_total_charge_ns'] == 18140230_000000
    assert p['phase_caps_ns'][recovery.INPUT]-p['retained_charged_ns_by_phase'][recovery.INPUT] == 18000*NS
    for key in ('retained_success_count', 'retained_success_ids_sha256', 'retained_generation_reservations',
                'additional_generation_allowance', 'effective_generation_limit', 'retry_work_id',
                'retry_old_reservation_sha256', 'original_workloads_sha256', 'scientific_source_root_sha256',
                'scientific_source_execution_sha256', 'scientific_source_approval_sha256'):
        assert p[key] == old[key]
    assert p['retained_charged_ns_by_phase'] == f['cost']['charged_ns_by_phase']
    assert p['human_source'] == human() and p['same_approved_caps'] is False
    assert p['authorizes_execution'] is False and p['authorizes_successful_reruns'] is False


def test_original_science_latest_floor_and_paid_admission_gate(expansion_fixture, tmp_path, monkeypatch):
    f = expansion_fixture
    binding = resource.build_binding(resource_policy=f['record'], science_reference=unpack(f['previous'])['science_snapshot'])
    _, source_cost, science, original, _ = partial._history(binding)
    assert source_cost == f['f']['q']['f']['cost'] and source_cost != f['cost']
    p = unpack(f['record'])
    directory = tmp_path/'unused-expansion-software-ledger'
    runtime = envelope(dict(protocol_sha256=p['protocol_sha256'], matrix_sha256=p['matrix_sha256'],
        ledger_directory=str(directory), predecessor=binding))
    contract = budget.contract_for_matrix(original['matrix'], protocol_sha256=p['protocol_sha256'],
        execution_sha256=digest('synthetic expanded execution'), runtime_manifest_sha256=runtime['sha256'],
        approval_sha256=digest('not actual B execution authority'), ledger_directory=directory, resource_policy=f['record'])
    _, actual_cost, actual_science = resource.resource_floor(runtime, contract)
    assert actual_cost == f['cost'] and actual_science == science
    monkeypatch.setattr(resource, 'verify_resource_predecessor', lambda record:record)  # Explicit synthetic live-proof seam.
    with budget.Ledger.create(directory, contract) as ledger:
        ledger.import_resource_floor(runtime)
        before = ledger.summary()
        assert before['generated_forecasts_reserved'] == 6272 and before['imported_success_count'] == 6298
        for key in recovery.COST_COLUMNS: assert before[key] == actual_cost[key]
        with pytest.raises(ValueError, match='complete metered domain admission'):
            ledger.reserve(prior.RETRY_WORK_ID)
        assert ledger.summary() == before


@pytest.mark.parametrize('field,value', [('id',recovery.HUMAN_ID), ('author_type','agent'),
    ('content','/goal resume 提高预算 5H 请继续。'), ('created_at','2026-10-04T00:35:45Z'),
    ('task_id','other-task'), ('session_id','other-session'), ('type','tool_call')])
def test_exact_user_source_not_ambiguous_grant_or_agent_proposal(field, value):
    message = human(); message[field] = value
    with pytest.raises(ValueError): module._human_source(message)


@pytest.mark.parametrize('change', ['refund', 'rephase', 'science', 'admission', 'spent_retry',
    'cap', 'halt', 'head', 'missing_tail', 'event_order'])
def test_latest_accounting_cannot_refund_relabel_or_expand_science(expansion_fixture, change):
    f = expansion_fixture; cost = deepcopy(f['cost']); args = deepcopy(f['args'])
    if change == 'refund': cost['charged_ns_by_phase'][recovery.INPUT] -= 1
    elif change == 'rephase': cost['measured_ns_by_phase']['method_forecasts'] += 1
    elif change == 'science': cost['work_dispositions'].pop(next(iter(cost['work_dispositions'])))
    elif change == 'admission': cost['event_types'][2] = 'bind_partial_imports'
    elif change == 'spent_retry': cost['generated_forecasts_reserved'] += 1
    elif change == 'cap': cost['phase_caps_ns'][recovery.INPUT] += NS
    elif change == 'halt': cost['halted_reason'] = 'control_credit_exhausted'
    elif change == 'head': cost['ledger_tip']['last_event_sha256'] = digest('wrong latest tip')
    elif change == 'missing_tail': cost['source_file_sha256'].pop('events/000004.json')
    else: cost['event_types'][1:3] = reversed(cost['event_types'][1:3])
    args['cost_reference'] = f['ref']('expansion-rejected-'+change, cost)
    with pytest.raises(ValueError): module.build_policy(**args)


def test_rehashed_policy_cannot_change_grant_total_retry_or_original_owner(expansion_fixture):
    payload = unpack(expansion_fixture['record'])
    for key,value in [('total_cap_ns',172800*NS), ('additional_generation_allowance',2),
                       ('retained_total_charge_ns',0), ('scientific_source_approval_sha256',digest('relabel'))]:
        with pytest.raises(ValueError): prior.validate_policy(envelope(dict(payload, **{key:value})))


def test_same_B_grant_cannot_be_applied_twice(expansion_fixture):
    f = expansion_fixture
    binding = resource.build_binding(resource_policy=f['record'], science_reference=unpack(f['previous'])['science_snapshot'])
    args = dict(f['args'], previous_binding_reference=f['ref']('reapplied-B-binding', unpack(binding)))
    with pytest.raises(ValueError, match='closed existing same-cap continuation'):
        module.build_policy(**args)


def test_projection_reuses_only_same_call_policy_and_public_reader_still_validates(expansion_fixture, tmp_path, monkeypatch):
    """Explicit synthetic policy seam: verify routing, not actual admission."""
    f = expansion_fixture; calls = []
    def checked(record):
        calls.append(record['sha256'])
        return deepcopy(unpack(record))
    monkeypatch.setattr(prior, 'validate_policy', checked)
    matrix = f['f']['q']['f']['matrix']
    p = unpack(f['record'])
    args = dict(protocol_sha256=p['protocol_sha256'], execution_sha256=digest('synthetic projection'),
        runtime_manifest_sha256=digest('synthetic runtime'), approval_sha256=digest('synthetic approval'),
        ledger_directory=tmp_path/'synthetic-reuse', resource_policy=f['record'])
    contract = budget.contract_for_matrix(matrix, **args)
    assert calls == [f['record']['sha256']]
    assert budget.contract_for_matrix(matrix, **args) == contract
    assert calls == [f['record']['sha256']]*2  # No cross-call memo.
    assert budget.validate_contract(contract) == contract
    assert calls == [f['record']['sha256']]*3  # Public reader independently validates.
    mismatch = deepcopy(p); mismatch['additional_generation_allowance'] += 1
    with pytest.raises(ValueError, match='same-call checked policy differs'):
        budget._validate_contract(contract, checked_policy=mismatch)
    with pytest.raises(TypeError): budget.validate_contract(contract, checked_policy=p)
    with pytest.raises(TypeError): budget.contract_for_matrix(matrix, **args, checked_policy=p)
