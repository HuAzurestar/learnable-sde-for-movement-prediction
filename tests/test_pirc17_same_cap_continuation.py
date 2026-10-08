"""Full-size SYNTHETIC metadata/accounting; not native/domain qualification."""
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.pirc17 import formal_budget as budget, formal_resource_policy as prior
from experiments.pirc17 import formal_resource_predecessor as resource, formal_partial_predecessor as partial
from experiments.pirc17 import formal_recovery_policy as recovery, formal_continuation_policy as module
from experiments.pirc17 import formal_entrypoint as entry, formal_partial_launch as launch
from experiments.pirc17.protocol_core import digest, envelope, unpack
from tests.test_pirc17_recovery_policy import make_recovery_fixture

NS = budget.NANOSECONDS


@pytest.fixture(scope='module')
def fixture(tmp_path_factory):
    q = make_recovery_fixture(tmp_path_factory.mktemp('same-cap-software'))
    approved = recovery.build_policy(**q['args'])
    previous = resource.build_binding(resource_policy=approved,
        science_reference=unpack(q['previous'])['science_snapshot'])
    ref = q['f']['ref']
    previous_ref = ref('same-cap-previous-binding', unpack(previous))
    directory = Path(q['cost']['ledger_directory']).parent/'closed-credit-failure'
    runtime = envelope(dict(schema_version=entry.RESOURCE_VERSION+'-runtime',
        protocol_sha256=q['f']['protocol']['sha256'], matrix_sha256=q['f']['matrix']['sha256'],
        predecessor=previous, predecessor_reference=previous_ref, ledger_directory=str(directory),
        input_paths=unpack(q['bundle']['runtime'])['input_paths']))
    execution = envelope(dict(protocol_sha256=q['f']['protocol']['sha256'],
        matrix_sha256=q['f']['matrix']['sha256'], runtime_manifest_sha256=runtime['sha256'], software_only=True))
    bundle = dict(schema_version=entry.RESOURCE_VERSION+'-bundle',
        protocol=q['f']['protocol'], matrix=q['f']['matrix'], execution=execution, runtime=runtime)
    contract = budget.contract_for_matrix(bundle['matrix'], protocol_sha256=bundle['protocol']['sha256'],
        execution_sha256=execution['sha256'], runtime_manifest_sha256=runtime['sha256'],
        approval_sha256=digest('synthetic credit failure approval'), ledger_directory=directory, resource_policy=approved)
    cost = deepcopy(q['cost'])
    events = ['resource_predecessor', 'control_open', 'control_credit', 'halt',
              'control_close', 'control_terminal_tail', 'control_late_overrun']
    tip = dict(root_sha256=digest(contract), event_count=len(events), last_event_sha256=digest('synthetic credit tip'))
    cost.update(ledger_directory=str(directory), ledger_root_sha256=digest(contract),
        ledger_tip=tip, head_sha256=digest(tip), execution_sha256=execution['sha256'],
        runtime_manifest_sha256=runtime['sha256'], approval_sha256=contract['approval_sha256'],
        phase_caps_ns=contract['phase_caps_ns'], halted_reason='control_credit_exhausted', event_types=events,
        source_file_sha256={name:digest(['software bytes', name]) for name in
            ['ledger.json','head.json',*(f'events/{i:06d}.json' for i in range(len(events)))]})
    delta = 3046266_000000
    for name in ('charged_ns_by_phase', 'conservatively_charged_ns_by_phase',
                 'control_charged_ns_by_phase', 'control_observed_ns_by_phase'):
        cost[name][recovery.INPUT] += delta
    cost['charged_total_ns'] = sum(cost['charged_ns_by_phase'].values())
    bundle_ref = ref('same-cap-bundle', bundle)
    terminal = dict(schema_version=entry.RESOURCE_VERSION+'-launch-terminal',
        ledger_root_sha256=cost['ledger_root_sha256'], bundle_sha256=bundle_ref['content_sha256'],
        candidate_complete=False, error_type='ControllerStopped', process_tree_closed=True,
        approval_verified=True, predecessor_floor_transferred_to_ledger=True,
        predecessor_charge_location='ledger', predecessor_charged_ns_by_phase=unpack(approved)['retained_charged_ns_by_phase'],
        cumulative_charge_lower_bound_ns=cost['charged_total_ns'])
    terminal_ref = ref('same-cap-terminal', terminal)
    cost['terminal_proof_sha256'] = digest({key:terminal_ref[key] for key in ('content_sha256','file_sha256')})
    args = dict(previous_binding_reference=previous_ref, cost_reference=ref('same-cap-cost', cost),
        bundle_reference=bundle_ref, terminal_reference=terminal_ref)
    record = module.build_policy(**args)
    binding = resource.build_binding(resource_policy=record, science_reference=unpack(previous)['science_snapshot'])
    return dict(q=q, ref=ref, args=args, approved=approved, cost=cost, terminal=terminal,
        record=record, binding=binding, previous=previous, contract=contract)


def test_new_floor_preserves_latest_charge_and_original_scientific_owner(fixture, tmp_path, monkeypatch):
    f = fixture
    p = prior.validate_policy(f['record'])
    old = unpack(f['approved'])
    assert p['phase_caps_ns'] == old['phase_caps_ns']
    assert p['total_cap_ns'] == 172800*NS
    assert p['human_source'] == old['human_source']  # No new human grant invented.
    assert p['retained_total_charge_ns'] == 17586496_000000
    assert p['phase_caps_ns'][recovery.INPUT] - p['retained_charged_ns_by_phase'][recovery.INPUT] == 553734_000000
    for key in ('effective_generation_limit','retained_generation_reservations','retry_work_id',
                'additional_generation_allowance','scientific_source_root_sha256','scientific_source_approval_sha256'):
        assert p[key] == old[key]
    b, science_cost, science, original, _ = partial._history(f['binding'])
    assert b['schema_version'] == resource.CONTINUATION_VERSION
    assert science_cost == f['q']['f']['cost'] and science_cost != f['cost']
    assert original == f['q']['f']['bundle']
    directory = tmp_path/'unused-software-ledger'
    runtime = envelope(dict(schema_version=entry.RESOURCE_VERSION+'-runtime',
        protocol_sha256=p['protocol_sha256'], matrix_sha256=p['matrix_sha256'],
        ledger_directory=str(directory), predecessor=f['binding']))
    contract = budget.contract_for_matrix(original['matrix'], protocol_sha256=p['protocol_sha256'],
        execution_sha256=digest('new synthetic scope'), runtime_manifest_sha256=runtime['sha256'],
        approval_sha256=digest('not genuine authority'), ledger_directory=directory, resource_policy=f['record'])
    _, actual_cost, actual_science = resource.resource_floor(runtime, contract)
    assert actual_cost == f['cost'] and actual_science == science
    assert entry.runtime_resource_policy(unpack(runtime)) == f['record']
    assert launch._floor(runtime, contract)[1] == actual_cost
    # Explicit live-proof seam: do not claim OS/source admission passed.
    monkeypatch.setattr(resource, 'verify_resource_predecessor', lambda record:record)
    with budget.Ledger.create(directory, contract) as ledger:
        ledger.import_resource_floor(runtime)
        before = ledger.summary()
        assert before['predecessor_floor']['requires_metered_source_admission'] is True
        assert before['generated_forecasts_reserved'] == 6272 and before['imported_success_count'] == 6298
        for key in recovery.COST_COLUMNS: assert before[key] == actual_cost[key]
        with pytest.raises(ValueError, match='complete metered domain admission'):
            ledger.reserve(prior.RETRY_WORK_ID)
        assert ledger.summary() == before


@pytest.mark.parametrize('change', ['refund', 'science', 'admission', 'spent_retry', 'cap', 'halt', 'head', 'missing_tail'])
def test_changed_closed_cost_cannot_buy_more_time_or_admit_science(fixture, change):
    f = fixture; cost = deepcopy(f['cost']); args = dict(f['args'])
    if change == 'refund': cost['charged_ns_by_phase'][recovery.INPUT] -= 1
    elif change == 'science': cost['work_dispositions'].pop(next(iter(cost['work_dispositions'])))
    elif change == 'admission': cost['event_types'][2] = 'partial_imports'
    elif change == 'spent_retry': cost['generated_forecasts_reserved'] += 1
    elif change == 'cap': cost['phase_caps_ns'][recovery.INPUT] += NS
    elif change == 'halt': cost['halted_reason'] = 'phase_cap_reached'
    elif change == 'head': cost['ledger_tip']['last_event_sha256'] = digest('wrong tip')
    else: cost['source_file_sha256'].pop('events/000006.json')
    args['cost_reference'] = f['ref']('same-cap-rejected-'+change, cost)
    with pytest.raises(ValueError): module.build_policy(**args)


def test_rehashed_policy_cannot_change_caps_retry_authority_or_cost(fixture):
    payload = unpack(fixture['record'])
    for key,value in [('additional_generation_allowance',2), ('retained_total_charge_ns',0),
                       ('human_source',dict(payload['human_source'], id='invented decision'))]:
        with pytest.raises(ValueError): prior.validate_policy(envelope(dict(payload, **{key:value})))


def test_latest_failed_scope_cannot_replace_original_science(fixture):
    b = deepcopy(unpack(fixture['binding']))
    b['ledger_directory'] = fixture['cost']['ledger_directory']
    with pytest.raises(ValueError): partial._history(envelope(b))
    with pytest.raises(ValueError):
        resource.build_binding(resource_policy=fixture['record'], science_reference=fixture['args']['cost_reference'])
