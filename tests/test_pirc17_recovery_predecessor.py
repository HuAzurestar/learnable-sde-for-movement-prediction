"""Synthetic accounting/ownership tests, NOT actual launch/source admission."""
from copy import deepcopy
from pathlib import Path

import pytest

from tests.test_pirc17_recovery_policy import fixture
from experiments.pirc17 import formal_budget as budget, formal_partial_predecessor as partial
from experiments.pirc17 import formal_resource_policy as prior, formal_recovery_policy as recovery
from experiments.pirc17 import formal_resource_predecessor as module
from experiments.pirc17.protocol_core import digest, envelope, unpack


def bound(f):
    p = recovery.build_policy(**f['args'])
    old = unpack(f['previous'])
    binding = module.build_binding(resource_policy=p, science_reference=old['science_snapshot'])
    return p, binding


def test_history_uses_original_model_scope_but_floor_debits_latest_accounting(fixture, tmp_path):
    f = fixture; p, binding = bound(f)
    b, source, science, original, old_contract = partial._history(binding)
    assert b['schema_version'] == module.RECOVERY_VERSION
    assert source == f['f']['cost'] and source != f['cost']
    assert b['ledger_directory'] == source['ledger_directory']
    assert science['original_execution_sha256'] == source['execution_sha256']
    assert science['original_approval_sha256'] == source['approval_sha256']
    assert old_contract == f['f']['contract'] and original == f['f']['bundle']
    directory = tmp_path/'unused-successor'
    runtime = envelope(dict(protocol_sha256=p['payload']['protocol_sha256'],
        matrix_sha256=p['payload']['matrix_sha256'], ledger_directory=str(directory), predecessor=binding))
    contract = budget.contract_for_matrix(f['f']['matrix'], protocol_sha256=p['payload']['protocol_sha256'],
        execution_sha256=digest('new scope, no empirical execution'), runtime_manifest_sha256=runtime['sha256'],
        approval_sha256=digest('software fixture is not human approval'), ledger_directory=directory, resource_policy=p)
    _, cost, actual_science = module.resource_floor(runtime, contract)
    assert cost == f['cost'] and cost != source and actual_science == science
    assert cost['charged_total_ns'] == 14540230_000000


def test_latest_floor_retains_costs_successes_retry_and_compulsory_admission(fixture, tmp_path, monkeypatch):
    f = fixture; p, binding = bound(f)
    directory = tmp_path/'synthetic-budget-ledger'
    runtime = envelope(dict(protocol_sha256=p['payload']['protocol_sha256'],
        matrix_sha256=p['payload']['matrix_sha256'], ledger_directory=str(directory), predecessor=binding))
    contract = budget.contract_for_matrix(f['f']['matrix'], protocol_sha256=p['payload']['protocol_sha256'],
        execution_sha256=digest('synthetic-new-execution'), runtime_manifest_sha256=runtime['sha256'],
        approval_sha256=digest('not real authority'), ledger_directory=directory, resource_policy=p)
    calls = []
    # Explicit software-unit seam; full actual closed-source verification is
    # mandatory in production and is not declared passed by this test.
    monkeypatch.setattr(module, 'verify_resource_predecessor', lambda record:calls.append(record['sha256']))
    with budget.Ledger.create(directory, contract) as ledger:
        ledger.import_resource_floor(runtime)
        summary = ledger.summary()
        assert summary['predecessor_floor']['requires_metered_source_admission'] is True
        assert summary['predecessor_floor']['ledger_root_sha256'] == f['cost']['ledger_root_sha256']
        assert summary['predecessor_floor']['retry_old_reservation_sha256'] == p['payload']['retry_old_reservation_sha256']
        for column in recovery.COST_COLUMNS:
            assert summary[column] == f['cost'][column]
        assert summary['imported_success_count'] == 6298 and summary['generated_forecasts_reserved'] == 6272
        assert ledger._state.imported_success == partial._history(f['previous'])[2]['completed_sources']
        assert ledger._state.carried_generation_reservations == {}
        assert ledger._state.halted is None and ledger._state.status.get(prior.RETRY_WORK_ID) is None
        with pytest.raises(ValueError, match='complete metered domain admission'): ledger.reserve(prior.RETRY_WORK_ID)
        assert ledger.summary() == summary
        with pytest.raises(ValueError): ledger.import_resource_floor(runtime)
        assert calls == [binding['sha256']]
        root, tip = ledger.root_sha256, ledger.tip
    def forbidden(*args, **kwargs): pytest.fail('cold floor replay invoked live verification')
    monkeypatch.setattr(module, 'verify_resource_predecessor', forbidden)
    with budget.Ledger.open(directory, expected_root_sha256=root, expected_tip=tip) as ledger:
        assert ledger.summary() == summary


@pytest.mark.parametrize('change', ['ledger', 'snapshot', 'previous', 'legacy_version'])
def test_recovery_cannot_relabel_science_as_latest_attempt(fixture, change):
    f = fixture; _, binding = bound(f)
    value = deepcopy(unpack(binding))
    if change == 'ledger': value['ledger_directory'] = f['cost']['ledger_directory']
    elif change == 'snapshot': value['science_snapshot'] = f['args']['cost_reference']
    elif change == 'previous': value['science_predecessor'] = f['args']['bundle_reference']
    else: value['schema_version'] = module.METADATA_VERSION
    with pytest.raises(ValueError): partial._history(envelope(value))


def test_new_policy_cannot_bind_a_different_scientific_inventory(fixture):
    f = fixture; p = recovery.build_policy(**f['args'])
    with pytest.raises(ValueError): module.build_binding(resource_policy=p, science_reference=f['args']['cost_reference'])


@pytest.mark.parametrize('fault', [None, 'event_bytes', 'cost_refund', 'active_latest'])
def test_live_verifier_checks_both_scopes_and_latest_journal_without_array_replay(fixture, monkeypatch, fault):
    """Explicit synthetic file/native/replay seams, NOT actual closure proof."""
    f = fixture; args = deepcopy(f['args']); latest = deepcopy(f['cost'])
    job = 'PIRC17-FORMAL-'+'c'*32
    control = envelope(dict(ledger_root_sha256=latest['ledger_root_sha256'], worker_job_name=job))
    event = envelope(dict(row=dict(control_sha256=control['sha256'])))
    latest['source_file_sha256']['events/000001.json'] = digest(event)
    args['cost_reference'] = f['f']['ref']('latest-synthetic-control-binding-'+str(fault), latest)
    p = recovery.build_policy(**args)
    binding = module.build_binding(resource_policy=p, science_reference=unpack(f['previous'])['science_snapshot'])
    source = f['f']['cost']; old_root = Path(source['ledger_directory'])
    original_files = {old_root/'ledger.json': envelope(f['f']['contract']),
                      old_root/'head.json': envelope(source['ledger_tip'])}
    monkeypatch.setattr(module, 'read_json', lambda path:original_files[Path(path)])
    cost_root = Path(latest['ledger_directory'])
    files = {cost_root/'events/000001.json': (event, digest('changed bytes') if fault == 'event_bytes' else digest(event)),
             cost_root/'controls'/(control['sha256']+'.json'): (control, digest(control))}
    monkeypatch.setattr(module.costs, '_read_bound', lambda path, bound:files[Path(path)])
    replays, jobs = [], []
    def inspect(directory, **expected):
        assert Path(directory) == cost_root
        assert expected == dict(expected_root_sha256=latest['ledger_root_sha256'],
            expected_head_sha256=latest['head_sha256'], expected_terminal_proof_sha256=latest['terminal_proof_sha256'])
        replays.append(expected)
        returned = deepcopy(latest)
        if fault == 'cost_refund': returned['charged_total_ns'] -= 1
        return envelope(returned)
    monkeypatch.setattr(module.costs, 'inspect_closed_ledger', inspect)
    from experiments.pirc17 import formal_partial_costs, formal_partial_science
    def closed(name):
        jobs.append(name)
        if fault == 'active_latest' and name == job:
            raise ValueError('synthetic latest Job still active')
    monkeypatch.setattr(formal_partial_costs, '_closed_job', closed)
    monkeypatch.setattr(formal_partial_science, 'inspect_partial_science',
        lambda *a, **kw:pytest.fail('scientific domains must remain in paid bootstrap'))
    if fault is None:
        assert module.verify_resource_predecessor(binding) == binding
        assert jobs == ['PIRC17-FORMAL-'+'a'*32, job, job, 'PIRC17-FORMAL-'+'a'*32]
        assert len(replays) == 1
    else:
        with pytest.raises(ValueError): module.verify_resource_predecessor(binding)


def test_second_reallocation_cannot_be_reused_as_a_third_grant(fixture):
    f = fixture; p = recovery.build_policy(**f['args'])
    previous = dict(unpack(f['previous']), resource_policy=p)
    args = dict(f['args'], previous_binding_reference=f['f']['ref']('unapproved-third-grant-source', previous))
    with pytest.raises(ValueError, match='only the prior metadata recovery'):
        recovery.build_policy(**args)
