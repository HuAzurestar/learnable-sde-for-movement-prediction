"""Frozen-plan CLI/deadline integration with explicit synthetic source seams.

No real source restoration, native worker launch or human authority. The real
entrypoint, budget projection, preparation and bootstrap clock guards run.
"""
from copy import deepcopy
from pathlib import Path
import sys
import time

import pytest

from tests.test_pirc17_formal_entrypoint import scientific_contract
from tests.test_pirc17_formal_environment import actual_environment
from tests.test_pirc17_recovery_policy import make_recovery_fixture
from tests.test_pirc17_resource_entrypoint import prepare
from experiments.pirc17 import formal_budget as budget, formal_controller as control
from experiments.pirc17 import formal_entrypoint as entry, formal_partial_launch as launch
from experiments.pirc17 import formal_recovery_policy as recovery, formal_resource_predecessor as resource
from experiments.pirc17 import formal_session as native
from experiments.pirc17.protocol_core import digest, envelope, publish, read_json, unpack


def test_frozen_plan_cli_projects_recovery_grant_and_never_resets_input_deadline(
        scientific_contract, actual_environment, tmp_path, monkeypatch, capsys):
    protocol, matrix = scientific_contract
    f = make_recovery_fixture(tmp_path/'synthetic', protocol=protocol, matrix=matrix)
    p = recovery.build_policy(**f['args'])
    binding = resource.build_binding(resource_policy=p, science_reference=unpack(f['previous'])['science_snapshot'])
    binding_path, _ = publish(tmp_path/'binding', unpack(binding))
    metadata = dict(source=f['f'], binding=binding, binding_path=binding_path, policy=p)
    path, record, b, r, auth = prepare(metadata, tmp_path/'unused', actual_environment, monkeypatch, capsys)
    assert b['protocol'] == protocol and b['matrix'] == matrix
    assert r['schema_version'] == entry.RESOURCE_VERSION+'-runtime'
    assert unpack(r['predecessor'])['schema_version'] == resource.RECOVERY_VERSION
    assert unpack(b['execution'])['phase_caps_seconds']['input_qualification_and_binding'] == 3600
    contract = entry.budget_contract(b, approval_sha256=auth['approval_sha256'])
    assert contract['phase_caps_ns'][recovery.INPUT] == 7200*budget.NANOSECONDS
    assert contract['phase_caps_ns'][recovery.EXPORT] == 3600*budget.NANOSECONDS
    assert contract['total_cap_ns'] == 172800*budget.NANOSECONDS
    assert len(contract['workloads']) == 11659 and contract['max_generated_forecasts'] == 11514
    charged, root = launch.launch_floor(b, approval_sha256=auth['approval_sha256'])
    assert root == f['cost']['ledger_root_sha256'] and charged == f['cost']['charged_ns_by_phase']
    assert sum(charged.values()) == 14540230_000000
    directory = tmp_path/'unused/ledger'
    assert not directory.exists() and not entry.launch_directory(directory).exists()
    monkeypatch.setattr(resource, 'verify_resource_predecessor', lambda value:value)  # Explicit synthetic source seam.
    with budget.Ledger.create(directory, contract) as ledger:
        ledger.import_resource_floor(b['runtime'])
        assert ledger._state.predecessor_floor['requires_metered_source_admission'] is True
        export_work = next(w for w in contract['workloads'] if w['phase'] == recovery.EXPORT)
        assert export_work['max_active_ns'] == 7200*budget.NANOSECONDS
        with pytest.raises(ValueError, match='complete metered domain admission'):
            ledger._state.reservation(export_work['work_id'])
        # Read-only arithmetic AFTER a hypothetical admission, explicit test
        # seam only. No reservation is journalled and no exporter is invoked.
        ledger._state.partial_imports = {'SOFTWARE not actual admission': True}
        try:
            allocation = ledger._state.reservation(export_work['work_id'])
            assert allocation['reserved_ns'] == 3600*budget.NANOSECONDS
        finally:
            ledger._state.partial_imports = None
        session_dir = directory/'session-000001'; session_dir.mkdir()
        session = dict(schema_version=native.VERSION, directory=str(session_dir), job_name='PIRC17-FORMAL-'+'a'*32,
            ledger_directory=str(directory), ledger_root_sha256=ledger.root_sha256,
            execution_sha256=b['execution']['sha256'], worker_command=entry.worker_command(
                path, record['sha256'], auth, b['runtime']))
        native._write(session_dir/'session.json', session)
        started = time.monotonic_ns()
        remaining = 3600*budget.NANOSECONDS
        (directory/'controls').mkdir()
        start = control._publish(directory/'controls', dict(schema_version=control.VERSION+'-control',
            ledger_root_sha256=ledger.root_sha256, phase=entry.PHASE_ORDER[0], started_ns=started,
            phase_deadline_ns=started+remaining, worker_job_name=session['job_name'],
            session_directory=str(session_dir), worker_command=session['worker_command']))
        ledger.open_control(start['sha256'], phase=entry.PHASE_ORDER[0], credit_ns=5*budget.NANOSECONDS)
        request = dict(schema_version=native.VERSION+'-bootstrap-request', session_sha256=digest(session),
            control_sha256=start['sha256'], started_ns=started, deadline_ns=started+remaining)
        native._write(session_dir/'bootstrap-request.json', request)
        native._bootstrap_request(envelope(request), session_sha256=digest(session), session=session, contract=contract)
        with pytest.raises(ValueError):
            native._bootstrap_request(envelope(dict(request, deadline_ns=started+7200*budget.NANOSECONDS)),
                session_sha256=digest(session), session=session, contract=contract)
        assert ledger.summary()['generated_forecasts_reserved'] == 6272
        assert ledger.summary()['imported_success_count'] == 6298


def test_unapproved_contract_still_rejects_work_above_its_phase(tmp_path):
    from tests.test_pirc17_formal_budget import fixture_contract
    contract = fixture_contract(tmp_path/'not-an-experiment')
    contract['phase_caps_ns']['fit'] = 500
    contract['total_cap_ns'] = 1000
    with pytest.raises(ValueError, match='work cap exceeds its own phase'):
        budget.validate_contract(contract)


@pytest.mark.parametrize('input_cap,old_input', [(7200_000000000, 3600_000000000),
                                               (3600_000000000, 2139205_000000)])
def test_public_run_uses_approved_input_cap_and_preserves_untransferred_startup_charge(
        tmp_path, monkeypatch, input_cap, old_input):
    """Real CLI/claim/terminal, explicit metadata/authority/no-ledger seams.

    Stop BEFORE constructing any real ledger, worker or scientific consumer.
    This control-flow unit does NOT validate a real policy or input source.
    Full frozen-plan policy/projection validation is tested separately above.
    It isolates both actual run guards and conservative claim accounting.
    """
    directory = tmp_path/'unused/ledger'
    directory.parent.mkdir()
    runtime = envelope(dict(schema_version=entry.RESOURCE_VERSION+'-runtime',
        ledger_directory=str(directory.resolve()), working_directory=str(Path.cwd().resolve()),
        environment=envelope(dict(python_executable=dict(path=str(Path(sys.executable).resolve()))))))
    b = dict(runtime=runtime, protocol=envelope(dict(software_fixture_only=True)),
             matrix=envelope(dict(software_fixture_only=True)), execution=envelope(dict(software_fixture_only=True)))
    path, record = publish(tmp_path/'bundle', b)
    approval_path, fake = publish(tmp_path/'SOFTWARE-authority', dict(software_only=True))
    auth = entry.authority_paths(approval_path=approval_path, approval_sha256=fake['sha256'],
        test_path=approval_path, review_path=approval_path, ledger_directory=directory)
    charges = {phase:0 for phase in entry.PHASE_ORDER}
    charges.update(input_qualification_and_binding=old_input, method_forecasts=10800027_000000)
    synthetic_contract = dict(phase_caps_ns={recovery.INPUT:input_cap}, software_fixture_only=True)
    monkeypatch.setattr(entry, 'validate_bundle', lambda value:(unpack(value), unpack(runtime)))
    monkeypatch.setattr(entry, 'budget_contract', lambda *args, **kwargs:deepcopy(synthetic_contract))
    monkeypatch.setattr(launch, 'launch_floor', lambda *args, **kwargs:(deepcopy(charges), digest('SOFTWARE source root')))
    monkeypatch.setattr(entry, '_approval', lambda *args: None)  # Explicit SOFTWARE authority seam.
    calls = []
    class SoftwareBoundary(RuntimeError):
        pass
    def stop_before_ledger(directory, contract):
        assert contract == synthetic_contract
        calls.append(str(directory))
        raise SoftwareBoundary('software stop before actual ledger or worker')
    monkeypatch.setattr(budget.Ledger, 'create', staticmethod(stop_before_ledger))
    args = ['run', '--bundle', str(path), '--bundle-sha256', record['sha256'],
        '--approval', auth['approval_path'], '--approval-sha256', auth['approval_sha256'],
        '--test', auth['test_path'], '--review', auth['review_path']]
    with pytest.raises(SoftwareBoundary, match='software stop before actual ledger'):
        entry.main(args)
    claim = entry.launch_directory(directory)
    start, terminal = (unpack(read_json(claim/name)) for name in ('start.json', 'terminal.json'))
    assert calls == [str(directory)] and not directory.exists()
    remaining = input_cap-old_input
    assert start['startup_reservation_ns'] == remaining  # NOT fresh phase allowance.
    assert start['predecessor_charged_ns_by_phase'] == terminal['predecessor_charged_ns_by_phase'] == charges
    assert terminal['approval_verified'] and terminal['process_tree_closed']
    assert terminal['error_type'] == 'SoftwareBoundary' and not terminal['candidate_complete']
    assert not terminal['startup_transferred_to_ledger'] and not terminal['predecessor_floor_transferred_to_ledger']
    assert terminal['startup_conservative_charge_ns'] >= remaining
    assert terminal['cumulative_charge_lower_bound_ns'] >= sum(charges.values())+remaining
    with pytest.raises(FileExistsError):
        entry.main(args)
    assert calls == [str(directory)]  # Consumed SOFTWARE claim cannot silently restart.
