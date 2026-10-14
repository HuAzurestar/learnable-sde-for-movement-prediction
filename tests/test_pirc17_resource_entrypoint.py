"""Frozen scientific plan, synthetic closed sources; NOT launch qualification.

Positive authority/environment/domain/native-Job seams are explicit. Real
prepare CLI, source seals, ledger, bootstrap-request and runtime readers remain
unmodified. No real saved data/model/array or empirical kernel is opened.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import os
import sys
import time

import pytest

from tests.test_pirc17_formal_entrypoint import scientific_contract
from tests.test_pirc17_formal_environment import actual_environment
from tests.test_pirc17_resource_policy import make_policy_fixture
from tests.test_pirc17_resource_predecessor import science_metadata
from experiments.pirc17 import formal_budget as budget, formal_controller as control
from experiments.pirc17 import formal_entrypoint as entry, formal_partial_launch as launch
from experiments.pirc17 import formal_partial_predecessor as partial
from experiments.pirc17 import formal_resource_policy as policy, formal_resource_predecessor as resource
from experiments.pirc17 import formal_session as native
from experiments.pirc17.protocol_core import digest, envelope, publish, read_json, unpack

NS = budget.NANOSECONDS


@pytest.fixture(scope='module')
def metadata(scientific_contract, tmp_path_factory):
    root = tmp_path_factory.mktemp('v5-SOFTWARE-metadata').resolve()
    p, m = scientific_contract
    f = make_policy_fixture(root, protocol=p, matrix=m)
    saved = science_metadata(f)
    approved = policy.build_policy(**f['args'])
    binding = resource.build_binding(resource_policy=approved, science_reference=f['ref']('science', saved))
    path, _ = publish(root/'binding', unpack(binding))
    return dict(source=f, binding=binding, binding_path=path, policy=approved, saved=saved)


def prepare(metadata, tmp_path, actual_environment, monkeypatch, capsys):
    from experiments.pirc17 import formal_environment
    class Environment:
        def __init__(self, p): self.hardware = deepcopy(unpack(actual_environment)['hardware'])
        def identity(self): return actual_environment
        def verify(self, value): assert value == actual_environment
        def check(self): return self.hardware
    monkeypatch.setattr(formal_environment, 'ConfiguredRuntime', Environment)
    assert entry.main(['prepare-resource', '--output-directory', str(tmp_path/'bundle'),
        '--ledger-directory', str(tmp_path/'ledger'), '--predecessor-reference',
        str(metadata['binding_path'])]) == 0
    report = json.loads(capsys.readouterr().out)
    path = Path(report['path']); record = read_json(path)
    assert record['sha256'] == report['bundle_sha256']
    assert report['final_eval_reads'] == 0 and report['final_eval_authorized'] is False
    ap, receipt = publish(tmp_path/'SOFTWARE-authority', dict(software_only=True))
    auth = entry.authority_paths(approval_path=ap, approval_sha256=receipt['sha256'],
        test_path=ap, review_path=ap, ledger_directory=tmp_path/'ledger')
    b, r = entry.validate_bundle(record)
    return path, record, b, r, auth


def metadata_predecessor(metadata, tmp_path):
    from experiments.pirc17 import formal_source_metadata
    s = dict(metadata['saved'], schema_version=formal_source_metadata.VERSION,
             resource_policy_sha256=metadata['policy']['sha256'],
             forecast_status_counts={'not_replayed':6271}, **formal_source_metadata.FLAGS)
    path, record = publish(tmp_path/'metadata-source', s)
    binding = resource.build_binding(resource_policy=metadata['policy'], science_reference=partial.reference(path))
    path, _ = publish(tmp_path/'metadata-binding', unpack(binding))
    return dict(metadata, saved=s, binding=binding, binding_path=path)


@pytest.mark.parametrize('metadata_mode', [False, True])
def test_public_resource_preparation_preserves_science_and_projects_only_approved_budget(
        metadata, tmp_path, actual_environment, monkeypatch, capsys, metadata_mode):
    if metadata_mode: metadata = metadata_predecessor(metadata, tmp_path)
    path, record, b, r, auth = prepare(metadata, tmp_path, actual_environment, monkeypatch, capsys)
    f = metadata['source']; e = unpack(b['execution'])
    assert b['protocol'] == f['protocol'] and b['matrix'] == f['matrix']
    assert r['schema_version'] == entry.RESOURCE_VERSION+'-runtime'
    assert unpack(record)['schema_version'] == entry.RESOURCE_VERSION+'-bundle'
    assert r['input_paths'] == unpack(f['bundle']['runtime'])['input_paths']
    assert e['phase_caps_seconds']['method_forecasts'] == 10800  # Original scientific plan seal.
    assert e['max_generated_forecasts'] == 11513 and e['scientific_forecasts'] == 11020
    contract = entry.budget_contract(b, approval_sha256=auth['approval_sha256'])
    assert contract['phase_caps_ns']['method_forecasts'] == 14400*NS
    assert contract['phase_caps_ns']['terrain_forecasts'] == 111600*NS
    assert contract['total_cap_ns'] == 172800*NS and contract['max_generated_forecasts'] == 11514
    assert len(contract['workloads']) == 11659 and contract['max_attempts_per_item'] == 1
    assert contract['resource_policy'] == metadata['policy']
    if metadata_mode:
        assert unpack(r['predecessor'])['schema_version'] == resource.METADATA_VERSION
    charges, root = launch.launch_floor(b, approval_sha256=auth['approval_sha256'])
    assert charges == f['cost']['charged_ns_by_phase'] and sum(charges.values()) == 13079435_000000
    assert root == f['cost']['ledger_root_sha256']
    assert entry.worker_command(path, record['sha256'], auth, b['runtime'])[1:5] == ['-u','-m',entry.MODULE,'worker']
    assert not (tmp_path/'ledger').exists() and not entry.launch_directory(tmp_path/'ledger').exists()
    with pytest.raises(ValueError, match='exact distinct predecessor'):
        launch.prepare_partial(output_directory=tmp_path/'wrong', ledger_directory=tmp_path/'wrong-ledger',
                               predecessor_reference_path=metadata['binding_path'])


def test_resource_native_bootstrap_routes_distinct_floor_and_original_input_deadline(
        metadata, tmp_path, actual_environment, monkeypatch, capsys):
    from experiments.pirc17 import formal_worker
    path, record, b, r, auth = prepare(metadata, tmp_path, actual_environment, monkeypatch, capsys)
    monkeypatch.setattr(entry, '_approval', lambda *a: None)  # SOFTWARE authority seam.
    monkeypatch.setattr(resource, 'verify_resource_predecessor', lambda b: b)  # No real source/Job.
    contract = entry.budget_contract(b, approval_sha256=auth['approval_sha256'])
    seen = []
    class Handler:
        def __init__(self, session, contract, *, input_options):
            assert input_options['partial_predecessor_reference'] == r['predecessor_reference']
            assert 'continuation_binding' not in input_options
        def bootstrap(self, output): seen.append('restore'); return dict(software_only=True)
        def __call__(self, work, output): seen.append(work['work_id']); return dict(software_only=True)
        def close(self): pass
    monkeypatch.setattr(formal_worker, 'FormalWorker', Handler)
    with budget.Ledger.create(tmp_path/'ledger', contract) as ledger:
        ledger.import_resource_floor(b['runtime'])
        first = entry.initial_floor_record(b['runtime'], ledger_root_sha256=ledger.root_sha256)[0]
        assert unpack(first)['type'] == 'resource_predecessor'
        assert entry.verify_initial_floor(b['runtime'], contract=contract,
            ledger_root_sha256=ledger.root_sha256) == first['sha256']
        assert ledger.summary()['imported_success_count'] == 6298
        assert ledger.summary()['generated_forecasts_reserved'] == 6272
        assert ledger.summary()['charged_ns_by_phase'] == metadata['source']['cost']['charged_ns_by_phase']
        with pytest.raises(ValueError): ledger.import_resource_floor(b['runtime'])
        directory = ledger.directory/'session-000001'; directory.mkdir()
        session = dict(schema_version=native.VERSION, directory=str(directory),
            job_name='PIRC17-FORMAL-'+'a'*32, ledger_directory=str(ledger.directory),
            ledger_root_sha256=ledger.root_sha256, execution_sha256=b['execution']['sha256'],
            worker_command=entry.worker_command(path, record['sha256'], auth, b['runtime']))
        native._write(directory/'session.json', session)
        now = time.monotonic_ns()
        remaining = 1460795_000000  # 3600s minus retained2139.205s; never a fresh input cap.
        (ledger.directory/'controls').mkdir()
        c = control._publish(ledger.directory/'controls', dict(schema_version=control.VERSION+'-control',
            ledger_root_sha256=ledger.root_sha256, phase=entry.PHASE_ORDER[0], started_ns=now,
            phase_deadline_ns=now+remaining, worker_job_name=session['job_name'],
            session_directory=str(directory), worker_command=session['worker_command']))
        ledger.open_control(c['sha256'], phase=entry.PHASE_ORDER[0], credit_ns=5*NS)
        request = dict(schema_version=native.VERSION+'-bootstrap-request', session_sha256=digest(session),
                       control_sha256=c['sha256'], started_ns=now, deadline_ns=now+remaining)
        native._write(directory/'bootstrap-request.json', request)
        native._bootstrap_request(envelope(request), session_sha256=digest(session), session=session, contract=contract)
        with pytest.raises(ValueError):
            native._bootstrap_request(envelope(dict(request, deadline_ns=now+3600*NS)),
                                      session_sha256=digest(session), session=session, contract=contract)
        wrapper = entry.RuntimeWorker(session, contract, bundle_path=path, bundle_sha256=record['sha256'], authority=auth)
        before = ledger.summary()
        assert wrapper.bootstrap(directory/'bootstrap') == dict(software_only=True)
        proof = entry.read_runtime_evidence(directory, contract=contract, session_sha256=digest(session),
            ledger_root_sha256=ledger.root_sha256, worker_pid=os.getpid(), phase=None, first_work_id=None)
        assert unpack(native._read(directory/'runtime/bootstrap.json'))['schema_version'] == entry.RESOURCE_VERSION+'-bootstrap-runtime'
        assert proof['startup']['content_sha256'] == wrapper.start['sha256'] and seen == ['restore']
        assert ledger.summary() == before
        work = next(w for w in unpack(b['matrix'])['workloads'] if w['work_id'] == policy.RETRY_WORK_ID)
        wrapper(work, directory/'outputs')
        assert entry.read_runtime_evidence(directory, contract=contract, session_sha256=digest(session),
            ledger_root_sha256=ledger.root_sha256, worker_pid=os.getpid(), phase=work['phase'], first_work_id=work['work_id'])
        assert seen == ['restore', policy.RETRY_WORK_ID]


def test_retired_resource_launch_never_replays_costs_or_consumes_a_claim(
        metadata, tmp_path, actual_environment, monkeypatch, capsys):
    path, record, b, r, auth = prepare(metadata, tmp_path, actual_environment, monkeypatch, capsys)
    args = dict(bundle_path=path, bundle_sha256=record['sha256'],
                **{k: auth[k] for k in ('approval_path','approval_sha256','test_path','review_path')})
    monkeypatch.setattr(entry, 'budget_contract', lambda *a, **kw: pytest.fail('old budget replay entered'))
    monkeypatch.setattr(control, 'Controller', lambda *a, **kw: pytest.fail('old controller started'))
    with pytest.raises(ValueError, match='restoration run is retired; use.*checkpoint_resume'): entry.run(**args)
    claim = entry.launch_directory(tmp_path/'ledger')
    assert not claim.exists() and not (tmp_path/'ledger').exists()
    with pytest.raises(ValueError, match='restoration run is retired'): entry.run(**args)
    assert not claim.exists()


def test_every_worker_controller_audit_and_export_projects_the_bound_resource_contract(
        metadata, tmp_path, actual_environment, monkeypatch, capsys):
    from experiments.pirc17 import formal_worker, formal_results, formal_reanalysis, formal_export
    from experiments.pirc17.formal_closed import ClosedOutputs
    _, _, b, r, auth = prepare(metadata, tmp_path, actual_environment, monkeypatch, capsys)
    contract = entry.budget_contract(b, approval_sha256=auth['approval_sha256'])
    scope = dict(protocol_sha256=b['protocol']['sha256'], execution_sha256=b['execution']['sha256'],
                 approval_sha256=auth['approval_sha256'], population_sha256=digest('SOFTWARE population'))
    # Constructor-only software input seam: no data restoration, domain or kernel.
    monkeypatch.setattr(formal_worker, 'InputWork', lambda **kw: SimpleNamespace(
        **{k: b[k] for k in ('protocol','execution','matrix')}, authority=auth))
    with budget.Ledger.create(tmp_path/'ledger', contract) as ledger:
        session = dict(directory=str(ledger.directory/'session-000001'), ledger_directory=str(ledger.directory),
                       ledger_root_sha256=ledger.root_sha256, execution_sha256=b['execution']['sha256'])
        worker = formal_worker.FormalWorker(session, contract, input_options={})
        assert worker.contract == contract and worker.prepared is None
        results = formal_results.ScientificResults(ledger, **{k: b[k] for k in ('protocol','execution','matrix')}, authority=auth)
        results._start()
        assert len(results.works) == 11659 and len(results.fit_ids) == 26
        closed = object.__new__(ClosedOutputs)  # Explicit constructor-only seam, not a closed output audit.
        closed.contract, closed.directory = contract, ledger.directory
        audit = formal_reanalysis.ReanalysisConsumers(closed=closed, **{k: b[k] for k in ('protocol','execution','matrix')},
            input_identity=envelope(scope), population=None, positions=None, prior=None, map_catalog=None,
            access_journal=auth['journal_directory'])
        export = formal_export.ExportConsumers(closed=closed, **{k: b[k] for k in ('protocol','execution','matrix')},
                                               scope=scope, access_journal=auth['journal_directory'])
        assert len(audit.work) == len(export.work) == 1
        assert len(audit.works) == len(export.works) == 11659


@pytest.mark.skipif(os.name != 'nt', reason='actual Windows owned Job')
@pytest.mark.parametrize('metadata_mode', [False, True])
def test_real_native_resource_bootstrap_and_closure_do_not_dispatch_or_refund_science(
        metadata, tmp_path, actual_environment, monkeypatch, capsys, metadata_mode):
    if metadata_mode: metadata = metadata_predecessor(metadata, tmp_path)
    _, _, b, r, auth = prepare(metadata, tmp_path, actual_environment, monkeypatch, capsys)
    monkeypatch.setattr(resource, 'verify_resource_predecessor', lambda b: b)  # Synthetic source; NOT actual admission.
    contract = entry.budget_contract(b, approval_sha256=auth['approval_sha256'])
    def no_work(*args): raise AssertionError('bootstrap must not authorize or validate prediction work')
    def validate(observation, output, tick):
        assert unpack(observation['barrier'])['value'] == dict(
            software_fixture_only=True, bootstraps=1, prediction_calls=0)
        assert (output/'software-only.bin').read_bytes() == b'not a model or forecast'
        tick()
        return dict(software_fixture_only=True, exact_fixture_bytes=True)
    worker = Path(__file__).parent/'fixtures/pirc17_session_worker.py'
    with budget.Ledger.create(tmp_path/'ledger', contract) as ledger:
        ledger.import_resource_floor(b['runtime'])
        old = ledger.summary()
        runner = control.Controller(ledger, [sys.executable, '-u', str(worker.resolve()),
            '--mode', 'bootstrap_counter'], authorize_work=no_work, validate_result=no_work,
            validate_bootstrap=validate)
        try:
            runner._open_phase(entry.PHASE_ORDER[0])
            runner._bootstrap()  # Exact full plan remains outstanding; no reduced run_all/matrix.
            proof = unpack(unpack(runner.bootstrap_receipt)['native_observation'])
            assert proof['accounting']['active_processes'] >= 2
            assert proof['deadline_ns']-proof['started_ns'] == 1460795_000000
            assert runner.session.sequence == 0
            new = ledger.summary()
            assert new['generated_forecasts_reserved'] == old['generated_forecasts_reserved'] == 6272
            assert new['imported_success_count'] == old['imported_success_count'] == 6298
            assert new['attempted_work_items'] == old['attempted_work_items']
            assert new['charged_ns_by_phase']['method_forecasts'] == 10800027_000000
            assert new['charged_ns_by_phase'][entry.PHASE_ORDER[0]] > old['charged_ns_by_phase'][entry.PHASE_ORDER[0]]
            assert not list((ledger.directory/'dispatches').glob('*.json'))
            if metadata_mode:
                # A micro software bootstrap is NOT scientific admission.
                # The real new first-event gate survives its truthy result.
                with pytest.raises(ValueError, match='complete metered domain admission'):
                    ledger.reserve(policy.RETRY_WORK_ID)
        finally:
            runner._close_phase(last=True, stopped=True)
            closure = runner.session.close()
        assert closure['process_tree_closed']
        current = native._query_job(runner.session.job_name)
        assert not current['exists'] or current['accounting']['active_processes'] == 0
        assert ledger.summary()['pending'] is None and ledger.summary()['active_control_sha256'] is None
