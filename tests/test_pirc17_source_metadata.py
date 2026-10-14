"""Source metadata is NOT domain evidence; fixtures never grant launch rights."""
from copy import deepcopy
from pathlib import Path

import pytest

from tests.test_pirc17_resource_policy import fixture as policy_fixture
from tests.test_pirc17_resource_predecessor import science_metadata
from tests.test_pirc17_formal_closed import completed, changed, view
from experiments.pirc17 import formal_budget as budget, formal_controller as control
from experiments.pirc17 import formal_resource_policy as policy, formal_resource_predecessor as resource
from experiments.pirc17 import formal_source_metadata as module, formal_session as native
from experiments.pirc17.protocol_core import digest, envelope, publish, unpack


def metadata_payload(f, p):
    s = science_metadata(f)
    return dict(s, schema_version=module.VERSION, resource_policy_sha256=p['sha256'],
                forecast_status_counts={'not_replayed': 6271}, **module.FLAGS)


@pytest.fixture
def fixture(policy_fixture, monkeypatch):
    f = policy_fixture
    # Synthetic historical Job name, never asserted to be actual closure.
    timeout = deepcopy(f['timeout'])
    observed = deepcopy(unpack(timeout['native_observation']))
    observed['closure']['job_name'] = 'PIRC17-FORMAL-'+'a'*32
    timeout['native_observation'] = envelope(observed)
    f['args']['timeout_reference'] = f['ref']('named-timeout', timeout)
    p = policy.build_policy(**f['args'])
    s = metadata_payload(f, p)
    binding = resource.build_binding(resource_policy=p, science_reference=f['ref']('source-metadata', s))
    old = Path(f['contract']['ledger_directory']); old.mkdir()
    budget._publish(old/'ledger.json', f['contract'])  # Ledger root is not the64KiB mailbox.
    native._write(old/'head.json', f['cost']['ledger_tip'])
    directory = old.parent/'software-metadata-successor'
    runtime = envelope(dict(protocol_sha256=f['protocol']['sha256'], matrix_sha256=f['matrix']['sha256'],
                            ledger_directory=str(directory), predecessor=binding))
    contract = budget.contract_for_matrix(f['matrix'], protocol_sha256=f['protocol']['sha256'],
        execution_sha256=digest('new-software-execution'), runtime_manifest_sha256=runtime['sha256'],
        approval_sha256=digest('new-software-not-human-approval'), ledger_directory=directory, resource_policy=p)
    return dict(f=f, p=p, s=s, binding=binding, old=old, directory=directory, runtime=runtime, contract=contract)


def test_current_byte_verification_stays_strict_and_metadata_never_claims_it(completed, monkeypatch):
    reader = view(completed)
    work = completed['contract']['workloads'][0]
    original = reader.read(work)
    path = original['directory']/'fixture.bin'
    with changed(path, b'CHANGED software fixture bytes'):
        with monkeypatch.context() as patch:
            patch.setattr(control, 'verify_artifacts', lambda *a: pytest.fail('metadata must not open owned value bytes'))
            item = reader.read_metadata(work)
        assert item['metadata_only'] is True and item['owned_source_bytes_rechecked'] is False
        assert item['artifacts'] == original['artifacts']
        assert item['settlement_sha256'] == original['settlement_sha256']
        with pytest.raises(ValueError): reader.read(work)


def test_complete_metadata_inventory_never_calls_full_reader_or_domain_loaders(fixture, monkeypatch):
    q = fixture; f = q['f']; s = q['s']; root = q['old']
    class MetadataReader:
        def __init__(self, directory, *, contract, tip):
            assert directory == root and contract == f['contract'] and tip == f['cost']['ledger_tip']
        def disposition(self, wid): return f['cost']['work_dispositions'].get(wid, 'unattempted')
        def read(self, *a): pytest.fail('full source must be deferred to paid bootstrap')
        def read_metadata(self, work):
            src = s['completed_sources'][work['work_id']]
            manifest, artifacts = {}, {}
            if src['artifact'] is None:
                manifest['context_sha256'] = s['context_sha256']
            else:
                a = src['artifact']; path = root/a['path']
                manifest = dict(artifact_path=str(path), artifact_sha256=a['content_sha256'])
                artifacts[path.name] = dict(file_sha256=a['file_sha256'], bytes=1)
            return dict(**{k: src[k] for k in ('result_sha256','settlement_sha256','observation_sha256','controller_elapsed_ns')},
                directory=root/'fixture-artifacts', manifest=manifest, artifacts=artifacts,
                metadata_only=True, owned_source_bytes_rechecked=False)
    monkeypatch.setattr(module, 'ClosedOutputs', MetadataReader)  # Complete-size synthetic metadata only.
    result = module.inspect_source_metadata(q['p'])
    assert unpack(result) == s
    assert len(s['completed_sources']) == 6298 and len(s['forecast_index']) == 6271 and len(s['fit_bindings']) == 26
    assert s['forecast_status_counts'] == {'not_replayed': 6271}


def test_metadata_floor_keeps_all_costs_but_cannot_reserve_before_paid_domain_admission(fixture, monkeypatch):
    q = fixture
    from experiments.pirc17 import formal_partial_costs, formal_partial_science
    jobs = []
    monkeypatch.setattr(formal_partial_costs, '_closed_job', lambda job: jobs.append(job))  # Explicit native seam.
    monkeypatch.setattr(formal_partial_science, 'inspect_partial_science',
                        lambda *a, **kw: pytest.fail('pre-controller domain replay forbidden'))
    assert unpack(q['binding'])['schema_version'] == resource.METADATA_VERSION
    with budget.Ledger.create(q['directory'], q['contract']) as ledger:
        ledger.import_resource_floor(q['runtime'])  # Real v2 verifier, exact synthetic closed head/root.
        before = ledger.summary()
        assert before['predecessor_floor']['requires_metered_source_admission'] is True
        assert before['imported_success_count'] == 6298 and before['generated_forecasts_reserved'] == 6272
        assert before['charged_ns_by_phase'] == q['f']['cost']['charged_ns_by_phase']
        for wid in (policy.RETRY_WORK_ID, next(w['work_id'] for w in unpack(q['f']['matrix'])['workloads']
                                             if w['phase'] == 'terrain_forecasts')):
            with pytest.raises(ValueError, match='complete metered domain admission'): ledger.reserve(wid)
        assert ledger.summary() == before
        ledger.open_control(digest('SOFTWARE metered input control'), phase='input_qualification_and_binding', credit_ns=5*budget.NANOSECONDS)
        assert ledger.summary()['generated_forecasts_reserved'] == 6272
        assert len(jobs) == 2
        # Internal state seam ONLY: real acceptance requires native/domain
        # bind_partial_imports, whose strict verifier is not bypassed in code.
        ledger._state.partial_imports = {'SOFTWARE simulated verified admission': True}
        ledger.close_control(digest('SOFTWARE metered input control'), observed_ns=0,
            evidence_sha256=digest('SOFTWARE close'), reason='phase_complete')
        ledger.reserve(policy.RETRY_WORK_ID)
        assert ledger.summary()['generated_forecasts_reserved'] == 6273
        assert ledger.summary()['predecessor_floor']['retry_old_reservation_sha256'] == unpack(q['p'])['retry_old_reservation_sha256']


@pytest.mark.parametrize('field,value', [('metadata_only',False), ('forecast_domains_verified',True),
    ('owned_source_bytes_rechecked',True), ('new_array_reads',1), ('requires_metered_domain_admission',False),
    ('forecast_status_counts',{'success':6271})])
def test_metadata_cannot_relabel_itself_as_qualified_or_skip_the_admission_gate(fixture, field, value):
    q = fixture; s = dict(q['s'], **{field:value})
    with pytest.raises(ValueError):
        resource.build_binding(resource_policy=q['p'], science_reference=q['f']['ref']('mutated-source-metadata', s))


def test_metadata_snapshot_cannot_be_smuggled_into_the_old_full_science_binding(fixture):
    q = fixture; b = dict(unpack(q['binding']), schema_version=resource.VERSION)
    with pytest.raises(ValueError): resource._history(envelope(b))
