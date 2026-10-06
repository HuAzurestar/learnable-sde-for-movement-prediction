"""Saved case data and graphics use the real read and supervised job boundaries."""

import json
from copy import deepcopy
from pathlib import Path
import sys

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_query import ResearchQuery
from infrastructure.research_store import ResearchStore, ResearchError, digest, encode
from tests.test_research_store import spec
from tests.test_research_web import service, request


def case_source(tmp_path, *, four_state=False, samples=True, visibility='synthetic'):
    store = ResearchStore(tmp_path, 'cases', initialize=True)
    value = spec()
    value['cells'][0]['visibility'] = visibility
    store.register(value, digest(value))
    result = {'schema_version': 'pirc25-result-v1', 'spec_hash': digest(value),
              'cell_hash': digest(value['cells'][0]), 'protocol_hash': value['protocol_hash'],
              'state_order': ['x', 'y', 'vx', 'vy'] if four_state else ['x', 'vx'],
              'units': ['m', 'm', 'm/s', 'm/s'] if four_state else ['m', 'm/s'],
              'time_unit': 's', 'metrics': {'fixture_error': 2.5}, 'qualification': 'fixture',
              'forecast': {'horizons': [1.0, 2.0], 'preview': {
                  'case_selection_rule': 'registered-synthetic-case', 'sample_selection_rule': 'all-saved',
                  'sample_ids': ['sample-0', 'sample-1'], 'generation_version': 'case-fixture-v1', 'n_samples': 2}}}
    if samples:
        result['forecast']['samples'] = ([[[0, 0, 1, 2], [1, 2, 3, 4]], [[0, 1, 2, 3], [2, 3, 4, 5]]]
                                         if four_state else [[[0, 1], [1, 2]], [[1, 2], [2, 3]]])
    artifact = store.artifact(encode(result), role='result', visibility=visibility,
                             study_id=value['study_id'], block_ids=['fixture-1'])
    run = store.register_run(value['study_id'], value['cells'][0])
    attempt = store.new_attempt(run)
    store.transition(attempt, 'RUNNING')
    store.transition(attempt, 'SUCCEEDED', artifact_id=artifact['artifact_id'])
    grant = {'authorization_id': 'case-viewer', 'study_id': value['study_id'],
             'expires_at': '2099-01-01T00:00:00+00:00', 'evidence_hash': digest('synthetic case permission'),
             'purposes': ['preview', 'export'], 'visibilities': [visibility], 'block_ids': ['fixture-1']}
    store.authorize(grant)
    return store, value, artifact, result, grant


def test_actual_case_endpoint_returns_saved_result_without_launching_a_job(tmp_path):
    with service(tmp_path) as (store, value, server):
        result = {'spec_hash': digest(value), 'cell_hash': digest(value['cells'][0]),
                  'protocol_hash': value['protocol_hash'], 'forecast': {}, 'metrics': {'error': 2.5}}
        artifact = store.artifact(encode(result), role='result', visibility='synthetic',
                                 study_id='synthetic', block_ids=['fixture-1'])
        assert request(server, '/api/artifacts/' + artifact['artifact_id'])[0] == 200
        status, _, body = request(server, '/api/cases/' + artifact['artifact_id'])
        assert status == 200, body
        case = json.loads(body)
        assert case['case_id'] == artifact['artifact_id'] and case['result'] == result
        assert case['figure_status'] == 'UNAVAILABLE'
        assert not any(e['event_kind'] in {'RESERVE', 'WORKER_STARTED'} for e in store.events())
        assert request(server, '/api/cases/' + artifact['artifact_id'], auth=False)[0] == 401
        assert request(server, '/api/cases/' + artifact['artifact_id'], method='POST')[0] == 405


@pytest.mark.parametrize('four_state', [False, True])
def test_real_graph_job_freezes_all_horizons_and_keeps_scores_and_arm_cost(tmp_path, four_state):
    from application.research_cases import CaseGraphRunner
    from application.research_case_figures import validate_case_package
    store, value, artifact, result, grant = case_source(tmp_path, four_state=four_state)
    before = BudgetLedger(store).balance('affine')['committed_ms']
    produced = CaseGraphRunner(store).run(artifact['artifact_id'], authorization_id=grant['authorization_id'],
                                        budget=BudgetSpec(20))
    assert produced['state'] == 'SUCCEEDED', produced
    viewed = ResearchQuery(store, grant['authorization_id']).case(artifact['artifact_id'])
    assert viewed['result'] == result and viewed['figure_status'] == 'AVAILABLE'
    index = viewed['figure_index']
    assert {entry['horizon_index'] for entry in index['figures']} == {None, 0, 1}
    assert index['source_artifact_id'] == artifact['artifact_id']
    assert viewed['computation_receipt']['cost']['arm_id'] == 'affine'
    assert viewed['computation_receipt']['cost']['charged_ms'] > 0
    assert BudgetLedger(store).balance('affine')['committed_ms'] > before
    figures = {}
    for entry in viewed['package']['figures']:
        content, media = ResearchQuery(store, grant['authorization_id']).artifact(entry['artifact_id'], export=True)
        assert media == 'image/svg+xml'
        assert entry['sha256'] == entry['artifact_id']
        figures[entry['filename']] = content.decode('utf-8')
        assert b'case-fixture-v1' in content
        if four_state:
            assert b'vx (m/s)' in content and b'vy (m/s)' in content
    validate_case_package(result, artifact['artifact_id'], index, figures)
    prior_events = store.events()
    reused = CaseGraphRunner(ResearchStore(tmp_path, 'cases')).run(artifact['artifact_id'],
                                                               authorization_id=grant['authorization_id'])
    assert reused['reused'] and reused['case'] == produced['case']
    assert sum(e['event_kind'] == 'WORKER_STARTED' for e in store.events()) == sum(
        e['event_kind'] == 'WORKER_STARTED' for e in prior_events)
    assert BudgetLedger(store).balance('affine')['committed_ms'] == before + viewed['computation_receipt']['cost']['charged_ms']


def test_missing_saved_paths_are_explicit_and_never_launch_a_graph_job(tmp_path):
    from application.research_cases import CaseGraphRunner
    store, _, artifact, result, grant = case_source(tmp_path, samples=False)
    produced = CaseGraphRunner(store).run(artifact['artifact_id'], authorization_id=grant['authorization_id'])
    assert produced['state'] == 'UNAVAILABLE'
    viewed = ResearchQuery(store, grant['authorization_id']).case(artifact['artifact_id'])
    assert viewed['result'] == result and viewed['figure_status'] == 'UNAVAILABLE'
    assert not any(e['event_kind'] in {'RESERVE', 'WORKER_STARTED'} for e in store.events())


@pytest.mark.parametrize('mutation', ['too_many_paths', 'bad_grid', 'wrong_width', 'boolean_coordinate', 'missing_policy'])
def test_source_plan_rejects_bad_or_overquota_inputs(mutation):
    from application.research_case_figures import case_plan
    result = {'state_order': ['x', 'vx'], 'units': ['m', 'm/s'], 'time_unit': 's',
              'forecast': {'horizons': [1, 2], 'samples': [[[0, 1], [1, 2]]], 'preview': {
                  'case_selection_rule': 'synthetic', 'sample_ids': ['s0'], 'n_samples': 1,
                  'generation_version': 'fixture'}}}
    if mutation == 'too_many_paths':
        result['forecast']['samples'] *= 65
    elif mutation == 'bad_grid':
        result['forecast']['horizons'] = [2, 1]
    elif mutation == 'wrong_width':
        result['forecast']['samples'][0][0] = [0, 1, 2]
    elif mutation == 'boolean_coordinate':
        result['forecast']['samples'][0][0][0] = True
    else:
        result['forecast'].pop('preview')
    with pytest.raises(ResearchError, match='RESOURCE_PLAN_REJECTED|CONTRACT_MISMATCH'):
        case_plan(result)


def test_graph_plan_joint_output_and_operation_caps_precede_worker(tmp_path):
    from application.research_cases import CaseGraphRunner
    store, _, artifact, _, grant = case_source(tmp_path)
    with pytest.raises(ResearchError, match='RESOURCE_PLAN_REJECTED'):
        CaseGraphRunner(store).run(artifact['artifact_id'], authorization_id=grant['authorization_id'], max_operations=1)
    assert not any(e['event_kind'] in {'RESERVE', 'WORKER_STARTED'} for e in store.events())


def test_preview_only_grant_never_becomes_graph_export_permission(tmp_path):
    from application.research_cases import CaseGraphRunner
    store, _, artifact, _, grant = case_source(tmp_path)
    CaseGraphRunner(store).run(artifact['artifact_id'], authorization_id=grant['authorization_id'], budget=BudgetSpec(20))
    store.authorize({**grant, 'authorization_id': 'preview-only', 'purposes': ['preview']})
    query = ResearchQuery(store, 'preview-only')
    viewed = query.case(artifact['artifact_id'])
    entry = viewed['package']['figures'][0]
    assert query.artifact(entry['artifact_id'])[1] == 'image/svg+xml'
    with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
        query.artifact(entry['artifact_id'], export=True)


def test_restricted_graph_cannot_be_declassified_by_original_case_metadata(tmp_path):
    from application.research_cases import CaseGraphRunner
    store, _, artifact, _, grant = case_source(tmp_path, visibility='restricted')
    made = CaseGraphRunner(store).run(artifact['artifact_id'], authorization_id=grant['authorization_id'], budget=BudgetSpec(20))
    assert all(store.manifest('artifact-' + e['artifact_id'])['visibility'] == 'restricted' for e in made['case']['figures'])
    store.authorize({**grant, 'authorization_id': 'synthetic-only', 'visibilities': ['synthetic']})
    with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
        ResearchQuery(store, 'synthetic-only').case(artifact['artifact_id'])


def test_case_queries_reject_wrong_roles_and_source_spec_binding(tmp_path):
    store, _, artifact, result, grant = case_source(tmp_path)
    forged = store.artifact(encode({**result, 'spec_hash': digest('wrong spec')}), role='result', visibility='synthetic',
                            study_id=grant['study_id'], block_ids=['fixture-1'])
    with pytest.raises(ResearchError, match='CONTRACT_MISMATCH|CORRUPT_ARTIFACT'):
        ResearchQuery(store, grant['authorization_id']).case(forged['artifact_id'])
    unrelated = store.artifact(b'{}', role='aggregate', visibility='synthetic', study_id=grant['study_id'], block_ids=['fixture-1'])
    with pytest.raises(ResearchError, match='CONTRACT_MISMATCH|UNAUTHORIZED_DATA'):
        ResearchQuery(store, grant['authorization_id']).case(unrelated['artifact_id'])


def test_case_job_caps_cannot_be_relaxed_even_before_source_read(tmp_path):
    from application.research_cases import CaseGraphRunner
    store, _, artifact, _, grant = case_source(tmp_path)
    before = store.events()
    with pytest.raises(ResearchError):
        CaseGraphRunner(store).run(artifact['artifact_id'], authorization_id=grant['authorization_id'], budget=BudgetSpec(7201))
    assert store.events() == before


def test_case_plan_bounds_escaped_metadata_and_encoded_package(tmp_path):
    from application.research_case_figures import case_plan, render_case_package
    _, _, artifact, result, _ = case_source(tmp_path)
    result['forecast']['samples'] = [result['forecast']['samples'][0] for _ in range(63)]
    result['forecast']['preview'].update(n_samples=63, sample_ids=['&' * 180 + str(i) for i in range(63)],
                                       case_selection_rule='&' * 200, generation_version='&' * 200)
    plan = case_plan(result)
    package = render_case_package(result, artifact['artifact_id'], {'fixture': 'reference'})
    assert len(encode(package)) <= plan['planned_output_bytes'], 'XML escaping and JSON envelope exceed preflight graph bound'


@pytest.mark.parametrize('name', ['request', 'source'])
def test_case_worker_actual_reads_use_its_smaller_source_cap(tmp_path, monkeypatch, name):
    from application.research_case_figures import case_plan, MAX_SOURCE_BYTES
    from experiments.pirc25 import case_worker
    _, _, artifact, result, _ = case_source(tmp_path / 'store')
    paths = {key: tmp_path / (key + '.json') for key in ('request', 'source')}
    paths['source'].write_bytes(encode(result))
    paths['request'].write_bytes(encode({'schema_version': 'pirc25-case-graph-request-v1',
        'source_artifact_id': artifact['artifact_id'], 'runtime_code_hash': 'fixture-code',
        'resource_plan': case_plan(result), 'computation_ref': {'fixture': 'reference'}}))
    monkeypatch.setattr(sys, 'argv', ['case_worker', str(paths['request']), str(paths['source']), str(tmp_path / 'result.json')])
    monkeypatch.setattr(case_worker, 'code_hash', lambda: 'fixture-code')
    original, reads = Path.open, []

    class GuardedStream:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return self.stream.__exit__(*args)
        def __getattr__(self, key):
            return getattr(self.stream, key)
        def read(self, size=-1):
            assert 0 <= size <= MAX_SOURCE_BYTES + 1, 'case worker reads beyond its admitted 2 MiB cap'
            reads.append(size)
            return self.stream.read(size)

    def guarded(path, *args, **kwargs):
        stream = original(path, *args, **kwargs)
        return GuardedStream(stream) if path == paths[name] else stream
    monkeypatch.setattr(Path, 'open', guarded)
    case_worker.main()
    assert reads and (tmp_path / 'result.json').is_file()


@pytest.mark.parametrize('quota', ['paths', 'points'])
def test_read_only_case_preview_preserves_too_large_error_contract(tmp_path, quota):
    store, _, _, result, grant = case_source(tmp_path)
    if quota == 'paths':
        result['forecast']['samples'] *= 33
    else:
        result['forecast']['horizons'] = list(range(1, 514))
        result['forecast']['samples'] = [[[0, 1]] * 513] * 2
    artifact = store.artifact(encode(result), role='result', visibility='synthetic',
                             study_id=grant['study_id'], block_ids=['fixture-1'])
    with pytest.raises(ResearchError, match='TOO_LARGE'):
        ResearchQuery(store, grant['authorization_id']).case(artifact['artifact_id'])
    assert not any(e['event_kind'] in {'RESERVE', 'WORKER_STARTED'} for e in store.events())


def test_legal_saved_preview_is_readable_when_full_graph_allocation_is_too_large(tmp_path):
    from application.research_cases import CaseGraphRunner
    from application.research_case_figures import case_plan
    store, _, _, result, grant = case_source(tmp_path, four_state=True)
    result['forecast']['horizons'] = list(range(1, 129))
    result['forecast']['samples'] = [[[0, 1, 2, 3]] * 128 for _ in range(64)]
    result['forecast']['preview'].update(n_samples=64, sample_ids=list(range(64)))
    artifact = store.artifact(encode(result), role='result', visibility='synthetic',
                             study_id=grant['study_id'], block_ids=['fixture-1'])
    with pytest.raises(ResearchError, match='RESOURCE_PLAN_REJECTED'):
        case_plan(result)
    with pytest.raises(ResearchError, match='RESOURCE_PLAN_REJECTED'):
        CaseGraphRunner(store).run(artifact['artifact_id'], authorization_id=grant['authorization_id'])
    viewed = ResearchQuery(store, grant['authorization_id']).case(artifact['artifact_id'])
    assert viewed['result'] == result and viewed['figure_status'] == 'UNAVAILABLE'
    assert not any(e['event_kind'] in {'RESERVE', 'WORKER_STARTED'} for e in store.events())


@pytest.mark.parametrize('mutation', ['foreign-source', 'wrong-ordinal', 'wrong-public-id'])
def test_managed_case_package_rejects_tampered_source_and_figure_bindings(tmp_path, mutation):
    from application.research_cases import CaseGraphRunner, verified_case_job
    store, _, artifact, result, grant = case_source(tmp_path)
    produced = CaseGraphRunner(store).run(artifact['artifact_id'], authorization_id=grant['authorization_id'], budget=BudgetSpec(20))
    assert produced['state'] == 'SUCCEEDED', produced
    package = deepcopy(produced['case'])
    if mutation == 'foreign-source':
        package['computation_ref']['source_artifact_id'] = digest('foreign source')
    elif mutation == 'wrong-ordinal':
        package['figure_index']['figures'][1]['horizon_index'] = 1
    else:
        package['figures'][0]['artifact_id'] = digest('foreign figure')
    with pytest.raises(ResearchError, match='CORRUPT_ARTIFACT'):
        verified_case_job(store, package, result)


def test_corrupt_successful_case_job_is_not_reused_or_silently_recomputed(tmp_path):
    from application.research_cases import CaseGraphRunner
    store, _, artifact, _, grant = case_source(tmp_path)
    runner = CaseGraphRunner(store)
    made = runner.run(artifact['artifact_id'], authorization_id=grant['authorization_id'], budget=BudgetSpec(20))
    assert made['state'] == 'SUCCEEDED', made
    before_starts = sum(e['event_kind'] == 'WORKER_STARTED' for e in store.events())
    before_cost = BudgetLedger(store).balance('affine')['committed_ms']
    (store.path / 'artifacts' / made['artifact_id']).write_bytes(b'corrupt synthetic case job output')
    with pytest.raises(ResearchError, match='CORRUPT_ARTIFACT'):
        runner.run(artifact['artifact_id'], authorization_id=grant['authorization_id'])
    with pytest.raises(ResearchError, match='CORRUPT_ARTIFACT'):
        ResearchQuery(store, grant['authorization_id']).case(artifact['artifact_id'])
    assert sum(e['event_kind'] == 'WORKER_STARTED' for e in store.events()) == before_starts
    assert BudgetLedger(store).balance('affine')['committed_ms'] == before_cost


def test_closed_source_arm_blocks_graph_jobs_but_not_read_only_saved_values(tmp_path):
    from application.research_cases import CaseGraphRunner
    store, value, artifact, result, grant = case_source(tmp_path)
    spending = {**value, 'study_id': 'case-spending'}
    store.register(spending, digest(spending))
    attempt = store.new_attempt(store.register_run(spending['study_id'], spending['cells'][0]))
    ledger = BudgetLedger(store)
    reservation = ledger.reserve(attempt, BudgetSpec(1))
    store.transition(attempt, 'RUNNING')
    ledger.settle(reservation['reservation_id'], 1000, outcome='TIMEOUT')
    store.transition(attempt, 'TIMEOUT', error_code='SYNTHETIC_TIMEOUT')
    with pytest.raises(ResearchError, match='BUDGET_EXHAUSTED'):
        CaseGraphRunner(store).run(artifact['artifact_id'], authorization_id=grant['authorization_id'])
    assert ResearchQuery(store, grant['authorization_id']).case(artifact['artifact_id'])['result'] == result
    assert ledger.balance('affine')['closed'] and ledger.balance('affine')['committed_ms'] == 1000
    assert not any(e['event_kind'] == 'WORKER_STARTED' for e in store.events())
