"""Saved case data and graphics use the real read and supervised job boundaries."""

import json

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
