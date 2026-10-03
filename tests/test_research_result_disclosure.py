"""All public result views need current authority after real reads and I/O."""

from datetime import datetime, timezone
import json

import pytest

import application.research_evidence as evidence_module
import infrastructure.research_store as store_module
from application.research_query import ResearchQuery
from infrastructure.research_store import ResearchError, digest, encode
from tests.test_research_cases import case_source
from tests.test_research_metadata_authorization import visible_comparison
from tests.test_research_web import service, request

VIEWS = ['preview', 'export', 'manifest', 'case', 'comparison']


@pytest.fixture
def frozen_result(tmp_path):
    source = case_source(tmp_path)
    store, value, artifact, result, grant = source
    package = visible_comparison(store, value)
    return store, artifact, result, ResearchQuery(store, grant['authorization_id']), package


def view_result(query, view, artifact, package):
    if view in {'preview', 'export'}:
        return query.artifact(artifact['artifact_id'], export=view == 'export')
    if view == 'manifest':
        return query.result_manifest(artifact['artifact_id'])
    if view == 'case':
        return query.case(artifact['artifact_id'])
    return query.comparison(package['aggregate_hash'])


@pytest.fixture
def permission_clock(monkeypatch):
    expired = [False]

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2100 if expired[0] else 2030, 1, 1, tzinfo=timezone.utc)

    monkeypatch.setattr(evidence_module, 'datetime', Clock)
    monkeypatch.setattr(store_module, 'datetime', Clock)
    return expired


def expire_after_read(store, permission_clock, monkeypatch, *, physical):
    original_read, original_events = store.read_artifact, store._events
    reads, expired_checks = [], []

    def read(*args, **kwargs):
        result = original_read(*args, **kwargs)
        reads.append(args[0])
        if not physical:
            permission_clock[0] = True
            expired_checks.append(True)
        return result

    def physical_check():
        result = original_events()
        if physical and reads and store._read_snapshot() is None:
            permission_clock[0] = True
            expired_checks.append(True)
        return result

    monkeypatch.setattr(store, 'read_artifact', read)
    monkeypatch.setattr(store, '_events', physical_check)
    return reads, expired_checks


@pytest.mark.parametrize('view', VIEWS)
def test_valid_public_result_view_preserves_exact_bytes_bindings_and_no_jobs(frozen_result,
                                                                          permission_clock, view):
    store, artifact, result, query, package = frozen_result
    response = view_result(query, view, artifact, package)
    if view in {'preview', 'export'}:
        assert response == (encode(result), 'application/json')
    elif view == 'manifest':
        assert response['artifact'] == artifact
        assert response['bindings']['spec_hash'] == result['spec_hash']
        assert 'forecast' not in response
    elif view == 'case':
        assert response['result'] == result
        assert response['figure_status'] == 'UNAVAILABLE'
    else:
        assert response['package'] == package
        assert response['aggregate'] == {'fixture': 'aggregate_id'}
    assert not any(event['event_kind'] in {'WORKER_STARTED', 'RESERVE'} for event in store.events())


@pytest.mark.parametrize('view', VIEWS)
@pytest.mark.parametrize('phase', ['read', 'physical'])
def test_expiry_after_actual_result_read_or_final_physical_check_denies_disclosure(frozen_result,
                                                                                permission_clock,
                                                                                monkeypatch, view, phase):
    store, artifact, _, query, package = frozen_result
    reads, expired_checks = expire_after_read(store, permission_clock, monkeypatch, physical=phase == 'physical')
    with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
        view_result(query, view, artifact, package)
    assert reads and expired_checks
    assert store.events()[-1]['event_kind'] == 'DISCLOSURE_DENIED'


@pytest.mark.parametrize('view', VIEWS)
def test_physical_grant_change_after_actual_read_is_not_hidden_by_entry_permission(frozen_result,
                                                                                 permission_clock,
                                                                                 monkeypatch, view):
    store, artifact, _, query, package = frozen_result
    original_read = store.read_artifact
    changed = []

    def corrupt_after_read(*args, **kwargs):
        result = original_read(*args, **kwargs)
        grant = store.manifest('authorization-' + query.authorization_id)
        (store.path / 'manifests' / ('authorization-' + query.authorization_id + '.json')).write_bytes(
            encode({**grant, 'purposes': []}))
        changed.append(True)
        return result

    monkeypatch.setattr(store, 'read_artifact', corrupt_after_read)
    with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
        view_result(query, view, artifact, package)
    assert changed
    assert store.events()[-1]['event_kind'] == 'DISCLOSURE_DENIED'


@pytest.mark.parametrize('view', VIEWS)
def test_actual_http_result_final_expiry_is_403_without_result_or_source_ids(tmp_path, permission_clock,
                                                                         monkeypatch, view):
    with service(tmp_path) as (store, value, server):
        result = {'spec_hash': digest(value), 'cell_hash': digest(value['cells'][0]),
                  'protocol_hash': value['protocol_hash'], 'forecast': {}, 'metrics': {'error': 2.5}}
        artifact = store.artifact(encode(result), role='result', visibility='synthetic',
                                 study_id=value['study_id'], block_ids=['fixture-1'])
        package = visible_comparison(store, value)
        artifact_path = '/api/artifacts/' + artifact['artifact_id']
        endpoint = {'preview': artifact_path, 'export': artifact_path + '?download=1',
                    'manifest': artifact_path + '?manifest=1', 'case': '/api/cases/' + artifact['artifact_id'],
                    'comparison': '/api/comparisons/' + package['aggregate_hash']}[view]
        status, _, content = request(server, endpoint)
        assert status == 200 and 'error' not in json.loads(content)
        reads, expired_checks = expire_after_read(store, permission_clock, monkeypatch, physical=True)
        status, _, content = request(server, endpoint)
        assert reads and expired_checks
        assert status == 403, content
        response = json.loads(content)
        assert set(response) == {'error'}
        assert response['error']['code'] == 'UNAUTHORIZED_DATA'
        assert artifact['artifact_id'].encode() not in content
        assert package['aggregate_hash'].encode() not in content
        assert store.events()[-1]['event_kind'] == 'DISCLOSURE_DENIED'
