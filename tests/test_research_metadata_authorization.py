"""Entry permission cannot authorize later private metadata disclosure."""

from datetime import datetime, timezone
import json

import pytest

import application.research_evidence as evidence_module
from application.research_query import ResearchQuery
from infrastructure.research_store import ResearchError, digest, encode
from tests.test_research_comparison_query_cost import listing_store
from tests.test_research_web import service, request


def visible_comparison(store, value):
    """Owner-published synthetic metadata fixture, not a scientific result."""
    aggregate_hash = digest('metadata authorization fixture')
    references = {}
    for key, role in (('aggregate_id', 'aggregate'), ('table', 'table'),
                      ('evidence-index', 'evidence-index')):
        artifact = store.artifact(encode({'fixture': key}), role=role,
            study_id=value['study_id'], visibility='synthetic',
            block_ids=sorted({cell['block_id'] for cell in value['cells']}))
        references[key] = artifact['artifact_id']
    package = {'study_id': value['study_id'], 'aggregate_hash': aggregate_hash, **references}
    store.publish('comparison-' + aggregate_hash, package)
    return package


@pytest.fixture
def metadata(tmp_path):
    store, value = listing_store(tmp_path)
    runs = [store.register_run(value['study_id'], cell) for cell in value['cells']]
    package = visible_comparison(store, value)
    return store, value, ResearchQuery(store, 'ui'), runs[0], package


@pytest.fixture
def permission_clock(monkeypatch):
    expired = [False]

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            year = 2100 if expired[0] else 2030
            return datetime(year, 1, 1, tzinfo=timezone.utc)

    monkeypatch.setattr(evidence_module, 'datetime', Clock)
    return expired


@pytest.mark.parametrize('kind', ['study', 'run', 'comparison'])
def test_valid_metadata_query_retains_original_scope_and_values(metadata, permission_clock, kind):
    store, value, query, run, package = metadata
    result = query.list(kind)
    assert len(result['items']) == {'study': 1, 'run': 4, 'comparison': 1}[kind]
    if kind == 'comparison':
        assert result['items'][0]['manifest'] == package
    if kind == 'run':
        assert all(row['manifest']['study_id'] == value['study_id'] for row in result['items'])
        assert all(row['budget']['committed_ms'] == 0 for row in result['items'])
    assert any(event['event_kind'] == 'DISCLOSURE_ALLOWED' for event in store.events())


def test_valid_run_metadata_preserves_provenance_and_evidence(metadata, permission_clock):
    store, value, query, run, package = metadata
    result = query.run(run)
    assert result['run']['run_id'] == run
    assert result['provenance']['code_hash'] == value['code_hash']
    assert result['paper_evidence'] == [package]
    assert result['budget']['committed_ms'] == 0


@pytest.mark.parametrize('kind', ['study', 'run', 'comparison'])
@pytest.mark.parametrize('phase', ['before', 'after'])
def test_expiry_during_metadata_projection_denies_disclosure(metadata, permission_clock,
                                                           monkeypatch, kind, phase):
    store, _, query, _, _ = metadata
    original = query._objects
    called = []

    def expire_at_projection(*args, **kwargs):
        called.append(True)
        if phase == 'before':
            permission_clock[0] = True
        result = original(*args, **kwargs)
        permission_clock[0] = True
        return result

    monkeypatch.setattr(query, '_objects', expire_at_projection)
    with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
        query.list(kind)
    assert called
    assert store.events()[-1]['event_kind'] == 'DISCLOSURE_DENIED'


@pytest.mark.parametrize('phase', ['run_metadata', 'paper_evidence'])
def test_run_detail_rechecks_permission_after_metadata_assembly(metadata, permission_clock,
                                                              monkeypatch, phase):
    store, _, query, run, _ = metadata
    method = '_run_metadata' if phase == 'run_metadata' else '_objects'
    original = getattr(query, method)
    called = []

    def expire_after_assembly(*args, **kwargs):
        result = original(*args, **kwargs)
        called.append(True)
        permission_clock[0] = True
        return result

    monkeypatch.setattr(query, method, expire_after_assembly)
    with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
        query.run(run)
    assert called
    assert store.events()[-1]['event_kind'] == 'DISCLOSURE_DENIED'


@pytest.mark.parametrize('endpoint', ['/api/studies', '/api/runs', '/api/comparisons', 'run-detail'])
def test_actual_http_metadata_expiry_is_403_without_private_response(tmp_path, permission_clock,
                                                                  monkeypatch, endpoint):
    with service(tmp_path) as (store, value, server):
        run = store.register_run(value['study_id'], value['cells'][0])
        visible_comparison(store, value)
        if endpoint == 'run-detail':
            endpoint = '/api/runs/' + run
        status, _, content = request(server, endpoint)
        assert status == 200
        assert 'error' not in json.loads(content)
        original = ResearchQuery._objects
        called = []

        def expire_before_read(query, *args, **kwargs):
            called.append(True)
            permission_clock[0] = True
            return original(query, *args, **kwargs)

        monkeypatch.setattr(ResearchQuery, '_objects', expire_before_read)
        status, _, content = request(server, endpoint)
        assert called
        assert status == 403, content
        response = json.loads(content)
        assert set(response) == {'error'}
        assert response['error']['code'] == 'UNAUTHORIZED_DATA'
        assert run.encode() not in content
        assert store.events()[-1]['event_kind'] == 'DISCLOSURE_DENIED'


@pytest.mark.parametrize('view', ['study', 'run', 'comparison', 'run-detail'])
def test_expiry_in_final_physical_validation_cannot_disclose_metadata(metadata, permission_clock,
                                                                    monkeypatch, view):
    store, _, query, run, _ = metadata
    original_objects, original_events = query._objects, store._events
    assembled, expired_checks = [], []

    def assembled_objects(*args, **kwargs):
        result = original_objects(*args, **kwargs)
        assembled.append(True)
        return result

    def expire_at_physical_check():
        result = original_events()
        if assembled and store._read_snapshot() is None:
            expired_checks.append(True)
            permission_clock[0] = True
        return result

    monkeypatch.setattr(query, '_objects', assembled_objects)
    monkeypatch.setattr(store, '_events', expire_at_physical_check)
    with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
        query.run(run) if view == 'run-detail' else query.list(view)
    assert expired_checks
    assert store.events()[-1]['event_kind'] == 'DISCLOSURE_DENIED'


def test_expiry_in_preprojection_journal_denies_before_private_index_read(metadata, permission_clock,
                                                                       monkeypatch):
    store, _, query, _, _ = metadata
    original_append, original_project = store.append, query._project_objects
    allowed, projections = [], []

    def expire_after_permission_journal(kind, *args, **kwargs):
        result = original_append(kind, *args, **kwargs)
        if kind == 'DISCLOSURE_ALLOWED':
            allowed.append(True)
            if len(allowed) == 2:
                permission_clock[0] = True
        return result

    def project(*args, **kwargs):
        projections.append(True)
        return original_project(*args, **kwargs)

    monkeypatch.setattr(store, 'append', expire_after_permission_journal)
    monkeypatch.setattr(query, '_project_objects', project)
    with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
        query.list('run')
    assert len(allowed) == 2
    assert not projections, 'expired permission journal was followed by a private metadata read'
    assert store.events()[-1]['event_kind'] == 'DISCLOSURE_DENIED'


def test_expiry_in_final_permission_journal_cannot_disclose_metadata(metadata, permission_clock,
                                                                  monkeypatch):
    store, _, query, _, _ = metadata
    original_append = store.append
    allowed = []

    def expire_after_permission_journal(kind, *args, **kwargs):
        result = original_append(kind, *args, **kwargs)
        if kind == 'DISCLOSURE_ALLOWED':
            allowed.append(True)
            if len(allowed) == 3:
                permission_clock[0] = True
        return result

    monkeypatch.setattr(store, 'append', expire_after_permission_journal)
    with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
        query.list('run')
    assert len(allowed) == 3
    assert store.events()[-1]['event_kind'] == 'DISCLOSURE_DENIED'
