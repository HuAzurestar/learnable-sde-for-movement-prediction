"""Real read events have a disposable, paged block/sequence audit projection.

An empty query is never permission or evidence of an untouched test block.
"""
import json
import sqlite3
import subprocess
import sys

import pytest

from application.research_data import EvaluationExposureLedger
from infrastructure.research_index import ResearchIndex
from infrastructure.research_store import ResearchError, digest
from tests.test_research_input_gates import final_eval, freeze_evidence, authorize


def actual_reads(tmp_path, *, source_block=None):
    store, content, protocol = final_eval(tmp_path, None)
    if source_block is not None:
        protocol['blocks'][0]['source_block_id'] = source_block
    protocol, _, _ = freeze_evidence(store, protocol)
    ledger = EvaluationExposureLedger(store)
    ledger.register_protocol(protocol, digest(protocol))
    authorize(store, protocol)
    assert ledger.read('reserved', 'test-block', purpose='evaluate',
                       authorization_id='evaluate', data_root=tmp_path) == content
    artifact = store.artifact(b'synthetic result', role='result', visibility='restricted',
                              block_ids=['test-block', 'other-block'], study_id='synthetic')
    grant = {'authorization_id': 'preview', 'study_id': 'synthetic',
             'expires_at': '2099-01-01T00:00:00+00:00', 'evidence_hash': digest('fixture permission'),
             'purposes': ['preview'], 'visibilities': ['restricted'],
             'block_ids': ['test-block', 'other-block']}
    store.authorize(grant)
    assert store.read_artifact(artifact['artifact_id'], purpose='preview', authorization=grant) == b'synthetic result'
    return store, artifact


def test_real_data_and_artifact_read_events_project_by_block_and_sequence(tmp_path):
    store, artifact = actual_reads(tmp_path)
    actual = store.events()
    expected = [e for e in actual if e['event_kind'] in {'READ_STARTED', 'READ_COMPLETED', 'READ_FAILED'}]
    index = ResearchIndex(store)
    index.rebuild()
    result = index.exposures(block_id='test-block')
    assert result['items'] == expected
    assert result['watermark'] == len(actual)
    assert result['next_cursor'] is None
    assert index.exposures(block_id='other-block')['items'] == [
        e for e in expected if e['payload'].get('artifact_id') == artifact['artifact_id']]
    assert index.exposures(block_id='absent')['items'] == []


def test_projection_has_actual_composite_block_sequence_index(tmp_path):
    store, _ = actual_reads(tmp_path)
    index = ResearchIndex(store)
    index.rebuild()
    with sqlite3.connect(index.path) as connection:
        indexes = connection.execute('PRAGMA index_list(exposure_blocks)').fetchall()
        columns = [tuple(row[2] for row in connection.execute('PRAGMA index_info(' + json.dumps(item[1]) + ')'))
                   for item in indexes]
        assert ('block_id', 'sequence') in columns
        plan = connection.execute('EXPLAIN QUERY PLAN SELECT sequence FROM exposure_blocks '
                                  'WHERE block_id=? AND sequence>? AND sequence<? ORDER BY sequence LIMIT ?',
                                  ('test-block', 0, 1000, 3)).fetchall()
        assert any('SEARCH' in row[3] and 'INDEX' in row[3] for row in plan), plan


def test_scope_query_pages_exact_original_events_without_duplicates(tmp_path):
    store, _ = actual_reads(tmp_path)
    index = ResearchIndex(store)
    index.rebuild()
    expected = [e for e in store.events() if e['event_kind'].startswith('READ_')]
    first = index.exposures(block_id='test-block', limit=1)
    second = index.exposures(block_id='test-block', limit=1, after=first['next_cursor'],
                             watermark=first['watermark'])
    last = index.exposures(block_id='test-block', after=second['next_cursor'])
    assert first['items'] + second['items'] + last['items'] == expected
    assert last['next_cursor'] is None
    assert index.exposures(block_id='test-block', before=expected[1]['sequence'])['items'] == expected[:1]


@pytest.mark.parametrize('change', ['append', 'missing', 'corrupt'])
def test_stale_or_unavailable_scope_projection_is_explicit_not_empty_permission(tmp_path, change):
    store, _ = actual_reads(tmp_path)
    index = ResearchIndex(store)
    index.rebuild()
    if change == 'append':
        store.append('NOTE', {'fixture': 'after watermark'})
    elif change == 'missing':
        index.path.unlink()
    else:
        index.path.write_bytes(b'corrupt synthetic SQLite')
    with pytest.raises(ResearchError, match='INDEX_STALE'):
        index.exposures(block_id='test-block')


def test_scoped_query_refuses_forged_selected_event_payload(tmp_path):
    store, _ = actual_reads(tmp_path)
    index = ResearchIndex(store)
    index.rebuild()
    with sqlite3.connect(index.path) as connection:
        connection.execute("UPDATE exposure_events SET event_json='{}'")
    with pytest.raises(ResearchError, match='INDEX_STALE'):
        index.exposures(block_id='test-block')


def test_scoped_query_refuses_forged_block_membership_even_with_original_event_json(tmp_path):
    store, _ = actual_reads(tmp_path)
    index = ResearchIndex(store)
    index.rebuild()
    data_start = next(e for e in store.events() if e['event_kind'] == 'READ_STARTED'
                      and 'artifact_id' not in e['payload'])
    with sqlite3.connect(index.path) as connection:
        connection.execute('INSERT INTO exposure_blocks VALUES (?, ?)', ('other-block', data_start['sequence']))
    with pytest.raises(ResearchError, match='INDEX_STALE'):
        index.exposures(block_id='other-block')


def test_source_block_alias_is_indexed_without_changing_original_event(tmp_path):
    store, _ = actual_reads(tmp_path, source_block='original-source')
    index = ResearchIndex(store)
    index.rebuild()
    expected = [e for e in store.events() if e['event_kind'].startswith('READ_')
                and 'artifact_id' not in e['payload']]
    assert index.exposures(block_id='original-source')['items'] == expected
    checks = index.exposures(block_id='original-source', include_checks=True)['items']
    assert [e['event_kind'] for e in checks] == ['EXPOSURE_ALLOWED', 'READ_STARTED', 'READ_COMPLETED']


def test_version_one_list_remains_readable_but_exposure_query_requires_rebuild(tmp_path):
    store, _ = actual_reads(tmp_path)
    index = ResearchIndex(store)
    index.rebuild()
    with sqlite3.connect(index.path) as connection:
        connection.executescript('DROP TABLE exposure_blocks; DROP TABLE exposure_events; PRAGMA user_version=1;')
    assert index.list(kind='protocol')['items'][0]['object_id'] == 'protocol-reserved'
    with pytest.raises(ResearchError, match='INDEX_STALE'):
        index.exposures(block_id='test-block')
    index.rebuild()
    assert len(index.exposures(block_id='test-block')['items']) == 4


@pytest.mark.parametrize('arguments', [{'limit': 0}, {'limit': 201}, {'limit': True}, {'after': -1},
                                      {'before': 0}, {'include_checks': 1}])
def test_scope_query_rejects_invalid_bounds(tmp_path, arguments):
    store, _ = actual_reads(tmp_path)
    with pytest.raises(ResearchError, match='CONTRACT_MISMATCH'):
        ResearchIndex(store).exposures(block_id='test-block', **arguments)


def test_independent_cli_queries_original_events_and_reports_lag_without_opening_data(tmp_path):
    store, _ = actual_reads(tmp_path)
    index = ResearchIndex(store)
    index.rebuild()
    original = store.events()
    command = [sys.executable, '-B', '-m', 'experiments.pirc25', '--root', str(store.path.parent),
               '--store-id', store.store_id, 'audit-exposures', '--block', 'test-block', '--limit', '1']
    completed = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout)
    expected = next(e for e in original if e['event_kind'] == 'READ_STARTED')
    assert result['items'] == [expected]
    assert result['next_cursor'] == expected['sequence']
    assert result['permission_decision'] == 'not-provided'
    assert store.events() == original, 'metadata query must not launch or journal a data read'
    store.append('NOTE', {'fixture': 'make projection stale'})
    failed = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert failed.returncode == 1
    assert json.loads(failed.stdout)['error']['code'] == 'INDEX_STALE'


def test_current_artifact_scope_mutation_cannot_be_hidden_by_old_sqlite(tmp_path):
    store, artifact = actual_reads(tmp_path)
    index = ResearchIndex(store)
    index.rebuild()
    path = store.path / ('manifests/artifact-' + artifact['artifact_id'] + '.json')
    damaged = json.loads(path.read_bytes())
    damaged['block_ids'] = ['forged-block']
    path.write_text(json.dumps(damaged), encoding='utf-8')
    with pytest.raises(ResearchError, match='CORRUPT_ARTIFACT'):
        index.exposures(block_id='other-block')


def test_old_scope_cursor_is_not_relabelled_after_rebuild(tmp_path):
    store, _ = actual_reads(tmp_path)
    index = ResearchIndex(store)
    index.rebuild()
    original = index.exposures(block_id='test-block', limit=1)
    store.append('NOTE', {'fixture': 'new snapshot'})
    index.rebuild()
    with pytest.raises(ResearchError, match='CURSOR_STALE'):
        index.exposures(block_id='test-block', after=original['next_cursor'], watermark=original['watermark'])


def test_selected_artifact_scope_checks_do_not_rescan_physical_chain_per_row(tmp_path, monkeypatch):
    store, _ = actual_reads(tmp_path)
    index = ResearchIndex(store)
    index.rebuild()
    original, calls = store._events, []
    def observed():
        calls.append(True)
        return original()
    monkeypatch.setattr(store, '_events', observed)
    assert len(index.exposures(block_id='test-block')['items']) == 4
    assert len(calls) == 1


def test_rebuild_verifies_one_actual_chain_and_hashes_legacy_artifact_scope(tmp_path, monkeypatch):
    store, artifact = actual_reads(tmp_path)
    index = ResearchIndex(store)
    original, calls = store._events, []
    def observed():
        calls.append(True)
        return original()
    monkeypatch.setattr(store, '_events', observed)
    index.rebuild()
    assert len(calls) == 1
    # Current disclosure events deliberately omit block_ids: rebuild must use
    # the actual hash-bound artifact manifest, not invent a public scope.
    assert all('block_ids' not in e['payload'] for e in original()
               if e['payload'].get('artifact_id') == artifact['artifact_id'])
    path = store.path / ('manifests/artifact-' + artifact['artifact_id'] + '.json')
    damaged = json.loads(path.read_bytes())
    damaged['block_ids'] = ['forged-block']
    path.write_text(json.dumps(damaged), encoding='utf-8')
    with pytest.raises(ResearchError, match='CORRUPT_ARTIFACT'):
        index.rebuild()
