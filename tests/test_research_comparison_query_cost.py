"""Avoid duplicate study disclosures; each artifact still needs fresh authority."""

from datetime import datetime, timezone

import pytest

from application.research_budget import BudgetSpec
from application.research_query import ResearchQuery
from infrastructure.research_store import ResearchError
from tests.test_research_comparison import source, compute


def test_managed_comparison_checks_study_once_and_authorizes_every_actual_read(source, monkeypatch):
    result = compute(source, budget=BudgetSpec(20))
    assert result['state'] == 'SUCCEEDED', result
    store = source[0]
    query = ResearchQuery(store, 'viewer')
    original_grant, original_read = query._grant, store.read_artifact
    grants, reads = [], []

    def checked_grant(*args, **kwargs):
        grants.append((args, kwargs))
        return original_grant(*args, **kwargs)

    def checked_read(artifact_id, *, purpose, authorization):
        reads.append((artifact_id, purpose, authorization))
        return original_read(artifact_id, purpose=purpose, authorization=authorization)

    monkeypatch.setattr(query, '_grant', checked_grant)
    monkeypatch.setattr(store, 'read_artifact', checked_read)
    previous_sequence = len(store.events())
    data = query.comparison(result['comparison']['aggregate_hash'])
    expected = {data['package'][key] for key in ('aggregate_id', 'computation-receipt', 'figure-index')}
    assert {artifact_id for artifact_id, _, _ in reads} == expected
    assert all(purpose == 'preview' and authorization == source[2] for _, purpose, authorization in reads)
    events = store.events()[previous_sequence:]
    for kind in ('EXPOSURE_ALLOWED', 'READ_STARTED', 'READ_COMPLETED'):
        assert {event['payload']['artifact_id'] for event in events if event['event_kind'] == kind} == expected
    assert len(grants) == 1, 'comparison redundantly revalidates/logs the same study grant for every artifact'


def test_expiry_between_comparison_artifact_reads_cannot_use_the_cached_study_grant(source, monkeypatch):
    result = compute(source, budget=BudgetSpec(20))
    assert result['state'] == 'SUCCEEDED', result
    store = source[0]
    import infrastructure.research_store as store_module
    original_read = store.read_artifact
    completed = []

    class AfterExpiry(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2100, 1, 1, tzinfo=timezone.utc)

    def expire_after_read(artifact_id, *, purpose, authorization):
        content = original_read(artifact_id, purpose=purpose, authorization=authorization)
        completed.append(artifact_id)
        monkeypatch.setattr(store_module, 'datetime', AfterExpiry)
        return content

    monkeypatch.setattr(store, 'read_artifact', expire_after_read)
    with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
        ResearchQuery(store, 'viewer').comparison(result['comparison']['aggregate_hash'])
    assert len(completed) == 1
    assert any(event['event_kind'] == 'EXPOSURE_DENIED' for event in store.events())
