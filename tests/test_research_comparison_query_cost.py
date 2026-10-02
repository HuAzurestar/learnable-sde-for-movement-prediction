"""Avoid duplicate study disclosures; each artifact still needs fresh authority."""

from datetime import datetime, timezone

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_query import ResearchQuery
from infrastructure.research_store import ResearchError, ResearchStore, digest
from tests.test_research_comparison import source, compute
from tests.test_research_store import spec


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


def listing_store(root):
    store = ResearchStore(root, 'listing', initialize=True)
    value = spec()
    value['cells'] = [{**value['cells'][0], 'seed': seed, 'visibility': 'synthetic'} for seed in (1, 2, 3, 4)]
    store.register(value, digest(value))
    store.authorize({'authorization_id': 'ui', 'study_id': value['study_id'],
        'expires_at': '2099-01-01T00:00:00+00:00', 'evidence_hash': digest('listing fixture'),
        'purposes': ['preview'], 'visibilities': ['synthetic'],
        'block_ids': sorted({cell['block_id'] for cell in value['cells']})})
    return store, value


def test_list_reuses_one_balance_per_arm_only_inside_one_query(tmp_path, monkeypatch):
    store, value = listing_store(tmp_path)
    original_balance = BudgetLedger.balance
    calls = []

    def count_balance(ledger, arm_id):
        calls.append(arm_id)
        return original_balance(ledger, arm_id)

    monkeypatch.setattr(BudgetLedger, 'balance', count_balance)
    query = ResearchQuery(store, 'ui')
    first = query.list('run')
    assert len(first['items']) == 4
    assert len(calls) == 1, 'each cell redundantly revalidates the same full arm ledger'
    attempt = store.new_attempt(store.register_run(value['study_id'], value['cells'][0]))
    BudgetLedger(store).reserve(attempt, BudgetSpec(1))
    calls.clear()
    second = query.list('run')
    assert len(calls) == 1
    assert all(row['budget']['committed_ms'] == 1000 for row in second['items'])
    assert all(row['budget']['committed_ms'] == 0 for row in first['items'])


def test_balance_update_during_list_cannot_publish_a_mixed_snapshot(tmp_path, monkeypatch):
    store, value = listing_store(tmp_path)
    attempt = store.new_attempt(store.register_run(value['study_id'], value['cells'][0]))
    original_balance = BudgetLedger.balance
    changed = []

    def update_after_balance(ledger, arm_id):
        result = original_balance(ledger, arm_id)
        if not changed:
            changed.append(True)
            ledger.reserve(attempt, BudgetSpec(1))
        return result

    monkeypatch.setattr(BudgetLedger, 'balance', update_after_balance)
    with pytest.raises(ResearchError, match='INDEX_STALE'):
        ResearchQuery(store, 'ui').list('run')


def test_recovery_hold_during_list_invalidates_the_budget_snapshot(tmp_path, monkeypatch):
    store, _ = listing_store(tmp_path)
    original_balance = BudgetLedger.balance
    changed = []

    def hold_after_balance(ledger, arm_id):
        result = original_balance(ledger, arm_id)
        if not changed:
            changed.append(True)
            store.quarantine_tail('synthetic snapshot recovery hold')
        return result

    monkeypatch.setattr(BudgetLedger, 'balance', hold_after_balance)
    with pytest.raises(ResearchError, match='INDEX_STALE'):
        ResearchQuery(store, 'ui').list('run')
    assert (store.path / 'recovery-hold.json').is_file()
