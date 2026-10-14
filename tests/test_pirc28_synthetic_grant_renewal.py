"""New owned synthetic grants may renew time without changing old grants/scope."""

from datetime import datetime, timedelta, timezone

import pytest

from experiments.pirc28.source_epoch import register_source_epoch
from infrastructure.research_store import ResearchError
from tests.test_pirc28_source_epoch import historical_source


def test_new_grants_extend_only_time_and_replay_cannot_change_registered_expiry(tmp_path):
    store, old, current, policy = historical_source(tmp_path)
    historical = {name: store.authorization('execute-' + name) for name in old.study_ids}
    expiry = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
    options = dict(execution_policy=policy, synthetic_grant_expires_at=expiry)
    release = register_source_epoch(store, old.definition_hash, current, **options)
    stable = ('purposes', 'visibilities', 'block_ids', 'data_root', 'test_authorization')
    for old_id, new_id in zip(old.study_ids, current.study_ids):
        grant = store.authorization('execute-' + new_id)
        assert grant['expires_at'] == expiry
        assert all(grant[field] == historical[old_id][field] for field in stable)
        assert grant['visibilities'] == ['synthetic'] and grant['test_authorization'] is False
        assert store.authorization('execute-' + old_id) == historical[old_id]
    scope = store.manifest('pirc28-synthetic-source-renewal-' + release['renewal_scope_hash'])
    assert scope['expires_at'] == expiry and scope['real_data_authorized'] is False
    before = store.events()
    assert register_source_epoch(store, old.definition_hash, current, **options) == release
    assert store.events() == before
    options['synthetic_grant_expires_at'] = (datetime.fromisoformat(expiry) + timedelta(hours=1)).isoformat()
    with pytest.raises(ResearchError, match='changed in place'):
        register_source_epoch(store, old.definition_hash, current, **options)
    assert store.events() == before


@pytest.mark.parametrize('expiry', ['2000-01-01T00:00:00+00:00', '2099-01-01T00:00:00+00:00',
                                   '2099-01-01T00:00:00', 'malformed'])
def test_invalid_renewal_date_is_refused_before_publication(tmp_path, expiry):
    store, old, current, policy = historical_source(tmp_path)
    before = store.events()
    with pytest.raises(ResearchError):
        register_source_epoch(store, old.definition_hash, current,
                               execution_policy=policy, synthetic_grant_expires_at=expiry)
    assert store.events() == before
