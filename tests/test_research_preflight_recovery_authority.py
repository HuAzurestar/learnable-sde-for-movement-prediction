"""Fail-closed metadata boundaries; isolated stores, no research workers."""

import pytest

from application.research_preflight_recovery import (SCHEMA, DECISION_SCHEMA,
    approve_preflight_recovery, adjudicate_preflight_recovery, prime_recovery)
from infrastructure.research_store import ResearchError, ResearchStore, digest
from tests.test_research_preflight_recovery import case


@pytest.mark.parametrize('damage', ['missing-decision', 'nonobject-decision', 'recursive-previous'])
def test_malformed_recovery_priming_is_a_typed_refusal(tmp_path, damage):
    store = ResearchStore(tmp_path, 'malformed-priming-fixture', initialize=True)
    if damage == 'missing-decision':
        value = {'schema_version': SCHEMA}
    elif damage == 'nonobject-decision':
        document = []
        store.publish('invalid-decision', document)
        value = {'schema_version': SCHEMA,
                 'decision': {'manifest_id': 'invalid-decision', 'sha256': digest(document)}}
    else:
        # A recovery cannot be its own previous retry schema. This must be
        # refused before recursively following an invalid nested decision.
        previous = {'schema_version': SCHEMA}
        store.publish('invalid-previous', previous)
        value = {'schema_version': DECISION_SCHEMA, 'scope': {
            'previous_adjudication': {'manifest_id': 'invalid-previous', 'sha256': digest(previous)},
            'source_bridge': {'manifest_id': 'never-read', 'sha256': 'a' * 64}}}
    before = store.events()
    with pytest.raises(ResearchError):
        prime_recovery(store, value)
    assert store.events() == before


@pytest.mark.parametrize('phase', ['approval', 'adjudication'])
def test_recovery_hold_prevents_new_decision_or_child_publication(case, phase):
    store, options, *_ = case
    decision = approve_preflight_recovery(store, **options) if phase == 'adjudication' else None
    (store.path / 'recovery-hold.json').write_text('{}', encoding='utf-8')
    before = store.events()
    with pytest.raises(ResearchError, match='RECOVERY_REQUIRED'):
        if decision is None:
            approve_preflight_recovery(store, **options)
        else:
            adjudicate_preflight_recovery(store, decision)
    assert store.events() == before


def test_later_closure_prevents_consuming_the_one_time_decision(case):
    store, options, *_ = case
    store.append('ARM_CLOSED', {'arm_id': 'pirc28-k1', 'reason': 'later independent closure'})
    before = store.events()
    with pytest.raises(ResearchError, match='BUDGET_EXHAUSTED'):
        approve_preflight_recovery(store, **options)
    assert store.events() == before
