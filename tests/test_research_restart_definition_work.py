"""Cold pure validation leaves the writer free; all authority stays fresh."""

from copy import deepcopy

import pytest

from application.research_restart_adjudication import (adjudicate_restart, prime_policy_definitions,
                                                       resolve_restart_adjudication)
from experiments.pirc25.affine import code_hash
from experiments.pirc28 import study
from infrastructure.research_store import ResearchError
from tests.test_pirc28_study import definition_value
from tests.test_research_restart_adjudication import epoch_fixture, interrupted_parent


def test_pure_definition_cache_is_content_bound_detached_and_bounded(monkeypatch):
    value = definition_value()
    study._validated_definition.cache_clear()
    original = study._validate
    calls = []

    def validate(document):
        calls.append(document['runtime_code_hash'])
        return original(document)

    monkeypatch.setattr(study, '_validate', validate)
    first = study.SyntheticStudyDefinition.from_dict(value)
    same = study.SyntheticStudyDefinition.from_dict(deepcopy(value))
    assert first is same and len(calls) == 1
    detached = same.to_dict()
    detached['forecast_horizons_s'][0] = 0.125
    assert first.to_dict() == value
    with pytest.raises(ResearchError):
        study.SyntheticStudyDefinition.from_dict(detached)
    for index in range(6):
        changed = {**value, 'runtime_code_hash': str(index) * 64}
        study.SyntheticStudyDefinition.from_dict(changed)
    assert study._validated_definition.cache_info().currsize == 4


def test_cold_restart_definitions_are_validated_outside_writer_scope_and_grants_stay_fresh(
        tmp_path, monkeypatch):
    store, historical, _, policy_ref, bridge = epoch_fixture(tmp_path)
    monkeypatch.setattr('infrastructure.process_tree.process_may_be_alive', lambda pid: False)
    parent, _ = interrupted_parent(store, historical)
    reference = adjudicate_restart(store, parent, source_bridge=bridge,
        execution_policy=policy_ref, source_code_hash=code_hash())
    # The low-level epoch fixture uses an arbitrary bridge ID. The release
    # discovery path intentionally indexes the production source-epoch prefix.
    store.publish('pirc28-source-epoch-bridge-' + bridge['sha256'], store.manifest(bridge['manifest_id']))
    study._validated_definition.cache_clear()
    original = study._validate
    scopes = []

    def validate(document):
        scopes.append(store._read_snapshot())
        assert store._read_snapshot() is None, 'cold whole-matrix validation owns the shared writer lock'
        return original(document)

    monkeypatch.setattr(study, '_validate', validate)
    prime_policy_definitions(store, policy_ref, code_hash())
    assert scopes == [None, None]
    resolve_restart_adjudication(store, reference, execution_policy=policy_ref, source_code_hash=code_hash())
    assert scopes == [None, None]
    before = store.events()
    authorization = store.authorization

    def expired(*args, **kwargs):
        return {**authorization(*args, **kwargs), 'expires_at': '2000-01-01T00:00:00+00:00'}

    monkeypatch.setattr(store, 'authorization', expired)
    with pytest.raises(ResearchError) as error:
        resolve_restart_adjudication(store, reference, execution_policy=policy_ref, source_code_hash=code_hash())
    assert error.value.code == 'UNAUTHORIZED_DATA'
    assert store.events() == before
