"""Select immutable authorization versions without a moving latest grant."""

import pytest

from application.research_query import ResearchQuery
from infrastructure.research_store import ResearchError, ResearchStore, digest
from tests.test_research_artifact_read_bounds import published
from tests.test_research_read_authorization import provider_source
from tests.test_research_store import spec


def test_same_id_versions_coexist_and_reopen_without_changing_old_grant(tmp_path):
    store, artifact, legacy = published(tmp_path)
    first = {**legacy, "version": "v1"}
    second = {**legacy, "version": "v2", "purposes": ["resume"]}
    store.authorize(first)
    store.authorize(second)
    reopened = ResearchStore(tmp_path, "read-bounds")
    assert reopened.authorization(legacy["authorization_id"], version="v1") == first
    assert reopened.authorization(legacy["authorization_id"], version="v2") == second
    assert reopened.authorization(legacy["authorization_id"]) == legacy
    assert reopened.read_artifact(artifact["artifact_id"], purpose="preview", authorization=first) == b'{"value":2}'
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        reopened.read_artifact(artifact["artifact_id"], purpose="preview", authorization=second)
    assert reopened.events()[-1]["event_kind"] == "EXPOSURE_DENIED"


def test_same_version_is_idempotent_but_changed_content_conflicts(tmp_path):
    store, _, grant = published(tmp_path)
    grant = {**grant, "version": "v1"}
    store.authorize(grant)
    before = store.events()
    store.authorize(grant)
    assert store.events() == before
    with pytest.raises(ResearchError, match="IDENTITY_CONFLICT"):
        store.authorize({**grant, "purposes": ["resume"]})
    assert store.authorization(grant["authorization_id"], version="v1") == grant


@pytest.mark.parametrize("version", [None, "", 1, True, [], "../v1"])
def test_invalid_explicit_version_is_rejected_without_publication(tmp_path, version):
    store, _, grant = published(tmp_path)
    before = store.events()
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        store.authorize({**grant, "authorization_id": "new", "version": version})
    assert store.events() == before


def test_missing_selected_version_never_falls_back_to_legacy_or_latest(tmp_path):
    store, _, legacy = published(tmp_path)
    grant = {**legacy, "authorization_id": "version-only", "version": "v2"}
    store.authorize(grant)
    for version in (None, "v1", "v3"):
        with pytest.raises(ResearchError, match="MISSING_INPUT"):
            store.authorization("version-only", version=version)


def test_provider_selects_exact_version_and_journals_its_binding(tmp_path):
    store, ledger, legacy, content = provider_source(tmp_path, "fit")
    first = {**legacy, "version": "v1", "purposes": ["validate"]}
    second = {**legacy, "version": "v2"}
    store.authorize(first)
    store.authorize(second)
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        ledger.read("reserved", "test-block", purpose="fit", authorization_id=legacy["authorization_id"],
            authorization_version="v1", data_root=tmp_path)
    assert ledger.read("reserved", "test-block", purpose="fit", authorization_id=legacy["authorization_id"],
        authorization_version="v2", data_root=tmp_path) == content
    request = store.events()[-1]["payload"]
    assert request["authorization_version"] == "v2"
    assert request["authorization_hash"] == digest(second)


def test_query_is_bound_to_selected_version_without_changing_legacy(tmp_path):
    store, _, legacy = published(tmp_path)
    value = spec()
    value["study_id"] = legacy["study_id"]
    value["cells"][0]["visibility"] = "synthetic"
    store.register(value, digest(value))
    first = {**legacy, "version": "v1", "purposes": ["resume"]}
    second = {**legacy, "version": "v2"}
    store.authorize(first)
    store.authorize(second)
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        ResearchQuery(store, legacy["authorization_id"], authorization_version="v1").list("study")
    query = ResearchQuery(store, legacy["authorization_id"], authorization_version="v2")
    assert query.list("study")["items"]
    assert query._grant() == second
    assert ResearchQuery(store, legacy["authorization_id"])._grant() == legacy
