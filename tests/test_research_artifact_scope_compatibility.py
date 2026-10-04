"""Actual old-format scopes, normalized aliases and owner-phase isolation."""

import hashlib
import math

import pytest

from application.research_data import EvaluationExposureLedger
from application.research_preregistration import PreregistrationGate, protocol_binding, same_source
from infrastructure.research_artifact_scope import ArtifactExposureFacts
from infrastructure.research_store import ResearchError, digest
from tests.research_file_observation import observe_file
from tests.test_research_artifact_scope_work import admitted
from tests.test_research_artifact_read_bounds import published
from tests.test_research_input_gates import final_eval, freeze_evidence, authorize


@pytest.mark.parametrize("kind", ["READ_STARTED", "READ_COMPLETED", "READ_FAILED", "EXPOSURE_ALLOWED"])
@pytest.mark.parametrize("variant", ["same", "same-content", "same-source", "unknown", "invalid-hash",
                                     "other", "artifact", "legacy-block"])
def test_first_source_read_preserves_original_actual_predicate_and_detachment(tmp_path, kind, variant):
    store, _, _ = published(tmp_path)
    identity = {"dataset_id": "dataset", "source_block_id": "source", "sha256": digest("target")}
    payload = dict(identity)
    if variant == "same-content":
        payload.update(dataset_id="renamed", source_block_id="renamed")
    elif variant == "same-source":
        payload.update(sha256=digest("older content"))
    elif variant in {"unknown", "invalid-hash"}:
        payload.update(sha256=None if variant == "unknown" else "not-a-hash", source_block_id="other")
    elif variant == "other":
        payload.update(dataset_id="other", source_block_id="other", sha256=digest("other"))
    elif variant == "artifact":
        payload = {"artifact_id": "legacy-artifact"}
    elif variant == "legacy-block":
        payload.pop("source_block_id")
        payload.update(block_id="source", sha256=digest("older content"))
    with store._read_transaction():
        early = store.append(kind, payload)
        before = store.append("BOUNDARY", {})["sequence"]
        store.append(kind, payload)
        actual = store.events()
        expected = next((event for event in actual if event["sequence"] < before
                         and event["event_kind"] in {"READ_STARTED", "READ_COMPLETED", "READ_FAILED"}
                         and same_source(event["payload"], identity)), None)
        first = store._first_source_read(identity, before)
        assert first == expected
        if first:
            first["payload"].clear()
            assert store.events()[early["sequence"] - 1]["payload"] == payload


@pytest.mark.parametrize("kind", ["READ_STARTED", "READ_COMPLETED", "READ_FAILED"])
@pytest.mark.parametrize("after_freeze", [False, True])
def test_unscoped_actual_read_dict_cannot_be_blind_history(tmp_path, monkeypatch, kind, after_freeze):
    store, _, protocol = final_eval(tmp_path, None)
    if not after_freeze:
        store.append(kind, {"purpose": "preview"})
    protocol, _, _ = freeze_evidence(store, protocol)
    if after_freeze:
        store.append(kind, {"purpose": "preview"})
    ledger = EvaluationExposureLedger(store)
    ledger.register_protocol(protocol, digest(protocol))
    authorize(store, protocol)
    opened = []
    observe_file(monkeypatch, tmp_path / "block.bin", before_open=lambda: opened.append(True))
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        ledger.read("reserved", "test-block", purpose="evaluate", authorization_id="evaluate", data_root=tmp_path)
    assert not opened
    assert store.events()[-1]["event_kind"] == "EXPOSURE_DENIED"


@pytest.mark.parametrize("alias", ["same-content", "same-source"])
def test_actual_result_only_alias_exposure_prevents_late_blind_plan(tmp_path, monkeypatch, alias):
    store, _, protocol = final_eval(tmp_path, None)
    block = protocol["blocks"][0]
    old_content = b"synthetic older source" if alias == "same-source" else (tmp_path / "block.bin").read_bytes()
    (tmp_path / "old-source.bin").write_bytes(old_content)
    old = {**block, "block_id": "foreign-window", "path": "old-source.bin", "release_id": "old",
           "source_block_id": "test-block" if alias == "same-source" else "foreign-window",
           "dataset_id": block["dataset_id"] if alias == "same-source" else "renamed-dataset",
           "sha256": hashlib.sha256(old_content).hexdigest(), "split_role": "train", "fit_scope": True}
    foreign = {"schema_version": "pirc25-data-protocol-v1", "protocol_id": "foreign-inputs",
               "study_id": "foreign-study", "blocks": [old]}
    ledger = EvaluationExposureLedger(store)
    ledger.register_protocol(foreign, digest(foreign))
    grant = {"authorization_id": "foreign-viewer", "study_id": "foreign-study", "purposes": ["preview"],
             "expires_at": "2099-01-01T00:00:00+00:00", "evidence_hash": digest("synthetic result permission"),
             "visibilities": ["synthetic"], "block_ids": ["foreign-window"]}
    store.authorize(grant)
    artifact = store.artifact(b"synthetic old result", role="qualification", visibility="synthetic",
                              block_ids=["foreign-window"], study_id="foreign-study")
    assert store.read_artifact(artifact["artifact_id"], purpose="preview", authorization=grant) == b"synthetic old result"
    assert all("dataset_id" not in event["payload"] for event in store.events() if event["event_kind"].startswith("READ_"))
    protocol, _, _ = freeze_evidence(store, protocol)
    ledger.register_protocol(protocol, digest(protocol))
    authorize(store, protocol)
    opened = []
    observe_file(monkeypatch, tmp_path / "block.bin", before_open=lambda: opened.append(True))
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        ledger.read("reserved", "test-block", purpose="evaluate", authorization_id="evaluate", data_root=tmp_path)
    assert not opened


def later_plan(store, ledger, protocol, plan):
    later = {**protocol, "protocol_id": "later-reserved"}
    value = {**plan, "selection_rule": "a new later plan", "protocol_bindings": [protocol_binding(later)]}
    later["preregistration_hash"] = PreregistrationGate(store).register_preregistration(value, digest(value))
    ledger.register_protocol(later, digest(later))
    store.authorize({**store.authorization("evaluate"), "authorization_id": "later-evaluate",
                     "protocol_hash": digest(later)})


def test_actual_interrupted_scope_update_drops_owner_prefix_before_next_blind_read(tmp_path, monkeypatch):
    store, ledger, _, protocol, plan, _, grant = admitted(tmp_path, 2)
    original = ArtifactExposureFacts._add_keys
    failed, opened = [], []

    def fail_once(facts, keys, sequence, **kwargs):
        result = original(facts, keys, sequence, **kwargs)
        if ("block", "test-block") in keys and not failed:
            failed.append(True)
            raise OSError("scope maintenance interrupted after real update")
        return result

    observe_file(monkeypatch, tmp_path / "block.bin", before_open=lambda: opened.append(True))
    with store._read_transaction():
        assert ledger._read_authority(protocol, protocol["blocks"][0], "evaluate", "evaluate")[0]
        artifact = store.artifact(b"synthetic interrupted result", role="qualification", visibility="synthetic",
                                  block_ids=["test-block"], study_id=grant["study_id"])
        monkeypatch.setattr(ArtifactExposureFacts, "_add_keys", fail_once)
        with pytest.raises(OSError, match="scope maintenance interrupted"):
            store.read_artifact(artifact["artifact_id"], purpose="preview", authorization=grant)
        assert store._read_snapshot()[1] is None and not hasattr(store._read_scope, "event_lookup")
        later_plan(store, ledger, protocol, plan)
        with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
            ledger.read("later-reserved", "test-block", purpose="evaluate",
                        authorization_id="later-evaluate", data_root=tmp_path)
    assert failed == [True] and not opened
    assert any(event["event_kind"] == "READ_STARTED" and event["payload"].get("artifact_id") == artifact["artifact_id"]
               for event in store.events())


def test_unresolved_post_freeze_artifact_is_not_retroactive_but_blocks_later_plan(tmp_path, monkeypatch):
    store, ledger, _, protocol, plan, _, _ = admitted(tmp_path, 2)
    opened = []
    observe_file(monkeypatch, tmp_path / "block.bin", before_open=lambda: opened.append(True))
    with store._read_transaction():
        assert ledger._read_authority(protocol, protocol["blocks"][0], "evaluate", "evaluate")[0]
        store.append("READ_FAILED", {"artifact_id": "missing-legacy-result", "purpose": "preview"})
        assert ledger._read_authority(protocol, protocol["blocks"][0], "evaluate", "evaluate")[0]
        later_plan(store, ledger, protocol, plan)
        with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
            ledger.read("later-reserved", "test-block", purpose="evaluate",
                        authorization_id="later-evaluate", data_root=tmp_path)
    assert not opened


@pytest.mark.parametrize("count", [16, 64])
def test_real_artifact_scope_key_access_is_logarithmic_and_rebuilt_per_physical_owner(tmp_path, count):
    store, ledger, _, protocol, _, _, _ = admitted(tmp_path, count)
    generations = []
    for _ in range(2):
        with store._read_transaction():
            assert ledger._read_authority(protocol, protocol["blocks"][0], "evaluate", "evaluate")[0]
            facts = store._event_lookup()._artifact_facts
            generations.append(facts)
            visits = []

            class ObservedKeys(list):
                def __getitem__(self, position):
                    visits.append(position)
                    return super().__getitem__(position)

            facts._keys = ObservedKeys(facts._keys)
            assert not store._artifact_read_before("test-block", {
                "dataset_id": "fixture", "source_block_id": "test-block", "sha256": digest("unrelated")}, len(store.events()) + 1)
            assert len(visits) <= 4 * (math.ceil(math.log2(len(facts._keys) + 1)) + 1)
        assert not hasattr(store._read_scope, "event_lookup")
    assert generations[0] is not generations[1]


@pytest.mark.parametrize("count", [16, 64])
def test_actual_metadata_view_does_not_use_global_result_candidates_per_cell(tmp_path, monkeypatch, count):
    from application.research_query import ResearchQuery
    from tests.test_research_data_access_view import access_fixture
    store, spec, _, preview = access_fixture(tmp_path)
    old = {"authorization_id": "old-viewer", "study_id": "unrelated-study", "purposes": ["preview"],
           "expires_at": "2099-01-01T00:00:00+00:00", "evidence_hash": digest("synthetic old view"),
           "visibilities": ["synthetic"], "block_ids": ["unrelated-" + str(n) for n in range(count)]}
    with store._read_transaction():
        store.authorize(old)
        for n in range(count):
            content = ("synthetic unrelated result " + str(n)).encode()
            artifact = store.artifact(content, role="qualification", visibility="synthetic",
                                      block_ids=["unrelated-" + str(n)], study_id=old["study_id"])
            assert store.read_artifact(artifact["artifact_id"], purpose="preview", authorization=old) == content
    original, candidates = store._prior_read_events, []

    def observed(identity, before):
        rows = original(identity, before)
        candidates.extend(event for event in rows if event["payload"].get("artifact_id"))
        return rows

    monkeypatch.setattr(store, "_prior_read_events", observed)
    value = ResearchQuery(store, preview["authorization_id"]).data_access()
    assert len(value["items"]) == len(spec["cells"])
    assert all(row["exposure"]["status"] == "NO_RECORDED_EXPOSURE" for row in value["items"])
    assert not candidates, "the metadata view re-enumerated all unrelated result reads for every cell"
