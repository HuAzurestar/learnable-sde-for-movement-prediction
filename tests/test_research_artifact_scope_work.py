"""Whole public legacy-disclosure work, separate from physical prefix scans.

Every result, file operation and exposure event is real. Only disposable
synthetic inputs are used; no permission, clock or work result is substituted.
"""

import pytest

from application.research_data import EvaluationExposureLedger
from application.research_preregistration import PreregistrationGate, protocol_binding
from infrastructure.research_store import ResearchError, digest, encode
from tests.research_file_observation import observe_file
from tests.test_research_input_gates import final_eval, freeze_evidence, authorize


def prior_artifacts(tmp_path, count):
    store, content, protocol = final_eval(tmp_path, None)
    grant = {"authorization_id": "old-viewer", "study_id": "unrelated-study",
             "expires_at": "2099-01-01T00:00:00+00:00",
             "evidence_hash": digest("synthetic prior view"), "purposes": ["preview"],
             "visibilities": ["synthetic"],
             "block_ids": ["test-block", *["unrelated-" + str(n) for n in range(count)]]}
    artifacts = []
    with store._read_transaction():
        store.authorize(grant)
        for number in range(count):
            data = ("synthetic unrelated result " + str(number)).encode()
            artifact = store.artifact(data, role="qualification", visibility="synthetic",
                                      block_ids=["unrelated-" + str(number)], study_id="unrelated-study")
            assert store.read_artifact(artifact["artifact_id"], purpose="preview", authorization=grant) == data
            artifacts.append(artifact)
    return store, content, protocol, artifacts, grant


def admitted(tmp_path, count):
    store, content, protocol, artifacts, old_grant = prior_artifacts(tmp_path, count)
    protocol, plan, _ = freeze_evidence(store, protocol)
    ledger = EvaluationExposureLedger(store)
    ledger.register_protocol(protocol, digest(protocol))
    authorize(store, protocol)
    return store, ledger, content, protocol, plan, artifacts, old_grant


@pytest.mark.parametrize("count", [16, 64])
@pytest.mark.parametrize("requests", [1, 4])
def test_whole_public_requests_do_not_repeat_global_legacy_artifact_manifest_work(
        tmp_path, monkeypatch, count, requests):
    store, ledger, content, _, _, _, _ = admitted(tmp_path, count)
    initial = len(store.events())
    original_json = store._json
    physical, manifests, opened = [], [], []

    def metadata(path):
        value = original_json(path)
        if path.parent == store.path / "events":
            physical.append(path.name)
        elif path.parent == store.path / "manifests" and path.name.startswith("artifact-"):
            manifests.append(path.name)
        return value

    monkeypatch.setattr(store, "_json", metadata)
    observe_file(monkeypatch, tmp_path / "block.bin", before_open=lambda: opened.append(True))
    with store._read_transaction():
        for _ in range(requests):
            assert ledger.read("reserved", "test-block", purpose="evaluate",
                               authorization_id="evaluate", data_root=tmp_path) == content
    physical_count = len(physical)
    events = store.events()
    assert len(opened) == requests
    assert [event["event_kind"] for event in events[initial:]] == [
        "EXPOSURE_ALLOWED", "READ_STARTED", "READ_COMPLETED"] * requests
    assert physical_count == initial + len(events), "retain both original complete physical prefix validations"
    # Permit one source-bound preprocessing pass at each physical boundary,
    # plus a constant number of selected-input guards per requested block.
    # This is a necessary whole-request bound, not a sufficient proof of the
    # complete O(B log E) contract or a wall-clock benchmark.
    assert len(manifests) <= 2 * count + 6 * requests, (
        f"{requests} actual provider requests reread {len(manifests)} unrelated artifact manifests "
        f"for {count} original managed disclosures")


def test_real_unresolved_legacy_scope_denies_before_reserved_provider(tmp_path, monkeypatch):
    store, ledger, _, _, _, artifacts, _ = admitted(tmp_path, 2)
    path = store.path / "manifests" / ("artifact-" + artifacts[0]["artifact_id"] + ".json")
    # This is an actual source-file change, not a fabricated lookup failure.
    path.write_bytes(encode({**artifacts[0], "block_ids": ["test-block"]}))
    opened = []
    observe_file(monkeypatch, tmp_path / "block.bin", before_open=lambda: opened.append(True))
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        ledger.read("reserved", "test-block", purpose="evaluate",
                    authorization_id="evaluate", data_root=tmp_path)
    assert not opened
    assert store.events()[-1]["event_kind"] == "EXPOSURE_DENIED"


def test_real_prior_scope_change_during_provider_never_returns_content(tmp_path, monkeypatch):
    store, ledger, _, _, _, artifacts, _ = admitted(tmp_path, 2)
    changed = []
    path = store.path / "manifests" / ("artifact-" + artifacts[0]["artifact_id"] + ".json")

    def corrupt(stream, content):
        if not changed:
            path.write_bytes(encode({**artifacts[0], "block_ids": ["test-block"]}))
            changed.append(True)

    observe_file(monkeypatch, tmp_path / "block.bin", after_read=corrupt)
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        ledger.read("reserved", "test-block", purpose="evaluate",
                    authorization_id="evaluate", data_root=tmp_path)
    assert changed == [True]
    assert [event["event_kind"] for event in store.events()][-2:] == ["EXPOSURE_DENIED", "READ_FAILED"]


def test_actual_appended_legacy_disclosure_blocks_later_plan_in_same_owner(tmp_path, monkeypatch):
    store, ledger, _, protocol, plan, _, old_grant = admitted(tmp_path, 2)
    opened = []
    observe_file(monkeypatch, tmp_path / "block.bin", before_open=lambda: opened.append(True))
    with store._read_transaction():
        assert ledger._read_authority(protocol, protocol["blocks"][0], "evaluate", "evaluate")[0]
        data = b"synthetic disclosed reserved metric"
        target = store.artifact(data, role="qualification", visibility="synthetic",
                                block_ids=["test-block"], study_id=old_grant["study_id"])
        assert store.read_artifact(target["artifact_id"], purpose="preview", authorization=old_grant) == data
        later = {**protocol, "protocol_id": "later-reserved"}
        later_plan = {**plan, "selection_rule": "a distinct later plan",
                      "protocol_bindings": [protocol_binding(later)]}
        later["preregistration_hash"] = PreregistrationGate(store).register_preregistration(
            later_plan, digest(later_plan))
        ledger.register_protocol(later, digest(later))
        grant = {**store.authorization("evaluate"), "authorization_id": "later-evaluate",
                 "protocol_hash": digest(later)}
        store.authorize(grant)
        with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
            ledger.read("later-reserved", "test-block", purpose="evaluate",
                        authorization_id="later-evaluate", data_root=tmp_path)
    assert not opened
    assert any(event["event_kind"] == "EXPOSURE_DENIED" for event in store.events())
