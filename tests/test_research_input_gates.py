"""Independent R1/R3 counterexamples and frozen input-gate contracts."""

import hashlib
from pathlib import Path

import pytest

from application.research_data import EvaluationExposureLedger
from application.research_preregistration import PreregistrationGate, protocol_binding, source_identity
from infrastructure.research_store import ResearchError, ResearchStore, digest


def final_eval(tmp_path, preregistration_hash):
    store = ResearchStore(tmp_path, "input-gates", initialize=True)
    data = b"synthetic reserved test data"
    (tmp_path / "block.bin").write_bytes(data)
    protocol = {"schema_version": "pirc25-data-protocol-v1", "protocol_id": "reserved",
                "study_id": "synthetic", "preregistration_hash": preregistration_hash,
                "history_status": "known", "blocks": [{"block_id": "test-block",
                "dataset_id": "fixture", "release_id": "v1", "split_role": "final-eval",
                "fit_scope": False, "path": "block.bin", "sha256": hashlib.sha256(data).hexdigest()}]}
    return store, data, protocol


def authorize(store, protocol):
    store.authorize({"authorization_id": "evaluate", "study_id": "synthetic",
        "expires_at": "2099-01-01T00:00:00+00:00", "evidence_hash": digest("synthetic permission"),
        "protocol_hash": digest(protocol), "purposes": ["evaluate"], "visibilities": ["restricted"],
        "block_ids": ["test-block"], "test_authorization": True})


@pytest.mark.parametrize("reference", [None, "", " ", "z" * 64, "0" * 64])
def test_r3_final_eval_rejects_absent_invalid_or_unresolved_preregistration(tmp_path, reference):
    store, _, protocol = final_eval(tmp_path, reference)
    ledger = EvaluationExposureLedger(store)
    ledger.register_protocol(protocol, digest(protocol))
    authorize(store, protocol)
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        ledger.read("reserved", "test-block", purpose="evaluate", authorization_id="evaluate", data_root=tmp_path)
    assert not any(e["event_kind"] == "READ_STARTED" for e in store.events())
    assert store.events()[-1]["event_kind"] == "EXPOSURE_DENIED"


def freeze_evidence(store, protocol, *, history_status="unexposed", mode="blind", wrong_scope=False):
    gate = PreregistrationGate(store)
    report = {"records": [{**source_identity(protocol["blocks"][0]), "status": history_status}]}
    history = {"schema_version": "pirc25-exposure-history-v1", "source": "synthetic external ledger fixture",
               "source_evidence": report, "source_evidence_hash": digest(report)}
    history_hash = gate.import_history(history, digest(history))
    plan = {"schema_version": "pirc25-preregistration-v1", "test_mode": mode,
            "study_ids": [protocol["study_id"]],
            "protocol_bindings": [digest("wrong protocol") if wrong_scope else protocol_binding(protocol)],
            "primary_metrics": ["error"], "selection_rule": "freeze before reserved read",
            "stopping_rule": "fixed registered matrix", "comparisons": [{"reference": "a", "candidate": "b"}]}
    prereg = gate.register_preregistration(plan, digest(plan))
    return {**protocol, "preregistration_hash": prereg, "history_hash": history_hash}, plan, history


def read(ledger, tmp_path):
    return ledger.read("reserved", "test-block", purpose="evaluate", authorization_id="evaluate", data_root=tmp_path)


def test_r3_valid_frozen_evidence_allows_repeat_reads_and_binds_audit(tmp_path):
    store, content, protocol = final_eval(tmp_path, None)
    protocol, plan, history = freeze_evidence(store, protocol)
    ledger = EvaluationExposureLedger(store)
    ledger.register_protocol(protocol, digest(protocol))
    authorize(store, protocol)
    assert read(ledger, tmp_path) == read(ledger, tmp_path) == content
    starts = [e for e in store.events() if e["event_kind"] == "READ_STARTED"]
    assert len(starts) == 2
    assert all(e["payload"]["preregistration_hash"] == digest(plan) and
               e["payload"]["history_hash"] == digest(history) and
               e["payload"]["sha256"] == protocol["blocks"][0]["sha256"] for e in starts)
    assert all(e["payload"]["frozen_sequence"] < e["sequence"] for e in starts)


@pytest.mark.parametrize("mutation", ["wrong-protocol", "missing-history", "unknown-history", "exposed-history",
                                     "changed-block", "forged-known", "changed-metrics"])
def test_r3_rejects_invalid_evidence_binding_before_data_access(tmp_path, mutation, monkeypatch):
    store, _, protocol = final_eval(tmp_path, None)
    status = {"unknown-history": "unknown", "exposed-history": "exposed"}.get(mutation, "unexposed")
    protocol, plan, _ = freeze_evidence(store, protocol, history_status=status, wrong_scope=mutation == "wrong-protocol")
    if mutation == "missing-history":
        protocol.pop("history_hash")
    elif mutation == "changed-block":
        protocol["blocks"][0]["sha256"] = "b" * 64
    elif mutation == "forged-known":
        protocol["history_hash"] = digest("not registered")
        protocol["history_status"] = "known"
    elif mutation == "changed-metrics":
        # Direct publication under the old hash must not forge a frozen record.
        plan["primary_metrics"] = ["post-hoc"]
        path = store.path / "manifests" / ("preregistration-" + protocol["preregistration_hash"] + ".json")
        from infrastructure.research_store import encode
        path.write_bytes(encode(plan))
    ledger = EvaluationExposureLedger(store)
    ledger.register_protocol(protocol, digest(protocol))
    authorize(store, protocol)
    from tests.research_file_observation import observe_file
    def forbid_data():
        pytest.fail("gate opened data before rejecting")
    observe_file(monkeypatch, tmp_path / "block.bin", before_open=forbid_data)
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        read(ledger, tmp_path)
    assert store.events()[-1]["event_kind"] == "EXPOSURE_DENIED"


def test_r3_cannot_reset_imported_history_by_registering_clean_report(tmp_path):
    store, _, protocol = final_eval(tmp_path, None)
    freeze_evidence(store, protocol, history_status="exposed")
    protocol, _, _ = freeze_evidence(store, protocol)
    ledger = EvaluationExposureLedger(store)
    ledger.register_protocol(protocol, digest(protocol))
    authorize(store, protocol)
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        read(ledger, tmp_path)


def test_r3_imported_exposure_of_identical_content_under_another_dataset_blocks_blind_read(tmp_path):
    store, _, protocol = final_eval(tmp_path, None)
    source = {**source_identity(protocol["blocks"][0]), "dataset_id": "old-dataset",
              "release_id": "old-release", "source_block_id": "old-window"}
    report = {"records": [{**source, "status": "exposed"}]}
    prior = {"schema_version": "pirc25-exposure-history-v1", "source": "synthetic prior report",
             "source_evidence": report, "source_evidence_hash": digest(report)}
    PreregistrationGate(store).import_history(prior, digest(prior))
    protocol, _, _ = freeze_evidence(store, protocol)
    ledger = EvaluationExposureLedger(store)
    ledger.register_protocol(protocol, digest(protocol))
    authorize(store, protocol)
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        read(ledger, tmp_path)


@pytest.mark.parametrize("renamed", ["none", "window", "dataset"])
def test_r3_new_plan_after_prior_read_cannot_claim_blindness(tmp_path, renamed):
    store, _, protocol = final_eval(tmp_path, None)
    source = source_identity(protocol["blocks"][0])
    # A controlled read under another study, including a failed/partial read,
    # predates this plan; a new study/block/release name must not clear it.
    if renamed == "window":
        source.update(source_block_id="old-window", release_id="old-release")
    elif renamed == "dataset":
        source.update(dataset_id="old-dataset", source_block_id="old-window", release_id="old-release")
    store.append("READ_STARTED", {**source, "study_id": "previous-study", "purpose": "select"})
    protocol, _, _ = freeze_evidence(store, protocol)
    ledger = EvaluationExposureLedger(store)
    ledger.register_protocol(protocol, digest(protocol))
    authorize(store, protocol)
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        read(ledger, tmp_path)


def test_r3_exploratory_mode_preserves_known_exposure_without_claiming_blindness(tmp_path):
    store, content, protocol = final_eval(tmp_path, None)
    protocol, _, _ = freeze_evidence(store, protocol, history_status="exposed", mode="exploratory")
    ledger = EvaluationExposureLedger(store)
    ledger.register_protocol(protocol, digest(protocol))
    authorize(store, protocol)
    assert read(ledger, tmp_path) == content
    assert store.events()[-1]["payload"]["test_mode"] == "exploratory"


def test_r3_audit_failure_prevents_read(tmp_path, monkeypatch):
    store, _, protocol = final_eval(tmp_path, None)
    protocol, _, _ = freeze_evidence(store, protocol)
    ledger = EvaluationExposureLedger(store)
    ledger.register_protocol(protocol, digest(protocol))
    authorize(store, protocol)
    append = store._append
    def fail(kind, *args, **kwargs):
        if kind == "READ_STARTED":
            raise OSError("audit unavailable")
        return append(kind, *args, **kwargs)
    monkeypatch.setattr(store, "_append", fail)
    from tests.research_file_observation import observe_file
    def forbid_data():
        pytest.fail("failed audit opened data")
    observe_file(monkeypatch, tmp_path / "block.bin", before_open=forbid_data)
    with pytest.raises(OSError, match="audit unavailable"):
        read(ledger, tmp_path)


@pytest.mark.parametrize("field,value", [("primary_metrics", []), ("selection_rule", " "),
    ("stopping_rule", ""), ("comparisons", []), ("protocol_bindings", ["z" * 64])])
def test_r3_incomplete_preregistration_cannot_be_frozen(tmp_path, field, value):
    store, _, protocol = final_eval(tmp_path, None)
    _, plan, _ = freeze_evidence(store, protocol)
    plan[field] = value
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        PreregistrationGate(store).register_preregistration(plan, digest(plan))


def test_r3_history_content_cannot_be_substituted_under_a_hash(tmp_path):
    store, _, protocol = final_eval(tmp_path, None)
    _, _, history = freeze_evidence(store, protocol)
    history["source_evidence"]["records"][0]["status"] = "exposed"
    with pytest.raises(ResearchError, match="content/hash"):
        PreregistrationGate(store).import_history(history, digest(history))


def test_r3_prior_result_disclosure_also_prevents_late_blind_registration(tmp_path):
    store, _, protocol = final_eval(tmp_path, None)
    artifact = store.artifact(b'{"metric":1}', role="result", visibility="restricted",
                              block_ids=["test-block"], study_id="older-study")
    grant = {"authorization_id": "older-preview", "study_id": "older-study",
        "expires_at": "2099-01-01T00:00:00+00:00", "evidence_hash": digest("older-permission"),
        "purposes": ["preview"], "visibilities": ["restricted"], "block_ids": ["test-block"]}
    store.authorize(grant)
    store.read_artifact(artifact["artifact_id"], purpose="preview", authorization=grant)
    protocol, _, _ = freeze_evidence(store, protocol)
    ledger = EvaluationExposureLedger(store)
    ledger.register_protocol(protocol, digest(protocol))
    authorize(store, protocol)
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        read(ledger, tmp_path)


def test_r3_incomplete_legacy_read_identity_is_not_assumed_unexposed(tmp_path):
    store, _, protocol = final_eval(tmp_path, None)
    store.append("READ_STARTED", {"dataset_id": "fixture", "release_id": "v1",
                                  "block_id": "legacy-window", "study_id": "old"})
    protocol, _, _ = freeze_evidence(store, protocol)
    ledger = EvaluationExposureLedger(store)
    ledger.register_protocol(protocol, digest(protocol))
    authorize(store, protocol)
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        read(ledger, tmp_path)


def test_r3_joint_plan_allows_only_studies_frozen_before_shared_reads(tmp_path):
    store, content, protocol = final_eval(tmp_path, None)
    _, plan, history = freeze_evidence(store, protocol)
    second = {**protocol, "study_id": "joint-study", "protocol_id": "joint-protocol"}
    plan["study_ids"].append("joint-study")
    plan["protocol_bindings"].append(protocol_binding(second))
    reference = PreregistrationGate(store).register_preregistration(plan, digest(plan))
    ledger = EvaluationExposureLedger(store)
    for value in (protocol, second):
        value.update(preregistration_hash=reference, history_hash=digest(history))
        ledger.register_protocol(value, digest(value))
        grant_id = "grant-" + value["study_id"]
        store.authorize({"authorization_id": grant_id, "study_id": value["study_id"],
            "expires_at": "2099-01-01T00:00:00+00:00", "evidence_hash": digest("joint permission"),
            "protocol_hash": digest(value), "purposes": ["evaluate"], "visibilities": ["restricted"],
            "block_ids": ["test-block"], "test_authorization": True})
        assert ledger.read(value["protocol_id"], "test-block", purpose="evaluate",
                           authorization_id=grant_id, data_root=tmp_path) == content
