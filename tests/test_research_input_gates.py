"""Independent R1/R3 counterexamples and frozen input-gate contracts."""

import hashlib

import pytest

from application.research_data import EvaluationExposureLedger
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
