"""Actual preparation, separate selection artifact and unchanged train fit."""

from copy import deepcopy
from datetime import datetime, timezone
import json

import pytest
import torch

from application.pirc26_data import read_block, decode_block
from application.pirc26_selection import selection_populations, selection_source
from application.research_data import EvaluationExposureLedger
from infrastructure.research_store import ResearchError, digest, encode
from models.phase_space import DynamicsSpec, AffineAccelerationDrift, PhaseSpaceSDE
from tests.test_pirc26_preparation import prepared_fixture, run
from tests.test_pirc26_dsde import contracts


def test_actual_selection_preserves_all_files_one_unit_and_requires_distinct_read_grant(tmp_path, contracts, monkeypatch):
    fixture = prepared_fixture(tmp_path, contracts, selection_files=2,
        consumer_study_ids=["selection-fixture-consumer"])
    store = fixture[0]
    outcome = run(fixture)
    assert outcome["state"] == "SUCCEEDED", outcome
    value = outcome["prepared"]
    selected = selection_populations(value["blocks"], value["normalizer"], study_id=fixture[1]["study_id"])
    assert len(selected) == 1
    document, metadata = selected[0]["document"], selected[0]["provenance"]
    assert metadata["independent_block_id"] == "selection-unit" and metadata["observations"] == 10
    assert [m["source_output_block_id"] for m in metadata["members"]] == ["prepared-2", "prepared-3"]
    assert len(document["segments"]) == 2 and len({s["segment_id"] for s in document["segments"]}) == 2
    for segment, original, mapping in zip(document["segments"], value["blocks"][2:], metadata["segments"]):
        assert {k:v for k,v in segment.items() if k != "segment_id"} == {
            k:v for k,v in original["document"]["segments"][0].items() if k != "segment_id"}
        assert mapping["point_membership_hash"] == digest(original["provenance"]["membership"][0])
    # A prospective selection consumer must not open the mixed-role result or
    # even the train-only derived geometry while inspecting source metadata.
    with monkeypatch.context() as patch:
        patch.setattr(store, "_verified_artifact_content", lambda *a,**k: pytest.fail("metadata lookup read private payload"))
        actual = selection_source(store, outcome["preparation_ref"], "selection-unit",
            consumer_study_id="selection-fixture-consumer")
    assert actual["provenance"] == metadata and actual["normalizer"] == value["normalizer"]
    assert actual["protocol_entry"]["source_block_id"] == "selection-unit"
    assert actual["protocol_entry"]["split_role"] == "selection" and not actual["protocol_entry"]["fit_scope"]
    protocol = {"schema_version":"pirc25-data-protocol-v1", "protocol_id":"selection-fixture",
        "study_id":"selection-fixture-consumer", "blocks":[actual["protocol_entry"]]}
    EvaluationExposureLedger(store).register_protocol(protocol, digest(protocol))
    grant = {"authorization_id":"selection-fixture-grant", "study_id":protocol["study_id"],
        "protocol_hash":digest(protocol), "expires_at":"2099-01-01T00:00:00+00:00",
        "evidence_hash":digest("explicit disposable derived selection permission"), "data_root":actual["data_root"],
        "purposes":["select"], "visibilities":["synthetic"], "block_ids":[document["block_id"]]}
    store.authorize(grant)
    transport = read_block(store, protocol["protocol_id"], document["block_id"],
        authorization_id=grant["authorization_id"], purpose="select")
    assert json.loads(transport["content_utf8"]) == document
    n = value["normalizer"]
    m = PhaseSpaceSDE(AffineAccelerationDrift(56), [[.1,0.],[0.,.1]], DynamicsSpec(
        metadata["coordinate_frame"], n["train_binding_hash"], digest(n), n["context_hash"], 56,
        tuple(n["means"]), tuple(n["scales"]))).to(dtype=torch.float64)
    dto = decode_block(transport, m)
    assert len(dto.segments) == 2
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        dto.transitions(batch_size=4)
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        selection_source(store, outcome["preparation_ref"], "selection-unit", consumer_study_id="unapproved-consumer")
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        selection_source(store, outcome["preparation_ref"], "train-unit")
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        selection_source(store, outcome["preparation_ref"], "selection-unit", arm={**fixture[1]["arms"][0],"budget_seconds":1})
    proof = store.manifest(outcome["preparation_ref"]["manifest_id"])
    real_manifest = store._manifest
    for fault in ("missing-file", "reorder", "normalizer", "role", "unit"):
        changed = deepcopy(proof)
        p = changed["selection_populations"][0]["provenance"]
        if fault == "missing-file":
            p["members"].pop()
        elif fault == "reorder":
            p["members"].reverse()
        elif fault == "normalizer":
            p["normalizer_hash"] = "f" * 64
        elif fault == "role":
            p["purpose"] = "fit"
        else:
            p["members"][0]["independent_block_id"] = "invented-unit"
        with monkeypatch.context() as patch:
            patch.setattr(store, "_manifest", lambda object_id: changed if object_id == outcome["preparation_ref"]["manifest_id"] else real_manifest(object_id))
            with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
                selection_source(store, outcome["preparation_ref"], "selection-unit")
    assert value["normalizer"]["independent_block_ids"] == ["train-unit"]
    before = len(store.attempts())
    reused = run(fixture)
    assert reused["reused"] and reused["prepared"] == value and len(store.attempts()) == before


def test_selection_publication_scope_and_quotas_preserve_train_transform(tmp_path, contracts, monkeypatch):
    fixture = prepared_fixture(tmp_path, contracts, selection_files=2)
    outcome = run(fixture)
    assert outcome["state"] == "SUCCEEDED", outcome
    value = outcome["prepared"]
    selected = selection_populations(value["blocks"], value["normalizer"], study_id=fixture[1]["study_id"])
    other = selection_populations(value["blocks"], value["normalizer"], study_id="another-source-study")
    assert other[0]["provenance"]["content_sha256"] != selected[0]["provenance"]["content_sha256"]
    assert other[0]["provenance"]["normalizer_hash"] == selected[0]["provenance"]["normalizer_hash"]
    import application.pirc26_selection as selection
    for key, quota in (("MAX_BYTES",1),("MAX_SEGMENTS",1),("MAX_OBSERVATIONS",9)):
        with monkeypatch.context() as patch:
            patch.setattr(selection,key,quota)
            with pytest.raises(ResearchError,match="RESOURCE_PLAN_REJECTED"):
                selection_populations(value["blocks"], value["normalizer"], study_id=fixture[1]["study_id"])
    changed = deepcopy(value["blocks"])
    changed[-1]["provenance"]["feature"]["independent_block_id"] = "train-unit"
    with pytest.raises(ResearchError,match="UNAUTHORIZED_DATA"):
        selection_populations(changed,value["normalizer"],study_id=fixture[1]["study_id"])


def test_original_selection_permission_is_rechecked_after_final_physical_scope(tmp_path, contracts, monkeypatch):
    fixture = prepared_fixture(tmp_path, contracts)
    outcome = run(fixture)
    assert outcome["state"] == "SUCCEEDED", outcome
    store = fixture[0]
    expired = [False]
    class PermissionClock(datetime):
        @classmethod
        def now(cls,tz=None):
            return datetime(2100,1,1,tzinfo=timezone.utc) if expired[0] else datetime.now(tz)
    import application.pirc26_preparation as preparation
    import application.research_data as data
    monkeypatch.setattr(preparation,"datetime",PermissionClock)
    monkeypatch.setattr(data,"datetime",PermissionClock)
    completion = store._read_completion
    def expire_after_authority(authority,expiry):
        def checked():
            authority()
            expired[0] = True
        completion(checked,expiry)
    monkeypatch.setattr(store,"_read_completion",expire_after_authority)
    with pytest.raises(ResearchError,match="UNAUTHORIZED_DATA"):
        selection_source(store,outcome["preparation_ref"],"selection-unit")
