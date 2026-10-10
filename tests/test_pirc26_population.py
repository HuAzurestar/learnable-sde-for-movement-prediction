"""Actual owned preparation; computational chunks are not independent units."""

from copy import deepcopy
import json

import pytest
import torch

from application.pirc26_population import training_population, validate_population, population_source
from application.pirc26_data import read_block, decode_block
from application.research_budget import BudgetLedger
from application.research_data import EvaluationExposureLedger
from infrastructure.research_store import ResearchError, digest, encode
from models.phase_space import DynamicsSpec, AffineAccelerationDrift, PhaseSpaceSDE
from tests.test_pirc26_preparation import prepared_fixture, run
from tests.test_pirc26_dsde import contracts


def test_actual_complete_pool_preserves_every_file_segment_point_and_original_unit(tmp_path, contracts, monkeypatch):
    fixture = prepared_fixture(tmp_path, contracts)
    store = fixture[0]
    outcome = run(fixture)
    assert outcome["state"] == "SUCCEEDED", outcome
    result = outcome["prepared"]
    pool = result["training_population"]
    d, p = pool["document"], pool["provenance"]
    assert len(result["blocks"]) == 3 and len(d["segments"]) == 2
    assert [m["source_output_block_id"] for m in p["members"]] == ["prepared-0", "prepared-1"]
    assert p["independent_block_ids"] == ["train-unit"]
    assert p["observations"] == 10 and p["transitions"] == 6
    assert len({s["segment_id"] for s in d["segments"]}) == 2  # original names collide
    for segment, original, mapping in zip(d["segments"], result["blocks"][:2], p["segments"]):
        original_segment = original["document"]["segments"][0]
        assert {k: v for k, v in segment.items() if k != "segment_id"} == {
            k: v for k, v in original_segment.items() if k != "segment_id"}
        assert mapping["source_segment_id"] == original_segment["segment_id"]
        assert mapping["point_membership_hash"] == digest(original["provenance"]["membership"][0])
    # Pure source lookup must not open the result, which contains selection.
    with monkeypatch.context() as patch:
        patch.setattr(store, "_verified_artifact_content", lambda *a: pytest.fail("metadata lookup read scientific payload"))
        source = population_source(store, outcome["preparation_ref"])
    assert source["provenance"] == p
    entry = source["protocol_entry"]
    assert entry["source_block_id"] != "train-unit"  # honest computational ID
    assert entry["independent_block_ids"] == ["train-unit"]
    assert entry["sha256"] == p["content_sha256"] and entry["size_bytes"] == p["size_bytes"]
    assert entry["training_population_ref"] == outcome["preparation_ref"]
    # No model consumer grant is invented by the production helper. This
    # explicit fixture-only grant exercises the real ledger/read/decode route.
    protocol = {"schema_version": "pirc25-data-protocol-v1", "protocol_id": "pool-input",
                "study_id": "pool-consumer-fixture", "blocks": [entry]}
    EvaluationExposureLedger(store).register_protocol(protocol, digest(protocol))
    store.authorize({"authorization_id": "pool-consumer-grant", "study_id": protocol["study_id"],
        "protocol_hash": digest(protocol), "data_root": source["data_root"],
        "expires_at": "2099-01-01T00:00:00+00:00", "evidence_hash": digest("synthetic pool fixture grant"),
        "block_ids": [entry["block_id"]], "purposes": ["fit"], "visibilities": ["synthetic"], "test_authorization": False})
    admitted = read_block(store, protocol["protocol_id"], entry["block_id"],
                          authorization_id="pool-consumer-grant", purpose="fit")
    assert json.loads(admitted["content_utf8"]) == d
    normalizer = result["normalizer"]
    model = PhaseSpaceSDE(AffineAccelerationDrift(56), [[.1, 0.], [0., .1]],
        DynamicsSpec(d["coordinate_frame"], d["train_binding_hash"], d["normalizer_hash"], d["context_hash"],
                     56, tuple(normalizer["means"]), tuple(normalizer["scales"]))).to(dtype=torch.float64)
    dto = decode_block(admitted, model)
    batches = dto.transitions(batch_size=8)
    assert len(batches) == 2 and sum(len(b.time) for b in batches) == 6
    for batch, segment in zip(batches, dto.segments):
        assert torch.equal(batch.state, segment["state"][:-1])
        assert torch.equal(batch.next_state, segment["state"][1:])
    # Reuse includes the same train-only artifact, with no additional worker,
    # budget charge or train/selection raw reads.
    cost = BudgetLedger(store).balance("affine")["committed_ms"]
    starts = len([e for e in store.events() if e["event_kind"] == "READ_STARTED"])
    reused = run(fixture)
    assert reused["reused"] and reused["prepared"]["training_population"] == pool
    assert population_source(store, reused["preparation_ref"]) == source
    assert BudgetLedger(store).balance("affine")["committed_ms"] == cost
    assert len([e for e in store.events() if e["event_kind"] == "READ_STARTED"]) == starts


def test_complete_population_refuses_omission_reorder_cross_role_and_tampering(tmp_path, contracts):
    fixture = prepared_fixture(tmp_path, contracts)
    outcome = run(fixture)
    assert outcome["state"] == "SUCCEEDED", outcome
    result = outcome["prepared"]
    blocks, binding, normalizer = result["blocks"], result["train_binding"], result["normalizer"]
    for changed in (blocks[1:], [blocks[1], blocks[0], blocks[2]]):
        with pytest.raises(ResearchError, match="complete frozen membership"):
            training_population(changed, binding, normalizer, study_id=fixture[1]["study_id"])
    changed = deepcopy(blocks)
    changed[1]["provenance"]["split_role"] = "selection"
    with pytest.raises(ResearchError):
        training_population(changed, binding, normalizer, study_id=fixture[1]["study_id"])
    for field in ("document", "provenance"):
        changed_pool = deepcopy(result["training_population"])
        if field == "document":
            changed_pool[field]["segments"][0]["position"][0][0] += 1.
        else:
            changed_pool[field]["segments"][0]["independent_block_id"] = "invented-unit"
        with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
            validate_population(changed_pool, blocks, binding, normalizer, study_id=fixture[1]["study_id"])


def test_complete_population_quota_rejects_without_silently_truncating(tmp_path, contracts, monkeypatch):
    fixture = prepared_fixture(tmp_path, contracts)
    outcome = run(fixture)
    assert outcome["state"] == "SUCCEEDED", outcome
    result = outcome["prepared"]
    import application.pirc26_population as population
    for field, limit in (("MAX_BYTES", 1), ("MAX_SEGMENTS", 1), ("MAX_OBSERVATIONS", 9)):
        with monkeypatch.context() as patch:
            patch.setattr(population, field, limit)
            with pytest.raises(ResearchError, match="RESOURCE_PLAN_REJECTED"):
                training_population(result["blocks"], result["train_binding"], result["normalizer"],
                                    study_id=fixture[1]["study_id"])


def test_population_publication_is_shared_within_source_study_not_across_studies(tmp_path, contracts):
    from application.pirc26_preparation import PreparationRunner
    from application.research_budget import BudgetSpec
    fixture = prepared_fixture(tmp_path, contracts)
    store, original, selections, settings = fixture
    first = run(fixture)
    second = PreparationRunner(store).run(original["study_id"], digest(original["cells"][1]),
        selections, settings, budget=BudgetSpec(60, category="smoke"))
    assert first["state"] == second["state"] == "SUCCEEDED", (first, second)
    assert first["preparation_ref"]["attempt_id"] != second["preparation_ref"]["attempt_id"]
    assert first["prepared"]["training_population"] == second["prepared"]["training_population"]
    for result in (first, second):
        source = population_source(store, result["preparation_ref"])
        artifact = store.manifest("artifact-" + source["protocol_entry"]["path"])
        assert artifact["study_id"] == original["study_id"]
    prepared = first["prepared"]
    other = training_population(prepared["blocks"], prepared["train_binding"], prepared["normalizer"], study_id="another-study")
    assert other["provenance"]["content_sha256"] != prepared["training_population"]["provenance"]["content_sha256"]
    assert other["document"]["train_binding_hash"] == prepared["training_population"]["document"]["train_binding_hash"]


def test_train_source_lookup_rechecks_original_authority_after_final_physical_scope(tmp_path, contracts, monkeypatch):
    from datetime import datetime, timezone
    import application.pirc26_preparation as preparation
    import application.research_data as data
    fixture = prepared_fixture(tmp_path, contracts)
    outcome = run(fixture)
    assert outcome["state"] == "SUCCEEDED", outcome
    store = fixture[0]
    expired = [False]
    class PermissionClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2100, 1, 1, tzinfo=timezone.utc) if expired[0] else datetime.now(tz)
    monkeypatch.setattr(preparation, "datetime", PermissionClock)
    monkeypatch.setattr(data, "datetime", PermissionClock)
    completion = store._read_completion
    def expire_after_authority(authority, expiry):
        def checked():
            authority()
            expired[0] = True
        completion(checked, expiry)
    monkeypatch.setattr(store, "_read_completion", expire_after_authority)
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        population_source(store, outcome["preparation_ref"])
