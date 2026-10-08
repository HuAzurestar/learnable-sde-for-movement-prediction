"""Disposable actual-worker proof; no production grant, data or store access."""

from copy import deepcopy
from datetime import datetime, timezone

import numpy as np
import pyarrow as pa
import pytest

from application.pirc26_dsde import ProjectionSpec
from application.pirc26_preparation import (PreparationRunner, preparation_settings, load_prepared,
    _validate_result, _verify_reads, MAX_PAIRS, NORMALIZER_POLICY)
from application.research_budget import BudgetLedger, BudgetSpec
from application.research_data import EvaluationExposureLedger
from infrastructure.research_store import ResearchStore, ResearchError, digest, encode
from tests.test_pirc26_dsde import contracts, sources, pair_for
from tests.test_research_store import spec


def prepared_fixture(tmp_path, contracts, *, selection_shift=0., missing_grant=False, cross_role_unit=False):
    store = ResearchStore(tmp_path / "ledger", "managed-preparation-fixture", initialize=True)
    pairs, selections, blocks = [], [], []
    for i, role in enumerate(("train", "train", "selection")):
        feature, condition, entry = sources(contracts, role=role)
        unit = "train-unit" if role == "train" or cross_role_unit else "selection-unit"
        file_id = "fixture-file-" + str(i)
        for name, value in {"file_id": file_id, "independent_block_id": unit}.items():
            feature = feature.set_column(feature.schema.get_field_index(name), name, pa.array([value] * len(feature)))
        condition = condition.set_column(condition.schema.get_field_index("file_id"), "file_id", pa.array([file_id] * len(condition)))
        # Separate train files, same original independent unit. Selection can
        # be moved without influencing train statistics or train membership.
        shift = i * .0001 if role == "train" else selection_shift
        condition = condition.set_column(condition.schema.get_field_index("lat"), "lat",
            pa.array([20. + j * .00001 + shift for j in range(len(condition))]))
        entry = {**entry, "file_id": file_id, "independent_block_id": unit, "source_block_id": unit}
        pair = pair_for(feature, condition, entry)
        f, c = pair.metadata()
        f = {**f, "block_id": "features-" + str(i), "path": str(i) + "-features.parquet"}
        c = {**c, "block_id": "conditions-" + str(i), "path": str(i) + "-conditions.parquet"}
        (tmp_path / f["path"]).write_bytes(pair.feature_bytes)
        (tmp_path / c["path"]).write_bytes(pair.condition_bytes)
        blocks.extend((f, c))
        pairs.append(pair)
        selections.append({"protocol_id": "fixture-sources", "feature_block_id": f["block_id"],
            "condition_block_id": c["block_id"], "authorization_id": "fixture-grant", "authorization_version": None,
            "output_block_id": "prepared-" + str(i), "purpose": "fit" if role == "train" else "select"})
    value = spec()
    value["study_id"] = "fixture-preparation"
    value["cells"] = [{"arm_id": "affine", "seed": 1, "block_id": unit, "visibility": "synthetic"}
                      for unit in sorted({b["independent_block_id"] for b in blocks})]
    protocol = {"schema_version": "pirc25-data-protocol-v1", "protocol_id": "fixture-sources",
                "study_id": value["study_id"], "blocks": blocks}
    value["protocol_hash"] = digest(protocol)
    EvaluationExposureLedger(store).register_protocol(protocol, digest(protocol))
    store.authorize({"authorization_id": "fixture-grant", "study_id": value["study_id"], "protocol_hash": digest(protocol),
        "expires_at": "2099-01-01T00:00:00+00:00", "evidence_hash": digest("synthetic test-only permission"),
        "data_root": str(tmp_path), "block_ids": [b["block_id"] for b in blocks if not missing_grant or b["block_id"] != "conditions-2"],
        "purposes": ["fit", "select"], "visibilities": ["synthetic"], "test_authorization": False})
    store.register(value, digest(value))
    settings = preparation_settings(*contracts, ProjectionSpec("fixture-metric-frame", 20., 110., .1))
    return store, value, selections, settings


def run(fixture):
    store, original, selections, settings = fixture
    return PreparationRunner(store).run(original["study_id"], digest(original["cells"][0]), selections, settings,
                                        budget=BudgetSpec(60, category="smoke"))


def test_actual_managed_preparation_normalizes_only_train_and_preserves_read_cost_and_units(tmp_path, contracts):
    fixture = prepared_fixture(tmp_path, contracts)
    store, original, selections, settings = fixture
    outcome = run(fixture)
    assert outcome["state"] == "SUCCEEDED", outcome
    result = load_prepared(store, outcome["preparation_ref"])
    assert result == outcome["prepared"]
    normalizer = result["normalizer"]
    assert normalizer["policy"] == NORMALIZER_POLICY
    assert normalizer["independent_block_ids"] == ["train-unit"]  # two files != two units
    assert normalizer["observations"] == 8
    assert len(result["blocks"]) == 3
    expected = []
    for block in result["blocks"][:2]:
        s = block["document"]["segments"][0]
        p, t, c = np.asarray(s["position"]), np.asarray(s["time"]), np.asarray(s["condition"])
        expected.append(np.column_stack((p[1:], np.diff(p, axis=0) / np.diff(t)[:, None], c[1:])))
    expected = np.concatenate(expected)
    np.testing.assert_allclose(normalizer["means"], expected.mean(axis=0), rtol=0, atol=1e-14)
    scales = expected.std(axis=0)
    scales[scales == 0] = 1.
    np.testing.assert_array_equal(normalizer["scales"], scales)
    assert all(b["document"]["normalizer_hash"] == digest(normalizer) for b in result["blocks"])
    assert result["scientific_qualification"] == "not-established"
    events = store.events()
    starts = [e for e in events if e["event_kind"] == "READ_STARTED"]
    completions = [e for e in events if e["event_kind"] == "READ_COMPLETED"]
    assert len(starts) == len(completions) == 6
    workers = [e for e in events if e["event_kind"] == "WORKER_STARTED"]
    assert len(workers) == 1 and completions[-1]["sequence"] < workers[0]["sequence"]
    proof = store.manifest(outcome["preparation_ref"]["manifest_id"])
    assert proof["cost"]["arm_id"] == original["arms"][0]["arm_id"]
    assert proof["cost"]["charged_ms"] > 0
    before = BudgetLedger(store).balance("affine")["committed_ms"]
    reused = run(fixture)
    assert reused["reused"] and reused["prepared"] == result
    assert BudgetLedger(store).balance("affine")["committed_ms"] == before
    assert len(store.attempts()) == len(workers) == 1
    assert len([e for e in store.events() if e["event_kind"] == "READ_STARTED"]) == 6


def test_selection_changes_cannot_fit_the_normalizer(tmp_path, contracts):
    left = run(prepared_fixture(tmp_path / "left", contracts, selection_shift=0.))
    right = run(prepared_fixture(tmp_path / "right", contracts, selection_shift=.01))
    assert left["state"] == right["state"] == "SUCCEEDED", (left, right)
    assert left["prepared"]["normalizer"] == right["prepared"]["normalizer"]
    assert left["prepared"]["train_binding"] == right["prepared"]["train_binding"]
    assert left["prepared"]["blocks"][-1]["document"]["segments"][0]["position"] != right["prepared"]["blocks"][-1]["document"]["segments"][0]["position"]


@pytest.mark.parametrize("fault", ["missing-grant", "cross-role-unit", "selection-only", "final-eval", "wrong-study", "excess-pairs", "duplicate-output"])
def test_preflight_rejects_bad_population_before_any_read_attempt_or_conversion(tmp_path, contracts, fault, monkeypatch):
    fixture = prepared_fixture(tmp_path, contracts, missing_grant=fault == "missing-grant", cross_role_unit=fault == "cross-role-unit")
    store, original, selections, settings = fixture
    if fault == "selection-only":
        selections[:] = selections[-1:]
    elif fault == "final-eval":
        selections[-1]["purpose"] = "evaluate"
    elif fault == "wrong-study":
        original = {**original, "study_id": "wrong-study"}
    elif fault == "excess-pairs":
        selections[:] = selections * (MAX_PAIRS + 1)
    elif fault == "duplicate-output":
        selections[1]["output_block_id"] = selections[0]["output_block_id"]
    monkeypatch.setattr("application.pirc26_preparation.read_source_pair", lambda *a, **k: pytest.fail("scientific read before whole-population preflight"))
    before = len(store.events())
    with pytest.raises(ResearchError):
        run((store, original, selections, settings))
    assert len(store.events()) == before and not store.attempts()


def test_worker_cannot_execute_without_actual_shared_owner(tmp_path, monkeypatch):
    from infrastructure.pirc26_preparation_worker import execute
    monkeypatch.setattr("infrastructure.pirc26_preparation_worker.read_input", lambda *a, **k: pytest.fail("unowned source read"))
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        execute(tmp_path / "result.json")


def test_prepared_output_is_compatible_with_the_actual_observed_consumer(tmp_path, contracts):
    import torch
    from application.pirc26_data import decode_block, read_block
    from models.phase_space import DynamicsSpec, AffineAccelerationDrift, PhaseSpaceSDE
    fixture = prepared_fixture(tmp_path, contracts)
    store = fixture[0]
    result = run(fixture)
    assert result["state"] == "SUCCEEDED", result
    prepared = result["prepared"]
    n = prepared["normalizer"]
    spec_value = DynamicsSpec("fixture-metric-frame", n["train_binding_hash"], digest(n), n["context_hash"],
                             56, tuple(n["means"]), tuple(n["scales"]))
    model = PhaseSpaceSDE(AffineAccelerationDrift(56), [[.1, 0.], [0., .1]], spec_value).to(dtype=torch.float64)
    # Fixture-only explicit derived protocol/grant. Production must resolve
    # actual lawful purpose and preregistration before such registrations.
    entries = []
    for i, b in enumerate(prepared["blocks"]):
        d, p = b["document"], b["provenance"]
        name = "prepared-" + str(i) + ".json"
        (tmp_path / name).write_bytes(encode(d))
        entries.append({"block_id": d["block_id"], "dataset_id": p["feature"]["dataset_id"],
            "release_id": "prepared-fixture", "source_block_id": p["feature"]["independent_block_id"],
            "sha256": p["content_sha256"], "size_bytes": p["size_bytes"], "path": name,
            "split_role": p["split_role"], "fit_scope": p["split_role"] == "train",
            "preparation_ref": result["preparation_ref"]})
    protocol = {"schema_version": "pirc25-data-protocol-v1", "protocol_id": "prepared-fixture",
                "study_id": "prepared-consumer-fixture", "blocks": entries}
    EvaluationExposureLedger(store).register_protocol(protocol, digest(protocol))
    store.authorize({"authorization_id": "prepared-fixture-grant", "study_id": protocol["study_id"],
        "protocol_hash": digest(protocol), "expires_at": "2099-01-01T00:00:00+00:00",
        "evidence_hash": digest(result["preparation_ref"]), "data_root": str(tmp_path),
        "block_ids": [e["block_id"] for e in entries], "purposes": ["fit", "select"],
        "visibilities": ["synthetic"], "test_authorization": False})
    for b in prepared["blocks"]:
        d, p = b["document"], b["provenance"]
        admission = read_block(store, "prepared-fixture", d["block_id"],
                              authorization_id="prepared-fixture-grant", purpose=p["purpose"])
        assert admission["source_identity"]["source_block_id"] == p["feature"]["independent_block_id"]
        block = decode_block(admission, model)
        assert len(block.segments) == 1
        if block.role == "train":
            assert sum(len(batch.state) for batch in block.transitions(batch_size=8)) == 3
        else:
            with pytest.raises(ResearchError, match="only admitted train"):
                block.transitions(batch_size=8)
    wrong = prepared["blocks"][-1]["document"]["block_id"]
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        read_block(store, "prepared-fixture", wrong, authorization_id="prepared-fixture-grant", purpose="fit")
    assert {e["payload"]["block_id"] for e in store.events() if e["event_kind"] == "READ_STARTED"
            and e["payload"].get("protocol_hash") == digest(protocol)} == {e["block_id"] for e in entries}


def test_read_receipts_and_normalizer_bindings_cannot_be_substituted(tmp_path, contracts):
    fixture = prepared_fixture(tmp_path, contracts)
    store = fixture[0]
    outcome = run(fixture)
    assert outcome["state"] == "SUCCEEDED", outcome
    request = store.manifest(outcome["preparation_ref"]["request_manifest_id"])
    wrong = deepcopy(request)
    wrong["read_event_hashes"][0] = wrong["read_event_hashes"][1]
    with store._read_transaction(), pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        _verify_reads(store, wrong)
    result = deepcopy(outcome["prepared"])
    result["normalizer"]["train_binding_hash"] = "a" * 64
    with pytest.raises(ResearchError):
        _validate_result(result, request)


@pytest.mark.parametrize("boundary", ["worker-validation", "outer-disclosure"])
def test_actual_worker_permission_is_fresh_at_validation_and_after_final_physical_scope(tmp_path, contracts, monkeypatch, boundary):
    import application.pirc26_preparation as preparation
    import application.research_data as data
    fixture = prepared_fixture(tmp_path, contracts)
    expired = [False]
    class PermissionClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2100, 1, 1, tzinfo=timezone.utc) if expired[0] else datetime.now(tz)
    monkeypatch.setattr(data, "datetime", PermissionClock)
    monkeypatch.setattr(preparation, "datetime", PermissionClock)
    if boundary == "worker-validation":
        actual_run = preparation.ResearchSupervisor.run
        def supervised(self, attempt_id, command, budget, **kwargs):
            validate = kwargs["result_validator"]
            def final_validate(value):
                expired[0] = True
                return validate(value)
            return actual_run(self, attempt_id, command, budget, **{**kwargs, "result_validator": final_validate})
        monkeypatch.setattr(preparation.ResearchSupervisor, "run", supervised)
        with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
            run(fixture)
        attempts = list(fixture[0].attempts().values())
        # Preserve the original supervisor's propagated validation error and
        # INTERRUPTED/SUPERVISOR_ERROR settlement; never relabel or retry it.
        assert len(attempts) == 1 and attempts[0]["state"] == "INTERRUPTED"
        assert attempts[0]["error_code"] == "SUPERVISOR_ERROR"
        assert BudgetLedger(fixture[0]).balance("affine")["committed_ms"] > 0
        assert not any(a.get("artifact_id") for a in fixture[0].attempts().values())
    else:
        outcome = run(fixture)
        assert outcome["state"] == "SUCCEEDED", outcome
        store = fixture[0]
        register_completion = store._read_completion
        def expire_after_authority(authority, expiry):
            def checked_authority():
                authority()
                expired[0] = True
            return register_completion(checked_authority, expiry)
        monkeypatch.setattr(store, "_read_completion", expire_after_authority)
        with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
            load_prepared(store, outcome["preparation_ref"])
