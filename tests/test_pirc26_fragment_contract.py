"""Synthetic-only causal eligibility; never production grants/data/results."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

from application.pirc26_dsde import context_binding
from application.pirc26_fragment_contract import SHORT_POLICY, fragment_summary, population_fragments
from application.pirc26_preparation import _validate_result, preparation_settings, validate_settings, train_binding
from application.pirc26_population import population_source, training_population
from application.pirc26_selection import selection_populations, selection_source
from application.pirc26_dsde import ProjectionSpec
from infrastructure.research_store import ResearchError, digest
from tests.test_pirc26_dsde import contracts, sources, pair_for, convert
from tests.test_pirc26_preparation import prepared_fixture, run


def fragments(contracts, lengths=(1, 4, 2), *, role="train", offsets=None):
    n = sum(lengths)
    feature, condition, entry = sources(contracts, role=role,
        offsets=tuple(i * 1_000_000_000 for i in range(n)) if offsets is None else offsets)
    labels = ["fragment-" + str(i) for i, length in enumerate(lengths) for _ in range(length)]
    feature = feature.set_column(feature.schema.get_field_index("segment_id"), "segment_id", pa.array(labels))
    return feature, condition, {**entry, "aligned_row_count": n}


def admitted(contracts, **kwargs):
    feature, condition, entry = fragments(contracts, **kwargs)
    return convert(pair_for(feature, condition, entry), contracts, fragment_policy=SHORT_POLICY)


def test_default_remains_strict_and_old_geometry_and_identity_unchanged(contracts):
    feature, condition, entry = fragments(contracts)
    with pytest.raises(ResearchError, match="source segment needs explicit bounded window policy"):
        convert(pair_for(feature, condition, entry), contracts)
    feature, condition, entry = fragments(contracts, lengths=(5,))
    pair = pair_for(feature, condition, entry)
    strict = convert(pair, contracts)
    explicit = convert(pair, contracts, fragment_policy=SHORT_POLICY)
    assert strict["document"] == explicit["document"]
    assert strict["provenance"]["membership"] == explicit["provenance"]["membership"]
    assert fragment_summary(strict, expected_policy="reject") is None
    assert "fragment_policy" not in strict["provenance"]


@pytest.mark.parametrize("role", ["train", "selection"])
@pytest.mark.parametrize("short", [1, 2])
def test_every_source_point_and_original_unit_is_accounted_for(contracts, role, short):
    result = admitted(contracts, lengths=(short, 4, short), role=role)
    p = result["provenance"]
    summary = fragment_summary(result, expected_policy=SHORT_POLICY)
    assert summary == {"policy": SHORT_POLICY, "receipt_hash": digest(p["fragment_disposition"]),
        "source_rows": short * 2 + 4, "included_points": 4, "duplicate_points": 0,
        "excluded_points": short * 2, "included_segments": 1, "excluded_segments": 2}
    assert p["feature"]["independent_block_id"] == "synthetic-unit"
    assert p["split_role"] == role and p["purpose"] == ("fit" if role == "train" else "select")
    assert len(result["document"]["segments"]) == len(p["membership"]) == 1
    assert p["membership"][0]["source_point_indices"] == list(range(short, short + 4))
    assert [point["source_point_index"] for record in p["fragment_disposition"]["segments"]
        for point in record["points"]] == list(range(short * 2 + 4))
    assert p["scientific_qualification"] == "not-established"


def test_duplicate_created_short_path_is_recorded_not_clipped_or_fabricated(contracts):
    feature, condition, entry = fragments(contracts, lengths=(1, 4, 3),
        offsets=tuple(i * 1_000_000_000 for i in (0, 1, 2, 3, 4, 5, 5, 6)))
    pair = pair_for(feature, condition, entry)
    with pytest.raises(ResearchError, match="duplicate source timestamp requires frozen policy"):
        convert(pair, contracts, fragment_policy=SHORT_POLICY)
    result = convert(pair, contracts, fragment_policy=SHORT_POLICY, duplicate_policy="keep-first-exact-time-v1")
    summary = fragment_summary(result)
    assert summary["source_rows"] == 8 and summary["included_points"] == 4
    assert summary["excluded_points"] == 3 and summary["duplicate_points"] == 1
    assert len(result["provenance"]["fragment_disposition"]["segments"][-1]["points"]) == 3


@pytest.mark.parametrize("fault", ["missing", "wrong", "boolean", "all-short", "too-long", "unknown-policy"])
def test_no_implicit_row_binding_whole_file_deletion_long_window_or_unknown_policy(contracts, fault):
    lengths = (1, 2, 2) if fault == "all-short" else (1, 4099) if fault == "too-long" else (1, 4, 2)
    feature, condition, entry = fragments(contracts, lengths=lengths)
    if fault == "missing": entry.pop("aligned_row_count")
    if fault == "wrong": entry["aligned_row_count"] += 1
    if fault == "boolean": entry["aligned_row_count"] = True
    with pytest.raises(ResearchError):
        convert(pair_for(feature, condition, entry), contracts,
            fragment_policy="silently-drop-unit" if fault == "unknown-policy" else SHORT_POLICY)


@pytest.mark.parametrize("fault", ["nan-coordinate", "outside-frame", "join", "nonfinite-context", "time-gap", "time-reversal", "bad-duplicate-coordinate"])
def test_exclusion_never_hides_invalid_source_points(contracts, monkeypatch, fault):
    offsets = None
    lengths = (2, 4)
    if fault in {"time-gap", "time-reversal"}:
        offsets = tuple(i * 1_000_000_000 for i in ((0, 61, 62, 63, 64, 65) if fault == "time-gap" else (1, 0, 2, 3, 4, 5)))
    if fault == "bad-duplicate-coordinate":
        offsets = tuple(i * 1_000_000_000 for i in (0, 0, 1, 2, 3, 4))
    feature, condition, entry = fragments(contracts, lengths=lengths, offsets=offsets)
    if fault in {"nan-coordinate", "outside-frame", "bad-duplicate-coordinate"}:
        values = condition["lat"].to_pylist()
        values[1 if fault == "bad-duplicate-coordinate" else 0] = float("nan") if fault != "outside-frame" else 21.
        condition = condition.set_column(condition.schema.get_field_index("lat"), "lat", pa.array(values))
    if fault == "join":
        indices = feature["point_index"].to_pylist()
        indices[0] = 1
        feature = feature.set_column(feature.schema.get_field_index("point_index"), "point_index", pa.array(indices))
    if fault == "nonfinite-context":
        import application.pirc26_dsde as dsde
        adapter = dsde._context_adapter(*contracts, entry)
        matrix = np.zeros((len(feature), context_binding(*contracts)["context_dim"]))
        matrix[0, 0] = np.nan
        adapter.transform = lambda *a, **k: SimpleNamespace(model_matrix=lambda: matrix)
        monkeypatch.setattr(dsde, "_context_adapter", lambda *a: adapter)
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        convert(pair_for(feature, condition, entry), contracts, fragment_policy=SHORT_POLICY,
            duplicate_policy="keep-first-exact-time-v1" if fault == "bad-duplicate-coordinate" else "reject")


@pytest.mark.parametrize("fault", ["missing-record", "unknown-disposition", "wrong-reason", "row-count", "duplicate-point", "reordered-index",
    "epoch", "missing-member", "document-time", "duplicate-count", "foreign-unit", "unfrozen-policy", "unknown-key"])
def test_structural_owner_check_refuses_forged_incomplete_or_overlapping_dispositions(contracts, fault):
    result = deepcopy(admitted(contracts))
    p, d = result["provenance"], result["document"]
    receipt = p["fragment_disposition"]
    if fault == "missing-record": receipt["segments"].pop()
    elif fault == "unknown-disposition": receipt["segments"][0]["disposition"] = "discard-quietly"
    elif fault == "wrong-reason": receipt["segments"][0]["reason_code"] = "low-score"
    elif fault == "row-count": receipt["source_rows"] -= 1
    elif fault == "duplicate-point": receipt["segments"][1]["points"][0]["point_id"] = receipt["segments"][0]["points"][0]["point_id"]
    elif fault == "reordered-index": receipt["segments"][1]["points"][0]["source_point_index"] = 0
    elif fault == "epoch": receipt["segments"][1]["points"][1]["absolute_epoch_ns"] += 1
    elif fault == "missing-member": p["membership"][0]["point_ids"].pop()
    elif fault == "document-time": d["segments"][0]["time"][1] += .5
    elif fault == "duplicate-count": p["removed_duplicate_timestamps"] = True
    elif fault == "foreign-unit": p["membership"][0]["independent_block_id"] = "foreign"
    elif fault == "unfrozen-policy": p["fragment_policy"] = "reject"
    else: receipt["segments"][0]["points"][0]["unknown"] = True
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        fragment_summary(result, expected_policy=SHORT_POLICY)


def test_policy_is_frozen_in_train_identity_not_fitted_from_selection(contracts):
    strict = preparation_settings(*contracts, ProjectionSpec("synthetic-frame", 20., 110., .1))
    explicit = preparation_settings(*contracts, ProjectionSpec("synthetic-frame", 20., 110., .1), fragment_policy=SHORT_POLICY)
    validate_settings(strict)
    validate_settings(explicit)
    assert "fragment_policy" not in strict and explicit["normalizer_policy"]["fragment_policy"] == SHORT_POLICY
    assert digest(train_binding([], strict)) != digest(train_binding([], explicit))
    for fault in ("unknown", "unfrozen-normalizer", "redundant-reject"):
        wrong = deepcopy(explicit)
        if fault == "unknown": wrong["fragment_policy"] = "drop-all"
        elif fault == "unfrozen-normalizer": wrong["normalizer_policy"].pop("fragment_policy")
        else: wrong = {**strict, "fragment_policy": "reject"}
        with pytest.raises(ResearchError):
            validate_settings(wrong)


@pytest.mark.parametrize("fault", ["provenance", "feature", "segments", "membership", "duplicate-policy", "removed-count"])
def test_malformed_receipt_transport_remains_a_closed_typed_error(contracts, fault):
    result = deepcopy(admitted(contracts))
    if fault == "provenance": result["provenance"] = None
    elif fault == "feature": result["provenance"]["feature"] = []
    elif fault == "segments": result["document"]["segments"] = None
    elif fault == "membership": result["provenance"].pop("membership")
    elif fault == "duplicate-policy": result["provenance"].pop("duplicate_policy")
    else: result["provenance"].pop("removed_duplicate_timestamps")
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        fragment_summary(result, expected_policy=SHORT_POLICY)


def test_actual_owned_worker_preserves_every_file_unit_exclusion_and_train_only_fit(tmp_path, contracts, monkeypatch):
    fixture = prepared_fixture(tmp_path, contracts, source_segments=["short", *(["main"] * 4)], fragment_policy=SHORT_POLICY,
        consumer_study_ids=["synthetic-selection-consumer"])
    store, original, _, settings = fixture
    outcome = run(fixture)
    assert outcome["state"] == "SUCCEEDED", outcome
    value = outcome["prepared"]
    request = store.manifest(outcome["preparation_ref"]["request_manifest_id"])
    assert len(value["blocks"]) == 3 and value["normalizer"]["observations"] == 6
    assert value["normalizer"]["policy"] == settings["normalizer_policy"]
    assert value["normalizer"]["independent_block_ids"] == ["train-unit"]
    pool = value["training_population"]
    assert pool["provenance"]["fragment_eligibility"]["source_rows"] == 10
    assert pool["provenance"]["fragment_eligibility"]["excluded_points"] == 2
    assert pool["provenance"]["observations"] == 8 and pool["provenance"]["transitions"] == 4
    selected = selection_populations(value["blocks"], value["normalizer"], study_id=original["study_id"])[0]
    assert selected["provenance"]["fragment_eligibility"]["source_rows"] == 5
    assert selected["provenance"]["fragment_eligibility"]["excluded_points"] == 1
    assert selected["provenance"]["independent_block_id"] == "selection-unit"
    expected = []
    for block in value["blocks"][:2]:
        segment = block["document"]["segments"][0]
        position, time, context = np.asarray(segment["position"]), np.asarray(segment["time"]), np.asarray(segment["condition"])
        expected.append(np.column_stack((position[1:], np.diff(position, axis=0) / np.diff(time)[:, None], context[1:])))
    np.testing.assert_allclose(value["normalizer"]["means"], np.concatenate(expected).mean(axis=0), rtol=0, atol=1e-14)
    assert population_source(store, outcome["preparation_ref"])["provenance"] == pool["provenance"]
    assert selection_source(store, outcome["preparation_ref"], "selection-unit",
        consumer_study_id="synthetic-selection-consumer")["provenance"] == selected["provenance"]
    # All source rows/exclusions are checked by the owner, not just geometry.
    wrong = deepcopy(value)
    wrong["blocks"][0]["provenance"]["fragment_disposition"]["segments"].pop(0)
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        _validate_result(wrong, request)
    wrong = deepcopy(value["blocks"])
    wrong[-1]["provenance"]["fragment_policy"] = "reject"
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        selection_populations(wrong, value["normalizer"], study_id=original["study_id"])
    wrong = deepcopy(value["blocks"])
    wrong[0]["provenance"]["fragment_disposition"]["segments"].pop(0)
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        training_population(wrong, value["train_binding"], value["normalizer"], study_id=original["study_id"])
    # Even a balanced forged summary cannot change the frozen source row count.
    proof = store.manifest(outcome["preparation_ref"]["manifest_id"])
    actual_manifest = store._manifest
    for role in ("training_population", "selection_populations"):
        changed = deepcopy(proof)
        metadata = changed[role]["provenance"] if role == "training_population" else changed[role][0]["provenance"]
        summary = metadata["members"][0]["fragment_disposition"]
        summary["source_rows"] += 1
        summary["duplicate_points"] += 1
        metadata["fragment_eligibility"] = population_fragments(metadata["members"])
        with monkeypatch.context() as patch:
            patch.setattr(store, "_manifest", lambda name: changed if name == outcome["preparation_ref"]["manifest_id"] else actual_manifest(name))
            with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
                if role == "training_population": population_source(store, outcome["preparation_ref"])
                else: selection_source(store, outcome["preparation_ref"], "selection-unit")
    assert len([e for e in store.events() if e["event_kind"] == "WORKER_STARTED"]) == 1
    assert value["scientific_qualification"] == "not-established"


def test_opted_in_missing_row_inventory_refuses_before_raw_read_attempt_or_reservation(tmp_path, contracts, monkeypatch):
    fixture = prepared_fixture(tmp_path, contracts, fragment_policy=SHORT_POLICY)
    store = fixture[0]
    before = store.events()
    monkeypatch.setattr("application.pirc26_preparation.read_source_pair", lambda *a, **k: pytest.fail("raw read before metadata preflight"))
    with pytest.raises(ResearchError, match="frozen aligned row count required before fragment preparation"):
        run(fixture)
    assert store.events() == before and not store.attempts()
