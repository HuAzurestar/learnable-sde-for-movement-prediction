"""Synthetic-only, admitted same-file conversion; no production source access."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from application.pirc26_dsde import (SourcePair, ProjectionSpec, context_binding,
    convert_source_pair, read_source_pair, _context_adapter)
from application.research_data import EvaluationExposureLedger
from infrastructure.research_store import ResearchStore, ResearchError, digest, encode

ROOT = Path(__file__).resolve().parents[1]
EPOCH = 1_700_000_000_000_000_000


@pytest.fixture
def contracts():
    spec = json.loads((ROOT / "tests/fixtures/pirc26_dsde_feature_spec.json").read_text(encoding="utf-8"))
    binding = json.loads((ROOT / "experiments/pirc22/benchmark_selection.consumer.json").read_text(encoding="utf-8"))
    return spec, binding


def parquet(table):
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


def sources(contracts, role="train", offsets=(0, 1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000)):
    spec, binding = contracts
    frozen = context_binding(spec, binding)
    n = len(offsets)
    entry = {"source_kind": "dsde-feature-parquet", "block_id": "features", "dataset_id": "synthetic-dataset",
        "release_id": "synthetic-feature-release", "source_block_id": "synthetic-unit", "file_id": "synthetic-file",
        "independent_block_id": "synthetic-unit", "source_split": "train" if role == "train" else "validation",
        "split_role": role, "fit_scope": role == "train", "path": "features.parquet", "sha256": "a" * 64,
        "size_bytes": 1, "feature_spec_sha256": frozen["feature_spec_sha256"]}
    adapter = _context_adapter(spec, binding, entry)
    identities = {"dataset_version": entry["dataset_id"], "file_id": entry["file_id"],
        "split": entry["source_split"], "independent_block_id": entry["independent_block_id"], "segment_id": "segment"}
    columns = {}
    for name in adapter._required_columns():
        if name in identities:
            columns[name] = [identities[name]] * n
        elif name == "point_id":
            columns[name] = [f"point-{i}" for i in range(n)]
        elif name == "point_index":
            columns[name] = list(range(n))
        elif name == "absolute_epoch_ns":
            columns[name] = [EPOCH + offset for offset in offsets]
        elif name.endswith("_status"):
            columns[name] = ["valid"] * n
        else:
            columns[name] = [10. if "worldcover" in name else 1.] * n
    feature = pa.table(columns)
    condition = pa.table({"file_id": [entry["file_id"]] * n,
        "t": pa.array([EPOCH + offset for offset in offsets], type=pa.timestamp("ns")),
        "lat": [20. + i * .00001 for i in range(n)], "lon": [110. + i * .00001 for i in range(n)]})
    return feature, condition, entry


def pair_for(feature, condition, entry):
    a, b = parquet(feature), parquet(condition)
    feature_meta = {**entry, "size_bytes": len(a), "sha256": hashlib.sha256(a).hexdigest()}
    condition_meta = {**entry, "source_kind": "dsde-condition-parquet", "block_id": "conditions",
        "release_id": "synthetic-condition-release", "path": "conditions.parquet", "size_bytes": len(b),
        "sha256": hashlib.sha256(b).hexdigest()}
    return SourcePair(digest("synthetic protocol"), digest("synthetic grant, not production consent"),
        encode(feature_meta).decode(), encode(condition_meta).decode(), a, b)


def convert(pair, contracts, **kwargs):
    spec, binding = contracts
    options = {"block_id": "observations", "train_binding_hash": "a" * 64, "normalizer_hash": "b" * 64,
        "context_hash": digest(context_binding(spec, binding))}
    options.update(kwargs)
    return convert_source_pair(pair, spec, binding, ProjectionSpec("synthetic-metric-frame", 20., 110., .1), **options)


def test_exact_accepted_context_without_snapshot_loader(contracts, monkeypatch):
    from experiments.nex326.pirc21_adapter import FeatureSnapshotAdapter
    f, c, entry = sources(contracts)
    monkeypatch.setattr(FeatureSnapshotAdapter, "__init__", lambda *a, **k: pytest.fail("whole snapshot constructor"))
    monkeypatch.setattr(FeatureSnapshotAdapter, "_load", lambda *a, **k: pytest.fail("filesystem loader"))
    actual = convert(pair_for(f, c, entry), contracts)
    segment = actual["document"]["segments"][0]
    assert np.asarray(segment["condition"]).shape == (5, 56)
    expected = _context_adapter(*contracts, entry)
    expected.table = f
    np.testing.assert_array_equal(segment["condition"], expected.transform("train", file_ids=[entry["file_id"]]).model_matrix())
    assert segment["time"] == [0., 1., 2., 3., 4.]
    assert segment["position"][0] == [0., 0.]
    assert segment["position"][1] == pytest.approx([1.0448905, 1.1119493], abs=1e-6)
    assert actual["provenance"]["scientific_qualification"] == "not-established"
    assert actual["provenance"]["membership"][0]["absolute_start_epoch_ns"] == EPOCH


def test_nanosecond_difference_before_float_conversion(contracts):
    f, c, entry = sources(contracts, offsets=(0, 1, 2, 3, 4))
    result = convert(pair_for(f, c, entry), contracts)
    assert result["document"]["segments"][0]["time"] == [0., 1e-9, 2e-9, 3e-9, 4e-9]


@pytest.mark.parametrize("role", ["train", "selection"])
def test_actual_ledger_reads_only_explicit_pair(tmp_path, contracts, role):
    f, c, entry = sources(contracts, role=role)
    pair = pair_for(f, c, entry)
    store = authorized_store(tmp_path, pair, role)
    result = read_source_pair(store, "synthetic-sources", "features", "conditions",
        authorization_id="synthetic-grant", purpose="fit" if role == "train" else "select")
    converted = convert(result, contracts)
    assert converted["provenance"]["split_role"] == role
    starts = [e for e in store.events() if e["event_kind"] == "READ_STARTED"]
    completed = [e for e in store.events() if e["event_kind"] == "READ_COMPLETED"]
    assert len(starts) == len(completed) == 2
    assert {e["payload"]["block_id"] for e in starts} == {"features", "conditions"}
    assert not any(e["payload"].get("block_id") == "sealed" for e in starts)


def authorized_store(tmp_path, pair, role, grant_blocks=None):
    store = ResearchStore(tmp_path / "ledger", "pirc26-conversion-fixture", initialize=True)
    f, c = pair.metadata()
    (tmp_path / "features.parquet").write_bytes(pair.feature_bytes)
    (tmp_path / "conditions.parquet").write_bytes(pair.condition_bytes)
    # An unrelated sealed block must never be visited; its file does not exist.
    sealed = {**f, "block_id": "sealed", "path": "do-not-open.parquet", "split_role": "final-eval", "fit_scope": False}
    protocol = {"schema_version": "pirc25-data-protocol-v1", "protocol_id": "synthetic-sources",
        "study_id": "synthetic-conversion", "blocks": [f, c, sealed]}
    EvaluationExposureLedger(store).register_protocol(protocol, digest(protocol))
    store.authorize({"authorization_id": "synthetic-grant", "study_id": protocol["study_id"],
        "protocol_hash": digest(protocol), "expires_at": "2099-01-01T00:00:00+00:00",
        "evidence_hash": digest("synthetic fixture grant, not production consent"), "data_root": str(tmp_path),
        "block_ids": grant_blocks if grant_blocks is not None else ["features", "conditions"],
        "purposes": ["fit" if role == "train" else "select"], "visibilities": ["synthetic"], "test_authorization": False})
    return store


@pytest.mark.parametrize("fault", ["missing-condition-grant", "expired", "wrong-purpose", "sealed"])
def test_pair_denied_before_either_scientific_open(tmp_path, contracts, monkeypatch, fault):
    f, c, entry = sources(contracts)
    pair = pair_for(f, c, entry)
    store = authorized_store(tmp_path, pair, "train", grant_blocks=["features"] if fault == "missing-condition-grant" else None)
    from application import research_data
    if fault == "expired":
        from datetime import datetime
        class FutureClock(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2100, 1, 1, tzinfo=tz)
        monkeypatch.setattr(research_data, "datetime", FutureClock)
    monkeypatch.setattr(research_data, "opened_regular_file", lambda *a, **k: pytest.fail("unauthorized byte read"))
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        read_source_pair(store, "synthetic-sources", "sealed" if fault == "sealed" else "features", "conditions",
            authorization_id="synthetic-grant", purpose="select" if fault == "wrong-purpose" else "fit")
    assert not any(e["event_kind"] == "READ_STARTED" for e in store.events())
    if fault in {"expired", "missing-condition-grant"}:
        assert store.events()[-1]["event_kind"] == "EXPOSURE_DENIED"


@pytest.mark.parametrize("fault", ["condition-root-escape", "condition-size", "cross-split"])
def test_entire_pair_preflight_before_first_payload(tmp_path, contracts, monkeypatch, fault):
    f, c, entry = sources(contracts)
    pair = pair_for(f, c, entry)
    _, condition = pair.metadata()
    if fault == "condition-root-escape":
        condition["path"] = "../outside.parquet"
    elif fault == "condition-size":
        condition["size_bytes"] += 1
    else:
        condition.update(split_role="selection", source_split="validation", fit_scope=False)
    pair = replace(pair, condition_metadata_utf8=encode(condition).decode())
    store = authorized_store(tmp_path, pair, "train")
    from application import research_data
    monkeypatch.setattr(research_data, "opened_regular_file", lambda *a, **k: pytest.fail("partial pair read"))
    with pytest.raises(ResearchError):
        read_source_pair(store, "synthetic-sources", "features", "conditions", authorization_id="synthetic-grant", purpose="fit")
    assert not any(e["event_kind"] == "READ_STARTED" for e in store.events())


@pytest.mark.parametrize("metadata", ['[]', '{"path":"a","path":"b"}', '{bad'])
def test_malformed_transport_metadata_is_typed_error(contracts, metadata):
    f, c, entry = sources(contracts)
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        convert(replace(pair_for(f, c, entry), feature_metadata_utf8=metadata), contracts)


def test_row_quota_refused_before_parquet_decode(contracts, monkeypatch):
    f, c, entry = sources(contracts)
    pair = pair_for(f, c, entry)
    monkeypatch.setattr(pq.ParquetFile, "read", lambda *a, **k: pytest.fail("quota exceeded before decode"))
    with pytest.raises(ResearchError, match="RESOURCE_PLAN_REJECTED"):
        convert(pair, contracts, max_observations=3)


def test_missing_factor_keeps_registered_validity_channel(contracts):
    f, c, entry = sources(contracts)
    rows = f.to_pylist()
    statuses = [name for name in f.column_names if name.endswith("_status")]
    for name in statuses:
        rows[2][name] = "source_missing"
    f = pa.Table.from_pylist(rows, schema=f.schema)
    values = convert(pair_for(f, c, entry), contracts)["document"]["segments"][0]["condition"]
    assert values[2] == [0.] * 56
    assert np.isfinite(values).all()


@pytest.mark.parametrize("fault", ["dataset", "file", "split", "unit", "point-index", "epoch", "duplicate-point", "missing-column", "null-time", "coordinate", "outside-frame", "segment-reentry"])
def test_bad_source_contract_fails_closed(contracts, fault):
    f, c, entry = sources(contracts)
    rows, raw = f.to_pylist(), c.to_pylist()
    keys = {"dataset": "dataset_version", "file": "file_id", "split": "split", "unit": "independent_block_id"}
    if fault in keys:
        rows[2][keys[fault]] = "foreign"
    elif fault == "point-index":
        rows[2]["point_index"] = 99
    elif fault == "epoch":
        rows[2]["absolute_epoch_ns"] += 1
    elif fault == "duplicate-point":
        rows[2]["point_id"] = rows[1]["point_id"]
    elif fault == "null-time":
        raw[2]["t"] = None
    elif fault == "coordinate":
        raw[2]["lat"] = float("nan")
    elif fault == "outside-frame":
        raw[2]["lat"] = 40.
    elif fault == "segment-reentry":
        rows[2]["segment_id"] = "other"
    f = pa.Table.from_pylist(rows, schema=f.schema)
    c = pa.Table.from_pylist(raw, schema=c.schema)
    if fault == "missing-column":
        f = f.drop(["point_index"])
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        convert(pair_for(f, c, entry), contracts)


def test_duplicates_require_explicit_frozen_policy(contracts):
    f, c, entry = sources(contracts, offsets=(0, 1_000_000_000, 1_000_000_000, 2_000_000_000, 3_000_000_000))
    pair = pair_for(f, c, entry)
    with pytest.raises(ResearchError, match="duplicate source timestamp"):
        convert(pair, contracts)
    result = convert(pair, contracts, duplicate_policy="keep-first-exact-time-v1")
    assert result["document"]["segments"][0]["time"] == [0., 1., 2., 3.]
    assert result["provenance"]["removed_duplicate_timestamps"] == 1
    assert result["provenance"]["membership"][0]["source_point_indices"] == [0, 1, 3, 4]


@pytest.mark.parametrize("offsets", [(0, 2, 1, 3, 4), (0, 1_000_000_000, 62_000_000_000, 63_000_000_000, 64_000_000_000)])
def test_refined_time_boundaries_never_silently_resplit(contracts, offsets):
    f, c, entry = sources(contracts, offsets=offsets)
    with pytest.raises(ResearchError, match="time boundary"):
        convert(pair_for(f, c, entry), contracts)


@pytest.mark.parametrize("fault", ["bytes", "context", "feature-spec", "row-quota", "output-quota", "invalid-parquet"])
def test_identity_and_resource_guards(contracts, fault):
    f, c, entry = sources(contracts)
    pair = pair_for(f, c, entry)
    options = {}
    if fault == "bytes":
        pair = replace(pair, feature_bytes=pair.feature_bytes[:-1] + b"x")
    elif fault == "context":
        options["context_hash"] = "d" * 64
    elif fault == "feature-spec":
        entry["feature_spec_sha256"] = "d" * 64
        pair = pair_for(f, c, entry)
    elif fault == "row-quota":
        options["max_observations"] = 3
    elif fault == "output-quota":
        options["max_output_bytes"] = 1
    else:
        metadata, _ = pair.metadata()
        metadata.update(size_bytes=3, sha256=hashlib.sha256(b"bad").hexdigest())
        pair = replace(pair, feature_bytes=b"bad", feature_metadata_utf8=encode(metadata).decode())
    with pytest.raises(ResearchError):
        convert(pair, contracts, **options)


def test_converted_document_uses_existing_causal_decoder(contracts):
    import torch
    from application.pirc26_data import decode_block
    from models.phase_space import AffineAccelerationDrift, DynamicsSpec, PhaseSpaceSDE
    from tests.test_pirc26_components_data import admitted
    f, c, entry = sources(contracts)
    result = convert(pair_for(f, c, entry), contracts)
    doc = result["document"]
    spec = DynamicsSpec(doc["coordinate_frame"], doc["train_binding_hash"], doc["normalizer_hash"], doc["context_hash"],
        56, (0.,) * 60, (1.,) * 60)
    model = PhaseSpaceSDE(AffineAccelerationDrift(56), [[.4, 0.], [.1, .3]], spec).to(dtype=torch.float64)
    decoded = decode_block(admitted(doc), model)
    request = decoded.forecast_request("segment", 2, (2., 3., 4.), sample_count=8, brownian_root_id="a" * 64)
    changed = deepcopy(doc)
    changed["segments"][0]["position"][3:] = [[999., -999.], [888., -888.]]
    assert decode_block(admitted(changed), model).forecast_request("segment", 2, (2., 3., 4.), sample_count=8,
        brownian_root_id="a" * 64) == request
    assert decoded.transitions(batch_size=16)[0].context.condition.shape[1] == 56
