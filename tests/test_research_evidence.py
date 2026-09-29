"""Complete-cell export preserves failures and denies unauthorized disclosure."""

import pytest

from application.research_evidence import export_evidence, accept_aggregate, accept_evidence_package
from infrastructure.research_store import ResearchStore, ResearchError, digest, encode
from tests.test_research_store import spec


def setup(tmp_path):
    store = ResearchStore(tmp_path, "evidence", initialize=True)
    value = spec()
    value["cells"] += [{"arm_id": "affine", "seed": 2, "block_id": "fixture-1"}]
    store.register(value, digest(value))
    grant = {"authorization_id": "export", "study_id": "synthetic", "expires_at": "2099-01-01T00:00:00+00:00",
             "evidence_hash": digest("fixture permission"), "purposes": ["export"],
             "visibilities": ["synthetic"], "block_ids": ["fixture-1"]}
    store.authorize(grant)
    return store, value, grant


def test_export_keeps_every_registered_cell_and_immutable_source(tmp_path):
    store, value, grant = setup(tmp_path)
    cell = value["cells"][0]
    run = store.register_run("synthetic", cell)
    attempt = store.new_attempt(run)
    store.transition(attempt, "RUNNING")
    result = {"spec_hash": digest(value), "cell_hash": digest(cell), "protocol_hash": value["protocol_hash"],
              "metrics": {"error": 1}, "metric_units": {"error": "m"}, "qualification": "fixture",
              "state_order": ["x", "y", "vx", "vy"], "units": ["m", "m", "m/s", "m/s"]}
    artifact = store.artifact(encode(result), role="result", visibility="synthetic", block_ids=["fixture-1"], study_id="synthetic")
    store.transition(attempt, "SUCCEEDED", artifact_id=artifact["artifact_id"])
    bundle = export_evidence(store, "synthetic", grant)
    assert len(bundle["expected_cells"]) == len(bundle["cells"]) == 2
    assert [c["status"] for c in bundle["cells"]] == ["SUCCEEDED", "MISSING"]
    assert bundle["cells"][0]["artifact_id"] == artifact["artifact_id"]
    assert "forecast" not in bundle["cells"][0]
    assert bundle == export_evidence(store, "synthetic", grant)


def test_export_cannot_use_preview_only_or_partial_block_grants(tmp_path):
    store, value, grant = setup(tmp_path)
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        export_evidence(store, "synthetic", {**grant, "purposes": ["preview"]})
    partial = {**grant, "authorization_id": "partial", "block_ids": []}
    store.authorize(partial)
    with pytest.raises(ResearchError, match="complete study matrix"):
        export_evidence(store, "synthetic", partial)


def test_aggregate_import_binds_hash_and_registered_spec(tmp_path):
    store, value, grant = setup(tmp_path)
    aggregate = {"schema_version": "pirc25-aggregate-v1", "study_id": "synthetic", "spec_hash": digest(value),
                 "protocol_hash": value["protocol_hash"], "arms": []}
    with pytest.raises(ResearchError):
        accept_aggregate(store, {**aggregate, "aggregate_hash": "0" * 64}, "synthetic")


def test_frozen_package_import_validates_all_attached_hashes(tmp_path):
    import hashlib

    store, value, grant = setup(tmp_path)
    bundle = export_evidence(store, "synthetic", grant)
    aggregate = {"schema_version": "pirc25-aggregate-v1", "study_id": "synthetic", "spec_hash": digest(value),
                 "protocol_hash": value["protocol_hash"], "source_bundle_hash": bundle["bundle_hash"],
                 "expected_cell_count": 2, "cell_dispositions": bundle["cells"], "arms": []}
    aggregate["aggregate_hash"] = digest(aggregate)
    table = b"aggregate_hash,metric,value\n"
    index = {"aggregate_hash": aggregate["aggregate_hash"], "table_sha256": hashlib.sha256(table).hexdigest()}
    files = {"aggregate.json": encode(aggregate), "metrics.csv": table, "PaperEvidenceIndex.json": encode(index)}
    manifest = {"schema_version": "pirc25-evidence-package-v1", "aggregate_hash": aggregate["aggregate_hash"],
                "files": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}}
    folder = tmp_path / "package"
    folder.mkdir()
    for name, content in {**files, "manifest.json": encode(manifest)}.items():
        (folder / name).write_bytes(content)
    imported = accept_evidence_package(store, folder, aggregate["aggregate_hash"])
    assert store.manifest("comparison-" + aggregate["aggregate_hash"]) == imported
    assert (store.path / "artifacts" / imported["table"]).read_bytes() == table
    (folder / "metrics.csv").write_bytes(b"changed")
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        accept_evidence_package(store, folder, aggregate["aggregate_hash"])
