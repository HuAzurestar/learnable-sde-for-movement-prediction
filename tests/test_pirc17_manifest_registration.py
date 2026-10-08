"""Metadata-only regression tests; no recovered research data or approval."""
from copy import deepcopy
import hashlib

import pytest

from experiments.pirc17 import manifest_registration as reg
from experiments.pirc17 import protocol_core as core


@pytest.fixture
def metadata():
    spec = {"feature_spec_id": "public-synthetic-spec"}
    rows = [{"split": "train", "file_id": "synthetic", "path": "train/synthetic.parquet",
             "sha256": "a"*64, "row_count": 7}]
    inventory = hashlib.sha256(("train/synthetic.parquet\0" + "a"*64 + "\0" + "7\n").encode()).hexdigest()
    snapshot = {"snapshot_id": "public-synthetic", "manifest_sha256": reg.OLD_MANIFEST,
        "counts": {"files": 1, "points": 7}, "files": 1, "history_window_points": 3,
        "feature_spec_id": spec["feature_spec_id"], "feature_spec_content_sha256": core.digest(spec),
        "content_inventory_sha256": inventory}
    original = {"dataset_id": "public-synthetic", "snapshot": snapshot,
                "final_eval_authorized": False, "untouched": {"seeds": [1, 2]}}
    original["sha256"] = core.digest(original)
    manifest = {**deepcopy(snapshot), "dataset_id": original["dataset_id"], "files": rows,
        "feature_spec_sha256": core.digest(spec), "coverage_report_sha256": reg.COVERAGE}
    proof = {"identities": {"manifest": reg.NEW_MANIFEST, "inventory": inventory, "coverage": reg.COVERAGE},
        "matches": {"manifest": False, "inventory": True, "coverage": True}, "exact_restoration": False,
        "manifest_inventory_matches_files": True, "files": 1, "rows": 7, "feature_bytes": 2450632099}
    process = {"worker_returncode": 0, "process_tree_closed": True, "accounting": {"active_processes": 0},
               "timed_out": False, "interrupted": False, "supervisor_error": None}
    reuse = {"reused_files": 0, "computed_files": 1, "valid_cache_files": 0}
    return original, manifest, spec, proof, process, reuse


def derive(metadata):
    return reg.derive_binding(*metadata, receipt_hashes=reg.RECEIPTS)


def test_only_manifest_and_provenance_change(metadata):
    before = deepcopy(metadata)
    value = derive(metadata)
    assert metadata == before
    assert not value["final_eval_authorized"]
    assert not value["manifest_reregistration"]["formal_evaluation_authorization"]
    assert value["sha256"] == core.digest({k: v for k, v in value.items() if k != "sha256"})
    value.pop("manifest_reregistration")
    value["snapshot"]["manifest_sha256"] = reg.OLD_MANIFEST
    value["sha256"] = metadata[0]["sha256"]
    assert value == metadata[0]


@pytest.mark.parametrize("case", ["inventory", "row", "spec", "history", "counts", "coverage",
    "exact_claim", "incomplete", "active", "timeout", "exit", "reuse", "original"])
def test_reject_changed_or_incomplete_evidence(metadata, case):
    original, manifest, spec, proof, process, reuse = metadata
    if case == "inventory": manifest["files"][0]["sha256"] = "b"*64
    if case == "row": manifest["files"][0]["row_count"] += 1
    if case == "spec": spec["extra"] = "changed"
    if case == "history": manifest["history_window_points"] = 2
    if case == "counts": manifest["counts"]["points"] = 8
    if case == "coverage": proof["matches"]["coverage"] = False
    if case == "exact_claim": proof["exact_restoration"] = True
    if case == "incomplete": process["process_tree_closed"] = False
    if case == "active": process["accounting"]["active_processes"] = 1
    if case == "timeout": process["timed_out"] = True
    if case == "exit": process["worker_returncode"] = 1
    if case == "reuse": reuse["computed_files"] = 0
    if case == "original": original["untouched"]["seeds"] = [3]
    with pytest.raises(ValueError):
        derive(metadata)


def test_wrong_receipt_identity_rejected(metadata):
    with pytest.raises(ValueError, match="receipt"):
        reg.derive_binding(*metadata, receipt_hashes={**reg.RECEIPTS, "start.json": "0"*64})


def test_exclusive_publication_keeps_old_and_partial_bytes(tmp_path, metadata):
    path = tmp_path/"binding.json"
    reg.publish_binding(path, derive(metadata))
    before = path.read_bytes()
    with pytest.raises(FileExistsError): reg.publish_binding(path, derive(metadata))
    assert path.read_bytes() == before
    partial = tmp_path/"partial.json"
    partial.write_bytes(b'{"sha256":')
    with pytest.raises(FileExistsError): reg.publish_binding(partial, derive(metadata))
    assert partial.read_bytes() == b'{"sha256":'


def test_renewed_protocol_retains_science_and_original_input():
    payload = reg.renewed_core_payload()
    assert payload["dataset_inputs"]["snapshot"]["manifest_sha256"] == reg.NEW_MANIFEST
    assert len(payload["source_sha256"]) == 165
    assert "PSDE-SDE/experiments/pirc17/formal_matrix.py" not in payload["source_sha256"]
    assert core.file_hash(reg.ROOT/reg.OLD_INPUT_PATH) == reg.OLD_INPUT_FILE_SHA256
    assert payload["source_sha256"]["PSDE-SDE/"+reg.OLD_INPUT_PATH] == reg.OLD_INPUT_FILE_SHA256
    assert not payload["final_eval_authorized"]
    assert payload["component_precedence"]["allowed_new_precheck_forecasts"] == 0
