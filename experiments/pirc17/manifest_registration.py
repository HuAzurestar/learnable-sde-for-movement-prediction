"""Register the authorized rebuilt metadata identity, never repair old hashes.

Reads only metadata and already closed recovery receipts. It neither opens
feature/trajectory/map data nor grants ACCEPT-01 or runs scientific work.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

from .protocol_core import canonical, digest, file_hash, publish, read_json, under, unpack
from .protocol_inputs import validate_snapshot_metadata

ROOT = Path(__file__).resolve().parents[2]
OLD_INPUT_PATH = "experiments/pirc17/evidence/protocol-inputs-v1.json"
OLD_INPUT_FILE_SHA256 = "7d4fcf9a158d59c868eb5682e9102e91ec4fb207d0dd2190a7747d83f0d74dc9"
NEW_INPUT_PATH = "experiments/pirc17/evidence/protocol-inputs-v2.json"
OLD_MANIFEST = "927f1d33df2e803d6f3aae403ea5e7046c888d792d1eedf2d2be23a015a08956"
NEW_MANIFEST = "56aabd7c20361bf8c5c01f0b534cd80823602129f473f534491c49a5c4e2665b"
COVERAGE = "32f65eea653f730ca7a5664088657199ee1c1c76d70d85947517b9b63d930d60"
RECEIPTS = {
    "verification.json": "f481383fedc4e18536a310ce649801df4b687f1920c9975c4be49d1e95b5cbce",
    "process-result.json": "eebce7db6bd4c8aaa6e78bd4f76618f11247d7dc8003d9ede1b3ef758cc97b5e",
    "start.json": "f1216588be32dd25b4a4fad726eda57a2f20743888860a18e1bbdc0812a4e26e",
    "reuse-result.json": "6e17eaecacae8ffe9a3c0b149a3a6cb51578352addf009ee07afb6c347292d19",
}
AUTHORIZATION = {
    "source": "resumed user-provided GOAL objective",
    "quote": "resume \u5141\u8bb8\u91cd\u65b0\u767b\u8bb0\u91cd\u5efa manifest\uff0c\u7ee7\u7eed GOAL\u3002",
    "recorded_at": "2026-10-01T13:06:04Z",
    "scope": "register rebuilt manifest only; not ACCEPT-01 or ACCEPT-02",
}


def derive_binding(original, manifest, spec, proof, process, reuse, *, receipt_hashes):
    """Pure metadata checks; public synthetic fixtures can exercise failures."""
    if original.get("sha256") != digest({k: v for k, v in original.items() if k != "sha256"}):
        raise ValueError("original input content identity mismatch")
    snapshot = original["snapshot"]
    if snapshot["manifest_sha256"] != OLD_MANIFEST:
        raise ValueError("original manifest identity changed")
    if (manifest["dataset_id"] != original["dataset_id"]
            or manifest["snapshot_id"] != snapshot["snapshot_id"]
            or manifest["counts"] != snapshot["counts"]
            or manifest["history_window_points"] != snapshot["history_window_points"]
            or manifest["feature_spec_id"] != snapshot["feature_spec_id"]
            or digest(spec) != snapshot["feature_spec_content_sha256"]
            or spec["feature_spec_id"] != snapshot["feature_spec_id"]):
        raise ValueError("snapshot scientific metadata changed")
    validate_snapshot_metadata(manifest, spec,
        expected_inventory_sha256=snapshot["content_inventory_sha256"], expected_files=snapshot["files"])
    if (manifest["coverage_report_sha256"] != COVERAGE
            or proof["identities"] != {"manifest": NEW_MANIFEST,
                "inventory": snapshot["content_inventory_sha256"], "coverage": COVERAGE}
            or proof["matches"] != {"manifest": False, "inventory": True, "coverage": True}
            or proof["exact_restoration"] is not False
            or proof["manifest_inventory_matches_files"] is not True
            or proof["files"] != snapshot["files"] or proof["rows"] != snapshot["counts"]["points"]
            or sum(r["row_count"] for r in manifest["files"]) != proof["rows"]
            or proof["feature_bytes"] != 2450632099):
        raise ValueError("complete rebuilt byte/row/inventory/coverage proof required")
    if (process["worker_returncode"] != 0 or process["process_tree_closed"] is not True
            or process["accounting"]["active_processes"] != 0
            or process["timed_out"] is not False or process["interrupted"] is not False
            or process["supervisor_error"] is not None):
        raise ValueError("closed successful producer required")
    if (reuse["reused_files"] + reuse["computed_files"] != snapshot["files"]
            or reuse["valid_cache_files"] != reuse["reused_files"]):
        raise ValueError("complete recovery accounting required")
    if receipt_hashes != RECEIPTS:
        raise ValueError("exact closed recovery receipt identities required")
    result = deepcopy(original)
    result["snapshot"]["manifest_sha256"] = NEW_MANIFEST
    result["manifest_reregistration"] = {
        "schema_version": "pirc17-rebuilt-manifest-registration-v1",
        "original_input_file_sha256": OLD_INPUT_FILE_SHA256,
        "original_manifest_sha256": OLD_MANIFEST, "rebuilt_manifest_sha256": NEW_MANIFEST,
        "content_inventory_sha256": snapshot["content_inventory_sha256"], "coverage_sha256": COVERAGE,
        "recovery_receipt_sha256": dict(receipt_hashes), "authorization": dict(AUTHORIZATION),
        "original_manifest_byte_restoration": False, "old_metadata_delta": "unknown: original bytes unavailable",
        "feature_content_changed": False, "scientific_parameters_changed": False,
        "formal_evaluation_authorization": False,
    }
    result["sha256"] = digest({k: v for k, v in result.items() if k != "sha256"})
    return result


def publish_binding(path, value):
    """Exclusive immutable publication, including on retry/partial residue."""
    with Path(path).open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def register(recovery_root):
    root = Path(recovery_root).resolve()
    original = read_json(ROOT/OLD_INPUT_PATH, expected_file_sha256=OLD_INPUT_FILE_SHA256)
    snapshot = under(root, "publication/snapshots/" + original["snapshot"]["snapshot_id"])
    manifest = read_json(snapshot/"manifest.json", expected_file_sha256=NEW_MANIFEST)
    spec = read_json(snapshot/"feature_spec.json",
        expected_file_sha256=original["snapshot"]["feature_spec_file_sha256"])
    if file_hash(snapshot/"coverage_report.json") != COVERAGE:
        raise ValueError("frozen coverage bytes changed")
    receipts = {name: read_json(under(root, name), expected_file_sha256=checksum)
                for name, checksum in RECEIPTS.items()}
    proof = receipts["verification.json"]
    if Path(proof["snapshot_directory"]).resolve() != snapshot:
        raise ValueError("verification applies to a different snapshot")
    value = derive_binding(original, manifest, spec, proof, receipts["process-result.json"],
        receipts["reuse-result.json"], receipt_hashes=RECEIPTS)
    path = ROOT/NEW_INPUT_PATH
    publish_binding(path, value)
    return {"path": NEW_INPUT_PATH, "content_sha256": value["sha256"], "file_sha256": file_hash(path),
        "scientific_calls": 0, "raw_feature_trajectory_map_reads": 0, "final_eval_authorized": False}


def renewed_core_payload():
    """Retain the original core scope; full executor catalog is a later seal.

The matrix pins the protocol, so adding the matrix itself to this core would
create a circular hash. It remains in the mandatory full execution catalog.
"""
    from . import protocol
    old = unpack(read_json(ROOT/"experiments/pirc17/protocols"/
        "d1c98c3d6b089d790bddd0a4776d20c11d9c0982d3b6181ac81c3536d2a83e54.json",
        expected_file_sha256="7a90d2c6d6f28a52f77a8d21240c67e914ef785950a54de10ecdf05bbc2dd641"),
        expected_sha256="d1c98c3d6b089d790bddd0a4776d20c11d9c0982d3b6181ac81c3536d2a83e54")
    current = protocol.protocol_semantics()
    if {k: v for k, v in old.items() if k not in {"source_sha256", "dataset_inputs"}} != {
            k: v for k, v in current.items() if k != "dataset_inputs"}:
        raise ValueError("scientific/authority/management contract changed")
    restored = deepcopy(current["dataset_inputs"])
    provenance = restored.pop("manifest_reregistration")
    if (provenance["authorization"] != AUTHORIZATION
            or provenance["recovery_receipt_sha256"] != RECEIPTS
            or provenance["formal_evaluation_authorization"] is not False
            or restored["snapshot"]["manifest_sha256"] != NEW_MANIFEST):
        raise ValueError("authorized registration provenance required")
    restored["snapshot"]["manifest_sha256"] = OLD_MANIFEST
    restored["sha256"] = digest({k: v for k, v in restored.items() if k != "sha256"})
    if restored != old["dataset_inputs"]:
        raise ValueError("registration changed other input fields")
    catalog = dict(old["source_sha256"])
    pointer = "PSDE-SDE/experiments/pirc17/protocol.py"
    protocol.verify_sources({k: v for k, v in catalog.items() if k != pointer})
    catalog[pointer] = file_hash(ROOT/"experiments/pirc17/protocol.py")
    catalog["PSDE-SDE/"+NEW_INPUT_PATH] = file_hash(ROOT/NEW_INPUT_PATH)
    return {**current, "source_sha256": catalog}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-root", type=Path)
    parser.add_argument("--renew-core", action="store_true")
    args = parser.parse_args()
    if args.renew_core:
        from .protocol import validate_protocol
        from .protocol_core import envelope
        payload = renewed_core_payload()
        validate_protocol(envelope(payload))
        path, record = publish(ROOT/"experiments/pirc17/protocols", payload)
        print(json.dumps({"path": path.relative_to(ROOT).as_posix(),
            "protocol_sha256": record["sha256"], "file_sha256": file_hash(path),
            "core_sources": len(payload["source_sha256"]), "final_eval_authorized": False}))
    elif args.recovery_root is not None:
        print(json.dumps(register(args.recovery_root)))
    else:
        parser.error("provide --recovery-root or --renew-core")
