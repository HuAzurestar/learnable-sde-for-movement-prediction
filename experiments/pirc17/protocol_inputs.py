"""Freeze existing input identities, not trajectories or evaluation outcomes.

Only release/snapshot metadata, identity-only sample manifests and previously
closed development summaries are read. Raw maps/condition/feature/trajectory
files are NOT opened. Runtime must verify each used raw file against this chain.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .protocol_core import decode, digest, file_hash, read_json, relative_path, sha256, under

VERSION = "pirc17-formal-input-binding-v1"
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "experiments/pirc17/evidence/protocol-inputs-v1.json"
LOCATION_PATH = "artifacts/pirc17/dev10/native-primary-p1024-missing-v1.extension/root.json"
LOCATION_SHA = "381c000b3ba9919d08325968255a4708a001d4a2a550940765428d0a37ecbc7d"
PREPARATION_PATH = "artifacts/pirc17/dev10/method-development-preparation-v1.json"
PREPARATION_SHA = "eec69a9d9ee502e6f5f574a1b7ebc9ea0ec561c7789bf23b1e3b142aa9e3201e"
TERRAIN_LEDGER = "artifacts/pirc17/dev10/small-budget-full-p512-h5-v1.jsonl"
TERRAIN_LEDGER_SHA = "5a67eb8b954382ee2526b8c5bb8b37b106969098254aae83b0a6061119a60d16"
DATASET_ID = "pirc20-r1t-nex326-midpoint-20260912-v1"
SNAPSHOT_ID = "pirc21-production-features-pirc18-20260921-v1"
SPLIT_COUNTS = {"train": 49845, "validation": 9831, "final_eval": 12370}


def validate_snapshot_metadata(manifest, spec, *, expected_inventory_sha256, expected_files):
    if manifest["feature_spec_sha256"] != digest(spec):
        raise ValueError("canonical feature-spec fingerprint changed")
    rows = manifest["files"]
    keys = [(r["split"], r["file_id"]) for r in rows]
    paths = [relative_path(r["path"]) for r in rows]
    if len(keys) != expected_files or len(keys) != len(set(keys)) or len(paths) != len(set(paths)):
        raise ValueError("snapshot file inventory identity/count changed")
    inventory = hashlib.sha256()
    for row in sorted(rows, key=lambda r: r["path"]):
        checksum = sha256(row["sha256"])
        if (row["split"] not in SPLIT_COUNTS or type(row["row_count"]) is not int
                or row["row_count"] < 0):
            raise ValueError("invalid snapshot role/row count")
        inventory.update(f"{row['path']}\0{checksum}\0{row['row_count']}\n".encode("utf-8"))
    if inventory.hexdigest() != manifest["content_inventory_sha256"] or inventory.hexdigest() != expected_inventory_sha256:
        raise ValueError("snapshot metadata inventory hash mismatch")
    return len(keys)


def identity_partition_audit(path, expected_sha256):
    """Only identity fields enter the audit; final targets/features never load."""
    if file_hash(path) != sha256(expected_sha256):
        raise ValueError("sample identity manifest changed")
    identity_names = ("sample_id", "segment_id", "independent_block_id")
    seen = {role: {name: set() for name in identity_names} for role in SPLIT_COUNTS}
    paired_identities = {role: [] for role in SPLIT_COUNTS}
    counts = dict.fromkeys(SPLIT_COUNTS, 0)
    with Path(path).open("rb") as stream:
        while raw := stream.readline(16385):
            if len(raw) > 16384:
                raise ValueError("sample identity row exceeds bound")
            row = decode(raw)
            role = row["split"]
            if role not in counts:
                raise ValueError("unregistered data split")
            counts[role] += 1
            if counts[role] > SPLIT_COUNTS[role]:
                raise ValueError("sample count exceeds frozen release")
            for name in identity_names:
                value = row[name]
                if not isinstance(value, str) or not value.strip() or len(value) > 1024:
                    raise ValueError("nonempty bounded sample/segment/block identity required")
                if name == "sample_id":
                    sha256(value)
                if name == "sample_id" and value in seen[role][name]:
                    raise ValueError("duplicate sample identity")
                seen[role][name].add(value)
            paired_identities[role].append([row[name] for name in identity_names])
    if counts != SPLIT_COUNTS:
        raise ValueError("sample identity population differs from release")
    pairs = [(a, b) for i, a in enumerate(SPLIT_COUNTS) for b in list(SPLIT_COUNTS)[i+1:]]
    intersections = {f"{a}:{b}": {name: len(seen[a][name] & seen[b][name]) for name in identity_names}
                     for a, b in pairs}
    if any(count for fields in intersections.values() for count in fields.values()):
        raise ValueError("train/validation/final-eval identity overlap")
    return {"counts": counts, "intersections": intersections,
        "split_identity_sha256": {role: digest({name: sorted(values) for name, values in fields.items()})
                                  for role, fields in seen.items()},
        "split_row_identity_sha256": {role: digest(sorted(rows)) for role, rows in paired_identities.items()},
        "fields_used": ["split", *identity_names], "label_prediction_metric_reads": 0}


def build_binding():
    locator = read_json(ROOT/LOCATION_PATH, expected_file_sha256=LOCATION_SHA)
    locations = locator["data_locations"]  # Locator only, NEVER a restart/authorization.
    release, snapshot, data_root = (Path(locations[k]).resolve() for k in ("release", "snapshot", "data_root"))
    preparation = read_json(ROOT/PREPARATION_PATH, expected_file_sha256=PREPARATION_SHA)
    identity = preparation["identity"]
    dataset = read_json(release/"dataset.json", expected_file_sha256=identity["dataset_sha256"])
    if dataset["dataset_id"] != DATASET_ID or identity["dataset_id"] != DATASET_ID:
        raise ValueError("registered real dataset identity required")
    artifacts = {relative_path(name): sha256(entry["sha256"]) for name, entry in dataset["artifacts"].items()}
    cohort = read_json(release/"cohort.json", expected_file_sha256=artifacts["cohort.json"])
    if cohort["cohort_id"] != DATASET_ID or cohort["sample_manifest_sha256"] != artifacts["samples.jsonl"]:
        raise ValueError("release/cohort/sample identity chain differs")
    partitions = identity_partition_audit(release/"samples.jsonl", artifacts["samples.jsonl"])
    eligibility = read_json(locations["eligibility"], expected_file_sha256=identity["eligibility_sha256"])
    if eligibility["dataset_id"] != DATASET_ID or eligibility["final_eval_label_prediction_metric_reads"] != 0:
        raise ValueError("existing development eligibility binding required")
    manifest = read_json(snapshot/"manifest.json")
    spec = read_json(snapshot/"feature_spec.json")
    if (manifest["dataset_id"] != DATASET_ID or manifest["snapshot_id"] != SNAPSHOT_ID
            or manifest["content_inventory_sha256"] != eligibility["snapshot"]["content_inventory_sha256"]
            or manifest["feature_spec_id"] != spec["feature_spec_id"]):
        raise ValueError("existing frozen feature snapshot identity required")
    # The manifest itself binds every final-eval feature-file identity; no such
    # file is opened. Full raw hashes were previously audited in DEV07, and each
    # actually used file must be reverified after authorization at runtime.
    file_count = validate_snapshot_metadata(manifest, spec,
        expected_inventory_sha256=eligibility["snapshot"]["content_inventory_sha256"], expected_files=7618)
    ledger_path = ROOT/TERRAIN_LEDGER
    if file_hash(ledger_path) != TERRAIN_LEDGER_SHA:
        raise ValueError("closed terrain map evidence changed")
    last = None
    with ledger_path.open("rb") as stream:
        for line in stream:
            last = line
    closed = decode(last)
    if closed["status"] != "complete" or closed["final_eval_label_prediction_metric_reads"] != 0:
        raise ValueError("closed development map ledger required")
    maps = closed["maps"]
    receipts, admitted_assets = {}, {}
    for absolute, expected in maps["receipt_sha256"].items():
        relative = Path(absolute).resolve().relative_to(data_root).as_posix()
        receipt = read_json(under(data_root, relative), expected_file_sha256=expected)
        receipts[relative] = expected
        # Freeze the entire existing receipt catalog, NOT just the ten assets
        # touched by three pilot origins. Snapshot raster parents are additional
        # only when proven by the hash-bound snapshot lineage at runtime.
        for asset in receipt["assets"]:
            name = asset.get("local_path", "")
            if (asset.get("status") != "valid" or asset.get("data_kind") != "raw" or not name
                    or not any(k in name for k in ("srtm", "copernicus_dem", "worldcover", "overture", "hydrorivers"))):
                continue
            name = relative_path(name)
            if asset.get("checksum", {}).get("algorithm") != "sha256":
                raise ValueError("raw map asset lacks SHA256")
            checksum = sha256(asset["checksum"]["value"])
            if admitted_assets.setdefault(name, checksum) != checksum:
                raise ValueError("conflicting map receipt asset identity")
    if len(receipts) != 5 or not admitted_assets:
        raise ValueError("complete existing map receipt catalog required")
    payload = {"schema_version": VERSION, "dataset_id": DATASET_ID,
        "dataset_file_sha256": identity["dataset_sha256"], "release_artifact_sha256": artifacts,
        "source_trajectory_sha256": sha256(dataset["source"]["trajectory"]["sha256"]),
        "partitions": partitions,
        "snapshot": {"snapshot_id": SNAPSHOT_ID, "manifest_sha256": file_hash(snapshot/"manifest.json"),
            "content_inventory_sha256": manifest["content_inventory_sha256"],
            "feature_spec_id": manifest["feature_spec_id"], "feature_spec_content_sha256": manifest["feature_spec_sha256"],
            "feature_spec_file_sha256": file_hash(snapshot/"feature_spec.json"),
            "history_window_points": manifest["history_window_points"], "files": file_count,
            "counts": manifest["counts"]},
        "development": {"eligibility_sha256": identity["eligibility_sha256"],
            "preparation_sha256": PREPARATION_SHA, "method_input_sha256": identity["sha256"],
            "method_population_identity": identity["population_identity"], "sample_counts": identity["sample_counts"],
            "observed_points": identity["observed_points"], "region_counts": identity["region_counts"],
            "qualification_splits": eligibility["splits"], "roles": ["train", "validation"],
            "outer_train_windows": 404, "validation_windows": 81},
        "online_maps": {"receipt_sha256": receipts, "admitted_receipt_assets_sha256": admitted_assets,
            "query_version": maps["query_version"], "raster_cache_limits": maps["raster_cache_limits"],
            "geometry_cache_limits": maps["geometry_cache_limits"],
            "snapshot_parent_rule": "Only exact registered-file identities read from hash-verified files in the frozen snapshot. Verify raw-file hash before use, retain every admitted parent in run provenance; no invented parent, download or new receipt.",
            "closed_pilot_touched_assets": len(maps["verified_assets"]),
            "catalog_scope": "All assets admitted by five existing receipt files, not a coverage guarantee for final predicted positions."},
        "history": {"prior_final_eval_exposed": True, "new_untouched_holdout_required": False,
            "forced_exploratory_downgrade": False,
            "user_decision": "2026-09-26: use already downloaded real data with separated train/evaluation; no new holdout or fabricated data.",
            "evidence": "MPA/project/PIRC-17/gists/DEV-07-exposure.md",
            "historical_release_final_eval_windows": 12370,
            "separate_historical_geolife_final_eval_windows": 17038,
            "geolife_used_by_this_protocol": False,
            "zero_read_scope": "This protocol preparation only; never a claim of no historical exposure."},
        "runtime_obligation": "After exact ACCEPT01, verify each used release/feature/condition/map/trajectory identity before decoding its values. Eligibility and selected paired populations freeze before performance. New formal fitted-model hashes are recorded after deterministic training, not invented here.",
        "final_eval_label_prediction_metric_reads": 0, "raw_feature_condition_trajectory_files_opened": 0,
        "formal_training_accepted": False, "final_eval_authorized": False}
    return {**payload, "sha256": digest(payload)}


if __name__ == "__main__":
    binding = build_binding()
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(binding, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"path": str(OUTPUT.relative_to(ROOT)), "sha256": binding["sha256"],
        "receipt_assets": len(binding["online_maps"]["admitted_receipt_assets_sha256"]),
        "final_eval_label_prediction_metric_reads": 0, "raw_feature_condition_trajectory_files_opened": 0}))
