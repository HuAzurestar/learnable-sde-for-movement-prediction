"""Describe retained map catalog metadata; never open/query raw map values."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from .checkpoint_resume import load
from .protocol_core import canonical, digest, file_hash, read_json, under, unpack

SOURCES = {
    "DSDE-SDE/trajectory/online_terrain.py": "cf87a8397c0cf68bd4216cdfdc93ffdcc14f4f1cd9133a5adbd14477e6c0b8e9",
    "DSDE-SDE/trajectory/batched_terrain.py": "c263249e17c6dc6bd12df1df382b2d17c78e2c073e7eb6ecf088bcbe0bd7c6d8",
    "DSDE-SDE/trajectory/cached_terrain.py": "5b93d5961dfd30388f9f8545004d618182be66bc02f4d64201eb7cd9a59dd66c",
    "DSDE-SDE/trajectory/multicell_terrain.py": "edfd184d3b664c87dafbe70096cfff19bf113a6e84d3177f549fff125869ff23",
    "DSDE-SDE/trajectory/linear_materialization.py": "7e7f59fc0f93d23639b4a3bc9d778cd4f2617d3da4f4ea555c7e74fac4d661c9",
    "DSDE-SDE/trajectory/feature_geometry.py": "5df4d18572b48fbf01844a1d4c3e993a722bb5a7f68045dfc1192c6da2a64255",
    "DSDE-SDE/map_data/terrain_features.py": "6a71ea06093db394a6a3be3cca5c2a86f22e14873f806d52f7566f7c04d13376",
    "DSDE-SDE/trajectory/terrain_expansion.py": "727a9f099827e84f9b1bb342b74da209f46d6fa30b5c2bae4a2f1b6f03fa7351",
    "PSDE-SDE/experiments/pirc17/formal_maps.py": "56a9edc4602a22f340c2385356591343626a3c724df66d78135823cfc339f163",
    "PSDE-SDE/experiments/pirc17/features.py": "7ce422a2a83add8ad8484d52f8a83a4669c05a259d3fcb8c55ab5e06de4cc26c",
}
FAMILIES = ("srtm", "copernicus_dem", "worldcover", "overture", "hydrorivers")


def family(relative):
    matches = [name for name in FAMILIES if name in relative.lower()]
    if len(matches) != 1:
        raise ValueError("one recognized map family required")
    return matches[0]


def receipt_assets(receipts, expected):
    """Same valid/raw admission and later-equal-hash replacement as provider."""
    assets = {}
    for receipt in receipts:
        for row in receipt["assets"]:
            relative = row.get("local_path", "")
            if row.get("status") != "valid" or not relative or row.get("data_kind") != "raw":
                continue
            if not any(name in relative for name in FAMILIES):
                continue
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("contained receipt path required")
            family(relative)
            checksum = row.get("checksum", {})
            sha = checksum.get("value", "")
            if checksum.get("algorithm") != "sha256" or not re.fullmatch("[0-9a-f]{64}", sha):
                raise ValueError("original raw asset hash required")
            if relative in assets and assets[relative]["checksum"]["value"] != sha:
                raise ValueError("conflicting original receipt hashes")
            assets[relative] = row
    if {k: v["checksum"]["value"] for k, v in assets.items()} != expected:
        raise ValueError("complete receipt catalog differs from original policy")
    return assets


def describe_catalog(assets, parents, catalog):
    original = {k: v["checksum"]["value"] for k, v in assets.items()}
    merged = dict(original)
    parent_paths = set()
    for parent in parents:
        match = re.fullmatch(r"registered-file:(.+):sha256:([0-9a-f]{64})", parent)
        if not match:
            raise ValueError("original snapshot parent binding required")
        relative, sha = match.groups()
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or family(relative) not in FAMILIES[:3]:
            raise ValueError("contained original raster parent required")
        if relative in merged and merged[relative] != sha:
            raise ValueError("snapshot parent and receipt hashes disagree")
        merged[relative] = sha
        parent_paths.add(relative)
    if merged != catalog:
        raise ValueError("retained whole catalog does not equal receipts plus parents")
    result = []
    for name in FAMILIES:
        rows = [r for p, r in assets.items() if family(p) == name]
        fields = {}
        for key in ("provider", "dataset", "version", "resolution", "unit", "nodata", "temporal_extent"):
            values = Counter(canonical(r[key]).decode() for r in rows if key in r)
            fields[key] = {"missing_records": len(rows)-sum(values.values()),
                           "recorded_values": [{"value": json.loads(value), "records": count}
                                               for value, count in sorted(values.items())]}
        result.append({"family": name, "receipt_assets": len(rows),
                       "additional_snapshot_parent_assets": sum(family(p) == name for p in set(merged)-set(original)),
                       "retained_catalog_assets": sum(family(p) == name for p in merged),
                       "parent_references": sum(family(p) == name for p in parent_paths),
                       "receipt_metadata_only": fields})
    return {"families": result, "receipt_assets": len(original), "snapshot_parent_assets": len(parent_paths),
            "snapshot_parents_already_in_receipts": len(parent_paths & set(original)),
            "additional_snapshot_parent_assets": len(set(merged)-set(original)),
            "retained_catalog_assets": len(merged), "retained_catalog_content_sha256": digest(merged)}


def project(runtime):
    settings, _ = load(runtime)
    bundle = unpack(read_json(settings["bundle"]), expected_sha256=settings["bundle_sha256"])
    protocol, execution = unpack(bundle["protocol"]), unpack(bundle["execution"])
    context = unpack(read_json(settings["context"]), expected_sha256=settings["context_sha256"])
    catalog = unpack(context["map_catalog"])
    if (catalog["protocol_sha256"] != bundle["protocol"]["sha256"]
            or catalog["execution_sha256"] != bundle["execution"]["sha256"]):
        raise ValueError("saved map catalog execution binding differs")
    workspace = Path(__file__).resolve().parents[3]
    for name, sha in SOURCES.items():
        if file_hash(workspace/name) != sha or execution["source_sha256"].get(name) != sha:
            raise ValueError("original provider source differs")
        if name != "PSDE-SDE/experiments/pirc17/formal_maps.py" and protocol["source_sha256"].get(name) != sha:
            raise ValueError("frozen scientific map source differs")
    if any(SOURCES.get(name) != sha for name, sha in catalog["source_sha256"].items()):
        raise ValueError("saved map catalog source differs")
    policy = protocol["dataset_inputs"]["online_maps"]
    root = Path(settings["input_paths"]["data_root"])
    receipts = [read_json(under(root, relative), expected_file_sha256=sha)
                for relative, sha in sorted(policy["receipt_sha256"].items())]
    if (catalog["static_identity"]["query_version"] != policy["query_version"]
            or catalog["static_identity"]["receipt_sha256"] != {
                str((root/relative).resolve()): sha for relative, sha in policy["receipt_sha256"].items()}):
        raise ValueError("original receipt/backend identity differs")
    assets = receipt_assets(receipts, policy["admitted_receipt_assets_sha256"])
    result = describe_catalog(assets, catalog["registered_parent_witnesses"], catalog["asset_sha256"])
    result.update({"schema_version": "pirc17-map-provider-description-v1",
                   "protocol_sha256": bundle["protocol"]["sha256"],
                   "execution_sha256": bundle["execution"]["sha256"],
                   "saved_map_catalog_sha256": context["map_catalog"]["sha256"],
                   "receipt_sha256": policy["receipt_sha256"], "source_sha256": SOURCES,
                   "query_version": policy["query_version"],
                   "scope": {"catalog_membership_proves_raw_asset_use": False,
                             "receipt_specs_are_measured_all_raster_geometry": False,
                             "raster_pixels_or_forecast_coordinate_arrays_decoded": False,
                             "new_fits_forecasts_particle_scores_or_map_queries": 0,
                             "original_provider_or_map_parameters_changed": False,
                             "all_forecast_per_factor_invalid_query_rates_available": False,
                             "whole_snapshot_or_raw_assets_reverified": False,
                             "map_vintages_matched_to_source_trajectory_time": False,
                             "physical_source_clock_or_publication_permission_certified": False,
                             "private_asset_paths_bbox_or_trajectory_identifiers_exported": False,
                             "independent_saved_output_audit_completed": False,
                             "all_review_items_or_paper_complete": False}})
    canonical(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = project(args.runtime)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(canonical(result)+b"\n")
    print("Described retained receipt and snapshot-parent catalog; zero raw map queries.")


if __name__ == "__main__":
    main()
