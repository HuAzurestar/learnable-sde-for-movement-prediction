"""Synthetic catalog metadata only; no query provider is constructed."""
from copy import deepcopy

import pytest

from experiments.pirc17.map_provider_description import describe_catalog, family, receipt_assets


def fixture():
    row = {"local_path": "map_data/srtm_priority/a.hgt.gz", "status": "valid", "data_kind": "raw",
           "checksum": {"algorithm": "sha256", "value": "a"*64}, "version": "saved", "unit": "metre"}
    return row, {row["local_path"]: "a"*64}


@pytest.mark.parametrize("name", ["srtm", "copernicus_dem", "worldcover", "overture", "hydrorivers"])
def test_family_is_catalog_type_not_route_location(name):
    assert family("map_data/"+name+"/tile") == name


@pytest.mark.parametrize("name", ["unknown/file", "srtm/worldcover/tile"])
def test_ambiguous_or_unknown_family(name):
    with pytest.raises(ValueError): family(name)


def test_receipt_merge_matches_provider_and_nonraw_or_invalid_not_admitted():
    row, expected = fixture()
    missing = {**row, "status": "failed"}
    processed = {**row, "data_kind": "processed"}
    newer = {**row, "version": "later equal-hash receipt"}
    result = receipt_assets([{"assets": [missing, processed, row]}, {"assets": [newer]}], expected)
    assert result[row["local_path"]]["version"] == newer["version"]


@pytest.mark.parametrize("key,value", [("local_path", "../srtm/tile"),
                                      ("checksum", {"algorithm": "md5", "value": "a"*64}),
                                      ("checksum", {"algorithm": "sha256", "value": "b"*64})])
def test_changed_asset_or_unsafe_binding_rejected(key, value):
    row, expected = fixture(); row[key] = value
    with pytest.raises(ValueError): receipt_assets([{"assets": [row]}], expected)


def test_conflicting_receipt_hash_is_not_last_writer_wins():
    row, expected = fixture(); other=deepcopy(row); other["checksum"]["value"]="b"*64
    with pytest.raises(ValueError, match="conflicting"):
        receipt_assets([{"assets": [row, other]}], expected)


def test_parent_overlap_and_additions_do_not_invent_product_metadata_or_use():
    row, expected = fixture(); extra = "map_data/worldcover_old/tile.tif"
    parents = ["registered-file:"+row["local_path"]+":sha256:"+"a"*64,
               "registered-file:"+extra+":sha256:"+"b"*64]
    r = describe_catalog({row["local_path"]: row}, parents, {**expected, extra: "b"*64})
    assert r["receipt_assets"] == 1 and r["retained_catalog_assets"] == 2
    assert r["snapshot_parents_already_in_receipts"] == r["additional_snapshot_parent_assets"] == 1
    wc = next(x for x in r["families"] if x["family"] == "worldcover")
    assert wc["receipt_assets"] == 0 and wc["additional_snapshot_parent_assets"] == 1
    assert wc["receipt_metadata_only"]["version"]["recorded_values"] == []
    s = next(x for x in r["families"] if x["family"] == "srtm")
    assert s["receipt_metadata_only"]["provider"]["missing_records"] == 1


@pytest.mark.parametrize("parent", ["invented", "registered-file:../srtm/tile:sha256:"+"a"*64,
                                    "registered-file:map_data/overture/tile:sha256:"+"a"*64,
                                    "registered-file:map_data/srtm_priority/a.hgt.gz:sha256:"+"b"*64])
def test_invalid_parent_or_conflict_rejected(parent):
    row, expected = fixture()
    with pytest.raises(ValueError): describe_catalog({row["local_path"]: row}, [parent], expected)


def test_missing_final_catalog_member_is_not_accepted():
    row, _ = fixture()
    with pytest.raises(ValueError): describe_catalog({row["local_path"]: row}, [], {})
