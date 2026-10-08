"""Frozen map catalog and durable, explicitly cumulative suffix observations.

Catalog membership is not an observation that an asset was used or verified.
Base-only and audit-only work must not invent terrain queries. Additional
visited assets must come from the same pinned receipts/snapshot parents.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re

from . import particle_suffix_contract as science

native, engine, admission = science.native, science.engine, science.admission
VERSION = science.VERSION + "-maps"
OBS_FIELDS = {"schema_version", "root_sha256", "start_sha256", "index", "row_sha256",
              "catalog_sha256", "map_identity", "scope"}
SCOPE = "cumulative current-attempt map identity after this forecast; null for no raw-map query"


def catalog(prepared, *, guard=lambda: None):
    """Reconstruct the registered catalog without any position query."""
    guard()
    expected = prepared.original_reference[-1]["maps"]
    root = Path(prepared.parent_root["data_locations"]["data_root"])
    fit = admission.read_closed(prepared.parent_root["source"]["fit"], prepared.spec["fit_sha256"])
    query_type, _ = engine.resolve_map_backend(prepared.spec["map_backend"])
    query = query_type(root, [root/p for p in engine.RECEIPTS])
    try:
        for parent in fit["snapshot_parent_assets"]:
            query.register_snapshot_parent(parent)
        static = {k: deepcopy(v) for k, v in query.identity.items() if k != "verified_assets"}
        if static != {k: v for k, v in expected.items() if k != "verified_assets"}:
            raise ValueError("suffix map backend, receipts or snapshot parents changed")
        assets = {p: v["checksum"]["value"] for p, v in query.assets.items()}
        for relative, digest in assets.items():
            path = Path(relative)
            if (path.is_absolute() or not (root/path).resolve().is_relative_to(root.resolve())
                    or not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest)):
                raise ValueError("suffix catalog requires contained immutable asset hashes")
        if any(assets.get(p) != sha for p, sha in expected["verified_assets"].items()):
            raise ValueError("suffix catalog omits or changes a previously used parent asset")
    finally:
        query.close()
    prepared.recheck(guard)
    return {"schema_version": VERSION+"-catalog", "static_identity": static, "asset_sha256": assets,
            "scope": "registered catalog only; NOT a record of map use or full asset verification"}


def identity_assets(identity, registered, data_root):
    if (not isinstance(identity, dict) or not isinstance(identity.get("verified_assets"), dict)
            or {k: v for k, v in identity.items() if k != "verified_assets"} != registered["static_identity"]):
        raise ValueError("suffix map observation changes static backend or receipt identity")
    assets = identity["verified_assets"]
    if any(registered["asset_sha256"].get(p) != sha for p, sha in assets.items()):
        raise ValueError("suffix observed an unregistered or differently hashed map asset")
    root = Path(data_root).resolve()
    return {root/p: sha for p, sha in assets.items()}


def observation(row, *, index, root_sha256, start_sha256, registered, identity):
    return {"schema_version": VERSION+"-observation", "index": index, "root_sha256": root_sha256,
            "start_sha256": start_sha256, "catalog_sha256": science._digest(registered),
            "row_sha256": science._digest({k: v for k, v in row.items() if k != "particle_artifact"}),
            "map_identity": deepcopy(identity), "scope": SCOPE}


def check_observation(value, row, *, index, root_sha256, start_sha256, registered, data_root):
    if (not isinstance(value, dict) or set(value) != OBS_FIELDS or type(value["index"]) is not int
            or value != observation(row, index=index, root_sha256=root_sha256, start_sha256=start_sha256,
                                    registered=registered, identity=value.get("map_identity"))):
        raise ValueError("suffix map observation changes its exact row, attempt or catalog")
    if row["configuration"] == "base":
        if value["map_identity"] is not None or row["suffix_accounting"]["new_raw_map_query_rows"] != 0:
            raise ValueError("base forecast cannot claim raw-map use")
        return {}
    return identity_assets(value["map_identity"], registered, data_root)


def verify_attempt(work, saved, *, root_sha256, start_sha256, registered, data_root, guard=lambda: None):
    """Verify each committed observation, then hash all distinct observed assets."""
    work = Path(work)
    catalog_path = work/"map-catalog.json"
    actual = admission.read_closed(catalog_path, native._hash(catalog_path))
    if actual != registered:
        raise ValueError("saved suffix map catalog differs from the frozen original")
    bound, committed, observations = {}, set(), []
    for record in saved.records:
        guard()
        row, index = record["row"], record["index"]
        if row["status"] != "success":
            continue
        path = work/"map-observations"/f"{index:06d}.json"
        value = admission.read_closed(path, native._hash(path))
        bound.update(check_observation(value, row, index=index, root_sha256=root_sha256,
            start_sha256=start_sha256, registered=registered, data_root=data_root))
        committed.add(path)
        observations.append({"index": index, "sha256": native._hash(path)})
    for path, digest in bound.items():
        guard()
        if native._hash(path) != digest:
            raise ValueError("observed suffix map asset changed")
    orphaned = [{"path": p.relative_to(work).as_posix(), "sha256": native._hash(p)}
                for p in sorted((work/"map-observations").rglob("*")) if p.is_file() and p not in committed]
    return {"committed_observations": observations, "uncommitted_observations": orphaned,
            "observed_asset_sha256": {str(p): sha for p, sha in bound.items()}}
