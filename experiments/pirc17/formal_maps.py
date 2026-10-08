"""Guarded static-map catalog and one retained exact-query provider.

Catalog admission is not a map query or proof every raw asset was used. Parent
admission uses the WHOLE frozen snapshot, independent of selected origins,
their future positions, scores or configuration. Raw values remain lazy and
hash-checked by the original map backend before use. No download or fallback.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from . import final_eval_guard as guard
from .development_rollout import resolve_map_backend
from .formal_eligibility import _bound_file
from .protocol_core import digest, envelope, file_hash, read_json, under, unpack
from .protocol_inputs import validate_snapshot_metadata

VERSION = "pirc17-formal-map-catalog-v1"
PARENT_COLUMNS = ("dem_surface_parent_asset_id", "worldcover_parent_asset_id")


def _open_query(root, policy, parents, execution):
    receipts = [_bound_file(root, relative, checksum) for relative, checksum in sorted(policy["receipt_sha256"].items())]
    query_type, modules = resolve_map_backend("multicell")
    sealed_sources = unpack(execution)["source_sha256"]
    source_hashes = {}
    for module in modules:
        key = "DSDE-SDE/"+module.__name__.replace(".", "/")+".py"
        checksum = file_hash(module.__file__)
        if sealed_sources.get(key) != checksum:
            raise ValueError("map backend differs from execution source binding")
        source_hashes[key] = checksum
    query = query_type(root, receipts, max_cells=policy["geometry_cache_limits"]["retained_cells"],
        raster_cache_bytes=policy["raster_cache_limits"]["retained_array_bytes"],
        raster_cache_readers=policy["raster_cache_limits"]["open_readers"])
    try:
        if {k: a["checksum"]["value"] for k, a in query.assets.items()} != policy["admitted_receipt_assets_sha256"]:
            raise ValueError("raw map receipt catalog changed")
        for parent in parents:
            query.register_snapshot_parent(parent)
        observed = query.identity
        if (observed["query_version"] != policy["query_version"]
                or observed["raster_cache_limits"] != policy["raster_cache_limits"]
                or observed["geometry_cache_limits"] != policy["geometry_cache_limits"]
                or observed["receipt_sha256"] != {str(p.resolve()): policy["receipt_sha256"][p.relative_to(root).as_posix()] for p in receipts}
                or observed["verified_assets"]):
            raise ValueError("map initialization changed policy or claimed unperformed raw use")
    except BaseException:
        query.close()
        raise
    return query, source_hashes


class RegisteredMaps:
    """Synchronous retained provider, owned/closed by the measured worker."""
    def __init__(self, query, catalog, *, root, policy, parents, execution):
        self.query, self.catalog = query, deepcopy(catalog)
        self.root, self.policy, self.parents, self.execution = root, deepcopy(policy), tuple(parents), deepcopy(execution)
        self.attempted_query_rows = self.completed_query_rows = 0
        self.closed = False

    def __call__(self, lonlat):
        if self.closed:
            raise ValueError("map provider is closed")
        xy = np.asarray(lonlat)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("map input must be predicted lon/lat rows")
        self.attempted_query_rows += len(xy)
        result = self.query(xy)
        if len(result) != len(xy):
            raise ValueError("map query changed row count")
        self.completed_query_rows += len(xy)
        return result

    def observation(self):
        if self.closed:
            raise ValueError("observe maps before closing the provider")
        identity = deepcopy(self.query.identity)
        catalog = unpack(self.catalog)
        if ({k: v for k, v in identity.items() if k != "verified_assets"} != catalog["static_identity"]
                or any(catalog["asset_sha256"].get(k) != v for k, v in identity["verified_assets"].items())):
            raise ValueError("used map identity is not in the admitted static catalog")
        return {"catalog_sha256": self.catalog["sha256"], "identity": identity,
            "attempted_query_rows": self.attempted_query_rows, "completed_query_rows": self.completed_query_rows,
            "scope": "cumulative retained-provider use, not all-catalog verification or this-forecast-only assets"}

    def fresh_provider(self):
        """For registered provider-cold trials only; caller must meter and close."""
        if self.closed:
            raise ValueError("cannot recreate a closed map owner")
        query, sources = _open_query(self.root, self.policy, self.parents, self.execution)
        catalog = unpack(self.catalog)
        if (sources != catalog["source_sha256"]
                or {k: v for k, v in query.identity.items() if k != "verified_assets"} != catalog["static_identity"]
                or {k: a["checksum"]["value"] for k, a in query.assets.items()} != catalog["asset_sha256"]):
            query.close()
            raise ValueError("provider-cold recreation changed the fixed map catalog")
        return RegisteredMaps(query, self.catalog, root=self.root, policy=self.policy, parents=self.parents, execution=self.execution)

    def close(self):
        if not self.closed:
            self.query.close()
            self.closed = True


def _prepare_maps(access, *, protocol, execution, snapshot, data_root):
    binding = unpack(protocol)["dataset_inputs"]
    if (access.access_kind != "final_eval_features" or access.population_sha256 is None
            or access.protocol_sha256 != protocol["sha256"] or access.execution_sha256 != execution["sha256"]
            or access.legacy_cohort_ack != binding["dataset_id"]):
        raise ValueError("approved exact population-bound map access required")
    frozen = binding["snapshot"]
    manifest = read_json(under(snapshot, "manifest.json"), expected_file_sha256=frozen["manifest_sha256"])
    spec = read_json(under(snapshot, "feature_spec.json"), expected_file_sha256=frozen["feature_spec_file_sha256"])
    if (manifest["dataset_id"] != binding["dataset_id"] or manifest["snapshot_id"] != frozen["snapshot_id"]
            or manifest["feature_spec_id"] != frozen["feature_spec_id"] or spec["feature_spec_id"] != frozen["feature_spec_id"]
            or digest(spec) != frozen["feature_spec_content_sha256"]
            or manifest["status"] != "valid"):
        raise ValueError("map-parent snapshot identity changed")
    validate_snapshot_metadata(manifest, spec, expected_inventory_sha256=frozen["content_inventory_sha256"],
                               expected_files=frozen["files"])
    parents, rows = {}, 0
    # Bounded batches of provenance strings only. Never select source tiles by
    # a forecast's hidden future path or admit new assets after seeing scores.
    for entry in sorted(manifest["files"], key=lambda r: r["path"]):
        path = _bound_file(snapshot, entry["path"], entry["sha256"])
        source = pq.ParquetFile(path)
        if source.metadata.num_rows != entry["row_count"]:
            raise ValueError("map-parent feature row count changed")
        for batch in source.iter_batches(batch_size=8192, columns=list(PARENT_COLUMNS)):
            if batch.schema.names != list(PARENT_COLUMNS):
                raise ValueError("frozen snapshot lacks explicit map-parent columns")
            rows += batch.num_rows
            for column in batch.columns:
                for parent in set(column.to_pylist()):
                    if parent is not None and not isinstance(parent, str):
                        raise ValueError("invalid snapshot parent identity")
                    if parent and parent.startswith("registered-file:"):
                        parents.setdefault(parent, {"feature_path": entry["path"], "feature_sha256": entry["sha256"]})
    root = Path(data_root).resolve()
    policy = binding["online_maps"]
    query, sources = _open_query(root, policy, sorted(parents), execution)
    try:
        catalog = envelope({"schema_version": VERSION, "protocol_sha256": protocol["sha256"],
            "execution_sha256": execution["sha256"], "approval_sha256": access.approval_sha256,
            "population_sha256": access.population_sha256, "access_started_sha256": access.access_started_sha256,
            "snapshot_inventory_sha256": frozen["content_inventory_sha256"], "parent_columns": list(PARENT_COLUMNS),
            "snapshot_files_inspected": len(manifest["files"]), "snapshot_rows_inspected": rows,
            "registered_parent_witnesses": parents, "source_sha256": sources,
            "static_identity": {k: v for k, v in query.identity.items() if k != "verified_assets"},
            "asset_sha256": {k: a["checksum"]["value"] for k, a in query.assets.items()},
            "catalog_selection": "entire frozen snapshot; independent of selected forecast origins, targets and performance",
            "raw_asset_value_queries": 0})
        return RegisteredMaps(query, catalog, root=root, policy=policy, parents=sorted(parents), execution=execution)
    except BaseException:
        query.close()
        raise


def prepare_formal_maps(*, protocol, execution, approval_path, approval_sha256, test_path, review_path,
                        journal_directory, population_path, population_sha256, eligibility_path, snapshot, data_root):
    return guard.guarded_call(access_kind="final_eval_features", protocol=protocol, execution=execution,
        approval_path=approval_path, approval_sha256=approval_sha256, test_path=test_path, review_path=review_path,
        journal_directory=journal_directory, population_path=population_path, population_sha256=population_sha256,
        eligibility_path=eligibility_path, operation=lambda access: _prepare_maps(access, protocol=protocol,
            execution=execution, snapshot=snapshot, data_root=data_root))
