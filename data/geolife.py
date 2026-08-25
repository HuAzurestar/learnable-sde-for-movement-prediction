"""GeoLife ingestion contract and independently reproducible audit helpers.

The cleaned GeoLife table is a local research asset and is never committed to
this repository. This module validates its schema, deterministic file-level
partitioning, cross-source leakage boundary, and locally registered provenance
without inventing a source or licence conclusion when evidence is absent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .validation import DataValidationError


PARTITIONS = ("train", "val", "eval")
REQUIRED_COLUMNS = frozenset({"file_id", "segment_id", "t", "x", "y"})


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"{label} missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DataValidationError(f"{label} must contain a JSON object: {path}")
    return value


def _parse_ratio(value: Any) -> tuple[float, float, float]:
    if isinstance(value, str):
        try:
            parts = tuple(float(part) for part in value.split("/"))
        except ValueError as exc:
            raise DataValidationError(f"invalid GeoLife split ratio: {value!r}") from exc
    elif isinstance(value, (list, tuple)):
        parts = tuple(float(part) for part in value)
    else:
        raise DataValidationError(f"invalid GeoLife split ratio: {value!r}")
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise DataValidationError(
            f"GeoLife split ratio must have three positive parts: {value!r}"
        )
    total = sum(parts)
    return tuple(part / total for part in parts)  # type: ignore[return-value]


def geolife_paths(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    return {
        "root": root,
        "table": root / "geolife_leg.parquet",
        "splits": root / "geolife_splits.json",
        "build_stats": root / "geolife_build_stats.json",
    }


def validate_geolife_schema(root: str | Path) -> dict[str, Any]:
    paths = geolife_paths(root)
    table_path = paths["table"]
    if not table_path.is_file():
        raise DataValidationError(f"GeoLife cleaned table missing: {table_path}")
    parquet = pq.ParquetFile(table_path)
    columns = set(parquet.schema_arrow.names)
    missing = sorted(REQUIRED_COLUMNS - columns)
    if missing:
        raise DataValidationError(f"GeoLife schema missing columns {missing}: {table_path}")
    return {
        "path": str(table_path),
        "rows": parquet.metadata.num_rows,
        "columns": parquet.schema_arrow.names,
        "required_columns_present": True,
    }


def geolife_file_id_splits(root: str | Path) -> dict[str, set[str]]:
    """Rebuild the registered file-level split and verify its file counts.

    The registered cleaned-data manifest stores aggregate counts rather than
    file-id lists. Therefore the adapter intentionally replays the registered
    algorithm: sorted unique IDs,
    ``RandomState(seed).shuffle``, then the registered ratio boundaries.
    """
    paths = geolife_paths(root)
    validate_geolife_schema(root)
    manifest = _read_json(paths["splits"], label="GeoLife split manifest")
    if manifest.get("split") != "file_id":
        raise DataValidationError("GeoLife split manifest must declare split='file_id'")
    try:
        seed = int(manifest["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataValidationError("GeoLife split manifest has no valid integer seed") from exc
    ratio = _parse_ratio(manifest.get("ratio"))

    frame = pd.read_parquet(paths["table"], columns=["file_id"])
    file_ids = np.array(sorted(frame["file_id"].astype(str).unique()))
    bad_ids = [file_id for file_id in file_ids if not file_id.startswith("GL_")]
    if bad_ids:
        raise DataValidationError(
            f"GeoLife file_id domain contains non-GL_ values: {bad_ids[:5]}"
        )
    rng = np.random.RandomState(seed)
    rng.shuffle(file_ids)
    first = int(ratio[0] * len(file_ids))
    second = int((ratio[0] + ratio[1]) * len(file_ids))
    result = {
        "train": set(file_ids[:first]),
        "val": set(file_ids[first:second]),
        "eval": set(file_ids[second:]),
    }
    if set.union(*result.values()) != set(file_ids):
        raise DataValidationError("GeoLife file split is not exhaustive")
    pairs = (("train", "val"), ("train", "eval"), ("val", "eval"))
    if any(result[left] & result[right] for left, right in pairs):
        raise DataValidationError("GeoLife file split is not disjoint")

    expected_files = manifest.get("files")
    if not isinstance(expected_files, Mapping):
        raise DataValidationError("GeoLife split manifest has no files count mapping")
    actual_files = {part: len(result[part]) for part in PARTITIONS}
    registered_files = {part: int(expected_files.get(part, -1)) for part in PARTITIONS}
    if actual_files != registered_files:
        raise DataValidationError(
            f"GeoLife split file counts disagree: actual={actual_files}, registered={registered_files}"
        )
    return result


def geolife_split_audit(root: str | Path) -> dict[str, Any]:
    """Recompute point/segment/file totals for every registered partition."""
    paths = geolife_paths(root)
    manifest = _read_json(paths["splits"], label="GeoLife split manifest")
    splits = geolife_file_id_splits(root)
    frame = pd.read_parquet(paths["table"], columns=["file_id", "segment_id"])
    frame["file_id"] = frame["file_id"].astype(str)
    actual: dict[str, dict[str, int]] = {}
    for part in PARTITIONS:
        selected = frame[frame["file_id"].isin(splits[part])]
        actual[part] = {
            "files": int(selected["file_id"].nunique()),
            "segments": int(selected[["file_id", "segment_id"]].drop_duplicates().shape[0]),
            "points": int(len(selected)),
        }
    for metric in ("files", "segments", "points"):
        registered = manifest.get(metric)
        if not isinstance(registered, Mapping):
            raise DataValidationError(f"GeoLife split manifest has no {metric} mapping")
        expected = {part: int(registered.get(part, -1)) for part in PARTITIONS}
        measured = {part: actual[part][metric] for part in PARTITIONS}
        if measured != expected:
            raise DataValidationError(
                f"GeoLife split {metric} disagree: actual={measured}, registered={expected}"
            )
    return {
        "algorithm": "sorted(file_id) -> RandomState(seed).shuffle -> registered ratio",
        "partitions": actual,
        "disjoint": True,
        "exhaustive": True,
    }


def assert_cross_source_no_leak(
    geolife_root: str | Path, osm_root: str | Path
) -> dict[str, int]:
    gl_path = geolife_paths(geolife_root)["table"]
    osm_path = Path(osm_root) / "unified_full_leg.parquet"
    if not osm_path.is_file():
        raise DataValidationError(f"OSM unified table missing: {osm_path}")
    gl_ids = set(
        pd.read_parquet(gl_path, columns=["file_id"])["file_id"].astype(str).unique()
    )
    osm_ids = set(
        pd.read_parquet(osm_path, columns=["file_id"])["file_id"].astype(str).unique()
    )
    overlap = gl_ids & osm_ids
    if overlap:
        raise DataValidationError(
            f"GeoLife/OSM file_id leakage detected: {sorted(overlap)[:5]}"
        )
    return {
        "geolife_file_ids": len(gl_ids),
        "osm_file_ids": len(osm_ids),
        "intersection": 0,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_audit(
    root: str | Path,
    registry_path: str | Path | None,
    *,
    verify_sha256: bool = True,
) -> dict[str, Any]:
    """Report registered provenance without claiming independent legal review."""
    if registry_path is None or not Path(registry_path).is_file():
        return {
            "status": "unconfirmed",
            "reason": "source/licence registry was not supplied or does not exist",
        }
    registry = _read_json(Path(registry_path), label="GeoLife source registry")
    required = ("registry_id", "dataset", "provider", "url", "version", "license")
    missing = [key for key in required if not registry.get(key)]
    if missing:
        return {"status": "unconfirmed", "reason": f"registry missing fields: {missing}"}
    product = registry.get("cleaned_product")
    if not isinstance(product, Mapping) or not product.get("sha256"):
        return {"status": "unconfirmed", "reason": "registry has no cleaned-product sha256"}
    result: dict[str, Any] = {
        "status": "registered_not_independently_legal_verified",
        "registry_id": registry["registry_id"],
        "dataset": registry["dataset"],
        "provider": registry["provider"],
        "url": registry["url"],
        "version": registry["version"],
        "license_record": registry["license"],
        "caveats": registry.get("caveats", []),
    }
    if verify_sha256:
        actual_sha256 = _sha256(geolife_paths(root)["table"])
        expected_sha256 = str(product["sha256"]).lower()
        if actual_sha256 != expected_sha256:
            raise DataValidationError(
                "GeoLife cleaned table sha256 mismatch: "
                f"actual={actual_sha256}, registered={expected_sha256}"
            )
        result["sha256"] = actual_sha256
        result["sha256_match"] = True
    return result


def audit_geolife_ingestion(
    geolife_root: str | Path,
    *,
    osm_root: str | Path | None = None,
    registry_path: str | Path | None = None,
    verify_sha256: bool = True,
) -> dict[str, Any]:
    """Return a JSON-serialisable, evidence-first ingestion audit."""
    schema = validate_geolife_schema(geolife_root)
    split = geolife_split_audit(geolife_root)
    leakage = (
        assert_cross_source_no_leak(geolife_root, osm_root)
        if osm_root is not None
        else {"status": "unconfirmed", "reason": "OSM root was not supplied"}
    )
    provenance = provenance_audit(
        geolife_root, registry_path, verify_sha256=verify_sha256
    )
    build_stats_path = geolife_paths(geolife_root)["build_stats"]
    quality = (
        _read_json(build_stats_path, label="GeoLife build stats")
        if build_stats_path.is_file()
        else {"status": "unconfirmed", "reason": "build stats are missing"}
    )
    failure_ledger = []
    for area, evidence in (
        ("leakage", leakage),
        ("provenance", provenance),
        ("quality", quality),
    ):
        if evidence.get("status") == "unconfirmed":
            failure_ledger.append(
                {"area": area, "status": "open", "reason": evidence["reason"]}
            )
    return {
        "status": "pass" if not failure_ledger else "provisional",
        "schema": schema,
        "split": split,
        "cross_source_leakage": leakage,
        "provenance": provenance,
        "quality_evidence": quality,
        "failure_ledger": failure_ledger,
    }
