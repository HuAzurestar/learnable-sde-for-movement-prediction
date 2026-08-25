"""GeoLife ingestion contract and independently reproducible audit helpers.

The cleaned GeoLife table is a local research asset and is never committed to
this repository. The audit distinguishes three claims that must not be mixed:

* exact schema/statistics/hash consistency (a technical ingestion gate),
* file-ID namespace disjointness (not a person- or content-level leakage proof),
* source registration versus independent legal/release approval.
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
CANONICAL_SCHEMA = (
    ("track_id", "int64"),
    ("file_id", "string"),
    ("cluster_A", "int64"),
    ("country", "string"),
    ("region", "string"),
    ("city", "string"),
    ("segment_id", "string"),
    ("t", "double"),
    ("x", "double"),
    ("y", "double"),
    ("z", "double"),
    ("vx", "double"),
    ("vy", "double"),
    ("speed", "double"),
)
CLEANED_LEG_GAP_DEFINITION_S = 60.0
WALK_FILTERED_GAP_DEFINITION_S = 300.0
_SUCCESSFUL_BUILD_STATUSES = {"pass"}


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
        "long_duration": root / "geolife_longduration_tiers.json",
    }


def _schema_signature(path: Path) -> list[tuple[str, str]]:
    schema = pq.ParquetFile(path).schema_arrow
    return [(field.name, str(field.type)) for field in schema]


def validate_geolife_schema(root: str | Path) -> dict[str, Any]:
    """Validate the complete 14-column contract, including every dtype."""
    table_path = geolife_paths(root)["table"]
    if not table_path.is_file():
        raise DataValidationError(f"GeoLife cleaned table missing: {table_path}")
    parquet = pq.ParquetFile(table_path)
    signature = _schema_signature(table_path)
    if signature != list(CANONICAL_SCHEMA):
        raise DataValidationError(
            "GeoLife schema does not match the canonical 14-column contract: "
            f"actual={signature}, expected={list(CANONICAL_SCHEMA)}"
        )
    return {
        "status": "pass",
        "file": table_path.name,
        "rows": parquet.metadata.num_rows,
        "columns": [name for name, _ in signature],
        "dtypes": {name: dtype for name, dtype in signature},
        "canonical_schema_match": True,
    }


def compare_geolife_osm_schema(
    geolife_root: str | Path, osm_root: str | Path | None
) -> dict[str, Any]:
    """Independently compare full names and dtypes against the OSM table."""
    if osm_root is None:
        return {
            "status": "unconfirmed",
            "reason": "OSM root was not supplied; cross-source schema equality was not checked",
        }
    osm_path = Path(osm_root) / "unified_full_leg.parquet"
    if not osm_path.is_file():
        return {
            "status": "failed",
            "reason": f"OSM unified table missing: {osm_path}",
        }
    gl_signature = _schema_signature(geolife_paths(geolife_root)["table"])
    osm_signature = _schema_signature(osm_path)
    matches = gl_signature == osm_signature
    return {
        "status": "pass" if matches else "failed",
        "identical_names_and_dtypes": matches,
        "geolife": [{"name": name, "dtype": dtype} for name, dtype in gl_signature],
        "osm": [{"name": name, "dtype": dtype} for name, dtype in osm_signature],
        **({} if matches else {"reason": "GeoLife and OSM schemas differ"}),
    }


def geolife_file_id_splits(root: str | Path) -> dict[str, set[str]]:
    """Rebuild the registered file-level split and verify its file counts.

    The manifest stores aggregate counts rather than file-ID lists. The adapter
    replays the registered sorted/shuffle/ratio algorithm. This proves file-ID
    partition disjointness only; it is not a user split.
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
        raise DataValidationError("GeoLife file-ID split is not exhaustive")
    pairs = (("train", "val"), ("train", "eval"), ("val", "eval"))
    if any(result[left] & result[right] for left, right in pairs):
        raise DataValidationError("GeoLife file-ID split is not disjoint")

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


def _user_from_file_id(file_id: str) -> str:
    parts = file_id.split("_", 2)
    if len(parts) < 2 or parts[0] != "GL" or not parts[1]:
        raise DataValidationError(f"cannot derive GeoLife user from file_id: {file_id!r}")
    return parts[1]


def geolife_split_audit(root: str | Path) -> dict[str, Any]:
    """Recompute point/segment/file totals and disclose user overlap."""
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

    users = {
        part: {_user_from_file_id(file_id) for file_id in splits[part]}
        for part in PARTITIONS
    }
    user_overlap = {
        "train_val": len(users["train"] & users["val"]),
        "train_eval": len(users["train"] & users["eval"]),
        "val_eval": len(users["val"] & users["eval"]),
        "all_three": len(users["train"] & users["val"] & users["eval"]),
    }
    return {
        "status": "pass",
        "algorithm": "sorted(file_id) -> RandomState(seed).shuffle -> registered ratio",
        "partition_unit": "file_id",
        "partitions": actual,
        "file_id_partitions_disjoint": True,
        "file_id_partitions_exhaustive": True,
        "user_partitioning": {
            "status": "boundary",
            "users": {part: len(users[part]) for part in PARTITIONS},
            "overlap": user_overlap,
            "user_independent": False,
            "claim": "user-independent generalization is not supported by this split",
        },
    }


def assert_cross_source_file_id_namespace_disjoint(
    geolife_root: str | Path, osm_root: str | Path
) -> dict[str, Any]:
    """Fail fast on file-ID collision, while stating the limited scope."""
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
            "GeoLife/OSM file-ID namespace overlap detected: "
            f"{sorted(overlap)[:5]}"
        )
    return {
        "status": "pass",
        "scope": "file_id_namespace_only",
        "geolife_file_ids": len(gl_ids),
        "osm_file_ids": len(osm_ids),
        "intersection": 0,
        "content_duplicate_check": {
            "status": "not_performed",
            "reason": "file-ID disjointness does not detect duplicated trajectories or people across sources",
        },
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
    """Verify registration while keeping legal/release approval separate."""
    if registry_path is None or not Path(registry_path).is_file():
        return {
            "status": "unconfirmed",
            "reason": "source/licence registry was not supplied or does not exist",
            "legal_review_status": "not_independently_verified",
        }
    registry = _read_json(Path(registry_path), label="GeoLife source registry")
    required = ("registry_id", "dataset", "provider", "url", "version", "license")
    missing = [key for key in required if not registry.get(key)]
    caveats = registry.get("caveats")
    if not isinstance(caveats, list) or not caveats:
        missing.append("caveats")
    if missing:
        return {
            "status": "unconfirmed",
            "reason": f"registry missing fields: {sorted(set(missing))}",
            "legal_review_status": "not_independently_verified",
        }
    product = registry.get("cleaned_product")
    if not isinstance(product, Mapping) or not product.get("sha256"):
        return {
            "status": "unconfirmed",
            "reason": "registry has no cleaned-product sha256",
            "legal_review_status": "not_independently_verified",
        }

    legal_review = registry.get("independent_legal_review")
    legal_verified = (
        isinstance(legal_review, Mapping)
        and legal_review.get("status") == "approved"
        and bool(legal_review.get("reviewer"))
        and bool(legal_review.get("reviewed_at"))
    )
    result: dict[str, Any] = {
        "status": (
            "registered_and_independently_legal_verified"
            if legal_verified
            else "registered_not_independently_legal_verified"
        ),
        "registry_id": registry["registry_id"],
        "dataset": registry["dataset"],
        "provider": registry["provider"],
        "url": registry["url"],
        "version": registry["version"],
        "license_record": registry["license"],
        "legal_review_status": "approved" if legal_verified else "not_independently_verified",
        "caveats": caveats,
    }
    if verify_sha256:
        actual_sha256 = _sha256(geolife_paths(root)["table"])
        expected_sha256 = str(product["sha256"]).lower()
        if actual_sha256 != expected_sha256:
            return {
                **result,
                "status": "failed",
                "reason": (
                    "GeoLife cleaned table sha256 disagrees with registry: "
                    f"actual={actual_sha256}, registered={expected_sha256}"
                ),
                "sha256": actual_sha256,
                "sha256_match": False,
            }
        result["sha256"] = actual_sha256
        result["sha256_match"] = True
    else:
        result["sha256_match"] = "not_checked"
    return result


def build_quality_audit(
    root: str | Path,
    *,
    schema: Mapping[str, Any],
    split: Mapping[str, Any],
    verify_sha256: bool = True,
) -> dict[str, Any]:
    """Cross-check every hard build-stat claim against the actual table."""
    paths = geolife_paths(root)
    stats_path = paths["build_stats"]
    if not stats_path.is_file():
        return {"status": "unconfirmed", "reason": "build stats are missing"}
    stats = _read_json(stats_path, label="GeoLife build stats")
    required = {
        "status",
        "n_failed",
        "schema_identical",
        "columns",
        "points_cleaned",
        "segments",
        "files_with_segments",
        "bytes",
        "sha256",
    }
    errors = [f"missing required field: {key}" for key in sorted(required - stats.keys())]

    build_status = stats.get("status")
    if "status" in stats and (
        not isinstance(build_status, str)
        or build_status.strip().lower() not in _SUCCESSFUL_BUILD_STATUSES
    ):
        errors.append(
            "status must be one of the successful build states "
            f"{sorted(_SUCCESSFUL_BUILD_STATUSES)}, got {build_status!r}"
        )
    if type(stats.get("n_failed")) is not int or stats.get("n_failed") != 0:
        errors.append(f"n_failed must be integer 0, got {stats.get('n_failed')!r}")
    if stats.get("schema_identical") is not True:
        errors.append(
            f"schema_identical must be true, got {stats.get('schema_identical')!r}"
        )

    actual_columns = list(schema["columns"])
    if stats.get("columns") != actual_columns:
        errors.append(
            f"columns disagree: actual={actual_columns}, registered={stats.get('columns')!r}"
        )
    partitions = split["partitions"]
    actual = {
        "points_cleaned": int(schema["rows"]),
        "segments": sum(int(partitions[part]["segments"]) for part in PARTITIONS),
        "files_with_segments": sum(int(partitions[part]["files"]) for part in PARTITIONS),
        "bytes": paths["table"].stat().st_size,
    }
    for key, value in actual.items():
        if stats.get(key) != value:
            errors.append(
                f"{key} disagree: actual={value}, registered={stats.get(key)!r}"
            )

    actual_sha256: str | None = None
    unconfirmed = []
    if verify_sha256:
        actual_sha256 = _sha256(paths["table"])
        if str(stats.get("sha256", "")).lower() != actual_sha256:
            errors.append(
                "build-stats sha256 mismatch: "
                f"actual={actual_sha256}, registered={stats.get('sha256')!r}"
            )
    else:
        unconfirmed.append("sha256 verification was explicitly skipped")

    status = "failed" if errors else ("unconfirmed" if unconfirmed else "pass")
    return {
        "status": status,
        "independently_measured": {
            **actual,
            "columns": actual_columns,
            "dtypes": schema["dtypes"],
            "sha256": actual_sha256 if actual_sha256 is not None else "not_checked",
        },
        "registered": stats,
        "errors": errors,
        "unconfirmed": unconfirmed,
        **({"reason": "; ".join(errors or unconfirmed)} if status != "pass" else {}),
    }


def long_duration_boundary_audit(root: str | Path) -> dict[str, Any]:
    """Report, and keep separate, the 60-second and 300-second gap tiers."""
    paths = geolife_paths(root)
    frame = pd.read_parquet(paths["table"], columns=["file_id", "segment_id", "t"])
    bounds = frame.groupby(["file_id", "segment_id"], sort=False)["t"].agg(["min", "max"])
    durations_h = (bounds["max"] - bounds["min"]) / 3600.0
    cleaned = {
        "gap_definition_s": CLEANED_LEG_GAP_DEFINITION_S,
        "segments_ge6h": int((durations_h >= 6.0).sum()),
        "max_duration_h": float(durations_h.max()) if len(durations_h) else 0.0,
        "source": "cleaned geolife_leg file-segment pairs",
    }
    tiers_path = paths["long_duration"]
    if not tiers_path.is_file():
        return {
            "status": "unconfirmed",
            "cleaned_leg_gap60s": cleaned,
            "reason": "walk-filtered long-duration tier registry is missing",
        }
    tiers = _read_json(tiers_path, label="GeoLife long-duration tiers")
    errors: list[str] = []
    try:
        gap_definition_s = float(tiers["gap_definition_s"])
        max_duration_h = float(tiers["max_duration_h"])
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"invalid gap/max-duration fields: {exc}")
        gap_definition_s = float("nan")
        max_duration_h = float("nan")
    if gap_definition_s != WALK_FILTERED_GAP_DEFINITION_S:
        errors.append(
            "walk-filtered tier gap must be explicitly registered as 300 seconds, "
            f"got {gap_definition_s!r}"
        )

    buckets: dict[str, dict[str, int]] = {}
    for key in ("ge6h", "ge8h", "ge10h", "ge12h"):
        value = tiers.get(key)
        if not isinstance(value, Mapping):
            errors.append(f"missing or invalid long-duration bucket: {key}")
            continue
        try:
            buckets[key] = {
                "segments": int(value["segments"]),
                "users": int(value["users"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid {key} bucket: {exc}")
    if "ge12h" in buckets and np.isfinite(max_duration_h):
        expected_zero = max_duration_h < 12.0
        if expected_zero != (buckets["ge12h"]["segments"] == 0):
            errors.append(
                ">=12h bucket is inconsistent with registered maximum duration: "
                f"ge12h={buckets['ge12h']}, max_duration_h={max_duration_h}"
            )

    return {
        "status": "failed" if errors else "pass",
        "cleaned_leg_gap60s": cleaned,
        "walk_filtered_gap300s": {
            "gap_definition_s": gap_definition_s,
            "buckets": buckets,
            "max_duration_h": max_duration_h,
            "note": tiers.get("note"),
        },
        "definitions_are_not_interchangeable": True,
        "errors": errors,
        **({"reason": "; ".join(errors)} if errors else {}),
    }


def _failed_namespace_evidence(exc: DataValidationError) -> dict[str, Any]:
    return {"status": "failed", "scope": "file_id_namespace_only", "reason": str(exc)}


def audit_geolife_ingestion(
    geolife_root: str | Path,
    *,
    osm_root: str | Path | None = None,
    registry_path: str | Path | None = None,
    verify_sha256: bool = True,
) -> dict[str, Any]:
    """Return a JSON-serialisable audit with technical and release gates."""
    schema = validate_geolife_schema(geolife_root)
    split = geolife_split_audit(geolife_root)
    schema_alignment = compare_geolife_osm_schema(geolife_root, osm_root)
    if osm_root is None:
        namespace = {
            "status": "unconfirmed",
            "scope": "file_id_namespace_only",
            "reason": "OSM root was not supplied",
            "content_duplicate_check": {
                "status": "not_performed",
                "reason": "cross-source content duplication was not checked",
            },
        }
    else:
        try:
            namespace = assert_cross_source_file_id_namespace_disjoint(
                geolife_root, osm_root
            )
        except DataValidationError as exc:
            namespace = _failed_namespace_evidence(exc)
    quality = build_quality_audit(
        geolife_root,
        schema=schema,
        split=split,
        verify_sha256=verify_sha256,
    )
    long_duration = long_duration_boundary_audit(geolife_root)
    provenance = provenance_audit(
        geolife_root, registry_path, verify_sha256=verify_sha256
    )

    technical_evidence = {
        "cross_source_schema": schema_alignment,
        "file_id_namespace": namespace,
        "quality": quality,
        "long_duration_registry": long_duration,
    }
    technical_statuses = {item["status"] for item in technical_evidence.values()}
    if "failed" in technical_statuses:
        technical_status = "failed"
    elif "unconfirmed" in technical_statuses:
        technical_status = "provisional"
    else:
        technical_status = "pass"

    if provenance.get("status") in {"failed", "unconfirmed"}:
        release_status = "blocked_missing_or_inconsistent_registration"
    elif provenance.get("legal_review_status") == "approved":
        release_status = "approved"
    else:
        release_status = "blocked_pending_independent_legal_review"

    failure_ledger: list[dict[str, Any]] = []
    for area, evidence in technical_evidence.items():
        if evidence["status"] in {"failed", "unconfirmed"}:
            failure_ledger.append(
                {
                    "area": area,
                    "status": "failed" if evidence["status"] == "failed" else "open",
                    "reason": evidence.get("reason", evidence["status"]),
                }
            )
    if provenance.get("status") in {"failed", "unconfirmed"}:
        failure_ledger.append(
            {
                "area": "provenance_registration",
                "status": "failed" if provenance["status"] == "failed" else "open",
                "reason": provenance.get("reason", provenance["status"]),
            }
        )
    if release_status != "approved":
        failure_ledger.append(
            {
                "area": "legal_release",
                "status": "blocked",
                "reason": release_status,
            }
        )

    content_boundary = namespace.get(
        "content_duplicate_check",
        {
            "status": "not_performed",
            "reason": "file-ID namespace failure prevented content-boundary reporting",
        },
    )
    boundary_ledger = [
        {"area": "user_partitioning", **split["user_partitioning"]},
        {"area": "cross_source_content_duplication", **content_boundary},
        {
            "area": "long_duration_definitions",
            "status": long_duration["status"],
            "cleaned_leg_gap60s": long_duration.get("cleaned_leg_gap60s"),
            "walk_filtered_gap300s": long_duration.get("walk_filtered_gap300s"),
        },
        {
            "area": "dataset_and_licence_caveats",
            "status": "registered" if provenance.get("caveats") else "unconfirmed",
            "caveats": provenance.get("caveats", []),
            "legal_review_status": provenance.get("legal_review_status"),
        },
    ]

    hard_failed = technical_status == "failed" or provenance.get("status") == "failed"
    if hard_failed:
        overall_status = "failed"
    elif technical_status != "pass" or release_status != "approved":
        overall_status = "provisional"
    else:
        overall_status = "pass"
    return {
        "status": overall_status,
        "technical_status": technical_status,
        "release_status": release_status,
        "schema": schema,
        "cross_source_schema": schema_alignment,
        "split": split,
        "cross_source_file_id_namespace": namespace,
        "provenance": provenance,
        "quality_evidence": quality,
        "long_duration_boundaries": long_duration,
        "failure_ledger": failure_ledger,
        "boundary_ledger": boundary_ledger,
    }
