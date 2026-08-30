#!/usr/bin/env python3
"""Byte-compare two OSM distance-field builds made from alternate extracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np


LAYERS = ("road_dist", "water_dist", "building_dist", "coverage")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare(left_manifest: Path, right_manifest: Path) -> dict:
    import rasterio

    left_doc = json.loads(left_manifest.read_text(encoding="utf-8"))
    right_doc = json.loads(right_manifest.read_text(encoding="utf-8"))
    result = {
        "schema_version": "osm-distance-field-comparison/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "left_manifest": f"{left_manifest.parent.name}/{left_manifest.name}",
        "right_manifest": f"{right_manifest.parent.name}/{right_manifest.name}",
        "layers": {},
    }
    all_equal = True
    for layer in LAYERS:
        left_path = left_manifest.parent / left_doc["outputs"][layer]["file"]
        right_path = right_manifest.parent / right_doc["outputs"][layer]["file"]
        left_file_sha = sha256_file(left_path)
        right_file_sha = sha256_file(right_path)
        with rasterio.open(left_path) as left, rasterio.open(right_path) as right:
            grid_equal = (
                left.crs == right.crs
                and left.transform == right.transform
                and left.shape == right.shape
                and left.dtypes == right.dtypes
                and left.nodata == right.nodata
            )
            left_array = left.read(1)
            right_array = right.read(1)
        array_equal = grid_equal and np.array_equal(left_array, right_array, equal_nan=True)
        left_array_sha = hashlib.sha256(left_array.tobytes(order="C")).hexdigest()
        right_array_sha = hashlib.sha256(right_array.tobytes(order="C")).hexdigest()
        file_equal = left_file_sha == right_file_sha
        result["layers"][layer] = {
            "grid_equal": grid_equal,
            "array_byte_equal": array_equal and left_array_sha == right_array_sha,
            "file_byte_equal": file_equal,
            "left_file_sha256": left_file_sha,
            "right_file_sha256": right_file_sha,
            "left_array_sha256": left_array_sha,
            "right_array_sha256": right_array_sha,
            "different_pixels": int(np.count_nonzero(left_array != right_array))
            if left_array.shape == right_array.shape
            else None,
        }
        all_equal = all_equal and grid_equal and array_equal and file_equal
    result["all_four_files_byte_equal"] = all_equal
    return result


def write_json_atomic(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as destination:
            json.dump(document, destination, ensure_ascii=False, indent=2)
            destination.write("\n")
        os.replace(temp_name, path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left_manifest", type=Path)
    parser.add_argument("right_manifest", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        result = compare(args.left_manifest.resolve(), args.right_manifest.resolve())
        if args.output:
            write_json_atomic(args.output.resolve(), result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["all_four_files_byte_equal"] else 2
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
