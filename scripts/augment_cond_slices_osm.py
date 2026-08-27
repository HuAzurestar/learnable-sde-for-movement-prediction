#!/usr/bin/env python3
"""Add audited OSM distance channels to existing condition-slice Parquet files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.osm import OSM_DISTANCE_COLUMNS, OSMDistanceCollection, augment_condition_frame


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> Path:
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if input_root == output_root:
        raise ValueError("input-root and output-root must differ; in-place mutation is not allowed")
    sources = sorted(input_root.rglob(args.pattern))
    if not sources:
        raise FileNotFoundError(f"no files matching {args.pattern!r} below {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    collection = OSMDistanceCollection(args.manifest)
    totals = {
        "files": 0,
        "points": 0,
        "covered_points": 0,
        "uncovered_points": 0,
        "by_split": {},
        "distance": {
            column: {"finite_points": 0, "sum_m": 0.0, "min_m": None, "max_m": None}
            for column in OSM_DISTANCE_COLUMNS
        },
    }
    outputs = []
    try:
        for source in sources:
            relative = source.relative_to(input_root)
            destination = output_root / relative
            if destination.exists() and not args.overwrite:
                raise FileExistsError(destination)
            frame = pd.read_parquet(source)
            augmented = augment_condition_frame(frame, collection)
            destination.parent.mkdir(parents=True, exist_ok=True)
            augmented.to_parquet(destination, index=False)
            covered = augmented["has_osm"].to_numpy() == 1
            split = relative.parts[0] if len(relative.parts) > 1 else "root"
            totals["files"] += 1
            totals["points"] += len(augmented)
            totals["covered_points"] += int(covered.sum())
            totals["uncovered_points"] += int((~covered).sum())
            totals["by_split"].setdefault(split, {"files": 0, "points": 0, "covered_points": 0})
            totals["by_split"][split]["files"] += 1
            totals["by_split"][split]["points"] += len(augmented)
            totals["by_split"][split]["covered_points"] += int(covered.sum())
            for column in OSM_DISTANCE_COLUMNS:
                values = augmented[column].to_numpy(dtype=np.float64)
                finite = values[np.isfinite(values)]
                stat = totals["distance"][column]
                stat["finite_points"] += len(finite)
                if len(finite):
                    stat["sum_m"] += float(finite.sum())
                    stat["min_m"] = float(finite.min()) if stat["min_m"] is None else min(stat["min_m"], float(finite.min()))
                    stat["max_m"] = float(finite.max()) if stat["max_m"] is None else max(stat["max_m"], float(finite.max()))
            outputs.append(
                {
                    "file": relative.as_posix(),
                    "sha256": sha256_file(destination),
                    "points": len(augmented),
                    "covered_points": int(covered.sum()),
                }
            )
    finally:
        collection.close()
    for stat in totals["distance"].values():
        count = stat["finite_points"]
        stat["mean_m"] = stat.pop("sum_m") / count if count else None
    totals["coverage_ratio"] = totals["covered_points"] / totals["points"] if totals["points"] else 0.0
    manifest = {
        "schema_version": "osm-cond-slices/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "distance_field_manifests": [str(path.resolve()) for path in args.manifest],
        "schema_additions": [*OSM_DISTANCE_COLUMNS, "has_osm"],
        "missing_semantics": "has_osm=0 and distance columns NaN; capped distances remain has_osm=1",
        "totals": totals,
        "outputs": outputs,
    }
    manifest_path = output_root / "osm_augmentation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--pattern", default="*_cond.parquet")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        manifest = run(parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
