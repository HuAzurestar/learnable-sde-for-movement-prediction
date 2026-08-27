#!/usr/bin/env python3
"""Build audited road/water/building distance rasters from a real OSM PBF.

The raster grid is projected, so Euclidean distance-transform values are metres.
The separate coverage raster is mandatory and prevents uncovered coordinates
from being interpreted as large observed distances.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


CHANNELS = ("road_dist", "water_dist", "building_dist")
ROAD_EXCLUDED = {
    "abandoned",
    "construction",
    "corridor",
    "elevator",
    "platform",
    "planned",
    "proposed",
    "raceway",
    "razed",
}
WATERWAY_INCLUDED = {"canal", "ditch", "drain", "river", "stream", "tidal_channel"}
INACTIVE_VALUES = {"abandoned", "construction", "demolished", "no", "planned", "proposed", "razed"}


def classify_way(tags: Mapping[str, str], is_closed: bool) -> tuple[str, ...]:
    """Map OSM way tags to distance channels using declared, testable rules."""

    classes: list[str] = []
    highway = tags.get("highway")
    if highway and highway not in ROAD_EXCLUDED:
        classes.append("road_dist")
    waterway = tags.get("waterway")
    is_water_area = is_closed and (
        tags.get("natural") == "water" or bool(tags.get("water")) or tags.get("landuse") == "reservoir"
    )
    if waterway in WATERWAY_INCLUDED or is_water_area:
        classes.append("water_dist")
    building = tags.get("building")
    if building and building not in INACTIVE_VALUES:
        classes.append("building_dist")
    return tuple(classes)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def align_bounds(bounds: Sequence[float], resolution: float, outward: bool = True) -> tuple[float, ...]:
    xmin, ymin, xmax, ymax = bounds
    low = math.floor if outward else math.ceil
    high = math.ceil if outward else math.floor
    return (
        low(xmin / resolution) * resolution,
        low(ymin / resolution) * resolution,
        high(xmax / resolution) * resolution,
        high(ymax / resolution) * resolution,
    )


def _geojson_geometries(document: dict) -> list[dict]:
    kind = document.get("type")
    if kind == "FeatureCollection":
        return [feature["geometry"] for feature in document.get("features", []) if feature.get("geometry")]
    if kind == "Feature":
        return [document["geometry"]]
    if kind in {"Polygon", "MultiPolygon"}:
        return [document]
    raise ValueError(f"unsupported coverage GeoJSON type: {kind!r}")


@dataclass
class BuildResult:
    manifest_path: Path
    report_path: Path


class FeatureCollector:
    """Pyosmium handler that retains only tagged geometry near one output grid."""

    def __init__(self, projected_bounds: Sequence[float], target_crs: str):
        import osmium
        from pyproj import Transformer
        from shapely.geometry import LineString, Polygon, box

        collector = self

        class _Handler(osmium.SimpleHandler):
            def way(self, way) -> None:
                collector.way(way)

        self._handler = _Handler()
        self._transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        self._box = box(*projected_bounds)
        self._LineString = LineString
        self._Polygon = Polygon
        self.geometries: dict[str, list] = {channel: [] for channel in CHANNELS}
        self.counts = Counter()

    def way(self, way) -> None:
        self.counts["ways_seen"] += 1
        tags = {tag.k: tag.v for tag in way.tags}
        is_closed = len(way.nodes) >= 4 and way.nodes[0].ref == way.nodes[-1].ref
        classes = classify_way(tags, is_closed)
        if not classes:
            return
        self.counts["candidate_ways"] += 1
        try:
            lon_lat = [(node.lon, node.lat) for node in way.nodes if node.location.valid()]
        except Exception:
            self.counts["invalid_location_ways"] += 1
            return
        if len(lon_lat) < 2:
            self.counts["invalid_location_ways"] += 1
            return
        x, y = self._transformer.transform(
            [coord[0] for coord in lon_lat], [coord[1] for coord in lon_lat]
        )
        coordinates = list(zip(x, y))
        try:
            geometry = (
                self._Polygon(coordinates)
                if is_closed and ("building_dist" in classes or "water_dist" in classes)
                else self._LineString(coordinates)
            )
            if not geometry.is_valid:
                geometry = geometry.buffer(0)
            if geometry.is_empty or not geometry.intersects(self._box):
                return
            clipped = geometry.intersection(self._box)
            if clipped.is_empty:
                return
        except Exception:
            self.counts["invalid_geometry_ways"] += 1
            return
        for channel in classes:
            self.geometries[channel].append(clipped)
            self.counts[f"{channel}_selected"] += 1
        self.counts["selected_vertices"] += len(coordinates)

    def apply(self, input_path: Path, index_spec: str) -> None:
        self._handler.apply_file(str(input_path), locations=True, idx=index_spec)


def _project_geometries(geometries: Iterable[dict], target_crs: str) -> list:
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform

    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    return [transform(transformer.transform, shape(geometry)) for geometry in geometries]


def _header_metadata(input_path: Path) -> dict:
    import osmium

    reader = osmium.io.Reader(str(input_path))
    try:
        header = reader.header()
        return {
            "replication_timestamp": header.get("osmosis_replication_timestamp") or None,
            "generator": header.get("generator") or None,
            "header_bounds": str(header.box()),
        }
    finally:
        reader.close()


def _write_raster(path: Path, array: np.ndarray, transform, crs: str, nodata, dtype: str) -> None:
    import rasterio

    height, width = array.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "DEFLATE",
        "predictor": 3 if dtype == "float32" else 2,
        "tiled": width >= 256 and height >= 256,
        "BIGTIFF": "IF_SAFER",
    }
    if profile["tiled"]:
        profile.update(blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(array.astype(dtype, copy=False), 1)
        destination.update_tags(
            source="OpenStreetMap contributors",
            license="ODbL-1.0",
            distance_unit="metre" if dtype == "float32" else "coverage_flag",
        )


def _quality_stats(distance: np.ndarray, feature_core: np.ndarray, coverage: np.ndarray, cap: float) -> dict:
    valid = coverage == 1
    values = distance[valid]
    if not len(values):
        raise ValueError("coverage mask contains no valid pixels")
    within_cap = values < cap - 1e-5
    return {
        "coverage_pixels": int(valid.sum()),
        "valid_observation_ratio": float(np.isfinite(values).mean()),
        "feature_pixels": int((feature_core[valid] == 1).sum()),
        "feature_pixel_ratio": float((feature_core[valid] == 1).mean()),
        "non_empty_ratio": float(within_cap.mean()),
        "at_cap_ratio": float((~within_cap).mean()),
        "distance_m": {
            "min": float(np.min(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        },
    }


def build(args: argparse.Namespace) -> BuildResult:
    try:
        import osmium
        import pyproj
        import rasterio
        import scipy
        import shapely
        from rasterio.features import rasterize
        from rasterio.transform import from_origin
        from rasterio.warp import transform_bounds
        from scipy.ndimage import distance_transform_edt
        from shapely.ops import unary_union
    except ImportError as exc:
        raise RuntimeError("install project optional dependencies with: pip install -e .[osm]") from exc

    started = time.time()
    input_path = args.pbf.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not (0 < args.resolution_m <= args.max_distance_m):
        raise ValueError("require 0 < resolution-m <= max-distance-m")
    if not args.coverage_geojson and not args.assume_bounds_covered:
        raise ValueError(
            "coverage is ambiguous: pass --coverage-geojson, or explicitly pass "
            "--assume-bounds-covered with --coverage-note for a verified interior window"
        )
    if args.assume_bounds_covered and not args.coverage_note:
        raise ValueError("--assume-bounds-covered requires a non-empty --coverage-note")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    managed_files = [output_dir / f"{channel}.tif" for channel in CHANNELS]
    managed_files += [output_dir / "osm_coverage.tif", output_dir / "manifest.json", output_dir / "quality_report.md"]
    existing = [path for path in managed_files if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing outputs: {existing}")

    actual_sha256 = sha256_file(input_path)
    if args.expected_sha256 and actual_sha256.lower() != args.expected_sha256.lower():
        raise ValueError(
            f"input checksum mismatch: expected {args.expected_sha256.lower()}, got {actual_sha256.lower()}"
        )
    header = _header_metadata(input_path)
    parent_input = None
    if args.parent_pbf:
        parent_path = args.parent_pbf.resolve()
        if not parent_path.is_file():
            raise FileNotFoundError(parent_path)
        parent_sha256 = sha256_file(parent_path)
        if (
            args.parent_expected_sha256
            and parent_sha256.lower() != args.parent_expected_sha256.lower()
        ):
            raise ValueError(
                "parent input checksum mismatch: expected "
                f"{args.parent_expected_sha256.lower()}, got {parent_sha256.lower()}"
            )
        parent_input = {
            "file": str(parent_path),
            "size_bytes": parent_path.stat().st_size,
            "sha256": parent_sha256,
            "source_url": args.parent_source_url,
            "pbf_header": _header_metadata(parent_path),
            "derivation_note": args.derivation_note,
        }

    requested_bounds = tuple(float(value) for value in args.bounds)
    min_lon, min_lat, max_lon, max_lat = requested_bounds
    if not (-180 <= min_lon < max_lon <= 180 and -90 <= min_lat < max_lat <= 90):
        raise ValueError(f"invalid EPSG:4326 bounds: {requested_bounds}")
    raw_core_bounds = transform_bounds(
        "EPSG:4326", args.crs, *requested_bounds, densify_pts=21
    )
    core_bounds = align_bounds(raw_core_bounds, args.resolution_m)
    xmin, ymin, xmax, ymax = core_bounds
    core_width = int(round((xmax - xmin) / args.resolution_m))
    core_height = int(round((ymax - ymin) / args.resolution_m))
    core_transform = from_origin(xmin, ymax, args.resolution_m, args.resolution_m)

    expansion = math.ceil(args.max_distance_m / args.resolution_m) * args.resolution_m
    work_bounds = (xmin - expansion, ymin - expansion, xmax + expansion, ymax + expansion)
    work_width = core_width + 2 * int(round(expansion / args.resolution_m))
    work_height = core_height + 2 * int(round(expansion / args.resolution_m))
    work_transform = from_origin(
        work_bounds[0], work_bounds[3], args.resolution_m, args.resolution_m
    )
    offset = int(round(expansion / args.resolution_m))

    if args.coverage_geojson:
        coverage_document = json.loads(args.coverage_geojson.read_text(encoding="utf-8"))
        coverage_wgs84 = _geojson_geometries(coverage_document)
        coverage_mode = "geojson"
    else:
        coverage_wgs84 = [
            {
                "type": "Polygon",
                "coordinates": [[
                    [min_lon, min_lat],
                    [max_lon, min_lat],
                    [max_lon, max_lat],
                    [min_lon, max_lat],
                    [min_lon, min_lat],
                ]],
            }
        ]
        coverage_mode = "verified_interior_bounds"
    projected_coverage = _project_geometries(coverage_wgs84, args.crs)
    coverage_geometry = unary_union(projected_coverage)
    coverage = rasterize(
        projected_coverage,
        out_shape=(core_height, core_width),
        transform=core_transform,
        fill=0,
        default_value=1,
        dtype="uint8",
    )
    if not np.any(coverage == 1):
        raise ValueError("coverage geometry does not intersect the output grid")

    collector = FeatureCollector(work_bounds, args.crs)
    index_note = args.location_index
    if args.location_index.endswith("_file_array"):
        with tempfile.TemporaryDirectory(prefix="osm-node-index-", dir=output_dir) as temp_dir:
            index_path = Path(temp_dir) / "locations.idx"
            collector.apply(input_path, f"{args.location_index},{index_path}")
    else:
        collector.apply(input_path, args.location_index)

    output_metadata: dict[str, dict] = {}
    channel_stats: dict[str, dict] = {}
    smoke_checks: dict[str, dict | None] = {}
    inverse = pyproj.Transformer.from_crs(args.crs, "EPSG:4326", always_xy=True)
    for channel in CHANNELS:
        geometries = collector.geometries[channel]
        if geometries:
            feature_work = rasterize(
                geometries,
                out_shape=(work_height, work_width),
                transform=work_transform,
                fill=0,
                default_value=1,
                all_touched=True,
                dtype="uint8",
            )
            distance_work = distance_transform_edt(
                feature_work == 0,
                sampling=(args.resolution_m, args.resolution_m),
            )
            np.minimum(distance_work, args.max_distance_m, out=distance_work)
        else:
            feature_work = np.zeros((work_height, work_width), dtype=np.uint8)
            distance_work = np.full(
                (work_height, work_width), args.max_distance_m, dtype=np.float64
            )
        rows = slice(offset, offset + core_height)
        cols = slice(offset, offset + core_width)
        distance = distance_work[rows, cols].astype(np.float32)
        feature_core = feature_work[rows, cols]
        distance[coverage == 0] = -9999.0
        path = output_dir / f"{channel}.tif"
        _write_raster(path, distance, core_transform, args.crs, -9999.0, "float32")
        channel_stats[channel] = _quality_stats(distance, feature_core, coverage, args.max_distance_m)

        check = None
        for geometry in geometries:
            candidate = geometry.representative_point()
            if (
                xmin <= candidate.x <= xmax
                and ymin <= candidate.y <= ymax
                and coverage_geometry.covers(candidate)
            ):
                row = int((ymax - candidate.y) // args.resolution_m)
                col = int((candidate.x - xmin) // args.resolution_m)
                if 0 <= row < core_height and 0 <= col < core_width:
                    lon, lat = inverse.transform(candidate.x, candidate.y)
                    check = {
                        "lon": round(float(lon), 7),
                        "lat": round(float(lat), 7),
                        "row": row,
                        "col": col,
                        "distance_m": round(float(distance[row, col]), 3),
                        "expected_upper_bound_m": round(args.resolution_m * math.sqrt(2), 3),
                        "passed": bool(distance[row, col] <= args.resolution_m * math.sqrt(2)),
                    }
                    break
        smoke_checks[channel] = check
        output_metadata[channel] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "dtype": "float32",
            "nodata": -9999.0,
            "unit": "metre",
        }

    coverage_path = output_dir / "osm_coverage.tif"
    _write_raster(coverage_path, coverage, core_transform, args.crs, 0, "uint8")
    output_metadata["coverage"] = {
        "file": coverage_path.name,
        "sha256": sha256_file(coverage_path),
        "size_bytes": coverage_path.stat().st_size,
        "dtype": "uint8",
        "nodata": 0,
        "meaning": {"1": "OSM extract coverage asserted", "0": "not covered; distances missing"},
    }

    manifest = {
        "schema_version": "osm-distance-field/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "region": args.region,
        "input": {
            "file": str(input_path),
            "size_bytes": input_path.stat().st_size,
            "sha256": actual_sha256,
            "source_url": args.source_url,
            "source_version": args.source_version or header["replication_timestamp"],
            "license": args.license,
            "attribution": "© OpenStreetMap contributors",
            "pbf_header": header,
            "parent_input": parent_input,
        },
        "grid": {
            "crs": args.crs,
            "requested_bounds_epsg4326": list(requested_bounds),
            "bounds_projected": list(core_bounds),
            "resolution_m": args.resolution_m,
            "width": core_width,
            "height": core_height,
            "max_distance_m": args.max_distance_m,
            "distance_algorithm": "scipy.ndimage.distance_transform_edt on all_touched raster",
        },
        "coverage": {
            "mode": coverage_mode,
            "source": str(args.coverage_geojson) if args.coverage_geojson else None,
            "note": args.coverage_note,
            "covered_pixels": int((coverage == 1).sum()),
            "grid_pixels": int(coverage.size),
            "covered_ratio": float((coverage == 1).mean()),
        },
        "feature_rules": {
            "road_dist": {
                "include": "ways with highway=*",
                "exclude_values": sorted(ROAD_EXCLUDED),
            },
            "water_dist": {
                "include_waterway_values": sorted(WATERWAY_INCLUDED),
                "include_closed_areas": "natural=water, water=*, or landuse=reservoir",
            },
            "building_dist": {
                "include": "ways with building=*",
                "exclude_values": sorted(INACTIVE_VALUES),
            },
            "known_limitations": [
                "relation-only multipolygons are not assembled in v1",
                "distance values are capped; cap is an observed lower bound, not missing coverage",
            ],
        },
        "feature_counts": dict(sorted(collector.counts.items())),
        "statistics": channel_stats,
        "smoke_checks": smoke_checks,
        "outputs": output_metadata,
        "toolchain": {
            "python": platform.python_version(),
            "osmium": importlib.metadata.version("osmium"),
            "pyproj": pyproj.__version__,
            "rasterio": rasterio.__version__,
            "scipy": scipy.__version__,
            "shapely": shapely.__version__,
            "location_index": index_note,
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        f"## {args.region} OSM distance-field quality report",
        "",
        f"Input SHA-256: `{actual_sha256}`",
        f"PBF replication timestamp: `{header['replication_timestamp']}`",
        f"Grid: `{args.crs}`, {args.resolution_m:g} m, {core_width} × {core_height}; cap {args.max_distance_m:g} m.",
        f"Coverage: {int((coverage == 1).sum()):,}/{coverage.size:,} pixels ({(coverage == 1).mean():.2%}); mode `{coverage_mode}`.",
        "",
        "| channel | selected ways | feature pixels | non-empty (< cap) | at cap | p50 m | p95 m |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for channel in CHANNELS:
        stats = channel_stats[channel]
        report_lines.append(
            f"| {channel} | {collector.counts[f'{channel}_selected']:,} | "
            f"{stats['feature_pixels']:,} | {stats['non_empty_ratio']:.2%} | "
            f"{stats['at_cap_ratio']:.2%} | {stats['distance_m']['p50']:.1f} | "
            f"{stats['distance_m']['p95']:.1f} |"
        )
    report_lines += [
        "",
        "Missing coverage is encoded by `osm_coverage=0` and distance nodata `-9999`; it is never replaced by the distance cap.",
        "The v1 extractor handles tagged ways and closed-way areas; relation-only multipolygons remain a declared limitation.",
    ]
    report_path = output_dir / "quality_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return BuildResult(manifest_path, report_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--bounds", type=float, nargs=4, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"), required=True)
    parser.add_argument("--crs", required=True, help="Projected CRS, e.g. EPSG:32651 or EPSG:27700")
    parser.add_argument("--resolution-m", type=float, default=100.0)
    parser.add_argument("--max-distance-m", type=float, default=10_000.0)
    parser.add_argument("--location-index", default="flex_mem", choices=("flex_mem", "sparse_mem_array", "dense_mem_array", "sparse_file_array", "dense_file_array"))
    parser.add_argument("--coverage-geojson", type=Path)
    parser.add_argument("--assume-bounds-covered", action="store_true")
    parser.add_argument("--coverage-note", default="")
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--source-version")
    parser.add_argument("--license", default="ODbL-1.0")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--parent-pbf", type=Path)
    parser.add_argument("--parent-source-url")
    parser.add_argument("--parent-expected-sha256")
    parser.add_argument("--derivation-note", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = build(parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(result.manifest_path)
    print(result.report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
