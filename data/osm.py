"""Sampling and condition-slice adapters for audited OSM distance fields.

Distance rasters contain metres in their native projected CRS.  A separate
coverage band is authoritative: outside coverage, distance values are missing
(``NaN``), not large observed distances.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


OSM_DISTANCE_COLUMNS = ("road_dist", "water_dist", "building_dist")


class OSMDistanceFieldError(ValueError):
    """Raised when an OSM distance-field manifest violates its contract."""


@dataclass(frozen=True)
class _RasterPaths:
    road_dist: Path
    water_dist: Path
    building_dist: Path
    coverage: Path


class OSMDistanceField:
    """Lazy sampler for one regional OSM distance-field manifest."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "osm-distance-field/v1":
            raise OSMDistanceFieldError(
                f"unsupported OSM manifest schema: {manifest.get('schema_version')!r}"
            )
        outputs = manifest.get("outputs", {})
        missing = [name for name in (*OSM_DISTANCE_COLUMNS, "coverage") if name not in outputs]
        if missing:
            raise OSMDistanceFieldError(f"manifest missing outputs: {missing}")
        base = self.manifest_path.parent
        self.paths = _RasterPaths(
            **{name: (base / outputs[name]["file"]).resolve() for name in (*OSM_DISTANCE_COLUMNS, "coverage")}
        )
        self.region = str(manifest.get("region", "unknown"))
        self._datasets = None
        self._transformer = None

    def _open(self):
        if self._datasets is not None:
            return
        try:
            import rasterio
            from pyproj import Transformer
        except ImportError as exc:  # pragma: no cover - depends on optional environment
            raise OSMDistanceFieldError(
                "OSM sampling requires the 'osm' optional dependencies"
            ) from exc
        datasets = {
            name: rasterio.open(getattr(self.paths, name))
            for name in (*OSM_DISTANCE_COLUMNS, "coverage")
        }
        reference = datasets["coverage"]
        for name, dataset in datasets.items():
            if (
                dataset.crs != reference.crs
                or dataset.transform != reference.transform
                or dataset.shape != reference.shape
            ):
                for opened in datasets.values():
                    opened.close()
                raise OSMDistanceFieldError(f"raster grid mismatch at {name}")
        self._datasets = datasets
        self._transformer = Transformer.from_crs("EPSG:4326", reference.crs, always_xy=True)

    def close(self) -> None:
        if self._datasets is not None:
            for dataset in self._datasets.values():
                dataset.close()
        self._datasets = None
        self._transformer = None

    def __enter__(self) -> "OSMDistanceField":
        self._open()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def sample(self, lon: Sequence[float], lat: Sequence[float]) -> pd.DataFrame:
        """Sample EPSG:4326 coordinates while preserving missing coverage.

        Returned columns are ``road_dist``, ``water_dist``, ``building_dist``
        and ``has_osm``.  Coordinates outside the coverage mask receive NaN in
        every distance channel and ``has_osm=0``.
        """

        self._open()
        lon_array = np.asarray(lon, dtype=np.float64)
        lat_array = np.asarray(lat, dtype=np.float64)
        if lon_array.shape != lat_array.shape or lon_array.ndim != 1:
            raise OSMDistanceFieldError("lon and lat must be equal-length one-dimensional arrays")
        x, y = self._transformer.transform(lon_array, lat_array)
        coordinates = list(zip(x, y))
        if not coordinates:
            return pd.DataFrame(
                {**{name: np.array([], dtype=np.float64) for name in OSM_DISTANCE_COLUMNS},
                 "has_osm": np.array([], dtype=np.int8)}
            )
        coverage_values = np.ma.asarray(
            list(self._datasets["coverage"].sample(coordinates, masked=True))
        ).astype(np.float64)[:, 0].filled(np.nan)
        has_osm = np.isfinite(coverage_values) & (coverage_values == 1)
        result: dict[str, np.ndarray] = {}
        for name in OSM_DISTANCE_COLUMNS:
            sampled = np.ma.asarray(
                list(self._datasets[name].sample(coordinates, masked=True))
            ).astype(np.float64)[:, 0].filled(np.nan)
            sampled[~has_osm] = np.nan
            result[name] = sampled
        result["has_osm"] = has_osm.astype(np.int8)
        return pd.DataFrame(result)


class OSMDistanceCollection:
    """Sample several non-overlapping regional fields in declared order."""

    def __init__(self, manifest_paths: Iterable[str | Path]):
        self.fields = [OSMDistanceField(path) for path in manifest_paths]
        if not self.fields:
            raise OSMDistanceFieldError("at least one OSM manifest is required")

    def close(self) -> None:
        for field in self.fields:
            field.close()

    def sample(self, lon: Sequence[float], lat: Sequence[float]) -> pd.DataFrame:
        lon_array = np.asarray(lon, dtype=np.float64)
        lat_array = np.asarray(lat, dtype=np.float64)
        result = pd.DataFrame(
            {name: np.full(len(lon_array), np.nan) for name in OSM_DISTANCE_COLUMNS}
        )
        result["has_osm"] = np.zeros(len(lon_array), dtype=np.int8)
        for field in self.fields:
            sampled = field.sample(lon_array, lat_array)
            take = (result["has_osm"].to_numpy() == 0) & (sampled["has_osm"].to_numpy() == 1)
            if take.any():
                result.loc[take, list(OSM_DISTANCE_COLUMNS)] = sampled.loc[
                    take, list(OSM_DISTANCE_COLUMNS)
                ].to_numpy()
                result.loc[take, "has_osm"] = 1
        return result


def augment_condition_frame(
    frame: pd.DataFrame, fields: OSMDistanceField | OSMDistanceCollection
) -> pd.DataFrame:
    """Return a copy with three OSM channels and an explicit coverage flag."""

    required = {"lat", "lon"}
    missing = required.difference(frame.columns)
    if missing:
        raise OSMDistanceFieldError(f"condition frame missing columns: {sorted(missing)}")
    sampled = fields.sample(frame["lon"].to_numpy(), frame["lat"].to_numpy())
    output = frame.copy()
    for column in (*OSM_DISTANCE_COLUMNS, "has_osm"):
        output[column] = sampled[column].to_numpy()
    return output
