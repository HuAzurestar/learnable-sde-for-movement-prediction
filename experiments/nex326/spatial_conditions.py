"""Leakage-safe DSDE terrain lookup for the NEX326 phase-space benchmark."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .cohort import Cohort, Segment
from .phase_space import ConditionField, PhaseSpaceError


EARTH_RADIUS_METRES = 6_371_000.0
DEGREES_TO_RADIANS = math.pi / 180.0
TERRAIN_CONDITION_NAMES = ("terrain_elevation", "terrain_slope")
DIRECTIONAL_TERRAIN_CONDITION_NAMES = (
    "terrain_elevation",
    "terrain_gradient_east",
    "terrain_gradient_north",
)
SUPPORTED_CONDITION_PROFILES = {
    TERRAIN_CONDITION_NAMES,
    DIRECTIONAL_TERRAIN_CONDITION_NAMES,
}
SPLIT_DIRECTORIES = {
    "train": "zhejiang_finetune",
    "adapt": "zhejiang_finetune",
    "validation": "zhejiang_val",
    "evaluation": "zhejiang_eval",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_id(segment: Segment) -> str:
    parts = segment.segment_id.split(":", 2)
    if len(parts) != 3 or not parts[1]:
        raise PhaseSpaceError(
            "DSDE terrain lookup requires split:file_id:source_segment ids: "
            f"{segment.segment_id}"
        )
    return parts[1]


def _tile_name(latitude: float, longitude: float) -> str:
    latitude_floor = math.floor(latitude)
    longitude_floor = math.floor(longitude)
    latitude_code = (
        f"N{latitude_floor:02d}" if latitude_floor >= 0 else f"S{-latitude_floor:02d}"
    )
    longitude_code = (
        f"E{longitude_floor:03d}"
        if longitude_floor >= 0
        else f"W{-longitude_floor:03d}"
    )
    return f"{latitude_code}{longitude_code}.hgt"


@dataclass(frozen=True)
class _Projection:
    latitude_origin: float
    longitude_origin: float
    condition_path: Path


class RasterTerrainField(ConditionField):
    """Evaluate registered SRTM features from one DSDE file's local coordinates."""

    def __init__(self, projection: _Projection, tile_loader, names):
        self._projection = projection
        self._tile_loader = tile_loader
        self._names = tuple(names)

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    def _geographic(self, position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.atleast_2d(np.asarray(position, dtype=float))
        if values.shape[1] != 2 or not np.isfinite(values).all():
            raise PhaseSpaceError("terrain lookup positions must have shape (..., 2)")
        latitude = (
            values[:, 1] / (EARTH_RADIUS_METRES * DEGREES_TO_RADIANS)
            + self._projection.latitude_origin
        )
        longitude = (
            values[:, 0]
            / (
                EARTH_RADIUS_METRES
                * math.cos(math.radians(self._projection.latitude_origin))
                * DEGREES_TO_RADIANS
            )
            + self._projection.longitude_origin
        )
        return latitude, longitude

    def evaluate(self, position: np.ndarray, time: float) -> np.ndarray:
        del time  # Terrain is static; time remains available for future fields.
        latitude, longitude = self._geographic(position)
        result = np.empty((len(latitude), len(self.names)), dtype=float)
        tile_names = np.asarray(
            [_tile_name(lat, lon) for lat, lon in zip(latitude, longitude)],
            dtype=object,
        )
        for tile_name in np.unique(tile_names):
            selected = tile_names == tile_name
            tile = self._tile_loader(str(tile_name))
            sampled = self._sample_tile(
                tile,
                latitude[selected],
                longitude[selected],
                str(tile_name),
            )
            feature_index = {
                "terrain_elevation": 0,
                "terrain_slope": 1,
                "terrain_gradient_east": 2,
                "terrain_gradient_north": 3,
            }
            result[selected] = sampled[
                :, [feature_index[name] for name in self.names]
            ]
        return result

    @staticmethod
    def _sample_tile(
        tile: np.ndarray,
        latitude: np.ndarray,
        longitude: np.ndarray,
        tile_name: str,
    ) -> np.ndarray:
        if tile.ndim != 2 or tile.shape[0] != tile.shape[1] or tile.shape[0] < 3:
            raise PhaseSpaceError(
                "SRTM HGT tile must be a square grid with at least three samples"
            )
        stem = Path(tile_name).stem
        try:
            tile_latitude = int(stem[1:3]) * (1 if stem[0] == "N" else -1)
            tile_longitude = int(stem[4:7]) * (1 if stem[3] == "E" else -1)
        except (IndexError, ValueError) as error:
            raise PhaseSpaceError(f"invalid SRTM tile name: {stem}") from error
        scale = tile.shape[0] - 1
        # Rasterio's SRTMHGT driver exposes the outer bounds half a post beyond
        # the integer-degree sample centres. Match the registered slice builder.
        half_post = 0.5 / scale
        row = (tile_latitude + 1.0 + half_post - latitude) * scale
        column = (longitude - (tile_longitude - half_post)) * scale
        row0 = np.clip(np.floor(row).astype(int), 0, scale)
        column0 = np.clip(np.floor(column).astype(int), 0, scale)
        row1 = np.clip(row0 + 1, 0, scale)
        column1 = np.clip(column0 + 1, 0, scale)
        row_fraction = row - row0
        column_fraction = column - column0
        elevation = (
            tile[row0, column0] * (1.0 - row_fraction) * (1.0 - column_fraction)
            + tile[row0, column1] * (1.0 - row_fraction) * column_fraction
            + tile[row1, column0] * row_fraction * (1.0 - column_fraction)
            + tile[row1, column1] * row_fraction * column_fraction
        )

        metres_per_cell_latitude = 111_320.0 / scale
        metres_per_cell_longitude = (
            metres_per_cell_latitude
            * math.cos(math.radians(tile_latitude + 0.5))
        )
        corner_rows = (row0, row0, row1, row1)
        corner_columns = (column0, column1, column0, column1)
        corner_slopes = []
        corner_east_gradients = []
        corner_north_gradients = []
        for corner_row, corner_column in zip(corner_rows, corner_columns):
            gradient_row = np.clip(corner_row, 1, scale - 1)
            gradient_column = np.clip(corner_column, 1, scale - 1)
            north_south = (
                tile[gradient_row - 1, gradient_column]
                - tile[gradient_row + 1, gradient_column]
            ) / (2.0 * metres_per_cell_latitude)
            east_west = (
                tile[gradient_row, gradient_column + 1]
                - tile[gradient_row, gradient_column - 1]
            ) / (2.0 * metres_per_cell_longitude)
            corner_slopes.append(
                np.degrees(np.arctan(np.hypot(north_south, east_west)))
            )
            corner_east_gradients.append(east_west)
            corner_north_gradients.append(north_south)
        weights = (
            (1.0 - row_fraction) * (1.0 - column_fraction),
            (1.0 - row_fraction) * column_fraction,
            row_fraction * (1.0 - column_fraction),
            row_fraction * column_fraction,
        )
        slope = (
            corner_slopes[0] * weights[0]
            + corner_slopes[1] * weights[1]
            + corner_slopes[2] * weights[2]
            + corner_slopes[3] * weights[3]
        )
        gradient_east = sum(
            gradient * weight
            for gradient, weight in zip(corner_east_gradients, weights)
        )
        gradient_north = sum(
            gradient * weight
            for gradient, weight in zip(corner_north_gradients, weights)
        )
        values = np.column_stack(
            [elevation, slope, gradient_east, gradient_north]
        )
        if not np.isfinite(values).all() or np.any(elevation <= -30_000):
            raise PhaseSpaceError("SRTM terrain lookup returned a void or non-finite value")
        return values


class DSDERasterConditionResolver:
    """Resolve file-specific local coordinates against registered SRTM rasters.

    Condition slices are used only to recover the exact projection origin used by
    the NEX-313 pipeline. Feature values always come from the raster, so no
    trajectory-point nearest-neighbour index (and therefore no future route
    geometry) is exposed.
    """

    names = TERRAIN_CONDITION_NAMES

    def __init__(
        self,
        cohort: Cohort,
        condition_root: Path | str,
        srtm_root: Path | str,
        names: tuple[str, ...] = TERRAIN_CONDITION_NAMES,
    ) -> None:
        self.names = tuple(names)
        if self.names not in SUPPORTED_CONDITION_PROFILES:
            raise PhaseSpaceError(
                f"unsupported DSDE terrain condition profile: {self.names}"
            )
        self._condition_root = Path(condition_root).resolve()
        self._srtm_root = Path(srtm_root).resolve()
        if not self._condition_root.is_dir() or not self._srtm_root.is_dir():
            raise PhaseSpaceError("condition and SRTM roots must both be directories")
        self._projections: dict[str, _Projection] = {}
        condition_paths: set[Path] = set()
        projection_cache: dict[Path, tuple[float, float]] = {}
        for split, segments in cohort.splits.items():
            if split == "animal_pretrain":
                continue
            directory = SPLIT_DIRECTORIES.get(split)
            if directory is None:
                raise PhaseSpaceError(f"no DSDE condition directory registered for {split}")
            for segment in segments:
                file_id = _file_id(segment)
                condition_path = (
                    self._condition_root / directory / f"{file_id}_cond.parquet"
                ).resolve()
                if not condition_path.is_relative_to(self._condition_root):
                    raise PhaseSpaceError("condition path escaped its registered root")
                if not condition_path.is_file():
                    raise PhaseSpaceError(
                        f"registered condition slice is absent: {condition_path}"
                    )
                if condition_path not in projection_cache:
                    frame = pd.read_parquet(
                        condition_path, columns=["file_id", "lat", "lon"]
                    )
                    if set(str(value) for value in frame["file_id"].unique()) != {
                        file_id
                    }:
                        raise PhaseSpaceError(
                            f"condition slice file identity mismatch: {condition_path}"
                        )
                    latitude_origin = float(frame["lat"].mean())
                    longitude_origin = float(frame["lon"].mean())
                    if not np.isfinite([latitude_origin, longitude_origin]).all():
                        raise PhaseSpaceError(
                            f"invalid projection origin in {condition_path}"
                        )
                    observed_tiles = {
                        _tile_name(float(latitude), float(longitude))
                        for latitude, longitude in zip(frame["lat"], frame["lon"])
                        if np.isfinite(latitude) and np.isfinite(longitude)
                    }
                    for tile_name in observed_tiles:
                        tile_path = (self._srtm_root / tile_name).resolve()
                        if not tile_path.is_relative_to(self._srtm_root):
                            raise PhaseSpaceError("SRTM path escaped its registered root")
                        if not tile_path.is_file():
                            raise PhaseSpaceError(
                                f"registered SRTM tile is absent: {tile_path}"
                            )
                    projection_cache[condition_path] = (
                        latitude_origin,
                        longitude_origin,
                    )
                latitude_origin, longitude_origin = projection_cache[condition_path]
                self._projections[segment.segment_id] = _Projection(
                    latitude_origin=latitude_origin,
                    longitude_origin=longitude_origin,
                    condition_path=condition_path,
                )
                condition_paths.add(condition_path)
        self._condition_paths = tuple(sorted(condition_paths))
        self._tiles: dict[Path, np.ndarray] = {}
        self._used_tile_paths: set[Path] = set()

    @staticmethod
    def _load_tile(path: Path) -> np.ndarray:
        samples = path.stat().st_size // 2
        side = math.isqrt(samples)
        if side * side != samples:
            raise PhaseSpaceError(f"SRTM tile does not contain a square int16 grid: {path}")
        return np.memmap(path, dtype=">i2", mode="r", shape=(side, side))

    def for_segment(self, segment: Segment) -> RasterTerrainField:
        try:
            projection = self._projections[segment.segment_id]
        except KeyError as error:
            raise PhaseSpaceError(
                f"no terrain projection registered for {segment.segment_id}"
            ) from error
        return RasterTerrainField(projection, self._tile, self.names)

    def _tile(self, tile_name: str) -> np.ndarray:
        path = (self._srtm_root / tile_name).resolve()
        if not path.is_relative_to(self._srtm_root):
            raise PhaseSpaceError("SRTM path escaped its registered root")
        if not path.is_file():
            raise PhaseSpaceError(f"SRTM tile required by rollout is absent: {path}")
        if path not in self._tiles:
            self._tiles[path] = self._load_tile(path)
        self._used_tile_paths.add(path)
        return self._tiles[path]

    def identity(self) -> dict[str, object]:
        return {
            "schema_version": "nex326-dsde-raster-condition-resolver-v2",
            "names": list(self.names),
            "projection": (
                "inverse NEX-313 file-mean equirectangular local projection; "
                "origin recovered from the registered condition slice"
            ),
            "feature_source": "SRTM HGT raster sampled at each simulated position",
            "future_route_point_index_used": False,
            "condition_files": [
                {
                    "path": str(path.relative_to(self._condition_root)).replace("\\", "/"),
                    "sha256": _sha256(path),
                }
                for path in self._condition_paths
            ],
            "terrain_tiles": [
                {"name": path.name, "sha256": _sha256(path)}
                for path in sorted(self._used_tile_paths)
            ],
        }


__all__ = [
    "DIRECTIONAL_TERRAIN_CONDITION_NAMES",
    "DSDERasterConditionResolver",
    "RasterTerrainField",
    "TERRAIN_CONDITION_NAMES",
]
