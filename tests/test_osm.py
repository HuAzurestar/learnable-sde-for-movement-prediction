import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.osm import OSM_DISTANCE_COLUMNS, OSMDistanceField, augment_condition_frame
from data.source import extract_features
from scripts.build_osm_distance_fields import align_bounds, classify_way


def test_classify_way_declares_active_feature_rules():
    assert classify_way({"highway": "footway"}, False) == ("road_dist",)
    assert classify_way({"highway": "construction"}, False) == ()
    assert classify_way({"waterway": "stream"}, False) == ("water_dist",)
    assert classify_way({"natural": "water"}, True) == ("water_dist",)
    assert classify_way({"natural": "water"}, False) == ()
    assert classify_way({"building": "house"}, True) == ("building_dist",)
    assert classify_way({"building": "demolished"}, True) == ()


def test_align_bounds_is_outward_and_grid_aligned():
    assert align_bounds((12.5, -9.9, 208.1, 101.0), 100.0) == (0.0, -100.0, 300.0, 200.0)


def _write_test_field(root: Path) -> Path:
    rasterio = pytest.importorskip("rasterio")
    pytest.importorskip("pyproj")
    from rasterio.transform import from_origin

    transform = from_origin(0.0, 200.0, 100.0, 100.0)
    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "crs": "EPSG:3857",
        "transform": transform,
    }
    arrays = {
        "road_dist": np.array([[0.0, -9999.0], [100.0, 200.0]], dtype=np.float32),
        "water_dist": np.array([[10.0, -9999.0], [110.0, 210.0]], dtype=np.float32),
        "building_dist": np.array([[20.0, -9999.0], [120.0, 220.0]], dtype=np.float32),
    }
    outputs = {}
    for name, array in arrays.items():
        path = root / f"{name}.tif"
        with rasterio.open(path, "w", **profile, dtype="float32", nodata=-9999.0) as dst:
            dst.write(array, 1)
        outputs[name] = {"file": path.name}
    coverage_path = root / "osm_coverage.tif"
    with rasterio.open(coverage_path, "w", **profile, dtype="uint8", nodata=0) as dst:
        dst.write(np.array([[1, 0], [1, 1]], dtype=np.uint8), 1)
    outputs["coverage"] = {"file": coverage_path.name}
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": "osm-distance-field/v1", "region": "test", "outputs": outputs}),
        encoding="utf-8",
    )
    return manifest_path


def test_sampler_separates_uncovered_from_large_distance(tmp_path):
    pyproj = pytest.importorskip("pyproj")
    manifest_path = _write_test_field(tmp_path)
    inverse = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon, lat = inverse.transform([50.0, 150.0, 150.0], [150.0, 150.0, 50.0])
    with OSMDistanceField(manifest_path) as field:
        sampled = field.sample(lon, lat)
    assert sampled["has_osm"].tolist() == [1, 0, 1]
    assert sampled.loc[0, list(OSM_DISTANCE_COLUMNS)].tolist() == [0.0, 10.0, 20.0]
    assert sampled.loc[1, list(OSM_DISTANCE_COLUMNS)].isna().all()
    assert sampled.loc[2, "road_dist"] == 200.0


def test_augmentation_preserves_terrain_and_marks_missing(tmp_path):
    pyproj = pytest.importorskip("pyproj")
    manifest_path = _write_test_field(tmp_path)
    inverse = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon, lat = inverse.transform([50.0, 150.0], [150.0, 150.0])
    frame = pd.DataFrame(
        {
            "lat": lat,
            "lon": lon,
            "dem_elev": [12.0, 13.0],
            "dem_slope": [2.0, 3.0],
            "landcover": [10, 20],
            "has_map": [1, 1],
        }
    )
    with OSMDistanceField(manifest_path) as field:
        augmented = augment_condition_frame(frame, field)
    assert augmented[["dem_elev", "dem_slope", "landcover", "has_map"]].equals(
        frame[["dem_elev", "dem_slope", "landcover", "has_map"]]
    )
    assert augmented["has_osm"].tolist() == [1, 0]


def test_osm_condition_aggregation_filters_uncovered_rows():
    frame = pd.DataFrame(
        {
            "road_dist": [100.0, np.nan],
            "water_dist": [200.0, np.nan],
            "building_dist": [300.0, np.nan],
            "has_osm": [1, 0],
        }
    )
    np.testing.assert_allclose(extract_features(frame, "osm"), [100.0, 200.0, 300.0])
