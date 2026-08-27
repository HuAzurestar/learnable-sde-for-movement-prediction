import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.osm import (
    OSM_DISTANCE_COLUMNS,
    OSMDistanceCollection,
    OSMDistanceField,
    OSMDistanceFieldError,
    augment_condition_frame,
)
from data.source import extract_features
from scripts.augment_cond_slices_osm import run as augment_slices
from scripts.build_osm_distance_fields import (
    _geojson_geometries,
    _publish_staged_directory,
    align_bounds,
    classify_way,
    validate_metric_projected_crs,
)


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

    root.mkdir(parents=True, exist_ok=True)

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


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [("road_dist", np.nan), ("water_dist", np.inf), ("building_dist", -np.inf)],
)
def test_osm_condition_rejects_any_nonfinite_covered_value(column, bad_value):
    frame = pd.DataFrame(
        {
            "road_dist": [100.0, 110.0],
            "water_dist": [200.0, 210.0],
            "building_dist": [300.0, 310.0],
            "has_osm": [1, 1],
        }
    )
    frame.loc[1, column] = bad_value
    assert extract_features(frame, "osm") is None


def test_crs_must_be_projected_and_meter_based():
    pyproj = pytest.importorskip("pyproj")
    assert validate_metric_projected_crs("EPSG:32651") == pyproj.CRS("EPSG:32651")
    with pytest.raises(ValueError, match="projected"):
        validate_metric_projected_crs("EPSG:4326")
    with pytest.raises(ValueError, match="must use metres"):
        validate_metric_projected_crs("EPSG:2263")


def test_coverage_geojson_requires_valid_polygon():
    pytest.importorskip("shapely")
    valid = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
    }
    assert _geojson_geometries(valid) == [valid]
    with pytest.raises(ValueError, match="Polygon/MultiPolygon"):
        _geojson_geometries({"type": "LineString", "coordinates": [[0, 0], [1, 1]]})
    with pytest.raises(ValueError, match="empty or invalid"):
        _geojson_geometries(
            {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]],
            }
        )


def test_collection_rejects_overlapping_manifests(tmp_path):
    pyproj = pytest.importorskip("pyproj")
    left = _write_test_field(tmp_path / "left")
    right = _write_test_field(tmp_path / "right")
    inverse = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon, lat = inverse.transform([50.0], [150.0])
    collection = OSMDistanceCollection([left, right])
    try:
        with pytest.raises(OSMDistanceFieldError, match="overlapping OSM manifests"):
            collection.sample(lon, lat)
    finally:
        collection.close()


def test_condition_augmentation_manifest_hashes_input_and_publishes_atomically(tmp_path):
    pyproj = pytest.importorskip("pyproj")
    pytest.importorskip("pyarrow")
    field_manifest = _write_test_field(tmp_path / "field")
    inverse = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon, lat = inverse.transform([50.0], [150.0])
    input_root = tmp_path / "input"
    source = input_root / "val" / "sample_cond.parquet"
    source.parent.mkdir(parents=True)
    pd.DataFrame({"lon": lon, "lat": lat}).to_parquet(source, index=False)
    output_root = tmp_path / "output"
    manifest_path = augment_slices(
        Namespace(
            input_root=input_root,
            output_root=output_root,
            manifest=[field_manifest],
            pattern="*_cond.parquet",
            overwrite=False,
        )
    )
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = document["outputs"][0]
    from scripts.augment_cond_slices_osm import sha256_file

    assert record["input_sha256"] == sha256_file(source)
    assert record["output_sha256"] == record["sha256"]
    assert not list(tmp_path.glob(".output.staging-*"))


def test_atomic_build_publication_refuses_unmanaged_destination(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "manifest.json").write_text("{}", encoding="utf-8")
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "keep.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(FileExistsError, match="unmanaged"):
        _publish_staged_directory(staging, destination, overwrite=True)
    assert (destination / "keep.txt").read_text(encoding="utf-8") == "user data"
