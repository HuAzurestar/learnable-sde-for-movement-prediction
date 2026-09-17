from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.nex326.pirc21_adapter import (
    FeatureSelection,
    FeatureSnapshotAdapter,
    PIRC21AdapterError,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _column(name: str) -> dict[str, object]:
    return {"name": name, "dtype": "float32", "unit": "1", "role": name}


def _factor(
    factor_id: str, columns: list[str], status: str
) -> dict[str, object]:
    return {
        "factor_id": factor_id,
        "value_columns": [_column(name) for name in columns],
        "status_column": status,
    }


def _variant(
    variant_id: str,
    factor_id: str,
    transform: str,
    inputs: list[str],
    outputs: list[str],
    *,
    fit_scope: str = "none",
    **parameters: object,
) -> dict[str, object]:
    return {
        "variant_id": variant_id,
        "factor_id": factor_id,
        "transform": transform,
        "parameters": {"input_columns": inputs, **parameters},
        "output_columns": [_column(name) for name in outputs],
        "output_dim": len(outputs),
        "fit_scope": fit_scope,
    }


def _spec() -> dict[str, object]:
    return {
        "schema_version": "pirc21-feature-spec-v1",
        "feature_spec_id": "pirc21-psde-adapter-fixture-v1",
        "processing_version": "fixture-v1",
        "factors": [
            _factor("dem_elevation", ["elevation_m"], "dem_elevation_status"),
            _factor("worldcover", ["worldcover_class"], "worldcover_status"),
            _factor(
                "overture_road",
                ["road_distance_m", "road_direction_east", "road_direction_north"],
                "overture_road_status",
            ),
            _factor(
                "historical_motion",
                ["history_direction_east", "history_direction_north"],
                "historical_motion_status",
            ),
        ],
        "variants": [
            _variant(
                "elevation.absolute",
                "dem_elevation",
                "identity",
                ["elevation_m"],
                ["v_elevation_absolute_m"],
            ),
            _variant(
                "elevation.absolute_standardized",
                "dem_elevation",
                "clipped_standardize",
                ["elevation_m"],
                ["v_elevation_absolute_z"],
                fit_scope="train_only",
                clip_sigma=6.0,
            ),
            _variant(
                "elevation.change_rate",
                "dem_elevation",
                "visible_time_difference",
                ["elevation_m"],
                ["v_elevation_change_rate_mps"],
                time_column="absolute_epoch_ns",
            ),
            _variant(
                "worldcover.grouped",
                "worldcover",
                "grouped_one_hot",
                ["worldcover_class"],
                [
                    "v_worldcover_built",
                    "v_worldcover_vegetated",
                    "v_worldcover_water_wetland",
                    "v_worldcover_bare_snow_unknown",
                ],
                groups=[
                    "built",
                    "vegetated",
                    "water_wetland",
                    "bare_snow_unknown",
                ],
            ),
            _variant(
                "road.distance_log1p",
                "overture_road",
                "log1p",
                ["road_distance_m"],
                ["v_road_distance_log1p"],
                scale_m=1.0,
            ),
            _variant(
                "road.direction",
                "overture_road",
                "unit_vector",
                ["road_direction_east", "road_direction_north"],
                ["v_road_direction_east", "v_road_direction_north"],
            ),
            _variant(
                "history.relative_road",
                "historical_motion",
                "relative_heading_sin_cos",
                ["history_direction_east", "history_direction_north"],
                ["v_history_road_relative_sin", "v_history_road_relative_cos"],
                reference_columns=["road_direction_east", "road_direction_north"],
            ),
        ],
        "compositions": [
            {
                "composition_id": "road.log_distance_x_direction",
                "input_variant_ids": ["road.distance_log1p", "road.direction"],
                "operator": "scalar_vector_product",
                "parameters": {},
                "output_columns": [
                    _column("c_road_log_direction_east"),
                    _column("c_road_log_direction_north"),
                ],
                "output_dim": 2,
            },
            {
                "composition_id": "worldcover.grouped_x_road_distance",
                "input_variant_ids": ["worldcover.grouped", "road.distance_log1p"],
                "operator": "scalar_vector_product",
                "parameters": {},
                "output_columns": [
                    _column("c_worldcover_built_x_road"),
                    _column("c_worldcover_vegetated_x_road"),
                    _column("c_worldcover_water_x_road"),
                    _column("c_worldcover_other_x_road"),
                ],
                "output_dim": 4,
            },
        ],
        "segment_aggregations": [
            {"aggregation_id": name, "operator": name}
            for name in ("mean", "std", "min", "max", "last", "valid_fraction")
        ],
    }


def _rows(split: str, values: list[float]) -> list[dict[str, object]]:
    base = 1_700_000_000_000_000_000
    rows = []
    for index, elevation in enumerate(values):
        road_valid = index != 2
        rows.append(
            {
                "dataset_version": "pirc20-fixture-v1",
                "point_id": f"{split}-point-{index}",
                "file_id": f"{split}-file-{index // 2}",
                "point_index": index,
                "absolute_epoch_ns": base + (index % 2) * 1_000_000_000,
                "segment_id": f"{split}-segment-{index // 2}",
                "split": split,
                "independent_block_id": f"{split}-block-{index // 2}",
                "elevation_m": elevation,
                "dem_elevation_status": "valid",
                "worldcover_class": (50, 10, 80, 999)[index % 4],
                "worldcover_status": "valid",
                "road_distance_m": float(index * 10) if road_valid else None,
                "road_direction_east": 3.0 if road_valid else None,
                "road_direction_north": 4.0 if road_valid else None,
                "overture_road_status": "valid" if road_valid else "not_materialized",
                "history_direction_east": 0.0,
                "history_direction_north": 1.0,
                "historical_motion_status": "valid",
            }
        )
    return rows


def _write_snapshot(tmp_path: Path) -> Path:
    root = tmp_path / "snapshot"
    root.mkdir()
    spec = _spec()
    (root / "feature_spec.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    coverage = {"schema_version": "pirc21-feature-snapshot-v1", "factors": {}}
    coverage_path = root / "coverage_report.json"
    coverage_path.write_text(
        json.dumps(coverage, sort_keys=True) + "\n", encoding="utf-8"
    )
    definitions = {
        "train": [0.0, 2.0, 4.0, 6.0],
        "validation": [100.0, 102.0],
        "final_eval": [200.0, 202.0],
    }
    files = []
    spec_fingerprint = _canonical_hash(spec)
    for split, values in definitions.items():
        relative = Path("features") / split / f"{split}.parquet"
        path = root / relative
        path.parent.mkdir(parents=True)
        table = pa.Table.from_pylist(_rows(split, values))
        table = table.replace_schema_metadata(
            {b"pirc21.feature_spec_sha256": spec_fingerprint.encode("ascii")}
        )
        pq.write_table(table, path)
        files.append(
            {
                "file_id": f"{split}-fixture",
                "split": split,
                "path": relative.as_posix(),
                "row_count": len(values),
                "sha256": _sha256(path),
            }
        )
    inventory = hashlib.sha256()
    for item in sorted(files, key=lambda value: value["path"]):
        inventory.update(
            f"{item['path']}\0{item['sha256']}\0{item['row_count']}\n".encode()
        )
    manifest = {
        "schema_version": "pirc21-feature-snapshot-v1",
        "snapshot_id": "fixture-snapshot-v1",
        "status": "valid",
        "dataset_id": "pirc20-fixture-v1",
        "feature_spec_id": spec["feature_spec_id"],
        "feature_spec_sha256": spec_fingerprint,
        "coverage_report_sha256": _sha256(coverage_path),
        "content_inventory_sha256": inventory.hexdigest(),
        "files": files,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def test_no_terrain_selection_is_a_real_zero_width_configuration(tmp_path):
    snapshot = _write_snapshot(tmp_path)
    adapter = FeatureSnapshotAdapter(snapshot, FeatureSelection())
    batch = adapter.transform("train")

    assert batch.columns == ()
    assert batch.values.shape == (4, 0)
    assert batch.model_matrix().shape == (4, 0)


def test_single_factor_and_composition_have_explicit_order_and_missingness(tmp_path):
    snapshot = _write_snapshot(tmp_path)
    selection = FeatureSelection(
        variant_ids=("road.distance_log1p", "road.direction"),
        composition_ids=("road.log_distance_x_direction",),
    )
    batch = FeatureSnapshotAdapter(snapshot, selection).transform("train")

    assert batch.columns == (
        "v_road_distance_log1p",
        "v_road_direction_east",
        "v_road_direction_north",
        "c_road_log_direction_east",
        "c_road_log_direction_north",
    )
    assert np.allclose(batch.values[1, :3], [np.log1p(10.0), 0.6, 0.8])
    assert not batch.valid[2].any()
    matrix = batch.model_matrix()
    assert matrix.shape == (4, 10)
    assert np.all(matrix[2, :5] == 0.0)
    assert np.all(matrix[2, 5:] == 0.0)
    with pytest.raises(PIRC21AdapterError, match="cannot be disabled"):
        batch.model_matrix(include_validity=False)


def test_train_only_fit_and_visible_difference_never_fit_on_validation(tmp_path):
    snapshot = _write_snapshot(tmp_path)
    selection = FeatureSelection(
        variant_ids=(
            "elevation.absolute_standardized",
            "elevation.change_rate",
        )
    )
    adapter = FeatureSnapshotAdapter(snapshot, selection)
    with pytest.raises(PIRC21AdapterError, match=r"fit\(\)"):
        adapter.transform("validation")
    adapter.fit()
    train = adapter.transform("train")
    validation = adapter.transform("validation")

    assert np.isclose(np.mean(train.values[:, 0]), 0.0)
    assert np.all(validation.values[:, 0] == 6.0)
    assert train.valid[:, 1].tolist() == [False, True, False, True]
    assert np.allclose(train.values[train.valid[:, 1], 1], [2.0, 2.0])
    repeated = FeatureSnapshotAdapter(snapshot, selection).fit()
    assert adapter.fit_state_fingerprint == repeated.fit_state_fingerprint
    assert validation.cache_identity == repeated.transform("validation").cache_identity
    assert adapter.identity_record["feature_spec_id"] == (
        "pirc21-psde-adapter-fixture-v1"
    )
    assert adapter.identity_record["fit_state_fingerprint"] == (
        adapter.fit_state_fingerprint
    )


def test_information_groups_and_segment_aggregates_are_configuration_only(tmp_path):
    snapshot = _write_snapshot(tmp_path)
    selection = FeatureSelection(
        variant_ids=("worldcover.grouped", "road.distance_log1p"),
        composition_ids=("worldcover.grouped_x_road_distance",),
        segment_aggregations=("mean", "last", "valid_fraction"),
    )
    adapter = FeatureSnapshotAdapter(snapshot, selection)
    point = adapter.transform("train")
    segment = adapter.transform_segments("train")

    assert point.values.shape == (4, 9)
    assert segment.segment_ids == ("train-segment-0", "train-segment-1")
    assert segment.values.shape == (2, 27)
    assert segment.columns[0] == "v_worldcover_built__mean"
    assert segment.columns[-1] == "c_worldcover_other_x_road__valid_fraction"
    assert segment.model_matrix().shape == (2, 54)


def test_final_eval_is_sealed_and_snapshot_tampering_is_rejected(tmp_path):
    snapshot = _write_snapshot(tmp_path)
    adapter = FeatureSnapshotAdapter(
        snapshot, FeatureSelection(variant_ids=("elevation.absolute",))
    )
    with pytest.raises(PIRC21AdapterError, match="sealed"):
        adapter.transform("final_eval")
    unlocked = adapter.transform(
        "final_eval", final_eval_unlock="pirc20-fixture-v1"
    )
    assert unlocked.values.shape == (2, 1)

    target = snapshot / "features" / "train" / "train.parquet"
    target.write_bytes(target.read_bytes() + b"tampered")
    with pytest.raises(PIRC21AdapterError, match="hash mismatch"):
        FeatureSnapshotAdapter(snapshot, FeatureSelection())


def test_cache_identity_changes_with_selected_factor_set(tmp_path):
    snapshot = _write_snapshot(tmp_path)
    empty = FeatureSnapshotAdapter(snapshot, FeatureSelection()).transform("train")
    elevation = FeatureSnapshotAdapter(
        snapshot, FeatureSelection(variant_ids=("elevation.absolute",))
    ).transform("train")
    assert empty.cache_identity != elevation.cache_identity
