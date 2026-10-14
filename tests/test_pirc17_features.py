import numpy as np
import pytest

from experiments.nex326.pirc21_adapter import FeatureSelection, FeatureSnapshotAdapter
from experiments.pirc17.features import CanonicalEncoder, LocalFrame, PredictedPositionFeatures
from experiments.pirc17.rollout import PredictedState
from tests.test_pirc21_adapter import _write_snapshot, _rows


def test_query_encoding_matches_offline_transforms_without_loading_rows(tmp_path, monkeypatch):
    adapter = FeatureSnapshotAdapter(_write_snapshot(tmp_path), FeatureSelection(
        variant_ids=("road.distance_log1p", "road.direction", "worldcover.grouped"),
        composition_ids=("road.log_distance_x_direction", "worldcover.grouped_x_road_distance")))
    expected = adapter.transform("train").model_matrix()
    def forbidden(*args, **kwargs):
        raise AssertionError("online encoding opened observed trajectory features")
    monkeypatch.setattr(adapter, "_load", forbidden)
    actual = CanonicalEncoder(adapter).encode(_rows("train", [0, 2, 4, 6])).model_matrix(4)
    np.testing.assert_array_equal(actual, expected)


def test_online_fitted_or_temporal_transform_is_not_silently_reinterpreted(tmp_path):
    adapter = FeatureSnapshotAdapter(_write_snapshot(tmp_path), FeatureSelection(
        variant_ids=("elevation.change_rate",)))
    with pytest.raises(ValueError, match="pointwise"):
        CanonicalEncoder(adapter)


def test_origin_anchored_projection_roundtrip():
    frame = LocalFrame(116.3, 39.9)
    xy = np.array([[0., 0.], [10., 20.], [-12., 75.]])
    np.testing.assert_allclose(frame.from_lonlat(frame.to_lonlat(xy)), xy, atol=1e-8)


def test_map_cannot_supply_truth_history_direction(tmp_path):
    adapter = FeatureSnapshotAdapter(_write_snapshot(tmp_path), FeatureSelection(
        variant_ids=("history.relative_road",)))
    captured = []
    def query(lonlat):
        captured.append(lonlat)
        return [{"history_direction_east": 99., "history_direction_north": 0.,
                 "historical_motion_status": "valid", "road_direction_east": 1.,
                 "road_direction_north": 0., "overture_road_status": "valid"}]
    provider = PredictedPositionFeatures(CanonicalEncoder(adapter), query, LocalFrame(0, 0))
    state = PredictedState(2, np.array([[0., 10.]]), np.array([[0., 5.]]),
        np.array([[[0., 0.], [0., 5.], [0., 10.]]]), np.array([0., 1., 2.]))
    result = provider(state)
    np.testing.assert_allclose(result.values[0], [-1, 0])
    np.testing.assert_allclose(captured[0], provider.frame.to_lonlat(state.positions_m))


def test_missing_canonical_fields_remain_masked(tmp_path):
    adapter = FeatureSnapshotAdapter(_write_snapshot(tmp_path), FeatureSelection(
        variant_ids=("road.distance_log1p",)))
    output = CanonicalEncoder(adapter).encode([{}])
    assert not output.valid.any()
    np.testing.assert_array_equal(output.model_matrix(1), [[0, 0]])
