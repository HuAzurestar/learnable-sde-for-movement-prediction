import numpy as np
import pytest

from experiments.pirc17.configurations import terrain_configurations, configuration_encoder, subset_matrix
from experiments.pirc17.development import window_from_rows
from experiments.pirc17.features import CanonicalEncoder
from experiments.nex326.pirc21_adapter import FeatureSelection, FeatureSnapshotAdapter
from tests.test_pirc21_adapter import _write_snapshot, _rows


def test_ten_configurations_keep_history_and_drop_owned_compositions():
    configs=terrain_configurations()
    assert len(configs)==10 and "loo-history" not in configs
    assert all("history.direction" in c["variant_ids"] for c in configs.values())
    assert configs["base"]["variant_ids"]==["history.direction"]
    assert "worldcover.grouped_x_road_distance" not in configs["loo-road"]["composition_ids"]
    assert "worldcover.grouped_x_road_distance" not in configs["loo-worldcover"]["composition_ids"]
    assert "surface.orientation_x_history" in configs["lio-surface"]["composition_ids"]


def test_selected_online_matrices_match_full_matrix_slices(tmp_path,monkeypatch):
    # Small existing adapter fixture tests slicing independently of the production
    # registry above; production dimensions remain covered by the handoff tests.
    configs={"both":{"variant_ids":["road.distance_log1p","road.direction","worldcover.grouped"],
        "composition_ids":["road.log_distance_x_direction","worldcover.grouped_x_road_distance"]},
        "road":{"variant_ids":["road.distance_log1p","road.direction"],
        "composition_ids":["road.log_distance_x_direction"]}}
    monkeypatch.setattr("experiments.pirc17.configurations.terrain_configurations",lambda:configs)
    selected=configs["both"]
    full=CanonicalEncoder(FeatureSnapshotAdapter(_write_snapshot(tmp_path),FeatureSelection(
        variant_ids=tuple(selected["variant_ids"]),composition_ids=tuple(selected["composition_ids"]))))
    rows=_rows("train",[0,2,4,6])
    matrix=full.encode(rows).model_matrix(len(rows))
    for name in configs:
        encoder=configuration_encoder(full,name)
        np.testing.assert_array_equal(encoder.encode(rows).model_matrix(len(rows)),
            subset_matrix(matrix,full.columns,encoder.columns))
    assert len(full.columns)>len(configuration_encoder(full,"road").columns)


def fixture(role="train"):
    times=[-10,-5,0]+list(range(5,1801,5))
    # Deliberately offset source indices: never confuse segment offset with source index.
    metadata=[{"point_index":i+7,"absolute_epoch_ns":int(t*1e9),"split":role,"independent_block_id":"b"} for i,t in enumerate(times)]
    sample={"sample_id":"fixture","split":role,"independent_block_id":"b","history_start":0,"history_end":2,
        "target_start":3,"target_end":len(times)-1}
    positions=np.column_stack((np.arange(len(times)+7)*.00001,np.zeros(len(times)+7)))
    return sample,metadata,positions


def test_visible_origin_never_uses_future_positions_and_source_indices_are_preserved():
    sample,metadata,positions=fixture()
    original=window_from_rows(sample,metadata,positions)
    changed=positions.copy()
    changed[10:,0]+=.001
    other=window_from_rows(sample,metadata,changed)
    np.testing.assert_array_equal(original.origin.velocity_mps,other.origin.velocity_mps)
    np.testing.assert_array_equal(original.origin.history_positions_m,other.origin.history_positions_m)
    assert not np.allclose(original.target_positions_m,other.target_positions_m)
    assert original.frame.longitude==positions[9,0]
    np.testing.assert_array_equal(original.horizon_seconds,[60,300,900,1800])


def test_development_loader_rejects_final_eval_and_role_mismatch():
    with pytest.raises(ValueError,match="forbids"):
        window_from_rows(*fixture("final_eval"))
    sample,metadata,positions=fixture()
    metadata[0]["split"]="validation"
    with pytest.raises(ValueError,match="role/block"):
        window_from_rows(sample,metadata,positions)
