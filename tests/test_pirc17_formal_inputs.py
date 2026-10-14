"""Causal final inputs from synthetic files; never an actual ACCEPT01 receipt."""
from dataclasses import fields

import numpy as np
import pyarrow.parquet as pq
import pytest

from experiments.pirc17 import formal_eligibility as eligible
from experiments.pirc17 import formal_inputs as inputs
from experiments.pirc17 import protocol_core as core
from tests.test_pirc17_formal_eligibility import authorize_fixture, check, disk_fixture, metadata_fixture


def window_fixture():
    s, rows = metadata_fixture()
    _, window = check(s, rows)
    return window, np.array([[110., 35.], [110.001, 35.001], [110.002, 35.002]])


def test_causal_prefix_frames_utc_solar_and_truth_type_separation():
    window, lonlat = window_fixture()
    prefix = inputs.bind_final_prefix(window, lonlat, population_sha256="3"*64, condition_sha256="4"*64)
    assert prefix.split == "final_eval"
    assert prefix.terrain_origin.mode == prefix.method_origin.mode == "causal_prefix"
    assert prefix.terrain_origin.epoch_seconds == prefix.method_origin.epoch_seconds == 0
    assert prefix.origin_epoch_ns == window["payload"]["origin_epoch_ns"]
    assert prefix.terrain_origin.velocity_error_mps is None
    np.testing.assert_array_equal(prefix.terrain_origin.position_m, [0, 0])
    np.testing.assert_allclose(prefix.to_scoring_frame(prefix.method_origin.history_positions_m),
                               prefix.terrain_origin.history_positions_m, atol=1e-8)
    np.testing.assert_array_equal(prefix.score_seconds, [60, 300, 900, 1800])
    assert prefix.scoring_frame.longitude == lonlat[-1, 0]
    assert prefix.condition_at.frame.longitude == lonlat[0, 0]
    solar = prefix.condition_at(prefix.method_origin.position_m[None], 0.)
    assert np.isfinite(solar).all() and not np.any(solar == -999.)
    assert not {"targets", "target_positions", "future", "assignment"} & {f.name for f in fields(prefix)}
    assert not prefix.score_seconds.flags.writeable
    assert not prefix.method_origin.history_positions_m.flags.writeable
    assert prefix.identity()["split"] == "final_eval"


@pytest.mark.parametrize("change", ["whole-route", "polar", "nonfinite", "target-in-prefix", "changed-time", "role", "source-duplicate"])
def test_causal_constructor_rejects_injected_future_or_invalid_boundaries(change):
    window, lonlat = window_fixture()
    p = window["payload"]
    if change == "whole-route": lonlat = np.tile(lonlat, (3, 1))
    elif change == "polar": lonlat[-1, 1] = 89.
    elif change == "nonfinite": lonlat[0, 0] = np.nan
    elif change == "target-in-prefix": p["prefix"].append(dict(p["targets"][0]))
    elif change == "changed-time": p["targets"][0]["absolute_epoch_ns"] += 1
    elif change == "role": p["sample"]["split"] = "validation"
    elif change == "source-duplicate": p["targets"][0]["source_point_index"] = p["prefix"][0]["source_point_index"]
    window = core.envelope(p)
    with pytest.raises(ValueError):
        inputs.bind_final_prefix(window, lonlat, population_sha256="3"*64, condition_sha256="4"*64)


def qualified_fixture(root, monkeypatch, **kwargs):
    binding, release, snapshot = disk_fixture(root, **kwargs)
    authority = authorize_fixture(root, binding, monkeypatch)
    result = eligible.qualify_final_inputs(**authority, release=release, snapshot=snapshot, output_directory=root/"output")
    q = core.unpack(core.read_json(result["result_path"]))
    return dict(**authority, population_path=root/"output"/q["population_path"],
        population_sha256=q["population_sha256"], eligibility_path=root/"output"/q["eligibility_path"],
        qualification_path=result["result_path"], qualification_sha256=result["result_sha256"],
        release=release, data_root=root/"data")


def test_guarded_original_position_load_and_exact_selected_order(tmp_path, monkeypatch):
    args = qualified_fixture(tmp_path, monkeypatch)
    original = pq.read_table
    reads = []
    def projected(path, *, columns, **kwargs):
        assert columns == ["file_id", "t", "lon", "lat"]
        reads.append(str(path))
        return original(path, columns=columns, **kwargs)
    monkeypatch.setattr(pq, "read_table", projected)
    loaded = inputs.load_final_positions(**args)
    selected = core.unpack(core.read_json(args["population_path"]))["selection"]["selected"]
    assert [p.sample_id for p in loaded.prefixes] == [r["sample_id"] for r in selected]
    assert [t.sample_id for t in loaded.targets] == [p.sample_id for p in loaded.prefixes]
    assert len(reads) == len(set(reads)) == 3
    for prefix, truth in zip(loaded.prefixes, loaded.targets):
        assert truth.window_sha256 == prefix.window_sha256
        assert prefix.population_sha256 == loaded.population_sha256 == args["population_sha256"]
        assert truth.positions_m.shape == (4, 2) and not truth.positions_m.flags.writeable
        assert np.linalg.norm(truth.positions_m[-1]) > 0
    events = [core.unpack(core.read_json(p)) for p in (tmp_path/"journal").glob("*.json")]
    assert len(events) == 4
    assert sum(e["access_kind"] == "final_eval_positions" for e in events) == 2


def test_future_coordinate_changes_cannot_change_causal_initial_states(tmp_path, monkeypatch):
    results = []
    for i, shift in enumerate((0., .005)):
        root = tmp_path/str(i); root.mkdir()
        args = qualified_fixture(root, monkeypatch, future_shift=shift)
        results.append(inputs.load_final_positions(**args))
    a, b = results
    for first, second in zip(a.prefixes, b.prefixes):
        np.testing.assert_array_equal(first.terrain_origin.history_positions_m, second.terrain_origin.history_positions_m)
        np.testing.assert_array_equal(first.method_origin.history_positions_m, second.method_origin.history_positions_m)
        np.testing.assert_array_equal(first.method_origin.velocity_mps, second.method_origin.velocity_mps)
        assert first.condition_at.identity() == second.condition_at.identity()
    assert any(not np.array_equal(t.positions_m, u.positions_m) for t, u in zip(a.targets, b.targets))


@pytest.mark.parametrize("missing", ["approval_sha256", "population_sha256", "population_path", "eligibility_path"])
def test_positions_cannot_open_without_exact_approval_and_population(tmp_path, monkeypatch, missing):
    args = qualified_fixture(tmp_path, monkeypatch)
    before = len(list((tmp_path/"journal").glob("*.json")))
    args[missing] = None
    monkeypatch.setattr(inputs, "_release_sources", lambda *a: pytest.fail("unapproved positions loader invoked"))
    with pytest.raises(ValueError):
        inputs.load_final_positions(**args)
    assert len(list((tmp_path/"journal").glob("*.json"))) == before


@pytest.mark.parametrize("mutation,match", [("condition-short", "out of range"),
    ("condition-bad-file", "another source identity"), ("condition-bad-truth", "nonpolar"),
    ("condition-origin-time", "UTC timestamp join"), ("condition-target-time", "UTC timestamp join"),
    ("condition-nontime", "datetime UTC")])
def test_selected_data_failure_is_not_replaced_or_silently_dropped(tmp_path, monkeypatch, mutation, match):
    args = qualified_fixture(tmp_path, monkeypatch, mutation=mutation)
    population_before = args["population_path"].read_bytes()
    with pytest.raises(ValueError, match=match):
        inputs.load_final_positions(**args)
    assert args["population_path"].read_bytes() == population_before
    events = [core.unpack(core.read_json(p)) for p in (tmp_path/"journal").glob("*.json")]
    assert any(e["event"] == "failed" and e["access_kind"] == "final_eval_positions" and e["possible_reads_retained"] for e in events)


@pytest.mark.parametrize("which", ["window", "condition", "qualification-hash"])
def test_position_artifact_tampering_fails_before_use(tmp_path, monkeypatch, which):
    args = qualified_fixture(tmp_path, monkeypatch)
    if which == "window":
        path = next((tmp_path/"output"/"windows").glob("*.json"))
        value = core.read_json(path)
        value["payload"]["prefix"][0]["source_point_index"] += 1
        path.write_bytes(core.canonical(core.envelope(value["payload"])))
    elif which == "condition":
        path = next((tmp_path/"data"/"cond_slices").glob("*.parquet"))
        path.write_bytes(path.read_bytes()+b" ")
    else:
        args["qualification_sha256"] = "0"*64
    with pytest.raises(ValueError, match="identity|changed"):
        inputs.load_final_positions(**args)
