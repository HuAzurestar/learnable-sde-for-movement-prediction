"""Software-only geometry and ledger fixtures; not research observations."""
import copy
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import duckdb
import numpy as np
import pytest

from trajectory.batched_terrain import BatchedLineIndex
from trajectory.linear_materialization import _SEARCH_RADII_M
from trajectory.multicell_terrain import MultiCellRawMapQuery
from experiments.pirc17 import geometry_transport_probe as probe
from experiments.pirc17.development_rollout import save_particle_artifact
from experiments.pirc17.features import LocalFrame


@pytest.fixture
def db():
    connection = duckdb.connect()
    connection.execute("LOAD spatial")
    connection.execute("SET threads=2")
    connection.execute("CREATE TABLE source_lines(subtype VARCHAR,class VARCHAR,geom GEOMETRY)")
    yield connection
    connection.close()


def test_resident_geometry_matches_every_reference_field_with_ties_and_cutoff(db):
    db.executemany("INSERT INTO source_lines VALUES(?,?,ST_GeomFromText(CAST(? AS VARCHAR)))", [
        ("road", "residential", "LINESTRING(25 -10,25 20)"),
        ("road", "residential", "LINESTRING(-25 -10,-25 20)"),
        ("road", "footway", "LINESTRING(1 -10,1 20)"),
        ("road", None, "LINESTRING(4000 0,4100 100)"),
        ("river", None, "MULTILINESTRING((0 50,10 50),(0 70,10 70))"),
    ])
    maximum = max(_SEARCH_RADII_M)
    points = np.array([(0, 0), (25, 0), (4050, 50), (maximum+25, 0),
                       (maximum+25+1e-4, 0), (900000, 0)])/100000
    pids = [90, 5, 7, 44, 23, 1]
    old = BatchedLineIndex(db, (100000., 100000., 0., 0.))
    new = probe.ResidentLineIndex(old)
    for _ in range(3):
        assert new.query(points, pids) == old.query(points, pids)
    assert next(r for r in new.query(points, pids) if r[:2] == (90, "road"))[3] == -1
    assert new.geometry_count == 4
    with pytest.raises(ValueError, match="already owned"):
        probe.ResidentLineIndex(old)
    assert new.query(points, pids) == old.query(points, pids)


def test_random_curves_nonzero_projection_are_exact(db):
    rng = np.random.default_rng(20260927)
    for i in range(48):
        xy = rng.normal(0, 3000, (15, 2))
        wkt = "LINESTRING("+",".join(f"{x:.14g} {y:.14g}" for x, y in xy)+")"
        db.execute("INSERT INTO source_lines VALUES(?,?,ST_GeomFromText(CAST(? AS VARCHAR)))",
                   ["road" if i % 2 else "river", "residential", wkt])
    projection = (73000., 111000., 15., 52.)
    old = BatchedLineIndex(db, projection)
    new = probe.ResidentLineIndex(old)
    points = rng.normal(0, 5000, (100, 2))/projection[:2]+projection[2:]
    assert new.query(points, list(range(100))) == old.query(points, list(range(100)))


def test_empty_family_and_rejected_queries(db):
    new = probe.ResidentLineIndex(BatchedLineIndex(db, (100000., 100000., 0., 0.)))
    assert new.query([[0, 0]], [4]) == []
    assert new.query(np.empty((0, 2)), []) == []
    with pytest.raises(ValueError, match="unique"):
        new.query([[0, 0], [0, 0]], [1, 1])
    with pytest.raises(ValueError, match="finite"):
        new.query([[np.nan, 0]], [1])
    with pytest.raises(ValueError, match="finite"):
        new.query([[0, 0]], [1, 2])


def test_exact_radius_boundary_with_single_line(db):
    db.execute("INSERT INTO source_lines VALUES ('road','residential',ST_GeomFromText('LINESTRING(0 -1,0 1)'))")
    old = BatchedLineIndex(db, (100000., 100000., 0., 0.))
    new = probe.ResidentLineIndex(old)
    maximum = max(_SEARCH_RADII_M)
    points = np.array([(maximum, 0), (maximum+1e-4, 0), (0, 0)])/100000
    result = new.query(points, [0, 1, 2])
    assert result == old.query(points, [0, 1, 2])
    assert [r[0] for r in result] == [0, 2]


@pytest.fixture
def linear_asset(tmp_path, db):
    import hashlib
    path = tmp_path/"overture.parquet"
    db.execute("COPY (SELECT 'road' subtype,'residential' AS class,"
               "ST_GeomFromText('LINESTRING(0.01 0,0.01 0.1)') geometry) TO ? (FORMAT PARQUET)", [str(path)])
    asset = {"asset_id": "software-fixture", "local_path": path.name, "data_kind": "raw", "status": "valid",
        "version": "software-v1", "spatial_extent": {"bbox": [0, 0, 3, 3], "crs": "EPSG:4326"},
        "checksum": {"algorithm": "sha256", "value": hashlib.sha256(path.read_bytes()).hexdigest()}}
    receipt = tmp_path/"receipt.json"
    receipt.write_text(json.dumps({"assets": [asset]}), encoding="utf-8")
    return path, receipt


def test_complete_provider_reuses_evicts_and_revalidates_original_assets(tmp_path, linear_asset):
    path, receipt = linear_asset
    old = MultiCellRawMapQuery(tmp_path, [receipt], max_cells=1)
    new = probe.ProbeMap(tmp_path, [receipt], max_cells=1)
    try:
        for points in ([[0, .05], [.009, .05]], [[4, .05]], [[0, .05]], [[0, .05]]):
            result = new(points)
            assert result == old(points)
            if points[0][0] < 3:
                assert result[0]["overture_road_status"] == "valid"
        assert len(new.resident_builds) == 3
        assert new.geometry_cache_measurements["retained_cells"] == 1
        assert new.geometry_cache_measurements["cell_evictions"] == 2
        assert [r["geometry_count"] for r in new.resident_builds] == [1, 0, 1]
        assert new.identity["reference_query_version"] == old.identity["query_version"]
        assert new.identity["query_version"] != old.identity["query_version"]
        path.write_bytes(b"changed software fixture")
        with pytest.raises(ValueError, match="hash mismatch"):
            new([[0, .05]])
    finally:
        old.close()
        new.close()


def test_shared_modes_use_one_original_index_and_recompute_changed_positions(tmp_path, linear_asset, monkeypatch):
    _, receipt = linear_asset
    shared = probe.SharedProbeMap(tmp_path, [receipt])
    oracle = MultiCellRawMapQuery(tmp_path, [receipt])
    try:
        first = [[0, .05]]
        initial = shared.query_mode("reference", first)
        assert initial == oracle(first) and initial[0]["overture_road_status"] == "valid"
        original = shared.line_index
        owner = shared.geometry_cells[0, 0]
        assert owner.line_index is original and shared.resident_builds == []
        calls, original_query = [], original.query

        def tracked_original(points, pids):
            calls.append(shared._active_backend)
            return original_query(points, pids)

        monkeypatch.setattr(original, "query", tracked_original)
        for mode, points in (("candidate", first), ("reference", [[.009, .05]]),
                             ("candidate", [[.005, .02]]), ("reference", first), ("candidate", first)):
            actual = shared.query_mode(mode, points)
            assert actual == oracle(points)
            assert shared.geometry_cells[0, 0] is owner
            assert owner.line_index.reference is original
            assert shared.connection is owner.connection is original.connection
            if points == [[.009, .05]]:
                assert actual[0]["road_distance_m"] != initial[0]["road_distance_m"]
        assert calls == ["reference", "reference"]
        assert len(shared.resident_builds) == 1
        assert shared.resident_builds[0]["geometry_count"] == 1
        assert shared.geometry_cache_measurements["cell_builds"] == 1
        assert shared.identity["query_answer_cache"] is False
        assert shared.identity["provider_layout"] == "shared"
    finally:
        shared.close()
        oracle.close()
    assert shared.line_index is None and not shared.geometry_cells


@pytest.mark.parametrize("changed_mode", ["reference", "candidate"])
def test_shared_eviction_and_both_modes_revalidate_sources(tmp_path, linear_asset, changed_mode):
    path, receipt = linear_asset
    shared = probe.SharedProbeMap(tmp_path, [receipt], max_cells=1)
    oracle = MultiCellRawMapQuery(tmp_path, [receipt], max_cells=1)
    try:
        for mode, points in (("candidate", [[0, .05]]), ("reference", [[4, .05]]),
                             ("reference", [[0, .05]]), ("candidate", [[0, .05]])):
            assert shared.query_mode(mode, points) == oracle(points)
        assert len(shared.resident_builds) == 2
        assert shared.geometry_cache_measurements["cell_builds"] == 3
        assert shared.geometry_cache_measurements["cell_evictions"] == 2
        path.write_bytes(b"changed software fixture")
        with pytest.raises(ValueError, match="hash mismatch"):
            shared.query_mode(changed_mode, [[0, .05]])
        assert shared._active_backend is None
        assert shared.geometry_cache_measurements["cell_failures"] == 1
        assert not shared.geometry_cells and shared.line_index is None
    finally:
        shared.close()
        oracle.close()


def test_shared_requires_explicit_nonreentrant_mode_and_restores_it_on_failure(tmp_path, monkeypatch):
    shared = probe.SharedProbeMap(tmp_path, [])
    try:
        with pytest.raises(RuntimeError, match="explicit backend"):
            shared(np.empty((0, 2)))
        with pytest.raises(RuntimeError, match="explicit query mode"):
            shared._linear_cell(0, 0)
        with pytest.raises(ValueError, match="explicit reference or candidate"):
            shared.query_mode("typo", np.empty((0, 2)))
        with pytest.raises(ValueError, match="finite"):
            shared.query_mode("candidate", [[np.nan, 0]])
        assert shared._active_backend is None and shared.resident_builds == []
        with monkeypatch.context() as nested:
            nested.setattr(shared, "_catalogs", lambda: shared.query_mode("reference", [[0, 0]]))
            with pytest.raises(RuntimeError, match="one synchronous"):
                shared.query_mode("candidate", [[0, 0]])
        assert shared._active_backend is None
        assert shared.query_mode("reference", np.empty((0, 2))) == []
    finally:
        shared.close()


def test_shared_raster_cache_and_new_parent_keep_exact_original_fields(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    path = tmp_path/"worldcover_ESA_WorldCover_N00E000.tif"
    values = np.full((16, 16), 10, dtype=np.uint8)
    values[:, 8:] = 20
    values[0, 0] = 0
    with rasterio.open(path, "w", driver="GTiff", width=16, height=16, count=1,
                       dtype="uint8", crs="EPSG:4326", transform=from_origin(0, 3, 3/16, 3/16),
                       tiled=True, blockxsize=16, blockysize=16, nodata=0) as raster:
        raster.write(values, 1)
    parent = f"registered-file:{path.name}:sha256:{probe._hash(path)}"
    shared, oracle = probe.SharedProbeMap(tmp_path, []), MultiCellRawMapQuery(tmp_path, [])
    try:
        assert shared.query_mode("candidate", [[.5, .5]])[0]["worldcover_status"] == "source_missing"
        assert shared.geometry_cache_measurements["retained_cells"] == 1
        for provider in (shared, oracle):
            provider.register_snapshot_parent(parent)
            assert not provider.geometry_cells
        points = [[.1, 2.9], [1.5, 1.5], [.5, .5], [3.1, .5]]
        for mode in ("reference", "candidate", "reference", "candidate"):
            actual = shared.query_mode(mode, points)
            assert actual == oracle(points)
            assert [r["worldcover_status"] for r in actual] == ["nodata", "valid", "valid", "source_missing"]
            assert [r["worldcover_class"] for r in actual[:3]] == [0, 20, 10]
        assert shared.worldcover_cache.reads == shared.worldcover_cache.opens == 1
        assert shared.worldcover_cache.hits == 3
        assert len(shared.resident_builds) == 3  # One invalidated cell, then two retained cells.
    finally:
        shared.close()
        oracle.close()
    assert shared.worldcover_cache.retained_bytes == 0 and not shared.worldcover_cache.readers


def query(points):
    return [{"value": float(x), "status": "valid"} for x, _ in points]


def batches():
    return [{"identity": {"batch": i}, "lonlat": np.array([[1., 2.], [3., 4.]])} for i in range(2)]


def test_benchmark_keeps_every_call_and_alternates_timed_order():
    records = []
    result = probe.benchmark(batches(), query, query, records.append)
    assert result["expected_call_count"] == result["attempted_call_count"] == 32
    assert result["all_calls_exact"] and result["certified"] is False
    assert len([r for r in records if r["phase"] == "warmup"]) == 8
    timed = [r for r in records if r["phase"] == "timed" and r["batch"] == {"batch": 0}]
    assert [r["backend"] for r in timed[:4]] == ["reference", "candidate", "candidate", "reference"]
    assert all(r["exactly_equal"] and r["output_row_count"] == 2 for r in records)


@pytest.mark.parametrize("fault", ["other_field", "nan", "exception", "lost_rows"])
def test_mismatch_and_failures_do_not_shrink_denominator(fault):
    def bad(points):
        rows = query(points)
        if fault == "other_field":
            rows[0]["status"] = "source_missing"
        elif fault == "nan":
            rows[0]["value"] = float("nan")
        elif fault == "exception":
            raise RuntimeError("software failure")
        else:
            return []
        return rows
    records = []
    result = probe.benchmark(batches(), query, bad, records.append)
    assert result["expected_call_count"] == result["attempted_call_count"] == 32
    assert not result["all_calls_exact"]
    assert result["mismatch_count" if fault == "other_field" else "failure_count"] == 16
    assert len(records) == 32


def test_resource_stop_retains_original_denominator():
    remaining = iter([probe.MINIMUM_FREE_BYTES+1, 0])
    records = []
    result = probe.benchmark(batches(), query, query, records.append, available_memory=lambda: next(remaining))
    assert result["resource_stopped"] and result["expected_call_count"] == 32
    assert result["attempted_call_count"] == len(records) == 1
    assert not result["all_calls_exact"]


def ledger():
    header = {"type": "header", "schema_version": "pirc17-development-rollout-v1",
        "purpose": "bounded_validation_engineering_pilot", "final_eval_label_prediction_metric_reads": 0,
        "sample_ids": ["software-a", "software-b", "software-c"], "configurations": ["base", "all-terrain"],
        "seeds": [20260814], "particle_counts": [512, 1024], "max_steps_seconds": [.625, .3125],
        "map_backend": "multicell", "selection_policy": "lexical_independent_blocks", "save_particles": True,
        "expected_run_count": 24, "brownian_driver": "pirc17-coupled-brownian-v2"}
    rows = [header]
    for sid, config, n, step in itertools.product(header["sample_ids"], header["configurations"],
                                                header["particle_counts"], header["max_steps_seconds"]):
        rows.append({"type": "run", "sample_id": sid, "configuration": config, "seed": 20260814,
            "particles": n, "max_step_seconds": step, "status": "success", "independent_block_id": sid,
            "actual_horizons_seconds": [1., 2., 3., 4.],
            "brownian_identity": {"version": header["brownian_driver"], "seed": 20260814,
                                  "max_particles": 1024, "stream_id": sid},
            "scores": {"time_weighted_energy_score_m": 0., "by_time": [
                {"elapsed_seconds": float(t), "energy_score_m": 0.} for t in range(1, 5)]}})
    return rows+[{"type": "completion", "attempted_run_count": 24, "success_count": 24, "failure_count": 0}]


@pytest.mark.parametrize("fault", ["incomplete", "missing", "failed", "final_eval", "settings", "blocks"])
def test_probe_rejects_changed_or_incomplete_original_ledger(fault):
    rows = ledger()
    if fault == "incomplete":
        rows.pop()
    elif fault == "missing":
        rows.pop(1)
    elif fault == "failed":
        rows[1]["status"] = "failure"
        rows[-1].update(success_count=23, failure_count=1)
    elif fault == "final_eval":
        rows[0]["final_eval_label_prediction_metric_reads"] = 1
    elif fault == "settings":
        rows[0]["map_backend"] = "legacy"
    else:
        for row in rows[1:-1]:
            row["independent_block_id"] = "one-software-block"
    with pytest.raises(ValueError):
        probe.select_probe_rows(rows)


def test_all_endpoint_particles_preserve_validation_frames_and_targets(tmp_path):
    selected = probe.select_probe_rows(ledger())
    windows = []
    times, targets = np.array([1., 2., 3., 4.]), np.zeros((4, 2))
    for i, row in enumerate(selected):
        row["particle_artifact"] = save_particle_artifact(tmp_path/"fixture.jsonl", row,
            np.zeros((1024, 4, 2)), targets, times)
        windows.append(SimpleNamespace(sample_id=row["sample_id"], block_id=row["independent_block_id"],
            role="validation", frame=LocalFrame(10.+i, 50.+i), horizon_seconds=times, target_positions_m=targets))
    result = probe.endpoint_batches(selected, tmp_path, windows)
    assert len(result) == 12 and all(len(b["lonlat"]) == 1024 for b in result)
    np.testing.assert_array_equal(result[0]["lonlat"], np.tile([10., 50.], (1024, 1)))
    for field, changed in (("role", "final_eval"), ("block_id", "wrong-block"),
                           ("target_positions_m", np.ones((4, 2)))):
        damaged = copy.deepcopy(windows)
        setattr(damaged[0], field, changed)
        with pytest.raises(ValueError, match="validation frame"):
            probe.endpoint_batches(selected, tmp_path, damaged)


def cli_args(tmp_path):
    return ["probe", *sum((["--"+name, str(tmp_path/name)] for name in
        ("ledger", "eligibility", "release", "snapshot", "data-root", "output")), []),
        "--ledger-sha256", "software-hash", "--eligibility-sha256", "software-hash"]


def test_cli_never_overwrites_and_preserves_preflight_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", cli_args(tmp_path))
    monkeypatch.setattr(probe, "prepare_batches", lambda _: (_ for _ in ()).throw(ValueError("changed source")))
    with pytest.raises(SystemExit) as error:
        probe.main()
    assert error.value.code == 1
    output = tmp_path/"output"
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[-1]["expected_call_count"] == 192 and rows[-1]["attempted_call_count"] == 0
    assert not rows[-1]["all_calls_exact"] and rows[-1]["error_message"] == "changed source"
    before = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        probe.main()
    assert error.value.code == 2 and output.read_bytes() == before


@pytest.fixture
def cli_system(tmp_path, monkeypatch):
    """CLI wiring only; geometric equivalence is exercised with actual cells above."""
    import psutil
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=4*1024**3))
    instances = []

    class Provider:
        def __init__(self, root, receipts):
            assert root.is_absolute() and all(p.is_absolute() for p in receipts)
            self.query_cwds, self.close_cwds, self.parents = [], [], []
            self.resident_builds = []
            self.identity = {"receipt_sha256": ["software-receipt"]}
            self.geometry_cache_measurements = {"retained_cells": 0}
            instances.append(self)

        def register_snapshot_parent(self, parent):
            self.parents.append(parent)

        def __call__(self, points):
            self.query_cwds.append(Path.cwd())
            return query(points)

        def query_mode(self, backend, points):
            assert backend in ("reference", "candidate")
            return self(points)

        def close(self):
            self.close_cwds.append(Path.cwd())

    def prepare(args):
        assert Path.cwd() == tmp_path
        assert all(getattr(args, name).is_absolute() for name in
                   ("ledger", "eligibility", "release", "snapshot", "data_root", "output"))
        return ([{"identity": {"batch": i, "point_count": 1024},
                  "lonlat": np.tile([1., 2.], (1024, 1))} for i in range(12)],
                ["software-parent"], [{"source_sha256": {}, "map_source_sha256": {},
                  "development_identity": "software-identity"},
                  {"maps": {"receipt_sha256": ["software-receipt"]}}])

    for name in ("MultiCellRawMapQuery", "ProbeMap", "SharedProbeMap"):
        monkeypatch.setattr(probe, name, Provider)
    monkeypatch.setattr(probe, "prepare_batches", prepare)
    return instances, Provider


@pytest.mark.parametrize("layout", ["independent", "shared"])
def test_cli_layout_keeps_full_calls_and_closes_each_owner_before_restoring_cwd(tmp_path, monkeypatch, cli_system, layout):
    instances, _ = cli_system
    argv = cli_args(Path("."))  # All supplied paths are intentionally relative.
    if layout == "shared":
        argv += ["--provider-layout", layout, "--scratch-directory", "scratch"]
    monkeypatch.setattr(sys, "argv", argv)
    probe.main()
    assert Path.cwd() == tmp_path
    expected_cwd = tmp_path/"scratch" if layout == "shared" else tmp_path
    assert len(instances) == (1 if layout == "shared" else 2)
    assert sum(len(p.query_cwds) for p in instances) == 192
    for owner in instances:
        assert owner.parents == ["software-parent"]
        assert owner.close_cwds == [expected_cwd]
        assert set(owner.query_cwds) == {expected_cwd}
    rows = [json.loads(line) for line in (tmp_path/"output").read_text().splitlines()]
    assert len(rows) == 196
    assert rows[-1]["all_calls_exact"] and rows[-1]["certified"] is False
    assert rows[-1]["attempted_call_count"] == rows[-1]["expected_call_count"] == 192
    for row in (rows[0], rows[1], rows[-2]):
        assert row["provider_layout"] == layout
        assert row["physical_map_owner_count"] == len(instances)
        assert row["maximum_retained_geometry_cells"] == 4*len(instances)
    if layout == "shared":
        assert "shared_cells" in rows[-2] and "reference_cells" not in rows[-2]
        assert rows[1]["scratch_directory"] == str(expected_cwd)
    else:
        assert "reference_cells" in rows[-2] and "shared_cells" not in rows[-2]
        assert rows[1]["scratch_directory"] is None


@pytest.mark.parametrize("fault", ["missing", "existing", "parent", "outside", "output"])
def test_cli_rejects_unsafe_shared_scratch_before_creating_evidence(tmp_path, monkeypatch, cli_system, fault):
    instances, _ = cli_system
    argv = cli_args(tmp_path)+["--provider-layout", "shared"]
    if fault != "missing":
        path = {"existing": tmp_path/"scratch", "parent": tmp_path,
                "outside": tmp_path.parent/"external-scratch", "output": tmp_path/"output"}[fault]
        if fault == "existing":
            path.mkdir()
        argv += ["--scratch-directory", str(path)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as error:
        probe.main()
    assert error.value.code == 2 and instances == []
    assert not (tmp_path/"output").exists() and Path.cwd() == tmp_path


@pytest.mark.parametrize("fault", ["receipt", "close", "candidate_constructor"])
def test_cli_failure_records_denominator_and_restores_cwd_after_cleanup(tmp_path, monkeypatch, cli_system, fault):
    instances, provider = cli_system
    layout = "independent" if fault == "candidate_constructor" else "shared"
    if fault == "receipt":
        monkeypatch.setattr(provider, "register_snapshot_parent",
                            lambda self, _: self.identity.update(receipt_sha256=["changed"]))
    elif fault == "close":
        close = provider.close

        def fail_close(self):
            close(self)
            raise RuntimeError("software cleanup failure")

        monkeypatch.setattr(provider, "close", fail_close)
    else:
        def fail_init(*args):
            raise RuntimeError("software candidate initialization failure")

        monkeypatch.setattr(probe, "ProbeMap", fail_init)
    monkeypatch.setattr(sys, "argv", cli_args(tmp_path)+[
        "--provider-layout", layout, "--scratch-directory", str(tmp_path/"scratch")])
    with pytest.raises(SystemExit) as error:
        probe.main()
    assert error.value.code == 1 and Path.cwd() == tmp_path
    assert len(instances) == 1 and instances[0].close_cwds == [tmp_path/"scratch"]
    rows = [json.loads(line) for line in (tmp_path/"output").read_text().splitlines()]
    assert rows[-1]["expected_call_count"] == 192
    assert rows[-1]["attempted_call_count"] == (192 if fault == "close" else 0)
    assert not rows[-1]["all_calls_exact"] and rows[-1]["certified"] is False
    assert "error_type" in rows[-1]
