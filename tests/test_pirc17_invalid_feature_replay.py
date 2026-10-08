"""Software-only observer and failure fixtures, not research observations."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.nex326.pirc21_adapter import FeatureSelection, FeatureSnapshotAdapter
from experiments.pirc17 import invalid_feature_replay as diagnostic
from experiments.pirc17.brownian import BrownianPath, integration_grid
from experiments.pirc17.features import CanonicalEncoder, LocalFrame, PredictedPositionFeatures
from experiments.pirc17.origins import known_velocity
from experiments.pirc17.rollout import Features, PredictedState, rollout
from tests.test_pirc17_geometry_transport_probe import ledger
from tests.test_pirc21_adapter import _write_snapshot


def state():
    return PredictedState(2.5, np.array([[1., 0.], [2., 0.], [3., 0.]]), np.ones((3, 2)),
        np.array([[[0., 0.], [1., 0.]], [[0., 0.], [2., 0.]], [[3., 0.], [3., 0.]]]),
        np.array([-2.5, 2.5]))


def test_observer_preserves_objects_inputs_and_nonfinite_invalid_values():
    features = Features(np.array([[1., 2.], [np.nan, 3.], [np.inf, 0.]]),
                        np.array([[True, True], [False, True], [False, False]]))
    source_rows = [{"road_distance_m": float(i), "overture_road_status": "valid",
                    "source_version": "software-fixture"} for i in range(3)]
    source_rows[1]["road_distance_m"] = float("nan")
    calls, queried, events = [], [], []

    def encode(rows):
        calls.append(rows)
        return features

    def query(points):
        queried.append(points.copy())
        return source_rows

    encoder = SimpleNamespace(columns=["road", "history"], encode=encode)
    observer = diagnostic.FeatureObserver(encoder, query, LocalFrame(0., 0.), events.append)
    original_state = state()
    positions, history = original_state.positions_m.copy(), original_state.history_positions_m.copy()
    before_values, before_valid = features.values.copy(), features.valid.copy()
    assert observer(original_state) is features
    assert len(calls) == len(queried) == 1
    assert "historical_motion_status" not in source_rows[0]
    np.testing.assert_array_equal(features.values, before_values)
    np.testing.assert_array_equal(features.valid, before_valid)
    np.testing.assert_array_equal(original_state.positions_m, positions)
    np.testing.assert_array_equal(original_state.history_positions_m, history)
    assert observer.state is None
    assert observer.query_times == [2.5]
    assert observer.query_rows == 3 and observer.invalid_rows == observer.event_count == 2
    assert [e["particle_index"] for e in events] == [1, 2]
    assert events[0]["invalid_columns"] == ["road"]
    assert events[1]["invalid_column_indexes"] == [0, 1]
    assert events[0]["canonical_row"]["road_distance_m"] == {"nonfinite": "nan"}
    assert events[1]["encoded_values"][0] == {"nonfinite": "inf"}
    assert events[1]["canonical_row"]["historical_motion_status"] == "insufficient_history"
    assert events[1]["predicted_history_positions_m"] == [[3., 0.], [3., 0.]]
    assert events[0]["elapsed_seconds"] == 2.5 and events[0]["query_index"] == 0
    json.dumps(events, allow_nan=False)


@pytest.mark.parametrize("fault", ["encoder", "mask", "shape", "reentry", "logging"])
def test_observer_resets_context_and_never_hides_failures(fault):
    def encode(rows):
        if fault == "encoder":
            raise ValueError("software encoding failure")
        if fault == "reentry":
            observer(state())
        shape = (3, 2) if fault == "shape" else (3, 1)
        valid = np.zeros(shape, dtype=int if fault == "mask" else bool)
        return Features(np.zeros(shape), valid)

    def emit(_):
        if fault == "logging":
            raise OSError("software logging failure")

    observer = diagnostic.FeatureObserver(SimpleNamespace(columns=["x"], encode=encode),
                                         lambda _: [{}, {}, {}], LocalFrame(0., 0.), emit)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        observer(state())
    assert observer.state is None and observer.event_count == 0
    with pytest.raises(RuntimeError, match="exactly one encode"):
        observer.encode([{}])


def test_observer_checks_memory_before_call_and_reports_periodic_counts():
    remaining, calls, events = [0], [], []

    def encode(rows):
        calls.append(rows)
        return Features(np.zeros((3, 1)), np.ones((3, 1), dtype=bool))

    observer = diagnostic.FeatureObserver(SimpleNamespace(columns=["x"], encode=encode),
        lambda _: [{}, {}, {}], LocalFrame(0., 0.), events.append, available_memory=lambda: remaining[0])
    with pytest.raises(MemoryError, match="2 GiB"):
        observer(state())
    assert not calls and observer.completed_calls == 0
    remaining[0] = diagnostic.probe.MINIMUM_FREE_BYTES
    for index in range(128):
        observer(replace(state(), elapsed_seconds=float(index)))
    assert len(calls) == 128 and len(events) == 1
    assert events[0]["type"] == "progress" and events[0]["feature_query_rows"] == 384
    assert events[0]["invalid_feature_rows"] == events[0]["recorded_invalid_events"] == 0
    assert observer.query_times == list(range(128))


@pytest.fixture
def fixture_inputs(tmp_path):
    adapter = FeatureSnapshotAdapter(_write_snapshot(tmp_path), FeatureSelection(
        variant_ids=("road.distance_log1p", "road.direction", "history.relative_road")))
    encoder = CanonicalEncoder(adapter)
    frame = LocalFrame(0., 0.)
    origin = known_velocity([0, 0], 0, [1, 0], source="software-fixture")
    times = np.array([1., 2.])
    driver = BrownianPath(times, [.5], history_step_seconds=1., particles=4, seed=8, stream_id="software-fixture")

    def query(points):
        return [{"road_distance_m": 5., "road_direction_east": float(i != 1), "road_direction_north": 0.,
                 "overture_road_status": "valid", "lon": float(x), "lat": float(y)}
                for i, (x, y) in enumerate(points)]

    model = SimpleNamespace(base_drift=lambda s: s.velocities_mps,
        diffusion=lambda s: np.broadcast_to(np.eye(2)*.1, (len(s.positions_m), 2, 2)),
        correction=lambda s, x: np.column_stack((x[:, 0]*.01, np.zeros(len(x)))))
    kwargs = dict(particles=4, seed=8, max_step_seconds=.5, history_step_seconds=1.,
                  base_drift=model.base_drift, diffusion=model.diffusion, conditioner=model.correction,
                  brownian_increments=driver)
    original = rollout(origin, times, terrain=PredictedPositionFeatures(encoder, query, frame), **kwargs)
    row = {"sample_id": "software-fixture", "configuration": "all-terrain", "seed": 8,
           "particles": 4, "max_step_seconds": .5, "independent_block_id": "software-block",
           "invalid_feature_rows": original.invalid_feature_rows, "feature_query_rows": original.feature_query_rows,
           "particle_artifact": {"path": "software-only.npz", "sha256": "software-hash"}}
    prepared = SimpleNamespace(encoder=encoder, window=SimpleNamespace(origin=origin, horizon_seconds=times, frame=frame),
        model=model, driver=driver, row=row, reference_positions=original.positions_m, reference_times=times,
        query_times=integration_grid(times, .5, 1.)[:-1], parents=["software-parent"],
        rows=[{"source_sha256": {}, "map_source_sha256": {}, "development_identity": "software"},
              {"maps": {"receipt_sha256": {"software": "hash"}}}])
    return prepared, query, kwargs


def test_actual_transforms_and_all_saved_coordinates_are_unchanged(fixture_inputs):
    prepared, query, kwargs = fixture_inputs
    events = []
    observer = diagnostic.FeatureObserver(prepared.encoder, query, prepared.window.frame, events.append)
    prediction = rollout(prepared.window.origin, prepared.reference_times, terrain=observer, **kwargs)
    assert prediction.invalid_feature_rows > 0
    assert diagnostic.compare_prediction(prediction, prepared, observer)["exact_replay"]
    assert observer.event_count == prediction.invalid_feature_rows
    assert all(e["particle_index"] == 1 and e["canonical_row"]["overture_road_status"] == "valid" for e in events)
    assert any("road" in c for e in events for c in e["invalid_columns"])


@pytest.mark.parametrize("fault", ["coordinate", "time", "count", "events", "grid", "shape", "nan"])
def test_complete_replay_rejects_every_kind_of_partial_or_changed_evidence(fixture_inputs, fault):
    prepared, query, kwargs = fixture_inputs
    observer = diagnostic.FeatureObserver(prepared.encoder, query, prepared.window.frame, lambda _: None)
    prediction = rollout(prepared.window.origin, prepared.reference_times, terrain=observer, **kwargs)
    if fault in ("coordinate", "nan"):
        damaged = prediction.positions_m.copy()
        damaged[0, 0, 0] = float("nan") if fault == "nan" else damaged[0, 0, 0]+1e-12
        prediction = replace(prediction, positions_m=damaged)
    elif fault == "time":
        prediction = replace(prediction, elapsed_seconds=np.array([1., 3.]))
    elif fault == "count":
        prediction = replace(prediction, feature_query_rows=prediction.feature_query_rows-1)
    elif fault == "events":
        observer.event_count -= 1
    elif fault == "grid":
        observer.query_times[1] += .01
    else:
        prediction = replace(prediction, positions_m=prediction.positions_m[:-1])
    result = diagnostic.compare_prediction(prediction, prepared, observer)
    assert not result["exact_replay"] and result["certified"] is False
    json.dumps(result, allow_nan=False)


def original_ledger_fixture():
    rows = ledger()
    rows[0]["physical_history_step_seconds"] = 5.
    rows[0]["sample_ids"][1] = diagnostic.SAMPLE_ID
    for row in rows[1:-1]:
        if row["sample_id"] == "software-b":
            row["sample_id"] = diagnostic.SAMPLE_ID
            row["brownian_identity"]["stream_id"] = diagnostic.SAMPLE_ID
        row["actual_horizons_seconds"] = [60., 300., 900., 1800.]
        for score, elapsed in zip(row["scores"]["by_time"], row["actual_horizons_seconds"]):
            score["elapsed_seconds"] = elapsed
        row["invalid_feature_rows"] = 0
        row["feature_query_rows"] = diagnostic.EXPECTED_QUERY_ROWS
    chosen = next(r for r in rows[1:-1] if r["sample_id"] == diagnostic.SAMPLE_ID
                  and r["configuration"] == "all-terrain" and r["particles"] == 1024 and r["max_step_seconds"] == .3125)
    chosen["invalid_feature_rows"] = diagnostic.EXPECTED_INVALID_ROWS
    return rows, chosen


@pytest.mark.parametrize("fault", [None, "incomplete", "other_invalid", "missing", "horizons", "counts", "clock"])
def test_selector_requires_full_original_ledger_and_registered_problem(fault):
    rows, chosen = original_ledger_fixture()
    if fault is None:
        assert diagnostic.select_workload(rows) is chosen
        return
    if fault == "incomplete":
        rows.pop()
    elif fault == "other_invalid":
        rows[1]["invalid_feature_rows"] = 1
    elif fault == "missing":
        rows.remove(chosen)
    elif fault == "horizons":
        chosen["actual_horizons_seconds"][0] = 59.
    elif fault == "counts":
        chosen["invalid_feature_rows"] = 50
    else:
        rows[0]["physical_history_step_seconds"] = 1.
    with pytest.raises(ValueError):
        diagnostic.select_workload(rows)


@pytest.mark.parametrize("fault", [None, "source", "fit_hash", "model_seed", "population", "selection",
                                  "role", "target", "times", "noise", "incomplete_audit"])
def test_preflight_binds_complete_evidence_model_population_targets_and_noise(tmp_path, monkeypatch, fault):
    rows, chosen = original_ledger_fixture()
    query_type, modules = diagnostic.resolve_map_backend("multicell")
    names = ("development_rollout.py", "rollout.py", "brownian.py", "dynamics.py", "checkpoints.py",
             "development.py", "features.py", "configurations.py", "metrics.py", "pilot_selection.py", "precision.py")
    rows[0]["source_sha256"] = {n: diagnostic._hash(Path(diagnostic.__file__).with_name(n)) for n in names}
    rows[0]["map_source_sha256"] = {m.__name__: diagnostic._hash(Path(m.__file__)) for m in modules}
    identity = {"sha256": "software-development"}
    rows[0]["development_identity"] = identity["sha256"]
    configs = diagnostic.terrain_configurations()
    model = SimpleNamespace(identity={"configuration_identity": configs["all-terrain"]["sha256"],
                                     "training_identity": identity["sha256"], "seed": 20260814})
    fitted = {"schema_version": "pirc17-development-fit-v1", "configurations": configs,
              "development_identity": identity, "models": {"all-terrain": {"20260814": {}}}}
    fit = tmp_path/"fit.json"
    fit.write_text(json.dumps(fitted), encoding="utf-8")
    rows[0]["fit_sha256"] = diagnostic._hash(fit)
    args = SimpleNamespace(ledger=tmp_path/"ledger", ledger_sha256="software-ledger", fit=fit,
        fit_sha256=rows[0]["fit_sha256"], snapshot=tmp_path/"snapshot", eligibility=tmp_path/"eligibility",
        eligibility_sha256="software-eligibility", release=tmp_path/"release", data_root=tmp_path)
    windows = [SimpleNamespace(sample_id=s, role="validation", block_id=b,
        horizon_seconds=np.array([60., 300., 900., 1800.]), target_positions_m=np.zeros((4, 2)))
        for s, b in zip(rows[0]["sample_ids"], ("software-a", "software-b", "software-c"))]
    encoder = SimpleNamespace(columns=["software-column"])
    driver_identity = copy.deepcopy(chosen["brownian_identity"])
    if fault == "source":
        rows[0]["source_sha256"]["rollout.py"] = "changed"
    elif fault == "fit_hash":
        args.fit_sha256 = "changed"
    elif fault == "model_seed":
        model.identity["seed"] += 1
    elif fault == "population":
        identity = {"sha256": "changed"}
    elif fault == "selection":
        windows.reverse()
    elif fault == "role":
        windows[1].role = "final_eval"
    elif fault == "target":
        windows[1].target_positions_m[0, 0] = 1.
    elif fault == "times":
        windows[1].horizon_seconds[0] = 59.
    elif fault == "noise":
        driver_identity["seed"] += 1
    replay_calls = []

    def replay_original(source_rows, directory, *, tolerance_m):
        replay_calls.append(len(source_rows)-2)
        assert tolerance_m == 10.27506475 and directory == tmp_path
        return {"runs": [None]*(23 if fault == "incomplete_audit" else 24)}

    monkeypatch.setattr(diagnostic, "load_ledger", lambda *_: rows)
    monkeypatch.setattr(diagnostic, "replay", replay_original)
    monkeypatch.setattr(diagnostic, "restore_dynamics", lambda _: model)
    monkeypatch.setattr(diagnostic.CanonicalEncoder, "frozen_pirc22", lambda _: encoder)
    monkeypatch.setattr(diagnostic, "load_development", lambda *_: ({"validation": windows}, [], identity))
    monkeypatch.setattr(diagnostic, "select_windows", lambda *_args, **_kwargs: windows)
    monkeypatch.setattr(diagnostic, "load_particle_evidence", lambda *_: (
        np.zeros((1024, 4, 2)), np.zeros((4, 2)), np.array([60., 300., 900., 1800.])))
    monkeypatch.setattr(diagnostic, "configuration_encoder", lambda *_: encoder)
    monkeypatch.setattr(diagnostic, "BrownianPath", lambda *_args, **_kwargs: SimpleNamespace(identity=driver_identity))
    if fault is None:
        prepared = diagnostic.prepare(args)
        assert prepared.row is chosen and prepared.query_type is query_type
        assert len(prepared.query_times) == 5760
        assert prepared.window.sample_id == diagnostic.SAMPLE_ID
    else:
        with pytest.raises(ValueError):
            diagnostic.prepare(args)
    assert replay_calls == ([] if fault == "source" else [24])


def cli_args():
    paths = ("ledger", "fit", "eligibility", "release", "snapshot", "data-root", "output", "scratch-directory")
    return ["diagnostic", *sum((["--"+n, n] for n in paths), []),
            *sum((["--"+n+"-sha256", "software-hash"] for n in ("ledger", "fit", "eligibility")), [])]


@pytest.mark.parametrize("fault", [None, "preflight", "receipts", "query", "close"])
def test_cli_records_complete_or_failed_replay_and_never_overwrites(tmp_path, monkeypatch, fixture_inputs, fault):
    import psutil
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=4*1024**3))
    prepared, query, kwargs = fixture_inputs
    # The small software fixture has no physical tick before 2s under main's 5s clock.
    # Recompute its own reference at that cadence; this is not changed research evidence.
    kwargs["history_step_seconds"] = 5.
    prepared.driver = BrownianPath(prepared.reference_times, [.5], history_step_seconds=5.,
                                  particles=4, seed=8, stream_id="software-fixture")
    kwargs["brownian_increments"] = prepared.driver
    original = rollout(prepared.window.origin, prepared.reference_times,
                       terrain=PredictedPositionFeatures(prepared.encoder, query, prepared.window.frame), **kwargs)
    prepared.reference_positions = original.positions_m
    prepared.row["invalid_feature_rows"] = original.invalid_feature_rows
    monkeypatch.setattr(diagnostic, "EXPECTED_QUERY_CALLS", 4)
    monkeypatch.setattr(diagnostic, "EXPECTED_QUERY_ROWS", 16)
    monkeypatch.setattr(diagnostic, "EXPECTED_INVALID_ROWS", original.invalid_feature_rows)
    owners = []

    class Maps:
        def __init__(self, root, receipts):
            assert root.is_absolute() and all(p.is_absolute() for p in receipts)
            self.identity = {"receipt_sha256": {"software": "hash"}}
            self.geometry_cache_measurements = {"retained_cells": 0}
            self.closed = []
            owners.append(self)

        def register_snapshot_parent(self, parent):
            assert parent == "software-parent"
            if fault == "receipts":
                self.identity["receipt_sha256"] = {"changed": "hash"}

        def __call__(self, points):
            assert Path.cwd() == tmp_path/"scratch-directory"
            if fault == "query":
                raise RuntimeError("software query failure")
            return query(points)

        def close(self):
            self.closed.append(Path.cwd())
            if fault == "close":
                raise RuntimeError("software cleanup failure")

    prepared.query_type = Maps

    def prepare(args):
        assert Path.cwd() == tmp_path
        assert all(getattr(args, n).is_absolute() for n in ("ledger", "fit", "eligibility", "snapshot", "release", "data_root"))
        if fault == "preflight":
            raise ValueError("software preflight failure")
        return prepared

    monkeypatch.setattr(diagnostic, "prepare", prepare)
    monkeypatch.setattr(sys, "argv", cli_args())
    if fault:
        with pytest.raises(SystemExit) as error:
            diagnostic.main()
        assert error.value.code == 1
    else:
        diagnostic.main()
    assert Path.cwd() == tmp_path
    assert len(owners) == (0 if fault == "preflight" else 1)
    assert all(p.closed == [tmp_path/"scratch-directory"] for p in owners)
    output = tmp_path/"output"
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    result = rows[-1]
    assert result["expected_diagnostic_runs"] == 1 and result["certified"] is False
    assert result["exact_replay"] == (fault is None)
    assert result["attempted_diagnostic_runs"] == (0 if fault in ("preflight", "receipts") else 1)
    if fault is None:
        assert result["recorded_invalid_events"] == original.invalid_feature_rows
        assert len([r for r in rows if r["type"] == "invalid_feature"]) == original.invalid_feature_rows
        assert result["feature_query_rows"] == 16
    else:
        assert "error_type" in result
    original_bytes = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        diagnostic.main()
    assert error.value.code == 2 and output.read_bytes() == original_bytes


@pytest.mark.parametrize("scratch", [".", "../outside", "output", "existing"])
def test_cli_rejects_unsafe_scratch_before_creating_output(tmp_path, monkeypatch, scratch):
    monkeypatch.chdir(tmp_path)
    (tmp_path/"existing").mkdir()
    argv = cli_args()
    argv[argv.index("--scratch-directory")+1] = scratch
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as error:
        diagnostic.main()
    assert error.value.code == 2 and not (tmp_path/"output").exists()
