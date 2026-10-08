"""Software-only counters, actual kernel oracle and whole-evidence gates."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from experiments.pirc17 import candidate_power as power
from experiments.pirc17 import direct_linear_rollout as engine
from experiments.pirc17 import query_demand as demand
from experiments.pirc17.brownian import BrownianPath, integration_grid
from experiments.pirc17.origins import known_velocity
from experiments.pirc17.rollout import Features, rollout
from tests.test_pirc17_candidate_power import build
from tests.test_pirc17_direct_linear_rollout import candidate_bytes, evidence, inputs, read_rows


def synthetic_sources():
    rows = []
    # Irregular outputs split integration intervals and physical history ticks.
    for sample, times in (("a", [1.2, 4.1]), ("b", [1.1, 4.3])):
        intervals = len(integration_grid(times, .7, 2.))-1
        for config in demand.FAMILY:
            for seed in demand.SEEDS:
                rows.append(dict(type="run", status="success", sample_id=sample,
                    independent_block_id="block-"+sample, configuration=config, seed=seed,
                    particles=4, max_step_seconds=.7, actual_horizons_seconds=list(times),
                    feature_query_rows=4*intervals, invalid_feature_rows=0,
                    rollout_wall_seconds_including_lazy_map_initialization=1. if config == "base" else 3.))
    return [[dict(expected_run_count=len(rows)), dict(configurations=list(demand.FAMILY),
        max_steps_seconds=[.7], particle_counts=[4], physical_history_step_seconds=2.),
        *rows, dict(status="complete")]]


def options():
    return dict(step_seconds=.7, source_particles=4, particle_counts=[4, 8], block_counts=[3, 30], seed_counts=[1, 5])


@pytest.mark.parametrize("step,history,times", [(.7, 2., [1.2, 4.1]), (.3125, 5., [1., 3.7, 5.1]),
    (4., 1., [2.2, 5.3]), (1., 5., [2., 5., 6.])])
def test_grid_count_matches_actual_causal_kernel(step, history, times):
    calls = []
    def terrain(state):
        calls.append(len(state.positions_m))
        return Features(np.zeros((4, 1)), np.ones((4, 1), dtype=bool))
    forecast = rollout(known_velocity([0, 0], 0, [1, 0], source="software-counter"), times,
        particles=4, seed=1, max_step_seconds=step, history_step_seconds=history,
        base_drift=lambda s:s.velocities_mps, diffusion=lambda s:np.zeros((4, 2, 2)),
        terrain=terrain, conditioner=lambda s,f:np.zeros((4, 2)))
    assert forecast.feature_query_rows == sum(calls) == 4*demand._intervals(tuple(times), step, history)


def test_complete_counts_baseline_separation_and_linear_scenario_oracle():
    sources = synthetic_sources()
    result = demand.project(sources, **options())
    assert result["whole_runs"] == result["selected_runs"] == 60
    assert result["source_independent_blocks"] == result["source_origins"] == 2
    assert result["selected_raw_map_query_rows"]*6 == result["selected_feature_rows"]*5
    assert len(result["scenarios"]) == 8
    for scenario in result["scenarios"]:
        runs = scenario["blocks"]*scenario["seed_count"]
        assert scenario["primary_runs"] == runs*6
        assert scenario["feature_rows_template_max"] == runs*6*scenario["particles"]*result["observed_integration_intervals_max"]
        assert scenario["raw_map_query_rows_template_max"]*6 == scenario["feature_rows_template_max"]*5
        assert scenario["linear_rollout_only_hours"] == pytest.approx(16*runs*scenario["particles"]/4/3600)
        assert not scenario["launch_authorized"]
    assert any("NEX326" in s for s in result["limits"])
    assert any("p50/p95" in s for s in result["limits"])


def test_brownian_values_only_memory_matches_real_driver_and_guard():
    result = demand.project(synthetic_sources(), **dict(options(), particle_counts=[4, 1_000_000_000]))
    scenario = result["scenarios"][0]
    times = [1.1, 4.3]
    driver = BrownianPath(times, [.7], history_step_seconds=2., particles=4, seed=1, stream_id="software")
    assert scenario["brownian_values_bytes_one_stream_template_max"] == driver.values.nbytes
    large = result["scenarios"][-1]
    assert not large["within_brownian_values_guard_for_templates"]
    with pytest.raises(ValueError, match="512 MiB"):
        BrownianPath(times, [.7], history_step_seconds=2., particles=1_000_000_000, seed=1, stream_id="software")


@pytest.mark.parametrize("fault", ["count", "boolean_count", "invalid", "wall_nan", "wall_negative", "failed",
    "partial", "duplicate", "missing_seed", "changed_block", "changed_time", "changed_runtime", "missing_setting"])
def test_rejects_changed_counts_denominators_and_identity(fault):
    sources = synthetic_sources()
    row = sources[0][2]
    if fault == "count": row["feature_query_rows"] -= 1
    if fault == "boolean_count": row["feature_query_rows"] = True
    if fault == "invalid": row["invalid_feature_rows"] = row["feature_query_rows"]+1
    if fault == "wall_nan": row[demand.WALL_FIELD] = float("nan")
    if fault == "wall_negative": row[demand.WALL_FIELD] = -1.
    if fault == "failed": row["status"] = "failure"
    if fault == "partial": sources[0][-1]["status"] = "running"
    if fault == "duplicate": sources.append(deepcopy(sources[0]))
    if fault == "missing_seed":
        sources[0].pop(2)
        sources[0][0]["expected_run_count"] -= 1
    if fault == "changed_block": row["independent_block_id"] = "other"
    if fault == "changed_time": row["actual_horizons_seconds"][0] = 1.15
    if fault == "changed_runtime":
        sources.append(deepcopy(sources[0]))
        sources[-1][0]["runtime"] = "different"
    if fault == "missing_setting": sources[0][1]["max_steps_seconds"] = [.8]
    with pytest.raises(ValueError):
        demand.project(sources, **options())


@pytest.mark.parametrize("field,value", [("particle_counts", [True]), ("block_counts", [0]),
    ("seed_counts", [6]), ("particle_counts", [4, 4]), ("block_counts", [3.0]), ("seed_counts", [])])
def test_rejects_invalid_scenario_axes(field, value):
    with pytest.raises(ValueError):
        demand.project(synthetic_sources(), **dict(options(), **{field:value}))


def test_cached_grid_does_not_bypass_boolean_validation():
    demand._intervals((1., 2.), 1., 1.)
    with pytest.raises(ValueError):
        demand._intervals((True, 2.), 1., 1.)


@pytest.fixture
def bound_inputs(inputs, evidence, monkeypatch):
    args, tracker = inputs
    # The upstream orchestration fixture uses one aggregate step. Supply its
    # exact synthetic counter separately; the real-kernel oracle is above.
    fake = engine.rollout
    def counted(origin, times, **kwargs):
        forecast = fake(origin, times, **kwargs)
        count = kwargs["particles"]*(len(integration_grid(times, kwargs["max_step_seconds"], kwargs["history_step_seconds"]))-1)
        return replace(forecast, feature_query_rows=count)
    monkeypatch.setattr(engine, "rollout", counted)
    monkeypatch.setattr(demand, "resolve_map_backend", engine.resolve_map_backend)
    power_args = build(args, evidence, overrides={0:{"particles":[4, 8]}})
    power.run(**power_args)
    return dict(**evidence, planning=power_args["output"], planning_sha256=demand._hash(power_args["output"]),
        ledgers=power_args["ledgers"], audits=power_args["audits"], particle_counts=[4, 8],
        block_counts=[3, 30], seed_counts=[1, 5], output=power_args["output"].with_name("query-demand.json")), tracker


def test_real_audit_chain_without_forecasting_overwriting_or_qualification(bound_inputs, monkeypatch):
    args, tracker = bound_inputs
    before = len(tracker["calls"])
    saved = {p:p.read_bytes() for p in [args["planning"], args["fit"], args["fit_ledger"], *args["ledgers"], *args["audits"]]}
    monkeypatch.setattr(engine, "run", lambda **kw:pytest.fail("budget analysis must not forecast"))
    result = demand.run(**args)
    assert json.loads(args["output"].read_text()) == result
    assert result["new_forecasts"] == result["final_eval_label_prediction_metric_reads"] == 0
    assert all(result[k] is False for k in ("certified", "numerically_qualified", "full_budget_qualified", "formal_training_accepted"))
    assert result["projection"]["selected_feature_rows"] == 60*360*4
    assert result["projection"]["selected_raw_map_query_rows"] == 50*360*4
    assert result["projection"]["whole_runs"] == 84
    assert result["source_numerical"][0]["out_of_tolerance"] > 0
    assert len(tracker["calls"]) == before and all(p.read_bytes() == data for p,data in saved.items())
    with pytest.raises(FileExistsError):
        demand.run(**args)
    with pytest.raises(FileExistsError):
        demand.run(**dict(args, output=args["ledgers"][0].with_suffix(".particles")/"new.json"))


@pytest.mark.parametrize("fault", ["planning_hash", "ledger_hash", "audit_hash", "missing_source", "late_array", "late_source"])
def test_bound_chain_and_post_calculation_mutations_fail_closed(bound_inputs, monkeypatch, fault):
    args, _ = bound_inputs
    if fault == "planning_hash": args["planning_sha256"] = "f"*64
    if fault == "ledger_hash": args["ledgers"][0].write_bytes(args["ledgers"][0].read_bytes()+b"\n")
    if fault == "audit_hash": args["audits"][0].write_bytes(args["audits"][0].read_bytes()+b" ")
    if fault == "missing_source": args["ledgers"] = args["ledgers"][:-1]
    if fault in {"late_array", "late_source"}:
        original = demand.project
        def mutate(*a, **kw):
            result = original(*a, **kw)
            if fault == "late_array":
                path = args["ledgers"][0].parent/read_rows(args["ledgers"][0])[2]["particle_artifact"]["path"]
                path.write_bytes(path.read_bytes()+b"changed")
            else:
                previous = demand.source_hashes()
                monkeypatch.setattr(demand, "source_hashes", lambda:{**previous, "query_demand.py":"f"*64})
            return result
        monkeypatch.setattr(demand, "project", mutate)
    with pytest.raises(ValueError):
        demand.run(**args)
    assert not args["output"].exists()
