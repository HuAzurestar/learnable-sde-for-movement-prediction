"""Synthetic dispatch/planning/resource tests; no empirical research runs."""
import copy
from dataclasses import replace
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.pirc17 import method_qualification as q
from experiments.pirc17 import method_qualification_analysis as qa
from experiments.pirc17.method_mechanisms import _digest, EXACT_REFERENCE, forecast_stream_binding
from experiments.pirc17.method_training import FittedMethod
from experiments.pirc17.method_rollout import MethodDynamics, MethodForecast, SWITCHING_MODELS, _grid
from experiments.pirc17.rollout import Forecast
from tests.test_pirc17_method_mechanisms import dynamics as fixture_dynamics
from tests.test_pirc17_method_cost_probe import prepared as fixture_prepared


def fake_diagnostics(slot, origin_id, seed, *, horizons=(60., 300., 900., 1800.)):
    scientific = "arm-01/full" if slot == EXACT_REFERENCE else slot
    c = q.required_slot(scientific)["components"]
    d = fixture_dynamics(scientific)
    binding = forecast_stream_binding(slot, "causal_prefix")
    stream = ([origin_id, "paired", binding["crn_pair_id"]] if binding["crn_pair_id"] else
              [origin_id, "independent", c["poa"], slot])
    grid = _grid(horizons, 5., c["dt_seconds"], c["dt_seconds"], 400)
    return {"dynamics": d.identity(), "model_kind": c["model"], "condition_names": c["condition"],
        "origin_mode": "causal_prefix", "origin_id": origin_id, "run_id": slot,
        "origin_epoch_seconds": 0., "velocity_source": "software-fixture", "velocity_observed_at_seconds": None,
        "velocity_error_mps": None, "particles": 512, "seed": seed, "max_step_seconds": 5.,
        "history_step_seconds": c["dt_seconds"], "mode_step_seconds": c["dt_seconds"],
        "integration_steps": len(grid), "history_ticks_seconds": [end for _, end, tick, _, _ in grid if tick],
        "mode_resampling_times_seconds": [end for _, end, _, tick, _ in grid
            if tick and end < horizons[-1] and c["model"] in SWITCHING_MODELS],
        "integrator": "exact" if slot == EXACT_REFERENCE else c["integrator"], "propagation": c["poa"],
        "random_stream_sha256": _digest(stream), "crn_pair_id": binding["crn_pair_id"]}


def synthetic_rows(plan):
    rows = []
    for b, (block, origins) in enumerate(plan["expected_origins_by_block"].items()):
        for origin in origins:
            for s, seed in enumerate(plan["seeds"]):
                for slot in plan["forecast_order_per_origin_seed"]:
                    effect = (b+1)*(1+.1*s)
                    score = 100+b if slot.endswith("/full") or slot == EXACT_REFERENCE else 100+b+effect
                    if slot == "arm-21/mc": score = 100+b+2*s
                    if slot == "arm-21/crn": score = 100+b+s
                    rows.append({"origin_id": origin, "block_id": block, "seed": seed, "slot_id": slot,
                        "split": "validation", "context_sha256": _digest(["software", origin]),
                        "elapsed_seconds": [60., 300., 900., 1800.], "status": "success", "score_m": score,
                        "diagnostics": fake_diagnostics(slot, origin, seed)})
    return rows


@pytest.fixture(scope="module")
def plan():
    return q.plan_definition()


@pytest.fixture(scope="module")
def rows(plan):
    return synthetic_rows(plan)


def test_exact_workload_reuses_five_fits_without_dropping_forecast_slots(plan):
    assert q.validate_plan(plan) == plan
    assert plan["reused_fits"] == 5 and plan["new_fits"] == 11
    assert plan["expected_forecasts"] == 3*5*(28+1) == 435
    assert plan["scientific_slot_forecasts"] == 420
    assert plan["score_diagnostic_extra_Gaussian_draws"] == 30720
    assert plan["prior_preparation_empirical_seconds"]+1060+240 < 1800
    assert len(plan["required_slots"]) == 28
    assert not plan["automatic_expansion"] and not plan["final_eval_authorized"]
    assert plan["secondary_origin_modes_in_this_batch"] == []


def test_saved_plan_matches_executable_definition(plan):
    assert json.loads(q.PLAN_PATH.read_text(encoding="utf-8")) == plan


@pytest.mark.parametrize("key,value", [
    ("particles", 1024), ("max_step_seconds", 2.5), ("seeds", [20260814]),
    ("nominal_seconds", [60, 300, 900, 3600]), ("planning_blocks", 62),
    ("expected_origins_by_block", {}), ("new_fit_slots", []), ("outer_wall_seconds", 1300),
    ("max_transitions", 160000), ("scientific_claim_authorized", True),
    ("automatic_retry", True), ("final_eval_authorized", True), ("source_sha256", {}),
])
def test_scope_mutations_rejected_before_execution(plan, key, value):
    changed = copy.deepcopy(plan)
    changed[key] = value
    with pytest.raises(ValueError, match="exact finite"):
        q.validate_plan(changed)


def test_analysis_uses_three_blocks_not_fifteen_seed_pseudoreplicates(plan, rows):
    result = qa.analyze(rows, expected_origins_by_block=q.ORIGINS, slots=plan["forecast_order_per_origin_seed"])
    assert result["successful_forecasts"] == 435 and result["unrecorded_forecasts"] == 0
    assert result["variance"]["gates"]["arm-21/crn"]["value"] == pytest.approx(4.)
    for family in result["families"].values():
        assert len(family["independent_block_ids"]) == 3
        for row in family["results"].values():
            assert np.asarray(row["per_block_per_seed_difference_m"]).shape == (3, 5)
            assert len(row["planning_scenarios"]) == 2
            assert row["planning_scenarios"][0]["assumed_true_improvement_m"] == 2*qa.DELTA_M
    first = result["families"]["method-model-structure"]["results"]["arm-02/pointwise"]
    assert first["paired_block_sd_m"] == pytest.approx(1.2)
    assert first["observed_development_improvement_m"] == pytest.approx(-2.4)
    assert first["simulation_mean_standard_error_m"] == pytest.approx(np.std([2, 2.2, 2.4, 2.6, 2.8], ddof=1)/np.sqrt(5))
    assert not result["certified"] and not result["numerically_qualified"] and not result["power_qualified"]


@pytest.mark.parametrize("kind", ["missing", "failed", "unavailable"])
def test_no_successful_intersection_power_or_variance(plan, rows, kind):
    changed = copy.deepcopy(rows)
    index = next(i for i, row in enumerate(changed) if row["slot_id"] == "arm-21/crn")
    if kind == "missing": changed.pop(index)
    else: changed[index].update(status=kind, reason="synthetic failed item")
    result = qa.analyze(changed, expected_origins_by_block=q.ORIGINS, slots=plan["forecast_order_per_origin_seed"])
    assert result["variance"]["status"] == "unavailable"
    assert result["families"]["method-numerical-propagation"]["status"] == "unavailable"
    assert result["families"]["method-model-structure"]["status"] == "computed"


@pytest.mark.parametrize("kind", ["duplicate", "split", "block", "seed", "context", "time", "N", "h", "stream", "clock", "nonfinite"])
def test_wrong_pairs_and_provenance_rejected(plan, rows, kind):
    changed = copy.deepcopy(rows)
    row = changed[1]
    if kind == "duplicate": changed.append(copy.deepcopy(row))
    elif kind == "split": row["split"] = "final_eval"
    elif kind == "block": row["block_id"] = "other-block"
    elif kind == "seed": row["seed"] = 20260819
    elif kind == "context": row["context_sha256"] = "e"*64
    elif kind == "time": row["elapsed_seconds"][-1] = 1799.
    elif kind == "N": row["diagnostics"]["particles"] = 1024
    elif kind == "h": row["diagnostics"]["max_step_seconds"] = 2.5
    elif kind == "stream": row["diagnostics"]["random_stream_sha256"] = "e"*64
    elif kind == "clock": row["diagnostics"]["history_ticks_seconds"] = []
    else: row["score_m"] = float("nan")
    with pytest.raises(ValueError):
        qa.analyze(changed, expected_origins_by_block=q.ORIGINS, slots=plan["forecast_order_per_origin_seed"])


def test_zero_variance_is_retained_not_promoted_to_qualification(plan, rows):
    changed = copy.deepcopy(rows)
    for row in changed: row["score_m"] = 100.
    result = qa.analyze(changed, expected_origins_by_block=q.ORIGINS, slots=plan["forecast_order_per_origin_seed"])
    assert result["variance"]["gates"]["arm-21/crn"]["passed"] is None
    assert not result["power_qualified"]
    assert result["families"]["method-model-structure"]["results"]["arm-02/pointwise"]["paired_block_sd_m"] == 0


def test_caps_before_work_and_exclusive_bounded_artifacts(plan, monkeypatch, tmp_path):
    monkeypatch.setattr(q, "available_physical_bytes", lambda: 2**40)
    budget = q.Budget(plan, io.StringIO())
    budget.started -= 1001
    with pytest.raises(TimeoutError): budget.run("fit", "x", lambda: pytest.fail("no over-budget work"))
    budget = q.Budget(plan, io.StringIO())
    monkeypatch.setattr(q, "available_physical_bytes", lambda: 0)
    with pytest.raises(MemoryError): budget.guard()
    path = tmp_path/"small.json"
    budget.save(path, {"software_only": True})
    with pytest.raises(FileExistsError): budget.save(path, {})
    with pytest.raises(OSError): budget.save(tmp_path/"large.json", {"too_big": "x"*(2*1024**2)})
    assert not (tmp_path/"large.json").exists()


def test_existing_reservation_is_not_restarted(plan, monkeypatch, tmp_path):
    monkeypatch.setattr(q, "OUTPUT", tmp_path)
    monkeypatch.setattr(q, "read_bound", lambda *a: plan)
    monkeypatch.setattr(q, "run_owned", lambda *a, **k: pytest.fail("no second attempt"))
    with pytest.raises(FileExistsError): q.supervise("plan", "a"*64)


@pytest.mark.parametrize("outcome", ["complete", "timeout", "interrupted", "worker_error", "tree_live"])
def test_supervisor_uses_owned_deadline_and_terminal_conditions(plan, monkeypatch, tmp_path, outcome):
    output = tmp_path/"new-attempt"
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        assert 1000 < kwargs["timeout"] <= 1020
        assert "--worker" in command and command[-1] == "a"*64
        return {"worker_returncode": 1 if outcome == "worker_error" else 0,
            "process_tree_closed": outcome != "tree_live", "timed_out": outcome == "timeout",
            "interrupted": outcome == "interrupted", "supervisor_error": None}
    monkeypatch.setattr(q, "OUTPUT", output)
    monkeypatch.setattr(q, "read_bound", lambda *a: plan)
    monkeypatch.setattr(q, "run_owned", run)
    assert q.supervise("software-plan", "a"*64) == (0 if outcome == "complete" else 1)
    assert len(calls) == 1 and (output/"start.json").exists() and (output/"supervision.json").exists()


def test_worker_cannot_load_outside_its_live_owned_job(plan, monkeypatch, tmp_path):
    monkeypatch.setattr(q, "OUTPUT", tmp_path)
    q.cost.publish(tmp_path/"start.json", {"software_only": True})
    monkeypatch.setattr(q, "read_bound", lambda path, sha: plan if path == "plan" else {
        "schema_version": q.VERSION, "plan_sha256": "a"*64, "job_name": "fake-job"})
    def reject(name): raise PermissionError("not owned")
    monkeypatch.setattr(q, "require_owned_job", reject)
    monkeypatch.setattr(q, "load_method_development", lambda *a, **k: pytest.fail("no unowned loading"))
    with pytest.raises(PermissionError, match="not owned"):
        q.worker("plan", "a"*64)
    assert not (tmp_path/"ledger.jsonl").exists()


def software_fit(slot, plan):
    original = fixture_dynamics(slot)
    tau = str(int(q.required_slot(slot)["components"]["dt_seconds"]))
    training = {"input_sha256": plan["expected_input_sha256"], "training_components": q.recipe(slot),
        "sample_counts": {"train": 328, "adapt": 76, "validation": 81},
        "transitions_by_role": plan["expected_transitions_by_interval"][tau], "numpy_version": np.__version__,
        "source_sha256": {k: plan["source_sha256"][k] for k in q.FIT_SOURCES},
        "formal_training_accepted": False, "software_only_not_a_real_fit": True}
    training["training_identity_sha256"] = _digest(training)
    identity = _digest({"training_identity_sha256": training["training_identity_sha256"], "model": original.model.to_dict()})
    d = MethodDynamics.bind(original.model, reference_interval_seconds=original.reference_interval_seconds,
        fit_identity=identity, noise_binding_rationale="software-only fitting provenance")
    return FittedMethod(d, training, slot, .001)


def test_fit_reuse_validates_training_content_not_just_saved_id(plan):
    fit = software_fit("arm-01/full", plan)
    row = q.fit_record(fit, seconds=.01, reused=True)
    assert q.check_fit(row, plan).dynamics.identity() == fit.dynamics.identity()
    row["training"]["software_only_not_a_real_fit"] = False
    with pytest.raises(ValueError, match="fitted recipe"):
        q.check_fit(row, plan)


def test_dispatch_all_slots_and_failure_disposition_no_refit_or_forecast_retry(plan, monkeypatch, tmp_path):
    source = fixture_prepared()
    base = source.prefixes[0]
    prefixes, segments = [], []
    for block, origins in q.ORIGINS.items():
        s = replace(base.assignment.sample, sample_id=origins[0], independent_block_id=block)
        prefixes.append(replace(base, assignment=replace(base.assignment, sample=s)))
        segments.append(source.segments[0])
    source = replace(source, prefixes=tuple(prefixes), segments=tuple(segments), identity={"sha256": plan["expected_input_sha256"]})
    monkeypatch.setattr(q, "ROOT", tmp_path.parent)
    monkeypatch.setattr(q, "available_physical_bytes", lambda: 2**40)
    fits, calls = [], []
    def fit(prepared, slot, **kwargs):
        fits.append(slot)
        if slot == "arm-02/pointwise": raise ValueError("deliberate software-only fit failure")
        return software_fit(slot, plan)
    def predict(d, origin, horizons, **kwargs):
        calls.append((kwargs["origin_id"], kwargs["seed"], kwargs["run_id"]))
        assert "target" not in kwargs and not hasattr(origin, "segment")
        diag = fake_diagnostics(kwargs["run_id"], kwargs["origin_id"], kwargs["seed"], horizons=horizons)
        diag["dynamics"] = d.identity()
        positions = np.broadcast_to(origin.position_m, (512, 4, 2)).copy()
        covariance = np.broadcast_to(np.eye(2), (512, 4, 2, 2)).copy()
        fp = kwargs["propagation"] == "fp"
        return MethodForecast(Forecast(horizons, positions, 0, 0, "software-only"),
            positions if fp else None, covariance if fp else None, diag)
    monkeypatch.setattr(q, "fit_development_method", fit)
    monkeypatch.setattr(q, "forecast_method", predict)
    monkeypatch.setattr(q, "score_path", lambda *a, **k: {"time_weighted_energy_score_m": 42.})
    reused = [(software_fit(slot, plan), {"slot_id": slot, "status": "success", "reused": True, "sha256": "a"*64})
              for slot in plan["cached_fit_slots"]]
    result = q.execute(source, plan, tmp_path, q.Budget(plan, io.StringIO()), reused)
    assert fits == plan["new_fit_slots"]
    assert len(calls) == len(set(calls)) == 420  # Fifteen pointwise predictions explicitly unavailable.
    assert len(result["forecasts"]) == 435 and len(result["fits"]) == 16
    assert sum(row["status"] == "unavailable" for row in result["forecasts"]) == 15
    assert result["analysis"]["families"]["method-model-structure"]["status"] == "unavailable"
    assert len(result["integration_gates"]) == 45 and len(result["score_gates"]) == 30
    assert len(list(tmp_path.glob("*.npz"))) == 420
