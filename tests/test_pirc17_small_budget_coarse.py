"""Software-only paired short/full path and bounded coarse-runner checks."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.pirc17 import small_budget_coarse as coarse
from experiments.pirc17 import small_budget_review as review
from experiments.pirc17.brownian import BrownianPath
from experiments.pirc17.origins import causal_prefix
from experiments.pirc17.rollout import rollout
from tests.test_pirc17_candidate_power import build
from tests.test_pirc17_direct_linear_rollout import candidate_bytes, evidence, inputs, read_rows


def test_full_reference_driver_and_real_kernel_preserve_short_prefix():
    times = np.array([55., 297., 894., 1796.])
    origin = causal_prefix([[-2., -1.], [-1., -.5], [0., 0.]], [-10., -5., 0.])
    saved = BrownianPath(times, [.625, .3125], history_step_seconds=5., particles=8, seed=20260814, stream_id="software")
    row = dict(actual_horizons_seconds=times.tolist(), max_step_seconds=.3125, particles=8,
        brownian_identity=saved.identity, sample_id="software", seed=20260814,
        independent_block_id="software-block", model_identity_sha256="software-model")
    driver = coarse.reference_driver(row)
    assert np.array_equal(driver.values, saved.values)
    # Fresh short-grid RNG draws are NOT the same reference stream.
    short_driver = BrownianPath(times[:2], [.3125], history_step_seconds=5., particles=8,
        seed=20260814, stream_id="software")
    assert not np.array_equal(short_driver(0., 55., 8, 2), driver(0., 55., 8, 2))
    model = SimpleNamespace(identity={"sha256": "software-model"})
    window = SimpleNamespace(sample_id="software", block_id="software-block", role="validation", horizon_seconds=times)
    for step in (5., 2.5):
        coarse.validate_pair(row, window, driver, 4, step, 2, model)
        kwargs = dict(seed=20260814, max_step_seconds=step, history_step_seconds=5., brownian_increments=driver,
            base_drift=lambda s: .2*s.velocities_mps+np.sin(s.positions_m)*.01,
            diffusion=lambda s: np.broadcast_to(np.eye(2)*.3, (len(s.positions_m), 2, 2)))
        full = rollout(origin, times, particles=8, **kwargs)
        short = rollout(origin, times[:2], particles=4, **kwargs)
        assert np.array_equal(short.positions_m, full.positions_m[:4, :2])
        with pytest.raises(ValueError, match="stream"):
            coarse.validate_pair(row, window, short_driver, 4, step, 2, model)
    bad = deepcopy(row)
    bad["brownian_identity"]["path_sha256"] = "f"*64
    with pytest.raises(ValueError, match="does not reproduce"):
        coarse.reference_driver(bad)


@pytest.fixture
def prepared(inputs, evidence, monkeypatch):
    original, tracker = inputs
    maps_type, _ = coarse.engine.resolve_map_backend("multicell")
    monkeypatch.setattr(maps_type, "identity", property(lambda self: dict(
        software_only=True, parents=self.parents, receipt_sha256={}, verified_assets={})))
    monkeypatch.setattr(maps_type, "assets", {}, raising=False)
    monkeypatch.setattr(coarse.MemoryObserver, "guard", lambda self: None)
    args = build(original, evidence, overrides={0: {"particles": [8]}, 1: {"particles": [8]}}, stem="coarse-parent")
    args["particles"] = 8
    review.parent.run(**args)
    plan = dict(schema_version=review.VERSION, new_rollouts=0, final_eval_authorized=False,
        particle_budgets=[4, 8], step_seconds=5., epsilon_m=1., alpha=.05, diagnostic_family_size=55, wall_seconds=600,
        reference_power=str(args["output"]), reference_power_sha256=review._hash(args["output"]),
        candidate_evidence={k: str(v) if isinstance(v, Path) else v for k, v in evidence.items()},
        inputs=[dict(ledger=str(p), ledger_sha256=s, audit=str(a), audit_sha256=h) for p, s, a, h in zip(
            args["ledgers"], args["ledger_sha256"], args["audits"], args["audit_sha256"])])
    offline_path = original["output"].with_name("offline-plan.json")
    offline_path.write_text(json.dumps(plan))
    offline_result = offline_path.with_name("offline-result.json")
    review.run(plan_path=offline_path, plan_sha256=review._hash(offline_path), output=offline_result)
    spec = dict(schema_version=coarse.VERSION, final_eval_authorized=False, scoring_slots=2,
        time_weights=[0., 1.], particles=4, step_seconds=5., seeds=list(coarse.SEEDS),
        sample_ids=["validation-000", "validation-004"], configurations=coarse.CONFIGURATIONS[:4],
        expected_run_count=40, map_backend="multicell", wall_seconds=900, outer_wall_seconds=960,
        new_empirical_total_cap_seconds=7200, prior_new_empirical_batches=[],
        offline_plan=str(offline_path), offline_plan_sha256=review._hash(offline_path),
        offline_result=str(offline_result), offline_result_sha256=review._hash(offline_result))
    path = original["output"].with_name("coarse-plan.json")
    path.write_text(json.dumps(spec))
    kwargs = {k: original[k] for k in ("eligibility", "release", "snapshot", "data_root")}
    kwargs.update(plan_path=path, plan_sha256=review._hash(path), output=path.with_name("coarse-run.jsonl"))
    return kwargs, spec, tracker


def test_complete_short_run_and_independent_saved_array_consumer(prepared):
    args, spec, tracker = prepared
    before = len(tracker["calls"])
    result = coarse.run(**args)
    assert result["status"] == "complete" and result["success_count"] == 40
    assert result["unattempted_run_count"] == result["failure_count"] == result["terminal_error_count"] == 0
    assert len(tracker["calls"])-before == 40 and all(m.closed for m in tracker["maps"])
    rows = read_rows(args["output"])
    report = json.loads(args["output"].with_suffix(".audit.json").read_text())
    assert len(report["paired_diagnostics"]) == 60 and len(report["aggregate_precision"]) == 6
    assert report["short_screen_only"] and not report["numerically_qualified"]
    assert report["missing_primary_comparisons"] == ["surface", "worldcover"]
    _, previous, _, references, _, _, _ = coarse.context(args["plan_path"], args["plan_sha256"])
    reproduced = coarse.analyze(rows[2:-1], args["output"].parent, spec, previous, references)
    assert all(report[k] == v for k, v in reproduced.items())
    for aggregate in report["aggregate_precision"]:
        if aggregate["kind"] == "coarse_minus_fine":
            assert aggregate["absolute_change_plus_margin_m"] == 0
        group = [x for x in report["paired_diagnostics"] if x["kind"] == aggregate["kind"] and x["comparison"] == aggregate["comparison"]]
        mean = np.mean([x["precision"]["by_time_energy_score_m"] for x in group], axis=0)
        cov = sum(np.array(x["precision"]["time_covariance_m2"]) for x in group)/100
        assert np.allclose(mean, aggregate["by_scoring_slot_difference_m"])
        assert np.allclose(cov, aggregate["slot_covariance_m2"])
    with pytest.raises(FileExistsError):
        coarse.run(**args)
    # No favorable-grid salvage, fake coupling, substituted targets or modified scores.
    for fault in ("missing", "duplicate", "stream", "model", "score", "resolution"):
        damaged = deepcopy(rows[2:-1])
        if fault == "missing": damaged.pop()
        elif fault == "duplicate": damaged[-1] = damaged[0]
        elif fault == "stream": damaged[0]["brownian_identity"]["path_sha256"] = "f"*64
        elif fault == "model": damaged[0]["model_identity_sha256"] = "f"*64
        elif fault == "score": damaged[0]["particle_precision"]["time_weighted_energy_score_m"] += 1
        else: damaged[0]["max_step_seconds"] = 2.5
        with pytest.raises(ValueError):
            coarse.analyze(damaged, args["output"].parent, spec, previous, references)


def test_full_horizon_keeps_complete_primary_family_without_claiming_acceptance(prepared):
    args, spec, _ = prepared
    spec.update(scoring_slots=4, time_weights=[.25]*4, configurations=coarse.CONFIGURATIONS, expected_run_count=60)
    args["plan_path"].write_text(json.dumps(spec))
    args["plan_sha256"] = coarse._hash(args["plan_path"])
    result = coarse.run(**args)
    assert result["status"] == "complete" and result["success_count"] == 60
    report = json.loads(args["output"].with_suffix(".audit.json").read_text())
    assert len(report["aggregate_precision"]) == 10 and not report["missing_primary_comparisons"]
    assert not report["short_screen_only"] and not report["numerically_qualified"]


def test_invalid_plan_and_changed_offline_result_are_rejected(prepared):
    args, original, tracker = prepared
    calls = len(tracker["calls"])
    for field, value in (("final_eval_authorized", True), ("wall_seconds", 901), ("outer_wall_seconds", 1000),
            ("seeds", list(coarse.SEEDS[:1])), ("scoring_slots", 3), ("step_seconds", 10),
            ("time_weights", [.5, .5]), ("particles", True), ("expected_run_count", 30),
            ("new_empirical_total_cap_seconds", 10000), ("configurations", ["base"]),
            ("sample_ids", ["validation-000"]*2)):
        changed = dict(original, **{field: value})
        with pytest.raises(ValueError): coarse.validate_plan(changed)
    offline_path = Path(original["offline_result"])
    offline_path.write_text(offline_path.read_text()+" ")
    with pytest.raises(ValueError): coarse.context(args["plan_path"], args["plan_sha256"])
    assert len(tracker["calls"]) == calls and not args["output"].exists()


def test_resource_stop_retains_failure_and_unattempted_denominator(prepared, monkeypatch):
    args, _, tracker = prepared
    before = len(tracker["calls"])
    old = coarse.engine.rollout
    counter = 0
    def stop(*a, **kw):
        nonlocal counter
        counter += 1
        if counter == 3:
            raise TimeoutError("software cap boundary")
        return old(*a, **kw)
    monkeypatch.setattr(coarse.engine, "rollout", stop)
    result = coarse.run(**args)
    assert result["status"] == "failed" and result["resource_stopped"]
    assert (result["success_count"], result["failure_count"], result["unattempted_run_count"]) == (2, 1, 37)
    assert len(tracker["calls"])-before == 2 and all(m.closed for m in tracker["maps"])
    assert not args["output"].with_suffix(".audit.json").exists()


def test_original_map_mutation_stops_before_new_forecast(prepared, monkeypatch, tmp_path):
    args, _, tracker = prepared
    asset = tmp_path/"ancestor-map.bin"
    asset.write_bytes(b"old")
    digest = coarse._hash(asset)
    asset.write_bytes(b"new")
    original = coarse.context
    def changed(*a):
        parts = list(original(*a))
        parts[-1][asset] = digest
        return parts
    monkeypatch.setattr(coarse, "context", changed)
    before = len(tracker["calls"])
    result = coarse.run(**args)
    assert result["status"] == "failed" and result["attempted_run_count"] == 0
    assert len(tracker["calls"]) == before


def test_empirical_cost_history_cannot_be_omitted_in_a_descendant_or_overrun(tmp_path):
    spec = dict(wall_seconds=900, new_empirical_total_cap_seconds=7200,
        offline_result_sha256="offline", prior_new_empirical_batches=[])
    path = tmp_path/"previous.jsonl"
    def prior(elapsed, predecessors):
        rows = [dict(schema_version=coarse.VERSION, plan=dict(spec, prior_new_empirical_batches=predecessors)),
                dict(type="completion", elapsed_seconds=elapsed)]
        path.write_text("".join(json.dumps(r)+"\n" for r in rows))
        return dict(ledger=str(path), ledger_sha256=coarse._hash(path))
    item = prior(100, [])
    assert coarse.prior_budget(dict(spec, prior_new_empirical_batches=[item])) == 100
    with pytest.raises(ValueError): coarse.prior_budget(dict(spec, prior_new_empirical_batches=[item, item]))
    item = prior(6400, [])
    with pytest.raises(ValueError, match="total empirical budget"):
        coarse.prior_budget(dict(spec, prior_new_empirical_batches=[item]))
    item = prior(100, [{"ledger": "omitted", "ledger_sha256": "old"}])
    with pytest.raises(ValueError, match="ordered"):
        coarse.prior_budget(dict(spec, prior_new_empirical_batches=[item]))
