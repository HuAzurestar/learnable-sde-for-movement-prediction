import copy
import itertools
import json

import numpy as np
import pytest

from experiments.pirc17.development_rollout import save_particle_artifact
from experiments.pirc17.metrics import energy_score
from experiments.pirc17.nested_precision import nested_energy_precision
from experiments.pirc17.precision_check import aggregate_precision, compare_reference, load_particle_evidence, main, replay
from experiments.pirc17.qualification import _hash


def fixture(tmp_path, *, particle_counts=(8,)):
    expected_runs = 4*len(particle_counts)
    header = {"type": "header", "schema_version": "pirc17-development-rollout-v1",
        "purpose": "bounded_validation_engineering_pilot", "final_eval_label_prediction_metric_reads": 0,
        "sample_ids": ["origin"], "configurations": ["base", "all-terrain"], "seeds": [20260814],
        "particle_counts": list(particle_counts), "max_steps_seconds": [.625, 1.25], "expected_run_count": expected_runs,
        "brownian_driver": "pirc17-coupled-brownian-v2", "time_weights": [.5, .5], "save_particles": True,
        "fit_sha256": "fixture-fit", "development_identity": "fixture-development", "rollout_version": "fixture-rollout",
        "physical_history_step_seconds": 5, "scoring_grid": "fixture-grid", "entropy_grid_m": {},
        "source_sha256": {name: "fixture" for name in ("rollout.py", "brownian.py", "dynamics.py", "checkpoints.py",
                            "features.py", "configurations.py", "metrics.py", "development.py")}}
    runs = []
    rng = np.random.default_rng(10)
    draws = rng.normal(size=(max(particle_counts), 2, 2))
    targets, times = np.zeros((2, 2)), np.array([1., 2.])
    for configuration in header["configurations"]:
        for step, particles in itertools.product(header["max_steps_seconds"], particle_counts):
            paths = draws[:particles]+step+(configuration == "all-terrain")
            score = [energy_score(paths[:, t], targets[t]) for t in range(2)]
            row = {"type": "run", "sample_id": "origin", "independent_block_id": "block",
                "configuration": configuration, "seed": 20260814, "particles": particles, "max_step_seconds": step,
                "actual_horizons_seconds": times.tolist(), "status": "success", "invalid_feature_rows": 0,
                "rollout_wall_seconds_including_lazy_map_initialization": 1,
                "brownian_identity": {"version": header["brownian_driver"], "seed": 20260814,
                    "stream_id": "origin", "max_particles": max(particle_counts), "path_sha256": "fixture-path"},
                "scores": {"time_weighted_energy_score_m": np.mean(score),
                           "by_time": [{"elapsed_seconds": t, "energy_score_m": v} for t, v in zip(times, score)]}}
            row["particle_artifact"] = save_particle_artifact(tmp_path/"ledger.jsonl", row, paths, targets, times)
            runs.append(row)
    return [header, *runs, {"type": "completion", "attempted_run_count": expected_runs,
                          "success_count": expected_runs, "failure_count": 0}]


def test_replay_derives_paired_precision_and_never_certifies(tmp_path):
    rows = fixture(tmp_path)
    report = replay(rows, tmp_path, tolerance_m=.1)
    assert len(report["runs"]) == len(report["paired_comparisons"]) == 4
    assert report["certified"] is False
    assert all(c["precision"]["time_weighted_standard_error_m"] > 0 for c in report["paired_comparisons"])
    assert len(report["aggregate_precision"]) == 4
    assert report["aggregate_precision"][0]["time_weighted_difference_m"] == pytest.approx(
        report["paired_comparisons"][0]["precision"]["time_weighted_energy_score_m"])


@pytest.mark.parametrize("budgets", [(8, 16), (8, 16, 32)])
def test_nested_particle_audit_replays_all_adjacent_budgets(tmp_path, budgets):
    rows = fixture(tmp_path, particle_counts=budgets)
    report = replay(rows, tmp_path, tolerance_m=.1)
    assert len(report["runs"]) == 4*len(budgets)
    assert len(report["paired_comparisons"]) == 4*len(budgets)+4*(len(budgets)-1)
    assert not report["particle_budget_unavailable"] and report["certified"] is False
    nested = [r for r in report["paired_comparisons"] if r["kind"] == "particle_budget_refinement"]
    assert len(nested) == 4*(len(budgets)-1)
    for comparison in nested:
        def paths(workload):
            return load_particle_evidence(next(row for row in rows[1:-1]
                if all(row[k] == v for k, v in workload.items())), tmp_path)
        large, targets, _ = paths(comparison["candidate_workload"])
        small, _, _ = paths(comparison["control_workload"])
        assert comparison["precision"] == nested_energy_precision(large, small, targets, rows[0]["time_weights"])
    aggregated = [r for r in report["aggregate_precision"] if r["kind"] == "particle_budget_refinement"]
    assert {(r["control_particles"], r["candidate_particles"]) for r in aggregated} == set(zip(budgets, budgets[1:]))
    assert all(r["candidate_step_seconds"] == r["control_step_seconds"] for r in aggregated)


def test_noninteger_budget_ratio_is_explicitly_unavailable(tmp_path):
    rows = fixture(tmp_path, particle_counts=(8, 12))
    report = replay(rows, tmp_path, tolerance_m=.1)
    assert len(report["particle_budget_unavailable"]) == 4
    assert not any(r["kind"] == "particle_budget_refinement" for r in report["paired_comparisons"])
    assert all("integer budget ratio" in r["reason"] for r in report["particle_budget_unavailable"])


def test_equal_scores_do_not_establish_nested_particle_pairing(tmp_path):
    rows = fixture(tmp_path, particle_counts=(8, 16))
    row = next(row for row in rows[1:-1] if row["particles"] == 16)
    paths, targets, times = load_particle_evidence(row, tmp_path)
    paths = paths[::-1]
    row["particle_artifact"] = save_particle_artifact(tmp_path/"reordered.jsonl", row, paths, targets, times)
    with pytest.raises(ValueError, match="exact shared prefix"):
        replay(rows, tmp_path, tolerance_m=.1)


@pytest.mark.parametrize("budgets", [(8, 16), (8, 12)])
def test_cli_binds_nested_source_and_fails_closed_when_ratio_is_unavailable(tmp_path, monkeypatch, budgets):
    rows = fixture(tmp_path, particle_counts=budgets)
    ledger, output = tmp_path/"ledger.jsonl", tmp_path/"report.json"
    ledger.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["precision_check", "--ledger", str(ledger),
        "--ledger-sha256", _hash(ledger), "--tolerance-m", ".1", "--output", str(output)])
    if budgets == (8, 12):
        with pytest.raises(SystemExit) as failure:
            main()
        assert failure.value.code == 1
    else:
        main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == "pirc17-particle-precision-audit-v3"
    assert len(report["source_sha256"]["nested_precision.py"]) == 64
    assert bool(report["particle_budget_unavailable"]) == (budgets == (8, 12))


@pytest.mark.parametrize("change", ["hash", "escape", "score", "grid", "pair"])
def test_tampered_or_mismatched_particle_evidence_rejected(tmp_path, change):
    rows = fixture(tmp_path)
    if change == "hash":
        rows[1]["particle_artifact"]["sha256"] = "0"*64
    elif change == "escape":
        rows[1]["particle_artifact"]["path"] = "../outside.npz"
    elif change == "score":
        rows[1]["scores"]["by_time"][0]["energy_score_m"] += 1
    elif change == "grid":
        for row in rows[1:-1]:
            row["actual_horizons_seconds"][1] = 3
            row["scores"]["by_time"][1]["elapsed_seconds"] = 3
    else:
        rows[1]["brownian_identity"]["path_sha256"] = "different"
    with pytest.raises(ValueError):
        replay(rows, tmp_path, tolerance_m=.1)


def test_backend_reference_checks_sources_scores_and_pairing(tmp_path):
    rows = fixture(tmp_path)
    reference = copy.deepcopy(rows)
    report = compare_reference(rows, reference, tolerance_m=.1)
    assert report["passed"] and report["particle_arrays_checked"] is False
    assert "not compared" in report["scope"]
    reference[1]["scores"]["by_time"][0]["energy_score_m"] += .01
    assert not compare_reference(rows, reference, tolerance_m=.1)["passed"]
    reference[0]["source_sha256"]["rollout.py"] = "changed"
    with pytest.raises(ValueError, match="source"):
        compare_reference(rows, reference, tolerance_m=.1)


def test_complete_backend_reference_checks_each_saved_array(tmp_path):
    rows, reference = fixture(tmp_path/"candidate"), fixture(tmp_path/"reference")
    report = compare_reference(rows, reference, tolerance_m=.1,
        directory=tmp_path/"candidate", reference_directory=tmp_path/"reference")
    assert report["passed"] and report["particle_arrays_checked"]
    assert all(r["all_particle_arrays_exactly_equal"] and r["max_coordinate_difference_m"] == 0
               for r in report["runs"])


@pytest.mark.parametrize("change", ["particle_order", "targets", "times", "shape", "nonfinite"])
def test_equal_scores_cannot_hide_different_or_invalid_saved_arrays(tmp_path, change):
    rows, reference = fixture(tmp_path/"candidate"), fixture(tmp_path/"reference")
    row = reference[1]
    with np.load(tmp_path/"reference"/row["particle_artifact"]["path"], allow_pickle=False) as saved:
        paths, targets, times = saved["positions_m"], saved["target_positions_m"], saved["elapsed_seconds"]
    if change == "particle_order":
        paths = paths[::-1]
    elif change == "targets":
        targets = targets+1
    elif change == "times":
        times = times+1
    elif change == "shape":
        targets = targets[:, :1]
    else:
        paths[0, 0, 0] = np.nan
    row["particle_artifact"] = save_particle_artifact(tmp_path/"reference"/"changed.jsonl", row, paths, targets, times)
    arguments = dict(tolerance_m=.1, directory=tmp_path/"candidate", reference_directory=tmp_path/"reference")
    if change in {"times", "shape", "nonfinite"}:
        with pytest.raises(ValueError):
            compare_reference(rows, reference, **arguments)
    else:
        report = compare_reference(rows, reference, **arguments)
        assert not report["passed"] and report["particle_arrays_checked"]
        assert report["runs"][0]["all_score_fields_exactly_equal"]
        assert not report["runs"][0]["all_particle_arrays_exactly_equal"]


def test_array_comparison_needs_both_directories_and_saved_evidence(tmp_path):
    rows = fixture(tmp_path)
    with pytest.raises(ValueError, match="both particle evidence directories"):
        compare_reference(rows, rows, tolerance_m=.1, directory=tmp_path)
    reference = copy.deepcopy(rows)
    reference[0]["save_particles"] = False
    with pytest.raises(ValueError, match="both saved"):
        compare_reference(rows, reference, tolerance_m=.1, directory=tmp_path, reference_directory=tmp_path)


def test_failed_workloads_remain_visible_and_have_no_precision_claim(tmp_path):
    rows = fixture(tmp_path)
    rows[1]["status"] = "failure"
    rows[-1].update(success_count=3, failure_count=1)
    result = replay(rows, tmp_path, tolerance_m=.1)
    assert result["ledger_audit"]["failures"] and not result["paired_comparisons"]


def aggregate_fixture():
    header = {"brownian_driver": "pirc17-coupled-brownian-v2", "sample_ids": ["a", "a-repeat", "b"],
              "seeds": [20260814, 20260815], "time_weights": [.5, .5]}
    rows = []
    for origin, seed in itertools.product(header["sample_ids"], header["seeds"]):
        rows.append({"kind": "integration_refinement", "comparison": "base", "independent_block_id": origin[0],
            "candidate_workload": {"sample_id": origin, "seed": seed, "particles": 8, "max_step_seconds": .5},
            "control_workload": {"sample_id": origin, "seed": seed, "particles": 8, "max_step_seconds": 1.},
            "precision": {"by_time_energy_score_m": [1, 1] if origin.startswith("a") else [3, 3],
                          "time_covariance_m2": [[4, 2], [2, 4]]}})
    return rows, header


def test_aggregate_uses_equal_blocks_squared_weights_and_time_covariance():
    rows, header = aggregate_fixture()
    report = aggregate_precision(rows, header)[0]
    # Four origin/seed rows in block A get weight 1/8, two in B get 1/4.
    squared_weights = 4/64+2/16
    assert report["time_weighted_difference_m"] == 2
    assert report["time_weighted_standard_error_m"] == pytest.approx(np.sqrt(3*squared_weights))
    assert report["independent_block_count"] == 2
    assert report["origin_count"] == 3


def test_aggregate_does_not_merge_different_control_particle_budgets():
    rows, header = aggregate_fixture()
    second = copy.deepcopy(rows)
    for row in second:
        row["control_workload"]["particles"] = 4
    report = aggregate_precision(rows+second, header)
    assert len(report) == 2
    assert {r["control_particles"] for r in report} == {4, 8}


@pytest.mark.parametrize("change", ["missing", "duplicate", "block", "driver"])
def test_invalid_aggregate_pairing_is_rejected(change):
    rows, header = aggregate_fixture()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif change == "block":
        rows[1]["independent_block_id"] = "wrong"
    else:
        header["brownian_driver"] = "pirc17-coupled-brownian-v1"
    with pytest.raises(ValueError):
        aggregate_precision(rows, header)
