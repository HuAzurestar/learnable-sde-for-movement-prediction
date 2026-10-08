"""Software fixtures only; no research observations or invented real ledgers."""
import json
import itertools

import numpy as np
import pytest

from experiments.pirc17.brownian import BrownianPath
from experiments.pirc17.development_rollout import save_particle_artifact
from experiments.pirc17.metrics import energy_score
from experiments.pirc17.particle_extension_check import compare_extension, main
from experiments.pirc17.precision_check import load_particle_evidence
from experiments.pirc17.qualification import _hash


def extension_fixture(folder, budgets):
    times, targets = np.array([1., 2.]), np.zeros((2, 2))
    header = {"type": "header", "schema_version": "pirc17-development-rollout-v1",
        "purpose": "bounded_validation_engineering_pilot", "final_eval_label_prediction_metric_reads": 0,
        "sample_ids": ["origin"], "configurations": ["base", "all-terrain"], "seeds": [20260814],
        "particle_counts": list(budgets), "max_steps_seconds": [.625, 1.25], "expected_run_count": 4*len(budgets),
        "brownian_driver": "pirc17-coupled-brownian-v2", "time_weights": [.5, .5], "save_particles": True,
        "fit_sha256": "fixture-fit", "development_identity": "fixture-development", "rollout_version": "fixture-rollout",
        "physical_history_step_seconds": 5., "scoring_grid": "fixture-grid", "entropy_grid_m": {},
        "source_sha256": {"fixture-predictor": "fixture-source"},
        "map_backend": "multicell", "map_source_sha256": {"fixture-map": "fixture-source"}}
    driver = BrownianPath(times, header["max_steps_seconds"], history_step_seconds=5.,
        particles=max(budgets), seed=20260814, stream_id="origin")
    draws = driver.values[[driver.index[t] for t in times]].transpose(1, 0, 2)
    rows = [header]
    for name, step, particles in itertools.product(header["configurations"], header["max_steps_seconds"], budgets):
        paths = draws[:particles]+step+(name == "all-terrain")
        scores = [energy_score(paths[:, i], targets[i]) for i in range(2)]
        row = {"type": "run", "sample_id": "origin", "independent_block_id": "block",
            "configuration": name, "seed": 20260814, "particles": particles, "max_step_seconds": step,
            "actual_horizons_seconds": times.tolist(), "status": "success", "invalid_feature_rows": 0,
            "rollout_wall_seconds_including_lazy_map_initialization": 1., "brownian_identity": dict(driver.identity),
            "scores": {"time_weighted_energy_score_m": float(np.mean(scores)),
                       "by_time": [{"elapsed_seconds": t, "energy_score_m": v} for t, v in zip(times, scores)]}}
        row["particle_artifact"] = save_particle_artifact(folder/"ledger.jsonl", row, paths, targets, times)
        rows.append(row)
    return [*rows, {"type": "completion", "attempted_run_count": 4*len(budgets),
                   "success_count": 4*len(budgets), "failure_count": 0}]


def compare(tmp_path, candidate, reference):
    return compare_extension(candidate, reference, directory=tmp_path/"new",
        reference_directory=tmp_path/"old", tolerance_m=.1)


@pytest.mark.parametrize("new_budgets", [(16,), (16, 32)])
def test_all_extended_budgets_match_every_full_reference_prefix(tmp_path, new_budgets):
    a, b = extension_fixture(tmp_path/"new", new_budgets), extension_fixture(tmp_path/"old", (4, 8))
    result = compare(tmp_path, a, b)
    assert result["passed"] and result["certified"] is False
    assert result["candidate_complete_run_count"] == 4*len(new_budgets)
    assert result["reference_complete_run_count"] == 8
    assert len(result["comparisons"]) == 8*len(new_budgets) and result["reference_particle_budget"] == 8
    assert {r["reference_workload"]["particles"] for r in result["comparisons"]} == {4, 8}
    assert result["reference_budgets_replayed"] == [4, 8]
    assert all(row["passed"] and row["max_prefix_coordinate_difference_m"] == 0 for row in result["comparisons"])


@pytest.mark.parametrize("change", ["missing", "failed", "bad_lower_budget_sidecar", "fit", "map", "source",
                                   "time_grid", "draw_order", "targets", "counts", "provenance",
                                   "driver_max", "missing_driver", "block"])
def test_incomplete_mixed_or_tampered_evidence_is_rejected(tmp_path, change):
    a, b = extension_fixture(tmp_path/"new", (16, 32)), extension_fixture(tmp_path/"old", (4, 8))
    if change == "missing":
        b.pop(-1)
    elif change == "failed":
        b[1]["status"] = "failure"
        b[-1].update(failure_count=1, success_count=7)
    elif change == "bad_lower_budget_sidecar":
        assert b[1]["particles"] == 4
        b[1]["particle_artifact"]["sha256"] = "0"*64
    elif change == "fit":
        a[0]["fit_sha256"] = "different-fit"
    elif change == "map":
        a[0]["map_source_sha256"] = {"changed": "map"}
    elif change == "source":
        a[0]["source_sha256"]["rollout.py"] = "changed"
    elif change in {"time_grid", "draw_order"}:
        for row in a[1:-1]:
            row["brownian_identity"]["time_grid_sha256" if change == "time_grid" else "draw_order"] = "changed"
    elif change == "targets":
        a[1]["actual_horizons_seconds"] = [1., 3.]
    elif change == "counts":
        a[0]["expected_run_count"] = 7
    elif change in {"driver_max", "missing_driver", "block"}:
        for row in a[1:-1]:
            if change == "driver_max":
                row["brownian_identity"]["max_particles"] += 1
            elif change == "missing_driver":
                row["brownian_identity"].pop("draw_order")
            else:
                row["independent_block_id"] = "different-block"
    else:
        a[0].pop("fit_sha256")
    with pytest.raises(ValueError):
        compare(tmp_path, a, b)


def reordered_candidates(tmp_path, rows):
    # Reorder every candidate budget consistently so internal nesting survives.
    for row in rows[1:-1]:
        paths, target, times = load_particle_evidence(row, tmp_path/"new")
        paths[[0, 1]] = paths[[1, 0]]
        row["particle_artifact"] = save_particle_artifact(tmp_path/"new"/"reordered.jsonl", row, paths, target, times)


def test_equal_scores_cannot_hide_changed_prefix_or_particle_order(tmp_path):
    a, b = extension_fixture(tmp_path/"new", (16, 32)), extension_fixture(tmp_path/"old", (4, 8))
    reordered_candidates(tmp_path, a)
    result = compare(tmp_path, a, b)
    assert not result["passed"]
    assert all(not row["arrays_exactly_equal"]["prefix_positions_m"] for row in result["comparisons"])
    assert all(row["arrays_exactly_equal"]["target_positions_m"] for row in result["comparisons"])


def test_equal_scores_cannot_hide_translated_targets_and_prefix(tmp_path):
    a, b = extension_fixture(tmp_path/"new", (16, 32)), extension_fixture(tmp_path/"old", (4, 8))
    for row in a[1:-1]:
        paths, target, times = load_particle_evidence(row, tmp_path/"new")
        row["particle_artifact"] = save_particle_artifact(tmp_path/"new"/"translated.jsonl",
            row, paths+3, target+3, times)
    result = compare(tmp_path, a, b)
    assert not result["passed"]
    assert all(not row["arrays_exactly_equal"]["target_positions_m"] for row in result["comparisons"])


def test_same_or_overlapping_budgets_are_not_an_extension(tmp_path):
    a, b = extension_fixture(tmp_path/"new", (8, 16)), extension_fixture(tmp_path/"old", (4, 8))
    with pytest.raises(ValueError, match="strictly exceed"):
        compare(tmp_path, a, b)


@pytest.mark.parametrize("changed_prefix", [False, True])
def test_cli_binds_original_ledgers_sources_and_refuses_overwrite(tmp_path, monkeypatch, changed_prefix):
    a, b = extension_fixture(tmp_path/"new", (16, 32)), extension_fixture(tmp_path/"old", (4, 8))
    if changed_prefix:
        reordered_candidates(tmp_path, a)
    candidate, reference, output = tmp_path/"new"/"ledger.jsonl", tmp_path/"old"/"ledger.jsonl", tmp_path/"report.json"
    for path, rows in ((candidate, a), (reference, b)):
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["particle_extension_check", "--ledger", str(candidate),
        "--ledger-sha256", _hash(candidate), "--reference", str(reference),
        "--reference-sha256", _hash(reference), "--tolerance-m", ".1", "--output", str(output)])
    if changed_prefix:
        with pytest.raises(SystemExit) as mismatch:
            main()
        assert mismatch.value.code == 1
    else:
        main()
    result = json.loads(output.read_text())
    assert result["passed"] is not changed_prefix
    assert result["ledger_sha256"] == _hash(candidate)
    assert result["reference_sha256"] == _hash(reference)
    assert len(result["audit_source_sha256"]["particle_extension_check.py"]) == 64
    digest = _hash(output)
    with pytest.raises(SystemExit):
        main()
    assert _hash(output) == digest
