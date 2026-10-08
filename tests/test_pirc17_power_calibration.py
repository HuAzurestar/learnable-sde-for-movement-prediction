import copy

import pytest

from experiments.pirc17.inference import SEEDS
from experiments.pirc17.power_calibration import calibrate


def ledger(seeds=SEEDS):
    origins = ["a", "a-repeat", "b", "c"]
    header = {"type": "header", "schema_version": "pirc17-development-rollout-v1",
        "purpose": "bounded_validation_engineering_pilot", "final_eval_label_prediction_metric_reads": 0,
        "sample_ids": origins, "configurations": ["base", "all-terrain"], "seeds": list(seeds),
        "particle_counts": [32], "max_steps_seconds": [.625], "expected_run_count": len(origins)*2*len(seeds),
        "brownian_driver": "pirc17-coupled-brownian-v2", "time_weights": [.5, .5],
        "fit_sha256": "fixture-fit", "development_identity": "fixture-development", "rollout_version": "fixture-rollout",
        "physical_history_step_seconds": 5, "scoring_grid": "fixture-grid", "entropy_grid_m": {}, "map_backend": "fixture",
        "map_source_sha256": {"fixture-map-source": "fixture"},
        "source_sha256": {name: "fixture" for name in ("rollout.py", "brownian.py", "dynamics.py", "checkpoints.py",
            "development.py", "features.py", "metrics.py", "configurations.py")}}
    rows = [header]
    for origin in origins:
        block = origin[0]
        for index, seed in enumerate(seeds):
            for configuration in header["configurations"]:
                # After averaging all five seeds, block deltas are -1, 0, +1.
                score = 100 + ((ord(block)-ord("b"))+index-2 if configuration == "all-terrain" else 0)
                rows.append({"type": "run", "sample_id": origin, "independent_block_id": block,
                    "configuration": configuration, "seed": seed, "particles": 32, "max_step_seconds": .625,
                    "actual_horizons_seconds": [1, 2], "status": "success",
                    "brownian_identity": {"version": header["brownian_driver"], "seed": seed,
                        "stream_id": origin, "max_particles": 32, "path_sha256": origin},
                    "scores": {"time_weighted_energy_score_m": score,
                        "by_time": [{"elapsed_seconds": t, "energy_score_m": score} for t in (1, 2)]}})
    rows.append({"type": "completion", "attempted_run_count": len(rows)-1, "success_count": len(rows)-1, "failure_count": 0})
    return rows


def summarize(rows):
    return calibrate(rows, step_seconds=.625, particles=32, delta_m=2,
                     planning_block_counts=[30, 62], comparisons=["all-vs-base"])


def test_sd_is_of_equal_blocks_and_seeds_not_repeated_origins_or_observed_effect():
    report = summarize([ledger()])
    assert report["source_independent_blocks"] == 3 and report["source_origin_count"] == 4
    result = report["results"]["all-vs-base"]
    assert result["paired_block_sd_m"] == pytest.approx(1)
    assert result["observed_development_improvement_m"] == 0
    assert all(s["assumed_true_improvement_m"] == 4 for s in result["planning_scenarios"])
    assert len(report["missing_primary_comparisons"]) == 4 and not report["certified"]


def test_disjoint_seed_ledgers_combine_without_inflating_block_counts():
    # Use rows from the same complete fixture so seed offsets remain unchanged.
    full = ledger()
    subsets = []
    for seeds in (SEEDS[:2], SEEDS[2:]):
        rows = copy.deepcopy([full[0], *[r for r in full[1:-1] if r["seed"] in seeds], full[-1]])
        rows[0]["seeds"] = list(seeds)
        rows[0]["expected_run_count"] = len(rows)-2
        rows[-1].update(attempted_run_count=len(rows)-2, success_count=len(rows)-2)
        subsets.append(rows)
    result = summarize(subsets)
    assert result["results"] == summarize([full])["results"]
    assert result["source_independent_blocks"] == 3


def test_partial_seeds_duplicates_and_unregistered_family_cannot_be_preregistered():
    with pytest.raises(ValueError, match="five"):
        summarize([ledger(SEEDS[:1])])
    with pytest.raises(ValueError, match="duplicate"):
        summarize([ledger(), ledger()])
    with pytest.raises(ValueError, match="primary family"):
        calibrate([ledger()], step_seconds=.625, particles=32, delta_m=2,
                  planning_block_counts=[30], comparisons=["all-vs-base"], family_size=1)


def test_final_eval_and_failed_or_differently_paired_evidence_is_rejected():
    rows = ledger()
    rows[0]["final_eval_label_prediction_metric_reads"] = 1
    with pytest.raises(ValueError):
        summarize([rows])
    rows = ledger()
    rows[1]["status"] = "failure"
    rows[-1].update(success_count=len(rows)-3, failure_count=1)
    with pytest.raises(ValueError, match="failed"):
        summarize([rows])
    rows = ledger()
    rows[1]["brownian_identity"]["path_sha256"] = "changed"
    with pytest.raises(ValueError):
        summarize([rows])


@pytest.mark.parametrize("field", ["map_source_sha256", "fit_sha256", "source_sha256"])
def test_mixed_predictor_or_map_identity_is_rejected(field):
    first, second = ledger(SEEDS[:2]), ledger(SEEDS[2:])
    if isinstance(second[0][field], dict):
        second[0][field][next(iter(second[0][field]))] = "changed"
    else:
        second[0][field] = "changed"
    with pytest.raises(ValueError, match="mixes"):
        summarize([first, second])


@pytest.mark.parametrize("rows", [[], [[]]])
def test_empty_evidence_is_rejected(rows):
    with pytest.raises(ValueError, match="ledgers"):
        summarize(rows)
