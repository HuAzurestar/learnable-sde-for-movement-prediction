"""Pure planning/metadata tests; fixture IDs are not research observations."""
import copy
import hashlib
import json

import pytest

from experiments.pirc17.delivery_policy import (
    PHASE_CAP_SECONDS, ROOT, digest, policy, select_identity_metadata,
    training_groups, validate_policy, workload_counts,
)


@pytest.fixture(scope="module")
def candidate():
    return policy()


def identities(n=60, per_block=2):
    return [{"sample_id": f"sample-{b:03d}-{i:03d}", "independent_block_id": f"block-{b:03d}",
             "split": "final_eval"} for b in range(n) for i in range(per_block)]


def test_snapshot_and_content_hash(candidate):
    saved = json.loads((ROOT/"experiments/pirc17/plans/full-delivery-policy-v1.json").read_text())
    assert saved == candidate
    payload = {k: v for k, v in candidate.items() if k != "sha256"}
    assert candidate["sha256"] == digest(payload)
    validate_policy(saved)


def test_selection_is_order_independent_and_not_pseudoreplication():
    rows = identities()
    result = select_identity_metadata(rows)
    assert result == select_identity_metadata(rows[::-1])
    assert result["eligible_block_count"] == 60
    assert len(result["selected"]) == 46
    assert len({r["independent_block_id"] for r in result["selected"]}) == 46
    assert all(r["sample_id"].endswith("-000") for r in result["selected"])
    assert result["secondary_selected"] == result["selected"][:6]
    assert result["shortfall_blocks"] == 0
    assert result["execution_admitted"] is False


@pytest.mark.parametrize("n", [0, 1, 29, 30, 45, 46, 100])
def test_shortfall_does_not_change_horizon_or_expand_scope(n):
    result = select_identity_metadata(identities(n))
    assert len(result["selected"]) == min(n, 46)
    assert result["shortfall_blocks"] == max(0, 46-n)
    assert result["minimum_30_blocks_met"] == (n >= 30)
    assert len(result["secondary_selected"]) == min(n, 6)
    assert result["execution_admitted"] is False


@pytest.mark.parametrize("extra", ["score_m", "passed", "eligible", "position_m", "seed"])
def test_selector_refuses_outcomes_or_implicit_eligibility(extra):
    rows = identities(1)
    rows[0][extra] = 0
    with pytest.raises(ValueError, match="identity metadata"):
        select_identity_metadata(rows)


@pytest.mark.parametrize("mutation", ["split", "sample_id", "independent_block_id", "duplicate"])
def test_invalid_identity_rows(mutation):
    rows = identities(1)
    if mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    else:
        rows[0][mutation] = "validation" if mutation == "split" else ""
    with pytest.raises(ValueError):
        select_identity_metadata(rows)


def test_fit_reuse_does_not_reduce_forecast_ledger():
    groups = training_groups()
    assert len(groups) == 16
    slots = [s for g in groups for s in g["slots"]]
    assert len(slots) == len(set(slots)) == 28
    shared = next(g for g in groups if "arm-01/full" in g["slots"])
    assert len(shared["slots"]) == 13
    assert {"arm-19/em", "arm-19/euler", "arm-21/mc", "arm-21/crn"} <= set(shared["slots"])
    assert all(not {"poa", "integrator", "seed"} & set(g["training_components"]) for g in groups)
    counts = workload_counts()
    assert counts["primary_method_forecasts"] == 6440
    assert counts["primary_terrain_forecasts"] == 2300
    assert counts["scientific_forecasts_all_modes"] == 11020
    assert counts["secondary_method_forecasts"] == 1680
    assert counts["secondary_terrain_forecasts"] == 600
    assert counts["inertial_deterministic_paths"] == 58
    assert counts["max_generated_stochastic_forecasts_including_audits"] == 11223
    assert counts["forecast_replay_items"] == 38
    assert counts["runtime_forecasts_including_warmup"] == 165
    assert counts["full_prediction_regeneration_passes"] == 0
    assert counts["full_saved_output_reanalysis_passes"] == 1


def test_phase_caps_are_explicit_not_estimates(candidate):
    budget = candidate["resource_budget"]
    assert sum(PHASE_CAP_SECONDS.values()) == 172800 == budget["total_active_compute_cap_seconds"]
    assert budget["max_attempts_per_work_item"] == 1
    assert budget["parallel_empirical_workers"] == 1
    assert budget["runtime_binding"]["torch_threads"] == 6
    assert budget["runtime_binding"]["numpy_openblas_threads"] == 12
    preparation = candidate["numerical_evidence"]["remaining_preparation_budget"]
    assert preparation["closed_probe_seconds"] + preparation["remaining_seconds"] == pytest.approx(1800)
    estimate = candidate["cost_projection"]
    assert estimate["primary_prediction_and_common_score_hours"] == pytest.approx(24.354035870129443)
    assert estimate["all_modes_prediction_and_common_score_hours"] == pytest.approx(
        estimate["primary_prediction_and_common_score_hours"]*58/46)
    assert estimate["secondary_origin_cost_measured"] is False
    assert estimate["not_an_upper_bound_or_project_eta"] is True


def test_projection_independent_arithmetic(candidate):
    """No import of the projection formula; derive directly from evidence."""
    cost = candidate["cost_projection"]
    evidence = {}
    for key, binding in cost["evidence"].items():
        raw = (ROOT/binding["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == binding["sha256"]
        evidence[key] = json.loads(raw)
    terrain = sum(v["mean_seconds"] for v in evidence["terrain"]["by_configuration"].values())
    methods = evidence["methods"]["by_forecast"]
    fp = max(methods[s]["forecast_seconds"] for s in
             ("arm-01/full", "arm-04/gmm_kernel", "arm-05/explicit_decomp"))
    per_method_repeat = 26*fp + methods["arm-21/mc"]["forecast_seconds"] + methods["arm-21/crn"]["forecast_seconds"]
    common_score = max(v["common_score_seconds"] for v in methods.values())
    assert cost["all_modes_prediction_and_common_score_hours"] == pytest.approx(
        58*5*(terrain+per_method_repeat+38*common_score)/3600, rel=1e-12)
    expected_replay = terrain+per_method_repeat
    expected_runtime = 11*(terrain+sum(v["forecast_seconds"] for v in methods.values()))
    assert cost["finite_runtime_and_reforecast_prediction_hours"] == pytest.approx(
        (expected_runtime+expected_replay)/3600, rel=1e-12)


@pytest.mark.parametrize("section,key,value", [
    ("forecast", "particles", 1024),
    ("forecast", "maximum_step_seconds", 2.5),
    ("selection", "requested_independent_blocks", 182),
    ("resource_budget", "max_attempts_per_work_item", 2),
    ("resource_budget", "total_active_compute_cap_seconds", 172801),
    ("numerical_evidence", "reference_is_exact", True),
    ("numerical_evidence", "automatic_parameter_expansion", True),
    ("cost_projection", "secondary_origin_cost_measured", True),
])
def test_modified_or_rehashed_scope_is_rejected(candidate, section, key, value):
    altered = copy.deepcopy(candidate)
    altered[section][key] = value
    altered["sha256"] = digest({k: v for k, v in altered.items() if k != "sha256"})
    with pytest.raises(ValueError, match="canonical"):
        validate_policy(altered)


@pytest.mark.parametrize("key", ["execution_admitted", "final_eval_authorized"])
def test_candidate_cannot_authorize_execution(candidate, key):
    assert candidate[key] is False
    altered = copy.deepcopy(candidate)
    altered[key] = True
    with pytest.raises(ValueError):
        validate_policy(altered)


def test_scientific_scope_limits_are_preserved(candidate):
    assert candidate["forecast"]["particles"] == 512
    assert candidate["forecast"]["maximum_step_seconds"] == 5.
    assert candidate["forecast"]["nominal_horizon_seconds"] == 1800.
    assert candidate["numerical_evidence"]["reference_is_exact"] is False
    assert candidate["numerical_evidence"]["epsilon_m"] == pytest.approx(10.27506475)
    assert len(candidate["numerical_evidence"]["limitations"]) == 5
    assert candidate["training"]["method_roles"] == {"train": 328, "adapt": 76, "validation": 81}
    assert candidate["origin_modes"]["causal_prefix"]["role"] == "primary"
    assert candidate["origin_modes"]["known_velocity"]["role"] == "secondary-descriptive"
    assert candidate["origin_modes"]["point_only"]["role"] == "secondary-descriptive"
    assert candidate["new_empirical_fits"] == candidate["new_empirical_forecasts"] == 0


def test_secondary_modes_cannot_become_primary_by_relabelling(candidate):
    altered = copy.deepcopy(candidate)
    altered["origin_modes"]["point_only"]["role"] = "primary"
    with pytest.raises(ValueError, match="canonical"):
        validate_policy(altered)
