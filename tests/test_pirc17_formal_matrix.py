"""Full inventory and factor-ownership tests without final-eval identities."""
from collections import Counter
from copy import deepcopy

import pytest

from experiments.pirc17 import formal_matrix as matrix
from experiments.pirc17 import formal_budget as budget
from experiments.pirc17.protocol_core import digest, envelope, unpack


@pytest.fixture(scope="module")
def protocol():
    return matrix.load_protocol()


@pytest.fixture(scope="module")
def complete(protocol):
    return matrix.build_matrix(protocol)


def test_complete_two_matrices_and_exact_forecast_count(complete):
    p = unpack(complete)
    assert Counter(r["disposition"] for r in p["method_ledger"]) == {"REQUIRED": 28, "EXCLUDED": 8}
    assert {r["arm_id"] for r in p["method_ledger"] if r["disposition"] == "EXCLUDED"} == {13, 17, 22}
    assert all(r["exclusion_reason"] and r["disposition_source"] and r["fit_identity"] is None
               for r in p["method_ledger"] if r["disposition"] == "EXCLUDED")
    assert len(p["terrain_ledger"]) == 10 and len(p["training_groups"]) == 16
    assert p["counts_by_kind"] == {"input_qualification_and_population": 1, "method_fit": 16, "terrain_fit": 10,
        "scientific_forecast": 11020, "same_grid_reference": 290, "inertial_path": 58, "common_scores": 58,
        "forecast_replay": 38, "runtime_cold": 75, "runtime_warmup": 15, "runtime_warm": 75,
        "mechanisms_and_inference": 1, "independent_reanalysis": 1, "aggregate_export": 1}
    assert len(p["workloads"]) == 11659
    assert p["max_generated_forecasts"] == sum(w["generated_forecasts"] for w in p["workloads"]) == 11513
    assert p["scoring_only_Gaussian_draws"] == 593920
    assert not p["final_eval_authorized"] and p["final_eval_reads"] == p["new_forecasts"] == p["new_fits"] == 0


def test_every_required_slot_and_config_has_all_registered_paired_axes(complete):
    p = unpack(complete)
    slots = {r["slot_id"] for r in p["method_ledger"] if r["disposition"] == "REQUIRED"}
    configs = set(p["terrain_ledger"])
    keys = {(r["matrix"], r["subject"], r["origin_mode"], r["origin_rank"], r["seed"])
            for r in p["workloads"] if r["kind"] == "scientific_forecast"}
    expected = {(group, subject, mode, rank, seed)
                for group, subjects in (("NEX326-methods", slots), ("terrain", configs))
                for subject in subjects for mode, count in p["rank_counts"].items()
                for rank in range(count) for seed in matrix.SEEDS}
    assert keys == expected and len(keys) == 11020
    assert p["rank_counts"] == {"causal_prefix": 46, "known_velocity": 6, "point_only": 6}
    assert Counter(r["matrix"] for r in p["workloads"] if r["scientific"]) == {"NEX326-methods": 8120, "terrain": 2900}


def test_four_score_instants_do_not_multiply_forecasts_and_fits_are_not_seed_replicates(complete):
    p = unpack(complete)
    assert p["counts_by_kind"]["scientific_forecast"] == (46+6+6)*5*(28+10)
    fits = [r for r in p["workloads"] if r["kind"] in {"method_fit", "terrain_fit"}]
    assert len(fits) == 26 and all(r["seed"] is None and r["generated_forecasts"] == 0 for r in fits)
    identities = {r["fit_identity"] for r in fits}
    assert all(r["fit_identity"] in identities for r in p["workloads"] if r["generated_forecasts"])


def test_terrain_dimensions_owner_closure_and_shared_interactions(complete):
    terrain = unpack(complete)["terrain_ledger"]
    assert {k: v["conditioner_input_dimension"] for k, v in terrain.items()} == {
        "base": 4, "all-terrain": 56, "loo-road": 38, "loo-river": 46, "loo-worldcover": 40,
        "loo-surface": 40, "lio-road": 14, "lio-river": 14, "lio-worldcover": 12, "lio-surface": 20}
    assert terrain["all-terrain"]["composition_owners"]["worldcover.grouped_x_road_distance"] == ["road", "worldcover"]
    for name, row in terrain.items():
        assert row["baseline_history_retained"] and row["retrain_independently"]
        assert "history.direction" in row["variant_ids"]
        assert row["numeric_input_dimension"] == row["validity_input_dimension"]
        for owners in row["composition_owners"].values():
            assert not set(owners) & set(row["removed_factors"])
        if name.startswith("lio-"):
            assert row["included_factors"] == [name.removeprefix("lio-")]


def test_owner_closure_rejects_retained_interactions_after_removal(protocol):
    configurations = deepcopy(unpack(protocol)["components"]["terrain_configurations"])
    configurations["loo-road"]["composition_ids"].append("worldcover.grouped_x_road_distance")
    with pytest.raises(ValueError, match="owner closure"):
        matrix.terrain_ownership(configurations)


def test_work_identity_and_phase_caps_complete(complete):
    p = unpack(complete)
    assert len({w["work_id"] for w in p["workloads"]}) == len(p["workloads"])
    assert set(p["counts_by_phase"]) == set(p["phase_caps_seconds"])
    assert sum(p["phase_caps_seconds"].values()) == 48*3600
    for w in p["workloads"]:
        assert w["work_id"] == digest({k: v for k, v in w.items() if k != "work_id"})
        assert 0 < w["max_active_seconds"] <= p["phase_caps_seconds"][w["phase"]]
        if w["kind"] in {"method_fit", "terrain_fit"}: assert w["max_active_seconds"] == 90
        if w["kind"] == "scientific_forecast": assert w["max_active_seconds"] == (30 if w["matrix"] == "NEX326-methods" else 180)


def test_runtime_replay_and_reference_are_bounded_separate_from_science(complete):
    p = unpack(complete)
    auxiliary = [w for w in p["workloads"] if w["generated_forecasts"] and not w["scientific"]]
    assert len(auxiliary) == 493
    assert sum(w["kind"].startswith("runtime_") for w in auxiliary) == 165
    for w in auxiliary:
        if w["kind"] != "same_grid_reference":
            assert (w["origin_mode"], w["origin_rank"], w["seed"]) == ("causal_prefix", 0, matrix.SEEDS[0])
        else:
            assert w["phase"] == "method_forecasts" and w["subject"] == matrix.EXACT_REFERENCE
    assert len({w["subject"] for w in auxiliary if w["kind"] == "forecast_replay"}) == 38


@pytest.mark.parametrize("fault", ["missing", "duplicate", "seed", "rank", "fit", "dimension", "exclusion", "caps"])
def test_even_rehashed_matrix_changes_are_rejected(complete, protocol, fault):
    p = deepcopy(unpack(complete))
    if fault == "missing": p["workloads"].pop()
    if fault == "duplicate": p["workloads"].append(deepcopy(p["workloads"][-1]))
    if fault == "seed": p["seeds"].append(20260819)
    if fault == "rank": p["rank_counts"]["causal_prefix"] = 47
    if fault == "fit": p["terrain_ledger"]["loo-road"]["fit_identity"] = "terrain-fit:all-terrain"
    if fault == "dimension": p["terrain_ledger"]["base"]["conditioner_input_dimension"] = 0
    if fault == "exclusion": next(r for r in p["method_ledger"] if r["disposition"] == "EXCLUDED")["disposition"] = "REQUIRED"
    if fault == "caps": p["phase_caps_seconds"]["terrain_forecasts"] += 1
    with pytest.raises(ValueError, match="matrix inventory changed"):
        matrix.validate_matrix(envelope(p), protocol)


def test_budget_projection_preserves_complete_capacity_but_grants_no_authority(complete, tmp_path):
    contract = budget.contract_for_matrix(complete, protocol_sha256=matrix.PROTOCOL_SHA256,
        execution_sha256=digest("NOT-REAL-EXECUTION"), runtime_manifest_sha256=digest("fixture-runtime"),
        approval_sha256=digest("NOT-REAL-APPROVAL"), ledger_directory=tmp_path/"fixture-only")
    assert len(contract["workloads"]) == 11659
    assert contract["total_cap_ns"] == 48*3600*budget.NANOSECONDS
    assert contract["max_generated_forecasts"] == 11513 and contract["max_attempts_per_item"] == 1
    assert not (tmp_path/"fixture-only").exists()  # Projection performs no writes/launch.
