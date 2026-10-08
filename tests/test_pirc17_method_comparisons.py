import copy

import numpy as np
import pytest

from experiments.pirc17.comparison_registry import comparison_registry, registered_family
from experiments.pirc17.inference import InferenceConfig, SEEDS, infer
from experiments.pirc17.method_comparisons import (
    DELTA_M, infer_method_family, method_comparison_registry, registered_method_family, registered_method_slot,
)


def test_36_slots_are_exhaustively_partitioned_without_erasing_exemptions_or_replays():
    registry = method_comparison_registry()
    assert registry["counts"] == {"arms": 22, "slots": 36, "required_slots": 28, "excluded_slots": 8,
        "shared_Full_anchors": 6, "identical_Full_replay_slots": 1,
        "predictive_comparisons": 21, "predictive_families": 5}
    assert len(registry["slots"]) == 36
    assert len({v["arm_id"] for v in registry["slots"].values()}) == 22
    excluded = [v for v in registry["slots"].values() if v["disposition"] == "EXCLUDED"]
    assert len(excluded) == 8 and {v["arm_id"] for v in excluded} == {13, 17, 22}
    assert all(v["exclusion_reason"] for v in excluded)
    assert not registry["final_eval_authorized"] and not registry["physical_reuse_approved"]
    assert not registry["numerically_qualified"]
    assert all(not v["mechanism_definition_is_predictive_effect_evidence"] for v in registry["slots"].values())
    assert registry["slots"]["arm-06/dt60"]["role"] == "identical-Full-replay"
    assert registry["fidelity"]["paper_equivalent"] is False


def test_controls_and_changes_are_fixed_before_results_and_all_families_fit_existing_b2000():
    registry = method_comparison_registry()
    candidates = []
    for name, family in registry["families"].items():
        assert registered_method_family(registry, name) == family
        config = InferenceConfig(**family["inference_config"])
        config.validate(len(family["contrasts"]))
        assert config.bootstrap_iterations == 2000 and config.delta_m == DELTA_M
        assert family["expected_draws_per_simultaneous_tail"] >= 10
        assert family["metric"] == "time_weighted_energy_score_m"
        assert family["terrain_factor_verdict_basis"] is False
        for contrast, (candidate, control) in family["contrasts"].items():
            assert contrast == candidate and candidate != control
            assert registry["slots"][candidate]["family_id"] == name
            assert registry["slots"][control]["role"] == "shared-Full-anchor"
            candidates.append(candidate)
    assert len(candidates) == len(set(candidates)) == 21
    assert sorted(len(f["contrasts"]) for f in registry["families"].values()) == [4, 4, 4, 4, 5]
    assert registry["families"]["method-observation-interval"]["contrasts"]["arm-06/dt600"] == ["arm-06/dt600", "arm-01/full"]


@pytest.mark.parametrize("mode", ["causal_prefix", "known_velocity", "point_only"])
def test_origin_mode_identities_and_roles_remain_distinct(mode):
    registry = method_comparison_registry(origin_mode=mode)
    assert registry["forecast_contract"]["origin_mode"] == mode
    role = "method-predictive" if mode == "causal_prefix" else "secondary-origin"
    assert all(f["role"] == role for f in registry["families"].values())
    assert len({method_comparison_registry(origin_mode=m)["sha256"] for m in
        ("causal_prefix", "known_velocity", "point_only")}) == 3


def test_both_matrices_reject_the_other_registry_and_diagnostics_are_not_primary_effects():
    method = method_comparison_registry()
    with pytest.raises(ValueError):
        registered_family(method, "weighted-es-primary", primary_factor="road")
    with pytest.raises(ValueError):
        registered_method_family(comparison_registry(), "method-model-structure")
    for diagnostic in method["diagnostic_comparisons"]:
        with pytest.raises(ValueError, match="unregistered"):
            registered_method_family(method, diagnostic)
    for slot, record in method["slots"].items():
        if record["role"] == "predictive-contrast-candidate":
            assert registered_method_slot(method, slot, predictive_change_claim=True) == record
        else:
            with pytest.raises(ValueError, match="cannot claim"):
                registered_method_slot(method, slot, predictive_change_claim=True)


@pytest.mark.parametrize("change", ["family", "control", "exclusion", "gate", "hash", "authorization", "numerical", "delta", "horizon", "source"])
def test_tampered_registration_cannot_authorize_an_inference_scope(change):
    registry = method_comparison_registry()
    family = registry["families"]["method-model-structure"]
    if change == "family":
        family["contrasts"].pop("arm-04/gmm_kernel")
    elif change == "control":
        family["contrasts"]["arm-02/pointwise"][1] = "arm-03/single_gaussian"
    elif change == "exclusion":
        registry["slots"]["arm-13/animal_pretrain"]["disposition"] = "REQUIRED"
    elif change == "gate":
        registry["slots"]["arm-03/single_gaussian"]["historical_mechanism_definition"]["threshold"] = -1
    elif change == "hash":
        registry["sha256"] = "0"*64
    elif change == "authorization":
        registry["final_eval_authorized"] = True
    elif change == "numerical":
        registry["numerically_qualified"] = True
    elif change == "delta":
        family["inference_config"]["delta_m"] = 1.
    elif change == "horizon":
        registry["forecast_contract"]["forecast_horizon_seconds"] = 600.
    else:
        registry["source_sha256"]["experiments/nex326/runner.py"] = "changed"
    with pytest.raises(ValueError, match="not canonical"):
        registered_method_family(registry, "method-model-structure")
    with pytest.raises(ValueError, match="not canonical"):
        registered_method_slot(registry, "arm-02/pointwise")


def test_lookup_copies_cannot_mutate_canonical_registry():
    registry = method_comparison_registry()
    original = copy.deepcopy(registry)
    registered_method_family(registry, "method-model-structure")["contrasts"].clear()
    registered_method_slot(registry, "arm-02/pointwise")["historical_mechanism_definition"].clear()
    assert registry == original
    with pytest.raises(ValueError):
        registered_method_slot(registry, "terrain-road")
    with pytest.raises(ValueError):
        method_comparison_registry(origin_mode="future-velocity")


def family_fixture(family, delta):
    """Known-effect SOFTWARE fixture, not predictions or empirical evidence."""
    candidates = {a for a, _ in family["contrasts"].values()}
    configurations = candidates | {b for _, b in family["contrasts"].values()}
    rows = []
    for block in range(30):
        for seed in SEEDS:
            for name in sorted(configurations):
                rows.append({"configuration": name, "origin_id": f"o{block}",
                    "independent_block_id": f"b{block}", "seed": seed, "status": "success",
                    "score_m": 1000.+(delta if name in candidates else 0.)})
    return rows


@pytest.mark.parametrize("delta,verdict", [(-100., "beneficial"), (100., "harmful"), (0., "equivalent"), (-DELTA_M, "inconclusive")])
def test_each_real_method_family_works_with_existing_block_inference(delta, verdict):
    registry = method_comparison_registry()
    for family_id in registry["families"]:
        family = registered_method_family(registry, family_id)
        result = infer_method_family(registry, family_id, family_fixture(family, delta),
            mechanism_passed={k: True for k in family["contrasts"]})
        assert result["independent_block_count"] == 30  # Not 30*5*Full anchors.
        assert result["registered_seeds"] == list(SEEDS)
        assert result["scientific_claim_authorized"] is False
        assert result["schema_version"] == family["inference_engine"]
        assert all(r["verdict"] == verdict for r in result["results"].values())
        np.testing.assert_allclose([r["delta_estimate_m"] for r in result["results"].values()], delta)


def test_missing_seed_or_failed_control_is_not_silently_removed():
    family = registered_method_family(method_comparison_registry(), "method-numerical-propagation")
    kwargs = {"config": InferenceConfig(**family["inference_config"]),
        "mechanism_passed": {k: True for k in family["contrasts"]}, "family": family["contrasts"]}
    rows = family_fixture(family, -100.)
    with pytest.raises(ValueError, match="complete matched"):
        infer(rows[:-1], **kwargs)
    control = next(r for r in rows if r["configuration"] == "arm-18/full")
    control["status"] = "failure"
    with pytest.raises(ValueError, match="failed"):
        infer(rows, **kwargs)
