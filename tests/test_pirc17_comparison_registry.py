import copy

import pytest

from experiments.pirc17.comparison_registry import GROUPS, comparison_registry, registered_family
from experiments.pirc17.configurations import terrain_configurations
from experiments.pirc17.inference import InferenceConfig, PRIMARY_FAMILY


def test_families_cover_all_ten_configurations_without_shrinking_primary_family():
    registry = comparison_registry()
    primary = registered_family(registry, "weighted-es-primary", inferential=True)
    lio = registered_family(registry, "weighted-es-lio", inferential=True)
    assert primary["contrasts"] == {k: list(v) for k, v in PRIMARY_FAMILY.items()}
    assert lio["contrasts"] == {g: [f"lio-{g}", "base"] for g in GROUPS}
    configurations = {name for family in (primary, lio) for pair in family["contrasts"].values() for name in pair}
    assert configurations == set(terrain_configurations())
    assert registry["bootstrap_iterations"] == 2000
    for family in (primary, lio):
        InferenceConfig(delta_m=41.100259).validate(len(family["contrasts"]))
    assert not registry["final_eval_authorized"] and registry["state"] == "candidate-unsealed"


@pytest.mark.parametrize("seconds", [60, 300, 900, 1800])
def test_scoring_slots_are_registered_and_cannot_be_promoted(seconds):
    registry = comparison_registry()
    name = f"scoring-slot-{seconds}s"
    family = registered_family(registry, name)
    assert family["nominal_scoring_seconds"] == seconds
    assert family["alpha"] is None and not family["factor_verdict_basis"]
    with pytest.raises(ValueError, match="descriptive"):
        registered_family(registry, name, inferential=True)
    with pytest.raises(ValueError, match="primary-mode LOO"):
        registered_family(registry, name, primary_factor="road")


@pytest.mark.parametrize("mode", ["causal_prefix", "known_velocity", "point_only"])
def test_origin_modes_stay_distinct_and_only_primary_loo_can_be_factor_basis(mode):
    registry = comparison_registry(origin_mode=mode)
    assert registry["sha256"] == comparison_registry(origin_mode=mode)["sha256"]
    assert len({comparison_registry(origin_mode=m)["sha256"] for m in
                ("causal_prefix", "known_velocity", "point_only")}) == 3
    with pytest.raises(ValueError, match="primary-mode LOO"):
        registered_family(registry, "weighted-es-lio", primary_factor="road")
    if mode == "causal_prefix":
        for group in GROUPS:
            assert registered_family(registry, "weighted-es-primary", primary_factor=group)["factor_verdict_basis"]
    else:
        with pytest.raises(ValueError, match="primary-mode LOO"):
            registered_family(registry, "weighted-es-primary", primary_factor="road")


@pytest.mark.parametrize("change", ["family", "role", "hash", "mode", "weights", "authorization"])
def test_tampered_registry_does_not_authorize_claims(change):
    registry = copy.deepcopy(comparison_registry())
    if change == "family":
        registry["families"]["weighted-es-primary"]["contrasts"].pop("river")
    elif change == "role":
        registry["families"]["weighted-es-lio"]["factor_verdict_basis"] = True
    elif change == "hash":
        registry["sha256"] = "0"*64
    elif change == "mode":
        registry["origin_mode"] = "point_only"
    elif change == "weights":
        registry["time_weights"] = [0, 0, 0, 1]
    else:
        registry["final_eval_authorized"] = True
    with pytest.raises(ValueError):
        registered_family(registry, "weighted-es-primary", primary_factor="road")


def test_lookup_does_not_mutate_registry_and_unknown_values_are_rejected():
    registry = comparison_registry()
    selected = registered_family(registry, "weighted-es-primary")
    selected["contrasts"].clear()
    assert len(registered_family(registry, "weighted-es-primary")["contrasts"]) == 5
    with pytest.raises(ValueError):
        registered_family(registry, "unregistered")
    with pytest.raises(ValueError):
        comparison_registry(origin_mode="future-position")
