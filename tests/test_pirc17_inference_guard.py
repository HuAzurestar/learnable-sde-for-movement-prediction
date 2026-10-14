import copy

import numpy as np
import pytest

from experiments.pirc17.inference import InferenceConfig, SEEDS, infer
from experiments.pirc17.comparison_registry import comparison_registry
from experiments.pirc17.inference_guard import infer_guarded, infer_terrain_family

FAMILY = {"effect": ("candidate", "control")}
DELTA = 41.100259


def records(deltas, *, baseline=1000., seed_changes=None):
    result = []
    for block, delta in enumerate(deltas):
        for s, seed in enumerate(SEEDS):
            for name, score in (("control", baseline), ("candidate", baseline+delta+(seed_changes[s] if seed_changes else 0.))):
                result.append({"configuration": name, "origin_id": f"o{block}",
                    "independent_block_id": f"b{block}", "seed": seed, "status": "success", "score_m": score})
    return result


def run(rows, *, mechanism=True):
    return infer_guarded(rows, config=InferenceConfig(delta_m=DELTA),
                         mechanism_passed={"effect": mechanism}, family=FAMILY)


@pytest.mark.parametrize("baseline", [100., 1000., 1e6])
@pytest.mark.parametrize("sign", [-1, 1])
def test_decimal_practical_boundary_is_never_a_conclusive_claim(baseline, sign):
    report = run(records([sign*DELTA]*30, baseline=baseline))
    assert report["results"]["effect"]["verdict"] == "inconclusive"
    assert report["config"]["delta_m"] == DELTA


@pytest.mark.parametrize("delta,expected", [(-100., "beneficial"), (100., "harmful"), (0., "equivalent"),
    (-DELTA-1e-5, "beneficial"), (DELTA+1e-5, "harmful"), (DELTA-1e-5, "equivalent")])
def test_clear_separation_is_not_lost_to_a_scientific_margin_change(delta, expected):
    result = run(records([delta]*30))
    assert result["results"]["effect"]["verdict"] == expected
    assert result["boundary_roundoff_guard"]["downgraded_comparisons"] == []
    assert result["boundary_roundoff_guard"]["tolerance_m"] < 1e-9


def test_original_intervals_scores_p_values_and_inputs_are_unchanged():
    rows = records(np.linspace(-160, -140, 35))
    original = copy.deepcopy(rows)
    config = InferenceConfig(delta_m=DELTA)
    base = infer(rows, config=config, mechanism_passed={"effect": True}, family=FAMILY)
    guarded = run(rows)
    assert rows == original
    assert guarded["schema_version"] == "pirc17-paired-block-inference-v3"
    assert guarded["base_inference_version"] == base["schema_version"]
    assert guarded["results"] == base["results"]
    for key, value in base.items():
        if key not in ("schema_version", "results"):
            assert guarded[key] == value


def test_boundary_reproducer_retains_raw_evidence_and_downgrade_trace():
    rows = records([-DELTA]*30)
    config = InferenceConfig(delta_m=DELTA)
    base = infer(rows, config=config, mechanism_passed={"effect": True}, family=FAMILY)
    guarded = run(rows)
    raw, result = base["results"]["effect"], guarded["results"]["effect"]
    assert raw["verdict"] == "beneficial"  # The discovered decimal-rounding regression.
    assert result["verdict"] == "inconclusive" and result["unprotected_v2_verdict"] == "beneficial"
    assert result["reason"] == "floating_point_practical_boundary"
    assert result["simultaneous_interval_m"] == raw["simultaneous_interval_m"]
    assert result["holm_adjusted_p_zero"] == raw["holm_adjusted_p_zero"]
    assert guarded["boundary_roundoff_guard"]["downgraded_comparisons"] == ["effect"]


def test_seed_values_indistinguishable_from_zero_are_not_directional_evidence():
    result = run(records([-200.]*30, seed_changes=[0., 0., 0., 200.-1e-12, 210.]))
    assert result["results"]["effect"]["verdict"] == "inconclusive"
    assert result["results"]["effect"]["reason"] == "floating_point_seed_boundary"


def test_existing_failure_precedence_is_preserved_and_no_inconclusive_result_is_promoted():
    assert run(records([-100.]*30), mechanism=False)["results"]["effect"]["reason"] == "mechanism_gate_failed"
    assert run(records([-100.]*29))["results"]["effect"]["reason"] == "insufficient_independent_blocks"
    assert run(records([0.]*29+[10000.]))["results"]["effect"]["verdict"] == "inconclusive"
    rows = records([-100.]*30)
    with pytest.raises(ValueError, match="duplicate"):
        run(rows+[rows[0]])
    with pytest.raises(ValueError, match="complete matched"):
        run(rows[:-1])
    rows[0]["status"] = "failure"
    with pytest.raises(ValueError, match="failed"):
        run(rows)


def test_terrain_family_uses_same_guard_without_changing_old_registry_or_numerical_artifacts():
    registry = comparison_registry()
    rows = []
    for block in range(30):
        for seed in SEEDS:
            for name in ("all-terrain", "base", "loo-road", "loo-river", "loo-worldcover", "loo-surface"):
                rows.append({"configuration": name, "origin_id": f"o{block}", "independent_block_id": f"b{block}",
                    "seed": seed, "status": "success", "score_m": 1000. if name == "all-terrain" else 1000.+DELTA})
    result = infer_terrain_family(registry, "weighted-es-primary", rows, config=InferenceConfig(delta_m=DELTA),
        mechanism_passed={name: True for name in registry["families"]["weighted-es-primary"]["contrasts"]})
    assert all(r["verdict"] == "inconclusive" for r in result["results"].values())
    assert result["schema_version"] == "pirc17-paired-block-inference-v3"
    assert result["factor_verdict_basis"] is True
    assert result["scientific_claim_authorized"] is False
    assert result["terrain_registry_sha256"] == registry["sha256"]
    assert registry == comparison_registry()


def test_descriptive_terrain_slots_or_changed_multiplicity_cannot_use_the_inferential_adapter():
    registry = comparison_registry()
    with pytest.raises(ValueError, match="descriptive"):
        infer_terrain_family(registry, "scoring-slot-60s", [],
            config=InferenceConfig(delta_m=DELTA), mechanism_passed={})
    with pytest.raises(ValueError, match="alpha/B"):
        infer_terrain_family(registry, "weighted-es-primary", [],
            config=InferenceConfig(delta_m=DELTA, alpha=.1), mechanism_passed={})
