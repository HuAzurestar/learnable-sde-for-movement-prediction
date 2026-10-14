import hashlib
import json
import math
from statistics import NormalDist

import pytest

from experiments.pirc17 import decision_policy as dp
from experiments.pirc17 import qualification_record as qr
from experiments.pirc17.inference import SEEDS
from experiments.pirc17.method_comparisons import DELTA_M, FAMILY_DEFINITIONS


def rows_for(family_id="weighted-es-primary", *, blocks=46, delta=-3*DELTA_M,
             partition="final_eval", mode="causal_prefix"):
    """Software fixture only; no real cohort or forecast data."""
    family = dp.registered_contrasts(family_id)
    controls = {b for a, b in family.values()}
    candidates = {a for a, b in family.values()}
    assert not controls & candidates
    configurations = controls | candidates
    population = {f"block-{b}": [f"origin-{b}"] for b in range(blocks)}
    matrix = "terrain" if family_id.startswith("weighted-es-") else "NEX326-methods"
    rows = [{"origin_id": origin, "independent_block_id": block, "seed": seed,
             "configuration": config, "status": "success",
             "score_m": 500. + (delta if config in candidates else 0.),
             "partition": partition, "origin_mode": mode, "matrix": matrix}
            for block, origins in population.items() for origin in origins
            for seed in SEEDS for config in sorted(configurations)]
    return rows, population


def infer_fixture(family_id="weighted-es-primary", *, mechanism=True, **kwargs):
    rows, population = rows_for(family_id, **kwargs)
    return dp.infer_qualified_family(rows, family_id=family_id,
        expected_origins_by_block=population,
        mechanism_passed={k: mechanism for k in dp.registered_contrasts(family_id)},
        evidence_partition=kwargs.get("partition", "final_eval"),
        origin_mode=kwargs.get("mode", "causal_prefix"))


def test_public_record_is_bound_bounded_and_contains_no_private_rows():
    e = dp.qualification_evidence()
    assert e["source_file_sha256"] == {k: v[1] for k, v in qr.BINDINGS.items()}
    assert e["independent_blocks"] == 3 and len(e["registered_seeds"]) == 5
    assert e["verified_method_counts"]["forecast_arrays_checked"] == 435
    assert e["verified_method_counts"]["observed_time_score_records_checked"] == 1740
    assert sum(map(len, e["method_families"].values())) == 21
    assert e["new_forecasts"] == e["final_eval_reads"] == 0
    assert not e["numerically_qualified"] and not e["scientific_claim_authorized"]
    text = json.dumps(e)
    for forbidden in ("sample_id", "origin_id", "block_id", "positions_m", "targets_m",
                      chr(69) + chr(58) + chr(92), chr(67) + chr(58) + chr(92)):
        assert forbidden not in text


def test_export_reader_rejects_changed_source_even_valid_json(tmp_path):
    path = tmp_path / "source.json"
    path.write_text('{"value":1}', encoding="utf-8")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert qr.read_bound(path, expected) == {"value": 1}
    path.write_text('{"value":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        qr.read_bound(path, expected)


def test_actual_primary_margins_are_recomputed_not_rounded_or_flag_read():
    e = dp.qualification_evidence()
    for name, row in e["terrain_primary"].items():
        margin = max(row["fine_mc_margin_m"], row["coarse_mc_margin_m"]) + row["step_change_plus_margin_m"]
        r = dp.numerical_applicability("weighted-es-primary", name)
        assert r["combined_pilot_margin_m"] == margin == row["combined_pilot_margin_m"]
        assert r["pilot_screen_supported"] and margin < DELTA_M/4
        assert not r["global_numerically_qualified"] and not r["precision_invariant_effect_qualified"]
    assert DELTA_M/4 - e["terrain_primary"]["river"]["combined_pilot_margin_m"] == pytest.approx(.3167157671805594)


@pytest.mark.parametrize("family,name,mode,status", [
    ("weighted-es-lio", "road", "causal_prefix", "LIO_has_no_same_fine_reference_study"),
    ("method-model-structure", "arm-04/gmm_kernel", "causal_prefix", "method_mechanism_diagnostics_not_total_numerical_bound"),
    ("weighted-es-primary", "road", "point_only", "secondary_origin_not_covered"),
])
def test_no_transfer_of_primary_numerical_certificate(family, name, mode, status):
    r = dp.numerical_applicability(family, name, origin_mode=mode)
    assert r["status"] == status
    assert not r["pilot_screen_supported"] and r["combined_pilot_margin_m"] is None


def test_power_is_independently_recomputed_and_does_not_expand46_to47():
    r = dp.planning_gate("method-observation-interval", "arm-06/dt600", blocks=46)
    sd = 84.35413023768275
    independent = NormalDist().cdf(DELTA_M*math.sqrt(46)/sd - NormalDist().inv_cdf(1-.05/8))
    assert r["scenarios"][0]["approximate_power"] == pytest.approx(independent, abs=1e-15)
    assert independent == pytest.approx(.790132480542189)
    assert r["scenarios"][0]["required_blocks_normal_approximation"] == 47
    assert r["scenarios"][1]["required_blocks_normal_approximation"] == 188
    assert not r["qualified"] and not r["automatic_expansion"]


def test_actual_2x_SD_sensitivity_is_reported_not_power_certification():
    low = []
    for family, comparisons in FAMILY_DEFINITIONS.items():
        for name in comparisons:
            r = dp.planning_gate(family, name, blocks=46)
            if r["scenarios"][1]["approximate_power"] < .8:
                low.append(name)
    assert set(low) == {"arm-03/single_gaussian", "arm-04/gmm_kernel", "arm-05/explicit_decomp", "arm-06/dt300", "arm-06/dt600"}


@pytest.mark.parametrize("name", ["arm-10/d2_mc", "arm-10/d2_closed"])
def test_zero_SD_formula_power_one_is_not_power_evidence(name):
    r = dp.planning_gate("method-objective-and-score", name, blocks=46)
    assert r["scenarios"][0]["approximate_power"] == 1.
    assert not r["qualified"] and r["reason"] == "zero_development_SD_is_not_power_evidence"


@pytest.mark.parametrize("blocks", [0, 47, True, 46.0])
def test_no_unregistered_block_growth(blocks):
    with pytest.raises(ValueError, match="one-to-46"):
        dp.planning_gate("weighted-es-primary", "river", blocks=blocks)


def test_minimum_blocks_and_shortfall_recalculate_same_prospective_gate():
    assert not dp.planning_gate("weighted-es-primary", "river", blocks=30)["qualified"]
    assert dp.planning_gate("weighted-es-primary", "river", blocks=46)["qualified"]
    assert dp.planning_gate("weighted-es-primary", "surface", blocks=29)["reason"] == "insufficient_independent_blocks"
    assert dp.planning_gate("weighted-es-lio", "road", blocks=46)["reason"] == "no_comparison_specific_development_SD"


def test_fixed_parameters_budget_counts_and_claim_boundaries():
    p = dp.decision_policy()
    assert p["inference_config"] == vars(dp.CONFIG)
    assert all(set(row) == {"value", "units", "rationale", "source"} and all(row[k] for k in ("units", "rationale", "source"))
               for row in p["parameters"].values())
    assert sum(p["parameters"]["phase_cap:"+phase]["value"] for phase in dp.PHASE_CAP_SECONDS) == 48*3600
    counts = p["workload_with_mechanisms"]
    assert counts["scientific_forecasts_unchanged"] == 11020
    assert counts["total_stochastic_forecasts_including_audits"] == 11513
    assert counts["additional_exact_kernel_forecasts"] == 290
    assert p["closed_preparation"]["further_prechecks"] == 0
    assert not p["final_eval_authorized"] and not p["numerically_qualified"]
    assert "absolute" in p["CRN_rule"] and "never floored" in p["CRN_rule"]
    assert p["sha256"] == qr.digest({k: v for k, v in p.items() if k != "sha256"})


@pytest.mark.parametrize("field", ["numerically_qualified", "final_eval_authorized", "scientific_claim_authorized"])
def test_candidate_flags_cannot_be_promoted_and_rehashed(field):
    p = dp.decision_policy()
    p[field] = True
    p["sha256"] = qr.digest({k: v for k, v in p.items() if k != "sha256"})
    with pytest.raises(ValueError, match="decision policy changed"):
        dp.validate_policy(p)


@pytest.mark.parametrize("delta,expected", [(-3*DELTA_M, "beneficial"), (3*DELTA_M, "harmful"),
    (0., "equivalent"), (-DELTA_M, "inconclusive"), (DELTA_M, "inconclusive")])
def test_known_effects_remain_conditional_not_global_numerical_claims(delta, expected):
    r = infer_fixture(delta=delta)
    assert r["independent_block_count"] == 46
    assert r["inference"]["registered_seeds"] == list(SEEDS)
    assert {v["verdict"] for v in r["results"].values()} == {expected}
    assert {v["precision_invariant_verdict"] for v in r["results"].values()} == {"inconclusive"}
    assert not r["scientific_claim_authorized"]


def test_strong_observed_effect_does_not_override_missing_or_low_planning_power():
    method = infer_fixture("method-observation-interval")
    r = method["results"]["arm-06/dt600"]
    assert r["raw_statistical_verdict"] == "beneficial"
    assert r["verdict"] == "inconclusive" and r["reason"] == "planning_power_below_target"
    assert infer_fixture("weighted-es-lio")["results"]["road"]["reason"] == "no_comparison_specific_development_SD"


@pytest.mark.parametrize("mutation", ["whole_origin_missing", "one_seed_missing", "failed"])
def test_missing_whole_origins_or_failed_rows_never_intersect_away(mutation):
    rows, population = rows_for()
    if mutation == "whole_origin_missing":
        rows = [r for r in rows if r["origin_id"] != "origin-0"]
    elif mutation == "one_seed_missing":
        rows.pop()
    else:
        rows[0].update(status="failed", reason="registered timeout", score_m=None)
    r = dp.infer_qualified_family(rows, family_id="weighted-es-primary", expected_origins_by_block=population,
        mechanism_passed={name: True for name in dp.PRIMARY_FAMILY}, evidence_partition="final_eval")
    assert r["inference"] is None
    assert all(value["verdict"] == "unavailable" for value in r["results"].values())
    assert r["independent_block_count"] == 46


@pytest.mark.parametrize("field,value", [("partition", "validation"), ("matrix", "NEX326-methods"),
    ("origin_mode", "point_only"), ("independent_block_id", "block-X"), ("seed", 9)])
def test_wrong_partition_matrix_pair_or_seed_is_rejected(field, value):
    rows, population = rows_for()
    rows[0][field] = value
    with pytest.raises(ValueError, match="evidence"):
        dp.infer_qualified_family(rows, family_id="weighted-es-primary", expected_origins_by_block=population,
            mechanism_passed={name: True for name in dp.PRIMARY_FAMILY}, evidence_partition="final_eval")


def test_no_secondary_hypothesis_tests_or_block_expansion():
    r = infer_fixture(mode="point_only", blocks=6)
    assert r["inference"] is None
    assert all(v["reason"] == "secondary_origin_descriptive_only" for v in r["results"].values())
    with pytest.raises(ValueError, match="six descriptive blocks"):
        infer_fixture(mode="point_only", blocks=7)
    with pytest.raises(ValueError, match="no secondary-time tests"):
        dp.registered_contrasts("scoring-slot-60s")


def test_failed_mechanism_overrides_strong_effect():
    r = infer_fixture(mechanism=False)
    assert {v["reason"] for v in r["results"].values()} == {"mechanism_gate_failed"}


def factor(primary, supporting, **kwargs):
    return dp.terrain_factor_conclusion(primary, supporting, "road", conflict_unresolved=kwargs.get("conflict", False),
        correlated_group_rule_passed=kwargs.get("owners", True))


def test_factor_verdicts_use_primary_with_supporting_missing_power_disclosed():
    lio = infer_fixture("weighted-es-lio")
    retain = factor(infer_fixture(), lio)
    redundant = factor(infer_fixture(delta=0), lio)
    harmful = factor(infer_fixture(delta=3*DELTA_M), infer_fixture("weighted-es-lio", delta=3*DELTA_M))
    assert retain["verdict"] == "retain" and harmful["verdict"] == "harmful"
    assert redundant["verdict"] == "redundant" and "not absence" in redundant["scope"]
    assert retain["supporting_power_reason"] == "no_comparison_specific_development_SD"
    assert not retain["scientific_claim_authorized"]


def test_conflicting_raw_LIO_is_not_hidden_by_its_missing_power():
    primary = infer_fixture()
    supporting = infer_fixture("weighted-es-lio", delta=3*DELTA_M)
    assert supporting["results"]["road"]["verdict"] == "inconclusive"
    assert factor(primary, supporting)["verdict"] == "inconclusive"
    assert factor(primary, infer_fixture("weighted-es-lio"), owners=False)["verdict"] == "inconclusive"


@pytest.mark.parametrize("which,field,value", [("primary", "partition", "validation"),
    ("primary", "matrix", "NEX326-methods"), ("supporting", "origin_mode", "point_only")])
def test_terrain_verdict_rejects_validation_method_and_secondary_substitution(which, field, value):
    reports = {"primary": infer_fixture(), "supporting": infer_fixture("weighted-es-lio")}
    reports[which][field] = value
    with pytest.raises(ValueError, match="only final-eval"):
        factor(reports["primary"], reports["supporting"])


def test_missing_LIO_cannot_disappear_from_factor_disposition():
    rows, population = rows_for("weighted-es-lio")
    supporting = dp.infer_qualified_family(rows[:-1], family_id="weighted-es-lio",
        expected_origins_by_block=population, mechanism_passed={name: True for name in dp.GROUPS},
        evidence_partition="final_eval")
    verdict = factor(infer_fixture(), supporting)
    assert verdict["verdict"] == "unavailable"
    assert not verdict["scientific_claim_authorized"]
    assert verdict["predictive_scope"] == dp.PREDICTIVE_SCOPE


def test_seed_instability_is_not_overridden_by_successful_power_planning():
    rows, population = rows_for()
    for row in rows:
        if row["configuration"] == "all-terrain" and row["seed"] in SEEDS[-2:]:
            row["score_m"] = 500.+10.
    report = dp.infer_qualified_family(rows, family_id="weighted-es-primary",
        expected_origins_by_block=population, mechanism_passed={name: True for name in dp.PRIMARY_FAMILY},
        evidence_partition="final_eval")
    assert all(r["planning"]["qualified"] for r in report["results"].values())
    assert all(r["verdict"] == "inconclusive" for r in report["results"].values())
    assert all(r["reason"] == "training_seed_direction_unstable" for r in report["results"].values())


def test_frozen_denominator_is_not_mutated_by_callers():
    rows, population = rows_for()
    report = dp.infer_qualified_family(rows, family_id="weighted-es-primary",
        expected_origins_by_block=population, mechanism_passed={name: True for name in dp.PRIMARY_FAMILY},
        evidence_partition="final_eval")
    population["block-0"][0] = "replaced"
    assert report["expected_origins_by_block"]["block-0"] == ["origin-0"]
