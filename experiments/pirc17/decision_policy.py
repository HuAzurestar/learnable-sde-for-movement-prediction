"""Pre-seal parameter, power and claim-scope binding for PIRC-17.

The statistical estimand is performance of the registered finite-budget
predictors. A predictive-effect verdict is NOT a claim of precision invariance,
an exact continuous-time model, or permission to access final evaluation.
Existing numerical artifacts keep their original uncertified flags.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from .comparison_registry import GROUPS, ORIGIN_MODES, comparison_registry
from .delivery_policy import BLOCK_LIMIT, PHASE_CAP_SECONDS, policy as delivery_policy
from .inference import InferenceConfig, PRIMARY_FAMILY, SEEDS, factor_verdict, planning_power
from .inference_guard import ROUNDOFF_MULTIPLIER, infer_guarded
from .method_comparisons import DELTA_M, FAMILY_DEFINITIONS
from .method_mechanisms import mechanism_registry
from .qualification_record import BINDINGS, OUTPUT as EVIDENCE_PATH, digest, read_bound

VERSION = "pirc17-pre-evaluation-decision-policy-v1"
ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_SHA256 = "780432b174cf59f4e6a2e243b00f6a33e831c9f546ad73c52242b4db0cbf81f0"
POLICY_PATH = ROOT / "experiments/pirc17/plans/pre-evaluation-decision-policy-v1.json"
CONFIG = InferenceConfig(delta_m=DELTA_M)
POWER_TARGET = .8
PREDICTIVE_SCOPE = "registered-finite-budget-predictor-performance"


def qualification_evidence():
    evidence = read_bound(EVIDENCE_PATH, EVIDENCE_SHA256)
    if (evidence["sha256"] != digest({k: v for k, v in evidence.items() if k != "sha256"})
            or evidence["source_file_sha256"] != {k: v[1] for k, v in BINDINGS.items()}):
        raise ValueError("closed development record changed")
    return evidence


def registered_contrasts(family_id):
    if family_id in FAMILY_DEFINITIONS:
        return {name: (name, control) for name, control in FAMILY_DEFINITIONS[family_id].items()}
    if family_id == "weighted-es-primary":
        return dict(PRIMARY_FAMILY)
    if family_id == "weighted-es-lio":
        return {group: (f"lio-{group}", "base") for group in GROUPS}
    raise ValueError("registered primary/supporting family required; no secondary-time tests")


def planning_gate(family_id, comparison, *, blocks, origin_mode="causal_prefix"):
    """Use only frozen development SD, never final observed effects/variance."""
    contrasts = registered_contrasts(family_id)
    if comparison not in contrasts or origin_mode not in ORIGIN_MODES:
        raise ValueError("unregistered comparison or origin mode")
    if type(blocks) is not int or not 1 <= blocks <= BLOCK_LIMIT:
        raise ValueError("fixed one-to-46 independent blocks required; no adaptive expansion")
    evidence = qualification_evidence()
    source = (evidence["terrain_primary"].get(comparison) if family_id == "weighted-es-primary"
              else evidence["method_families"].get(family_id, {}).get(comparison))
    sd = source["paired_block_sd_m"] if source is not None else None
    scenarios = [] if sd is None else [{"sd_multiplier": multiplier,
        **planning_power(paired_sd_m=sd*multiplier, blocks=blocks, true_improvement_m=2*DELTA_M,
                         delta_m=DELTA_M, family_size=len(contrasts), target=POWER_TARGET)}
        for multiplier in (1., 2.)]
    reason = ("secondary_origin_descriptive_only" if origin_mode != "causal_prefix"
              else "insufficient_independent_blocks" if blocks < CONFIG.minimum_blocks
              else "no_comparison_specific_development_SD" if sd is None
              else "zero_development_SD_is_not_power_evidence" if sd == 0
              else "planning_power_below_target" if scenarios[0]["approximate_power"] < POWER_TARGET
              else "registered_1x_SD_planning_requirement_met")
    return {"family_id": family_id, "comparison": comparison, "blocks": blocks,
        "origin_mode": origin_mode, "paired_block_sd_m": sd, "scenarios": scenarios,
        "qualified": reason == "registered_1x_SD_planning_requirement_met", "reason": reason,
        "evidence_sha256": EVIDENCE_SHA256, "automatic_expansion": False,
        "scope": "Prospective normal planning at assumed2delta, not observed post-hoc power or guaranteed bootstrap power. 2xSD is sensitivity, not a new sample target."}


def numerical_applicability(family_id, comparison, *, origin_mode="causal_prefix"):
    if comparison not in registered_contrasts(family_id) or origin_mode not in ORIGIN_MODES:
        raise ValueError("unregistered numerical applicability scope")
    evidence = qualification_evidence()
    terrain = evidence["terrain_primary"].get(comparison) if family_id == "weighted-es-primary" else None
    if terrain is not None and origin_mode == "causal_prefix":
        margin = max(terrain["fine_mc_margin_m"], terrain["coarse_mc_margin_m"]) + terrain["step_change_plus_margin_m"]
        supported = margin <= DELTA_M/4
        status = "bounded_development_primary_screen_passed" if supported else "bounded_development_primary_screen_failed"
    else:
        margin, supported = None, False
        status = ("secondary_origin_not_covered" if origin_mode != "causal_prefix"
                  else "LIO_has_no_same_fine_reference_study" if family_id == "weighted-es-lio"
                  else "method_mechanism_diagnostics_not_total_numerical_bound")
    return {"status": status, "pilot_screen_supported": supported, "combined_pilot_margin_m": margin,
        "epsilon_m": DELTA_M/4, "evidence_sha256": EVIDENCE_SHA256,
        "global_numerically_qualified": False, "precision_invariant_effect_qualified": False,
        "scope": "Keep fixed-budget performance separate from resolution-invariant/model-limit claims. Three-block finite-reference evidence does not certify all final blocks, metrics or model variants."}


def _parameter(value, units, rationale, source):
    return {"value": value, "units": units, "rationale": rationale, "source": source}


def decision_policy():
    """Canonical supplement; never mutates the historical finite-delivery plan."""
    base = delivery_policy()
    mechanisms = mechanism_registry()
    evidence = qualification_evidence()
    families = ["weighted-es-primary", "weighted-es-lio", *FAMILY_DEFINITIONS]
    parameters = {
        "particles": _parameter(512, "paths/forecast", "Closed bounded development screen; not global optimality.", EVIDENCE_SHA256),
        "max_step": _parameter(5., "s", "Chosen coarse grid; no automatic halving.", EVIDENCE_SHA256),
        "nominal_horizon": _parameter(1800., "s", "Existing qualified common support;60min has only4 development blocks.", base["sha256"]),
        "scoring_times": _parameter([60., 300., 900., 1800.], "s from origin", "Four original-observation slots within one forecast.", base["sha256"]),
        "time_weights": _parameter([.25]*4, "fraction", "Equal registered scoring-time weights.", comparison_registry()["sha256"]),
        "target_tolerance": _parameter(30., "s", "Nearest original observation, earlier ties; no interpolated truth.", base["sha256"]),
        "observation_gap_max": _parameter(60., "s", "Frozen data-eligibility gap criterion.", base["sha256"]),
        "seeds": _parameter(list(SEEDS), "seed labels", "Five fixed paired simulation streams, not independent data blocks.", "HC-03/EC-09"),
        "delta": _parameter(DELTA_M, "m weighted ES", "5% of validation inertial ES; not a validated operational-utility threshold.", "8f7289c79c1625e57b6904943ae8e9724e3f9558d451c1fb82edf04c1a41818d"),
        "alpha": _parameter(.05, "family probability", "Separate nominal95% families; no across-family FWER claim.", "HC-03/EC-10"),
        "bootstrap_iterations": _parameter(2000, "resamples/check", "Studentized paired-block bootstrap with independent fixed-seed tail check.", "HC-03/EC-09"),
        "bootstrap_seeds": _parameter([20260926, 20260927], "seed labels", "Prespecified primary and independent Monte Carlo tail checks.", "experiments/pirc17/inference.py"),
        "minimum_blocks": _parameter(30, "independent blocks", "Minimum for categorical population inference, not a power guarantee.", "HC-03/EC-09"),
        "primary_block_cap": _parameter(46, "independent blocks", "Maximum1x-SD terrain planning requirement; do not grow to47/188 for method power.", BINDINGS["terrain_power"][1]),
        "secondary_block_cap": _parameter(6, "independent blocks/mode", "Two bounded descriptive mode diagnostics, not population hypothesis tests.", base["sha256"]),
        "agreeing_seeds": _parameter(4, "of5 fixed seeds", "Strict direction or within-delta stability, including roundoff guard.", "HC-03/EC-09"),
        "tail_shift_max": _parameter(DELTA_M/4, "m", "Independent-check interval endpoint shift tolerance; failure remains inconclusive.", "experiments/pirc17/inference.py"),
        "tail_draws_min": _parameter(10, "expected draws/tail", "B*alpha/(2*family_size); no underresolved simultaneous bands.", "experiments/pirc17/inference.py"),
        "roundoff_multiplier": _parameter(ROUNDOFF_MULTIPLIER, "float64 epsilon times absolute-score scale", "Strict boundary guard, not an error or significance allowance.", "experiments/pirc17/inference_guard.py"),
        "power_target": _parameter(.8, "probability", "Prospective1x-SD normal planning gate;2xSD sensitivity always reported.", BINDINGS["terrain_power"][1]),
        "power_alternative": _parameter(2*DELTA_M, "m improvement", "Fixed assumed alternative, never observed final/development effect.", BINDINGS["terrain_power"][1]),
        "numerical_pilot_epsilon": _parameter(DELTA_M/4, "m weighted-contrast error diagnostic", "Existing aggregate pilot allowance;not every trajectory/time or global truth bound.", BINDINGS["terrain_coarse"][1]),
        "numerical_pilot_normal_quantile": _parameter(3.3172473615524347, "standard-normal units", "Original55-diagnostic family, unchanged.", BINDINGS["terrain_coarse"][1]),
        "numerical_pilot_family_size": _parameter(55, "diagnostics", "Retain original diagnostic multiplicity and failures.", BINDINGS["terrain_coarse"][1]),
    }
    for phase, seconds in PHASE_CAP_SECONDS.items():
        parameters[f"phase_cap:{phase}"] = _parameter(seconds, "active compute s across attempts/resumes", "Fixed phase envelope; no transfers or reset; executable enforcement belongs to DEV-04.", base["sha256"])
    payload = {"schema_version": VERSION, "state": "candidate-unsealed", "parameters": parameters,
        "inference_config": asdict(CONFIG), "closed_evidence_sha256": EVIDENCE_SHA256,
        "base_delivery_policy_sha256": base["sha256"], "method_mechanism_registry_sha256": mechanisms["sha256"],
        "workload_with_mechanisms": mechanisms["pre_seal_workload_supplement"],
        "family_rules": {family: {"contrasts": {k: list(v) for k, v in registered_contrasts(family).items()},
            "planning_at46": {name: planning_gate(family, name, blocks=46) for name in registered_contrasts(family)},
            "numerical_applicability": {name: numerical_applicability(family, name) for name in registered_contrasts(family)}}
            for family in families},
        "predictive_estimand": PREDICTIVE_SCOPE,
        "claim_rule": "Any predictive categorical result is conditional on the sealed finite-budget algorithm, observed cohort and horizon. It is NOT precision-invariant or a continuous-time-model effect. Numerical convergence status is reported separately and never promoted from the predictive verdict.",
        "effect_priority": ["Missing/failed complete paired evidence: unavailable, no success intersection.",
            "Secondary modes: descriptive only, no hypothesis tests or factor verdicts.",
            "Unresolved ownership/LOO-LIO conflict: inconclusive.",
            "Mechanism failure, low/missing/zero-SD planning power, minimum-block failure: inconclusive.",
            "Unbounded or unstable bootstrap tails, practical/seed boundary overlap: inconclusive.",
            "Otherwise the guarded simultaneous-interval/Holm/4of5 rule, with mandatory finite-budget scope."],
        "numerical_claim_rule": "A resolution-invariant or globally numerically-converged effect remains inconclusive: no available artifact establishes that claim. Existing bounded terrain primary screen and separately defined method diagnostics remain reportable as such, not blanket qualifications.",
        "LIO_rule": "Supporting LIO has no comparison-specific planning SD; categorical LIO inference remains inconclusive. Retain effect/intervals, do not replace primary LOO. Check opposite raw practical directions conservatively even when LIO power is missing.",
        "aliases_rule": "Zero-SD identical score-only predictors are not powered predictive-change experiments. Keep both rows/common-score results and separately report score-estimator diagnostics. Full anchors and EM aliases never increase independent counts.",
        "CRN_rule": "Always retain absolute MC-FP and CRN-FP variances in m^2 with any ratio; zero denominator is unavailable, never floored. Even a tiny positive denominator does not establish speedup or universal efficiency; no post-hoc near-zero cutoff.",
        "correlated_group_rule": "Owner closure and independent refits must be verified from the exact matrix. Shared interaction owners remain joint; no unique/additive/synergy attribution. Unresolved conflicts force inconclusive; equivalence refers only to increment in the full model.",
        "numerical_parameters_by_reference": {"forecast_input_training_and_runtime": base["sha256"],
            "method_statistic_operator_threshold_units": mechanisms["sha256"],
            "required_metric_registry_sha256": hashlib.sha256((ROOT/"experiments/pirc17/metric_registry.json").read_bytes()).hexdigest()},
        "closed_preparation": {"empirical_cap_seconds": 1800., "used_seconds": 842.8162042999876,
            "remaining_seconds": 957.1837957000124, "further_prechecks": 0,
            "scope": "Separate bounded preparation account, not all historical PIRC17 work; unused balance is not authorization."},
        "reference_cost_update": {"already_completed_reference_forecasts": 15, "mean_seconds": .6240022866637446,
            "conditional_290_reference_seconds": 180.96066313248593, "source": BINDINGS["methods"][1],
            "scope": "Existing saved timings, no new probe, not isolated benchmark/upper bound; inside old method phase cap."},
        "required_before_execution": ["DEV-03 exact data/code/protocol seal with exposure account",
            "DEV-04 full executable matrix and durable phase caps", "TEST-01/REVIEW-01", "human ACCEPT-01"],
        "required_after_execution": ["Complete independent saved-output audit", "all evidence cards and bounded conclusions",
            "full editable manuscript and PDF", "REVIEW-02 and human ACCEPT-02"],
        "source_limits": evidence["limitations"], "automatic_parameter_expansion": False,
        "final_eval_authorized": False, "numerically_qualified": False, "scientific_claim_authorized": False}
    return {**payload, "sha256": digest(payload)}


def validate_policy(value):
    if value != decision_policy():
        raise ValueError("decision policy changed; fixed scope, evidence and rules required")


def infer_qualified_family(rows, *, family_id, expected_origins_by_block, mechanism_passed,
                           evidence_partition, origin_mode="causal_prefix"):
    """Complete-population statistical consumer, NOT an access/acceptance guard.

The sealed runner owns authorization, raw provenance, real mechanism recompute
and forecast identities. This layer rejects partition/matrix substitution and
missing entire origins before inference; a caller-supplied True is not proof of
raw mechanism validity. Every output remains unauthorized for publication.
"""
    family = registered_contrasts(family_id)
    if evidence_partition not in {"validation", "final_eval"} or origin_mode not in ORIGIN_MODES:
        raise ValueError("explicit evidence partition and origin mode required")
    if origin_mode != "causal_prefix" and len(expected_origins_by_block) > 6:
        raise ValueError("secondary origin mode is limited to six descriptive blocks")
    matrix = "terrain" if family_id.startswith("weighted-es-") else "NEX326-methods"
    if (not expected_origins_by_block or len(expected_origins_by_block) > BLOCK_LIMIT
            or any(not isinstance(b, str) or not b or not isinstance(origins, (list, tuple))
                   or len(origins) != 1 for b, origins in expected_origins_by_block.items())):
        raise ValueError("frozen nonempty blocks with one origin per block required")
    origins = {origin: block for block, values in expected_origins_by_block.items() for origin in values}
    if len(origins) != len(expected_origins_by_block) or any(not isinstance(o, str) or not o for o in origins):
        raise ValueError("unique nonempty frozen origin identities required")
    if set(mechanism_passed) != set(family) or any(type(v) is not bool for v in mechanism_passed.values()):
        raise ValueError("explicit per-comparison mechanism status required")
    configurations = {v for pair in family.values() for v in pair}
    expected = {(o, seed, config) for o in origins for seed in SEEDS for config in configurations}
    rows = list(rows)
    seen, failures = set(), []
    for row in rows:
        key = row["origin_id"], row["seed"], row["configuration"]
        if (key not in expected or key in seen or type(row["seed"]) is not int
                or row["independent_block_id"] != origins[row["origin_id"]]
                or row.get("partition") != evidence_partition or row.get("matrix") != matrix
                or row.get("origin_mode") != origin_mode):
            raise ValueError("duplicate, extra, wrong-block/partition/matrix/mode evidence")
        seen.add(key)
        if row["status"] != "success":
            if row["status"] not in {"failed", "unavailable"} or not isinstance(row.get("reason"), str) or not row["reason"].strip():
                raise ValueError("failed required row needs explicit status/reason")
            failures.append(key)
        elif (isinstance(row["score_m"], bool) or not isinstance(row["score_m"], (int, float))
              or not math.isfinite(row["score_m"])):
            raise ValueError("finite common score required")
    missing = sorted(expected-seen)
    result = {"schema_version": VERSION, "family_id": family_id, "matrix": matrix,
        "partition": evidence_partition, "origin_mode": origin_mode, "predictive_scope": PREDICTIVE_SCOPE,
        "expected_origins_by_block": copy.deepcopy(expected_origins_by_block),
        "expected_rows": len(expected), "missing_rows": [list(k) for k in missing],
        "failed_rows": [list(k) for k in failures], "scientific_claim_authorized": False,
        "independent_block_count": len(expected_origins_by_block), "results": {}, "inference": None}
    if missing or failures:
        result["results"] = {name: {"verdict": "unavailable", "reason": "incomplete_prespecified_paired_family"} for name in family}
        return result
    if origin_mode != "causal_prefix":
        result["results"] = {name: {"verdict": "inconclusive", "reason": "secondary_origin_descriptive_only"} for name in family}
        return result
    inference = infer_guarded(rows, config=CONFIG, mechanism_passed=mechanism_passed, family=family)
    result["inference"] = inference  # Preserve raw estimates, intervals, tails and p-values.
    for name, raw in inference["results"].items():
        power = planning_gate(family_id, name, blocks=len(expected_origins_by_block))
        numeric = numerical_applicability(family_id, name)
        # Existing numerical failures are never changed to qualified. Categorical
        # predictive inference names the actual registered finite algorithm only.
        reason, verdict = raw["reason"], raw["verdict"]
        if not mechanism_passed[name]:
            reason, verdict = "mechanism_gate_failed", "inconclusive"
        elif not power["qualified"]:
            reason, verdict = power["reason"], "inconclusive"
        result["results"][name] = {"verdict": verdict, "reason": reason,
            "raw_statistical_verdict": raw["verdict"], "planning": power,
            "numerical_applicability": numeric, "predictive_scope": PREDICTIVE_SCOPE,
            "precision_invariant_verdict": "inconclusive"}
    return result


def terrain_factor_conclusion(primary, supporting, group, *, conflict_unresolved,
                              correlated_group_rule_passed, unavailable_reason=None):
    """No method-table/validation substitution or favorable-family selection."""
    if group not in GROUPS:
        raise ValueError("registered terrain factor required; history has no terrain verdict")
    for report, family_id in ((primary, "weighted-es-primary"), (supporting, "weighted-es-lio")):
        if (report["matrix"] != "terrain" or report["family_id"] != family_id
                or report["partition"] != "final_eval" or report["origin_mode"] != "causal_prefix"
                or report.get("predictive_scope") != PREDICTIVE_SCOPE):
            raise ValueError("only final-eval primary-mode terrain LOO/LIO can support a factor verdict")
    if primary["expected_origins_by_block"] != supporting["expected_origins_by_block"]:
        raise ValueError("LOO/LIO population mismatch")
    loo, lio = primary["results"][group], supporting["results"][group]
    if any(r["verdict"] == "unavailable" for r in (loo, lio)):
        unavailable = factor_verdict(None, None, conflict_unresolved=conflict_unresolved,
            correlated_group_rule_passed=correlated_group_rule_passed, power_qualified=False,
            unavailable_reason=unavailable_reason or "required_LOO_or_LIO_evidence_unavailable")
        return {**unavailable, "predictive_scope": PREDICTIVE_SCOPE,
            "precision_invariant_verdict": "unavailable", "scientific_claim_authorized": False}
    raw_loo = primary["inference"]["results"][group]["verdict"]
    raw_lio = supporting["inference"]["results"][group]["verdict"]
    if type(conflict_unresolved) is not bool or type(correlated_group_rule_passed) is not bool:
        raise ValueError("explicit conflict and owner-closure status required")
    conflict = conflict_unresolved or {raw_loo, raw_lio} == {"beneficial", "harmful"}
    output = factor_verdict(loo, lio, conflict_unresolved=conflict,
        correlated_group_rule_passed=correlated_group_rule_passed,
        power_qualified=loo["planning"]["qualified"])
    return {**output, "predictive_scope": PREDICTIVE_SCOPE,
        "supporting_power_reason": lio["reason"], "precision_invariant_verdict": "inconclusive",
        "scientific_claim_authorized": False}


if __name__ == "__main__":
    value = decision_policy()
    with POLICY_PATH.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"path": str(POLICY_PATH.relative_to(ROOT)), "sha256": value["sha256"],
                      "final_eval_authorized": False}))
