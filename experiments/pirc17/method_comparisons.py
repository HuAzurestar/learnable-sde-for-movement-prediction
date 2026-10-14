"""Candidate NEX326 comparison contract, separate from terrain attribution.

This freezes comparison roles, not evaluation permission or scientific results.
The historical runner still needs the explicit PIRC-17 forecast-task adapter.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from experiments.nex326.fidelity import build_fidelity_report
from experiments.nex326.specification import load_experiment_spec
from .comparison_registry import comparison_registry
from .inference import InferenceConfig, SEEDS
from .inference_guard import INFERENCE_VERSION, infer_guarded
from .workload import method_inventory

VERSION = "pirc17-method-comparison-families-v1"
DELTA_M = 41.100259
ROOT = Path(__file__).resolve().parents[2]

# Groups follow the frozen study questions, not observed effect sizes. All Full
# anchors denote NEX326-FULL-v2; do not stack them as independent observations.
FAMILY_DEFINITIONS = {
    "method-model-structure": {
        "arm-02/pointwise": "arm-01/full",
        "arm-03/single_gaussian": "arm-01/full",
        "arm-04/gmm_kernel": "arm-01/full",
        "arm-05/explicit_decomp": "arm-01/full",
    },
    "method-observation-interval": {
        "arm-06/dt30": "arm-01/full",
        "arm-06/dt120": "arm-01/full",
        "arm-06/dt300": "arm-01/full",
        "arm-06/dt600": "arm-01/full",
    },
    "method-objective-and-score": {
        "arm-08/qmle": "arm-07/full",
        "arm-09/mixed": "arm-07/full",
        "arm-09/pure_es": "arm-07/full",
        "arm-10/d2_mc": "arm-07/full",
        "arm-10/d2_closed": "arm-07/full",
    },
    "method-transfer-adaptation": {
        "arm-12/scratch": "arm-11/full",
        "arm-14/reptile": "arm-11/full",
        "arm-15/drift_only": "arm-11/full",
        "arm-15/two_step": "arm-11/full",
    },
    "method-numerical-propagation": {
        "arm-19/em": "arm-18/full",
        "arm-19/euler": "arm-18/full",
        "arm-21/mc": "arm-20/full",
        "arm-21/crn": "arm-20/full",
    },
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def method_comparison_registry(*, origin_mode="causal_prefix"):
    forecast = comparison_registry(origin_mode=origin_mode)
    inventory = method_inventory()
    slots = {r["slot_id"]: r for r in inventory["slots"]}
    spec = load_experiment_spec()
    mechanisms = {}
    for arm, subconfig in spec.executions:
        slot = f"arm-{arm.arm_id:02d}/{subconfig['subconfig_id']}"
        mechanisms[slot] = {**arm.mechanism_gate, **subconfig.get("mechanism_gate", {})}
    families, candidate_family = {}, {}
    config = InferenceConfig(delta_m=DELTA_M)
    for family_id, candidates in FAMILY_DEFINITIONS.items():
        config.validate(len(candidates))
        contrasts = {}
        for candidate, control in candidates.items():
            if (candidate in candidate_family or slots[candidate]["disposition"] != "REQUIRED"
                    or not slots[control]["full_anchor"]
                    or slots[control]["disposition"] != "REQUIRED"):
                raise ValueError("method contrasts require unique required candidates and registered Full anchors")
            candidate_family[candidate] = family_id
            contrasts[candidate] = [candidate, control]
        families[family_id] = {
            "role": "method-predictive" if origin_mode == "causal_prefix" else "secondary-origin",
            "metric": "time_weighted_energy_score_m", "direction": "lower_is_better",
            "contrasts": contrasts, "inference": "paired-block-bootstrap-t-and-Holm",
            "alpha": config.alpha, "inference_config": asdict(config),
            "inference_engine": INFERENCE_VERSION,
            "expected_draws_per_simultaneous_tail": config.bootstrap_iterations*config.alpha/(2*len(contrasts)),
            "terrain_factor_verdict_basis": False,
        }
    full_slots = {k for k, v in slots.items() if v["full_anchor"]}
    excluded = {k for k, v in slots.items() if v["disposition"] == "EXCLUDED"}
    replay = {"arm-06/dt60"}
    if (len(candidate_family) != 21 or len(full_slots) != 6 or len(excluded) != 8
            or set(slots) != set(candidate_family) | full_slots | excluded | replay
            or slots["arm-06/dt60"]["components"] != slots["arm-01/full"]["components"]):
        raise ValueError("all 36 method slots must retain exactly one declared comparison role")
    slot_registry = {}
    for name, row in slots.items():
        role = ("policy-excluded" if name in excluded else "shared-Full-anchor" if name in full_slots
                else "identical-Full-replay" if name in replay else "predictive-contrast-candidate")
        slot_registry[name] = {
            "arm_id": row["arm_id"], "disposition": row["disposition"], "role": role,
            "family_id": candidate_family.get(name), "exclusion_reason": row["exclusion_reason"],
            "component_identity_sha256": row["component_identity_sha256"],
            "historical_mechanism_definition": mechanisms[name],
            "mechanism_definition_is_predictive_effect_evidence": False,
        }
    fidelity = build_fidelity_report(spec)
    if fidelity["failed_route_count"]:
        raise ValueError("method fidelity routes changed")
    sources = ("experiments/nex326/experiment.json", "experiments/nex326/pirc19_scope_policy.json",
               "experiments/nex326/runner.py", "experiments/nex326/model.py",
               "experiments/nex326/fidelity.py", "experiments/pirc17/inference.py",
               "experiments/pirc17/inference_guard.py", "experiments/pirc17/method_comparisons.py")
    payload = {
        "schema_version": VERSION, "matrix": "NEX326-methods", "state": "candidate-unsealed",
        "final_eval_authorized": False, "numerically_qualified": False,
        "forecast_contract": {k: forecast[k] for k in ("origin_mode", "primary_origin_mode",
            "forecast_horizon_seconds", "nominal_scoring_seconds", "time_weights",
            "target_tolerance_seconds", "target_rule")},
        "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        "registered_seeds": list(SEEDS), "slots": slot_registry, "families": families,
        "counts": {"arms": 22, "slots": 36, "required_slots": 28, "excluded_slots": 8,
                   "shared_Full_anchors": 6, "identical_Full_replay_slots": 1,
                   "predictive_comparisons": 21, "predictive_families": 5},
        "delta_source": {"value_m": DELTA_M, "rule": "5 percent of the existing equal-block validation inertial weighted ES, rounded to six decimals",
            "validation_scale_sha256": "8f7289c79c1625e57b6904943ae8e9724e3f9558d451c1fb82edf04c1a41818d",
            "limitation": "Shared task-scale margin, not a validated operational search-utility threshold."},
        "sign_convention": "candidate_ES_minus_control_ES; negative favours candidate",
        "uncertainty_unit": "independent PIRC-20 block; equal matched origins and five seeds within block",
        "multiplicity_scope": "Holm zero tests and Bonferroni bootstrap-t intervals within each named family; nominal 95 percent per family, no global across-family or across-matrix claim",
        "common_scoring_rule": "All predictive comparisons use the same PIRC-17 weighted 2D ensemble ES estimator. Arm-specific score estimators are separate diagnostics, never different primary scorers on opposite sides of a contrast.",
        "descriptive_scoring_slots": list(forecast["nominal_scoring_seconds"]),
        "diagnostic_comparisons": {
            "Full-dt60-replay": {"pair": ["arm-06/dt60", "arm-01/full"],
                "role": "control-reproducibility-only", "predictive_change_claim_allowed": False},
            "EM-alias-consistency": {"pair": ["arm-19/em", "arm-19/euler"],
                "role": "same-implementation-alias-check", "independent_replication_claim_allowed": False},
            "d2-score-estimator": {"pair": ["arm-10/d2_closed", "arm-10/d2_mc"],
                "role": "numerical-score-diagnostic", "predictive_improvement_implied": False},
            "MC-CRN": {"pair": ["arm-21/crn", "arm-21/mc"],
                "role": "paired-estimator-variance-and-cost", "predictive_improvement_implied": False},
        },
        "physical_reuse_approved": False,
        "shared_control_rule": "Keep every slot and current arm-seeded execution identity; do not pool repeated Full anchors or aliases to increase independent block/seed counts.",
        "fidelity": fidelity,
        "mechanism_evidence_limits": {
            "recompute": "Recompute statistic/operator/threshold from the bound raw evidence; never trust a saved passed flag.",
            "variance_reduction_factor": "Legacy covariance-based generated-draw diagnostic, not measured variance reduction of the actual multi-horizon forecast contrast.",
            "split_exact_error": "Legacy exact label uses one propagation substep; it is not an analytic continuous-time solution.",
            "density_mass_error": "Legacy statistic checks mixture-weight normalization, not a numerical PDE mass-conservation study.",
            "closed_mc_relative_error": "Legacy first-prediction, moment-matched-Gaussian diagnostic generates 8192 samples; it is not full-cohort qualification and must not run implicitly under the new bounded budget.",
        },
        "before_final_evaluation": ["Bind explicit causal origins, actual horizons, complete paired populations and permissible conditions.",
            "Freeze fitted-model identity, requested/effective method samples and step policy; the terrain N/h certificate does not certify method routes.",
            "Qualify numerical/mechanism and per-comparison power evidence in permitted development data, with fixed finite cost caps.",
            "Bind this candidate registry to the sealed protocol and exact ACCEPT-01 authorization; this module provides neither."],
        "terrain_scope": "Ten terrain configurations and their primary LOO/supporting LIO evidence remain separate and required.",
    }
    return {**payload, "sha256": _digest(payload)}


def registered_method_family(registry, family_id):
    """Canonical scope lookup only; does not authorize data access or a claim."""
    mode = registry.get("forecast_contract", {}).get("origin_mode")
    if registry != method_comparison_registry(origin_mode=mode):
        raise ValueError("method comparison registry is not canonical")
    if family_id not in registry["families"]:
        raise ValueError("unregistered method predictive family")
    return copy.deepcopy(registry["families"][family_id])


def registered_method_slot(registry, slot_id, *, predictive_change_claim=False):
    mode = registry.get("forecast_contract", {}).get("origin_mode")
    if registry != method_comparison_registry(origin_mode=mode):
        raise ValueError("method comparison registry is not canonical")
    if slot_id not in registry["slots"]:
        raise ValueError("unregistered method slot")
    slot = registry["slots"][slot_id]
    if predictive_change_claim and slot["role"] != "predictive-contrast-candidate":
        raise ValueError("excluded slots, Full anchors and identical replays cannot claim a method change")
    return copy.deepcopy(slot)


def infer_method_family(registry, family_id, rows, *, mechanism_passed):
    """Run the canonical statistical rule, not final scientific acceptance.

The caller must separately bind input identity, actual forecast-task pairing,
mechanism evidence, numerical/power qualification and final-eval permission.
"""
    family = registered_method_family(registry, family_id)
    result = infer_guarded(rows, config=InferenceConfig(**family["inference_config"]),
                          mechanism_passed=mechanism_passed, family=family["contrasts"])
    result.update(method_registry_sha256=registry["sha256"], family_id=family_id,
                  matrix="NEX326-methods", scientific_claim_authorized=False)
    return result
