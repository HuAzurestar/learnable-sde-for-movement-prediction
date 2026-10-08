"""Candidate terrain comparison families; never a final-eval authorization.

Primary and supporting families are distinct. Secondary scoring-time results
are descriptive, with no unregistered horizon-specific practical verdicts.
NEX326 method comparisons remain in their separate method registry.
"""
from __future__ import annotations

import copy
import hashlib
import json

from .calibration import NOMINAL_SECONDS, TARGET_TOLERANCE_SECONDS, TIME_WEIGHTS
from .inference import PRIMARY_FAMILY, SEEDS

REGISTRY_VERSION = "pirc17-terrain-comparison-families-v1"
GROUPS = ("road", "river", "worldcover", "surface")
ORIGIN_MODES = ("causal_prefix", "known_velocity", "point_only")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def comparison_registry(*, origin_mode="causal_prefix"):
    if origin_mode not in ORIGIN_MODES:
        raise ValueError("explicit supported origin mode required")
    primary_origin = origin_mode == "causal_prefix"
    families = {
        "weighted-es-primary": {
            "role": "primary" if primary_origin else "secondary-origin",
            "metric": "time_weighted_energy_score_m", "nominal_scoring_seconds": None,
            "contrasts": {k: list(v) for k, v in PRIMARY_FAMILY.items()},
            "inference": "paired-block-bootstrap-t-and-Holm", "alpha": .05,
            "factor_verdict_basis": primary_origin,
        },
        "weighted-es-lio": {
            "role": "supporting-LIO", "metric": "time_weighted_energy_score_m",
            "nominal_scoring_seconds": None,
            "contrasts": {g: [f"lio-{g}", "base"] for g in GROUPS},
            "inference": "paired-block-bootstrap-t-and-Holm", "alpha": .05,
            "factor_verdict_basis": False,
        },
    }
    for seconds in NOMINAL_SECONDS:
        families[f"scoring-slot-{int(seconds)}s"] = {
            "role": "secondary-descriptive", "metric": "energy_score_m",
            "nominal_scoring_seconds": seconds,
            "contrasts": {k: list(v) for k, v in PRIMARY_FAMILY.items()},
            "inference": "none-descriptive-only", "alpha": None,
            "factor_verdict_basis": False,
            "scope": "slot in the same 30-minute forecast, not a separately fitted shorter-horizon experiment",
        }
    payload = {
        "schema_version": REGISTRY_VERSION, "state": "candidate-unsealed",
        "final_eval_authorized": False, "origin_mode": origin_mode,
        "primary_origin_mode": "causal_prefix", "forecast_horizon_seconds": 1800.,
        "nominal_scoring_seconds": list(NOMINAL_SECONDS), "time_weights": list(TIME_WEIGHTS),
        "target_tolerance_seconds": TARGET_TOLERANCE_SECONDS,
        "target_rule": "distinct original observations nearest each nominal slot; earlier observation wins ties; no interpolation",
        "registered_seeds": list(SEEDS), "bootstrap_iterations": 2000,
        "families": families,
        "multiplicity_scope": "separate registered primary and LIO families; no global across-family error-control claim",
        "secondary_metric_policy": "report all metric-registry secondary diagnostics; no hypothesis tests or terrain verdicts from them",
        "factor_rule": "primary LOO plus separately evaluated supporting LIO/conflict/power/mechanism rules; no best-family selection",
        "method_matrix_scope": "NEX326 is separate; these families cannot substitute for method-arm evidence",
    }
    return {**payload, "sha256": _digest(payload)}


def registered_family(registry, family_id, *, inferential=False, primary_factor=None):
    """Return only a canonical family; do not trust caller-supplied role flags.

    This checks family scope only. Sealed-protocol authorization, numerical
    qualification and raw evidence validation remain separate prerequisites.
    """
    if registry != comparison_registry(origin_mode=registry.get("origin_mode")):
        raise ValueError("comparison registry changed or is not canonical")
    if family_id not in registry["families"]:
        raise ValueError("unregistered comparison family")
    family = registry["families"][family_id]
    if inferential and family["inference"] == "none-descriptive-only":
        raise ValueError("secondary scoring slots are descriptive, not an inferential family")
    if primary_factor is not None:
        if (primary_factor not in GROUPS or not family["factor_verdict_basis"]
                or family["contrasts"].get(primary_factor) != ["all-terrain", f"loo-{primary_factor}"]):
            raise ValueError("primary terrain-factor basis must be the registered primary-mode LOO")
    return copy.deepcopy(family)
