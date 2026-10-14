"""Conservative floating-point boundary guard for the formal inference path.

The v2 resampling, p-values and intervals remain byte-for-byte reproducible.
Only otherwise conclusive decisions near practical/seed sign boundaries may be
downgraded. This is a rounding guard, not another statistical error allowance or
a change to the scientific margin. Old empirical audit source hashes stay valid.
"""
from __future__ import annotations

import math
import sys

from .inference import INFERENCE_VERSION as BASE_VERSION
from .inference import PRIMARY_FAMILY, InferenceConfig, infer

INFERENCE_VERSION = "pirc17-paired-block-inference-v3"
ROUNDOFF_MULTIPLIER = 128


def infer_guarded(rows, *, config: InferenceConfig, mechanism_passed, family=PRIMARY_FAMILY):
    rows = list(rows)
    report = infer(rows, config=config, mechanism_passed=mechanism_passed, family=family)
    # The absolute score scale matters: subtracting scores near 1000 m loses
    # more absolute precision than subtracting scores near zero, even when the
    # paired difference is identical. v2 has already rejected nonfinite rows.
    scale = max(1., config.delta_m, max(abs(float(row["score_m"])) for row in rows))
    guard = ROUNDOFF_MULTIPLIER*sys.float_info.epsilon*scale
    changed = []
    for name, result in report["results"].items():
        verdict = result["verdict"]
        if verdict == "inconclusive":
            continue  # Never override existing mechanism/block/tail failures.
        lower, upper = result["simultaneous_interval_m"]
        seeds = result["seed_delta_m"]
        if (lower is None or upper is None or not all(math.isfinite(x) for x in (lower, upper, *seeds))):
            raise ValueError("a conclusive base result requires finite interval and seed effects")
        if verdict == "beneficial":
            margin_ok = upper < -config.delta_m-guard
            seed_ok = sum(x < -guard for x in seeds) >= config.required_agreeing_seeds
        elif verdict == "harmful":
            margin_ok = lower > config.delta_m+guard
            seed_ok = sum(x > guard for x in seeds) >= config.required_agreeing_seeds
        elif verdict == "equivalent":
            margin_ok = lower > -config.delta_m+guard and upper < config.delta_m-guard
            seed_ok = sum(abs(x) < config.delta_m-guard for x in seeds) >= config.required_agreeing_seeds
        else:
            raise ValueError("unknown conclusive inference verdict")
        if not margin_ok or not seed_ok:
            result.update(unprotected_v2_verdict=verdict, unprotected_v2_reason=result["reason"],
                verdict="inconclusive", reason="floating_point_practical_boundary" if not margin_ok
                else "floating_point_seed_boundary")
            changed.append(name)
    report.update(schema_version=INFERENCE_VERSION, base_inference_version=BASE_VERSION,
        boundary_roundoff_guard={"absolute_score_scale_m": scale, "tolerance_m": guard,
            "float64_epsilon_multiplier": ROUNDOFF_MULTIPLIER,
            "rule": "Require strict practical and seed-direction separation beyond 128*float64_epsilon*max(1,delta,max_abs_input_score).",
            "scope": "Conservative decision-only rounding guard; unchanged raw estimates, intervals, p-values, bootstrap draws and scientific delta. Not a statistical/numerical-convergence bound.",
            "downgraded_comparisons": changed})
    return report


def infer_terrain_family(registry, family_id, rows, *, config: InferenceConfig, mechanism_passed):
    """Use the same conservative engine for a canonical terrain family.

This scope adapter does not validate raw forecasts, power or human acceptance.
The sealed protocol must bind the full config, including its practical margin.
"""
    from .comparison_registry import registered_family

    family = registered_family(registry, family_id, inferential=True)
    if config.alpha != family["alpha"] or config.bootstrap_iterations != registry["bootstrap_iterations"]:
        raise ValueError("terrain inference alpha/B differ from registered family")
    result = infer_guarded(rows, config=config, mechanism_passed=mechanism_passed, family=family["contrasts"])
    result.update(terrain_registry_sha256=registry["sha256"], family_id=family_id,
        matrix="terrain", factor_verdict_basis=family["factor_verdict_basis"], scientific_claim_authorized=False)
    return result
