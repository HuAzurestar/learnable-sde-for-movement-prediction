"""Finite development-method evidence, never a final scientific verdict.

Saved complete five-seed pairs provide planning SD and simulation variability.
Neither three development blocks nor an exact affine-kernel comparison proves
global numerical accuracy, adequate final power, or an untouched holdout.
"""
from __future__ import annotations

import math
import numpy as np

from .inference import SEEDS, paired_blocks, planning_power
from .method_comparisons import DELTA_M, FAMILY_DEFINITIONS
from .method_mechanisms import (EXACT_REFERENCE,
                                forecast_stream_binding, variance_diagnostic, _digest)
from .method_rollout import SWITCHING_MODELS, _grid
from .method_training import required_slot

VERSION = "pirc17-bounded-method-qualification-analysis-v1"


def check_rows(rows, expected_origins_by_block, slots):
    """Retain failures/missing rows, but reject extra or mislabeled evidence."""
    rows = tuple(rows)
    origins = {origin: block for block, values in expected_origins_by_block.items() for origin in values}
    if not origins or len(origins) != sum(map(len, expected_origins_by_block.values())):
        raise ValueError("unique expected origins and independent blocks required")
    expected = {(origin, seed, slot) for origin in origins for seed in SEEDS for slot in slots}
    lookup, contexts = {}, {}
    for row in rows:
        key = row["origin_id"], row["seed"], row["slot_id"]
        if (type(row["seed"]) is not int or key not in expected or key in lookup
                or row["block_id"] != origins[row["origin_id"]] or row["split"] != "validation"):
            raise ValueError("duplicate, extra, mislabeled or non-validation method row")
        lookup[key] = row
        if row["status"] != "success":
            if row["status"] not in {"failed", "unavailable"} or not row.get("reason"):
                raise ValueError("explicit failure/unavailable reason required")
            continue
        if isinstance(row["score_m"], bool) or not math.isfinite(row["score_m"]):
            raise ValueError("finite common score required")
        context = row["context_sha256"]
        if (not isinstance(context, str) or len(context) != 64
                or any(c not in "0123456789abcdef" for c in context)):
            raise ValueError("bound context hash required")
        times = np.asarray(row["elapsed_seconds"], dtype=float)
        if (times.shape != (4,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
                or np.any(np.abs(times-np.array([60, 300, 900, 1800])) > 30)):
            raise ValueError("registered four actual observed scoring instants required")
        identity = (context, tuple(times))
        if contexts.setdefault(row["origin_id"], identity) != identity:
            raise ValueError("method origin context or scoring instants changed")
        slot = row["slot_id"]
        c = required_slot("arm-01/full" if slot == EXACT_REFERENCE else slot)["components"]
        binding = forecast_stream_binding(slot, "causal_prefix")
        d = row["diagnostics"]
        expected_d = {"origin_id": row["origin_id"], "seed": row["seed"], "run_id": slot,
            "origin_mode": "causal_prefix", "particles": 512, "max_step_seconds": 5.,
            "history_step_seconds": float(c["dt_seconds"]), "mode_step_seconds": float(c["dt_seconds"]),
            "model_kind": c["model"], "condition_names": c["condition"], "propagation": c["poa"],
            "integrator": "exact" if slot == EXACT_REFERENCE else c["integrator"],
            "crn_pair_id": binding["crn_pair_id"]}
        if any(d.get(k) != v for k, v in expected_d.items()):
            raise ValueError("method row changed registered model, origin, stream, N, h or clocks")
        stream_key = ([row["origin_id"], "paired", binding["crn_pair_id"]] if binding["crn_pair_id"]
                      else [row["origin_id"], "independent", c["poa"], slot])
        if d["random_stream_sha256"] != _digest(stream_key):
            raise ValueError("method random-stream identity changed")
        grid = _grid(times, 5., c["dt_seconds"], c["dt_seconds"], 400)
        if (d["integration_steps"] != len(grid)
                or d["history_ticks_seconds"] != [end for _, end, tick, _, _ in grid if tick]
                or d["mode_resampling_times_seconds"] != [end for _, end, _, tick, _ in grid
                    if tick and end < times[-1] and c["model"] in SWITCHING_MODELS]):
            raise ValueError("actual method integration/history/mode grid changed")
    return lookup, expected


def analyze(rows, *, expected_origins_by_block, slots, planning_blocks=46):
    """Whole declared pairs only; never estimate power from successful subsets."""
    if planning_blocks != 46:
        raise ValueError("fixed prospective 46-block candidate required; no adaptive sample search")
    if len(expected_origins_by_block) != 3 or any(len(v) != 1 for v in expected_origins_by_block.values()):
        raise ValueError("fixed three independent development blocks and one origin each required")
    rows = tuple(rows)
    lookup, expected = check_rows(rows, expected_origins_by_block, slots)
    good = {k for k, r in lookup.items() if r["status"] == "success"}
    families = {}
    for family_id, contrasts in FAMILY_DEFINITIONS.items():
        configurations = {s for pair in contrasts.items() for s in pair}
        needed = {k for k in expected if k[2] in configurations}
        missing = sorted(needed-good)
        if missing:
            families[family_id] = {"status": "unavailable", "missing_or_failed_count": len(missing),
                "reason": "complete prespecified block/origin/five-seed family required; no intersection deletion"}
            continue
        selected = [{"origin_id": r["origin_id"], "independent_block_id": r["block_id"],
            "seed": r["seed"], "configuration": r["slot_id"], "status": "success", "score_m": r["score_m"]}
            for key, r in lookup.items() if key in needed]
        blocks, names, differences, counts = paired_blocks(selected, {c: [c, a] for c, a in contrasts.items()})
        means = differences.mean(axis=1)
        results = {}
        for i, name in enumerate(names):
            sd = float(means[:, i].std(ddof=1))
            seed_means = differences[:, :, i].mean(axis=0)
            results[name] = {"paired_block_sd_m": sd,
                "observed_development_improvement_m": float(-means[:, i].mean()),
                "per_block_per_seed_difference_m": differences[:, :, i].tolist(),
                "seed_mean_difference_m": seed_means.tolist(),
                "simulation_mean_standard_error_m": float(seed_means.std(ddof=1)/np.sqrt(len(SEEDS))),
                "planning_scenarios": [{"planning_blocks": planning_blocks, "sd_multiplier": multiplier,
                    **planning_power(paired_sd_m=sd*multiplier, blocks=planning_blocks,
                        true_improvement_m=2*DELTA_M, delta_m=DELTA_M, family_size=len(contrasts), target=.8)}
                    for multiplier in (1., 2.)]}
        families[family_id] = {"status": "computed", "independent_block_ids": blocks,
            "origin_count_by_block": counts, "results": results}
    variance_slots = {"arm-20/full", "arm-21/mc", "arm-21/crn"}
    variance_expected = {k for k in expected if k[2] in variance_slots}
    variance = ({"status": "computed", "gates": variance_diagnostic(
        [r for k, r in lookup.items() if k in variance_expected], expected_origins_by_block=expected_origins_by_block)}
        if variance_expected <= good else {"status": "unavailable",
            "reason": "incomplete prespecified MC/CRN/FP five-seed grid",
            "missing_or_failed_count": len(variance_expected-good)})
    return {"schema_version": VERSION, "expected_forecasts": len(expected), "successful_forecasts": len(good),
        "failed_forecasts": sum(r["status"] == "failed" for r in rows),
        "unavailable_forecasts": sum(r["status"] == "unavailable" for r in rows),
        "unrecorded_forecasts": len(expected-set(lookup)), "families": families, "variance": variance,
        "planning_blocks": planning_blocks, "registered_seeds": list(SEEDS), "delta_m": DELTA_M,
        "alternative": "2*delta fixed before results, not the observed development effect",
        "scope": "Three existing development blocks and five original seeds, including calibration-exposed targets; not final performance or a global numerical certificate.",
        "sd_limit": "1x/2x observed paired SD are sensitivity assumptions, not confidence bounds; three blocks give an imprecise SD.",
        "simulation_se_limit": "Descriptive five-seed Monte Carlo SE on this fixed development set; not a total numerical-error bound or a particle-refinement certificate.",
        "zero_sd_rule": "Normal-planning formula may return power1 for zero observed SD; this is degenerate small-sample evidence, not qualification.",
        "certified": False, "numerically_qualified": False, "power_qualified": False,
        "scientific_claim_authorized": False, "final_eval_reads": 0, "automatic_expansion": False}
