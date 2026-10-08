"""Exact conditional bootstrap-tail oracles on software-only two-point fixtures.

The bootstrap count distribution is enumerated analytically; no large-Monte-
Carlo reference is treated as truth. These are not research samples or a proof
of population coverage. No dataset loader or final-eval evidence is accessed.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np

from . import inference
from .inference import InferenceConfig, PRIMARY_FAMILY, SEEDS, infer
from .qualification import _hash

CASES = ((30, 1), (30, 3), (30, 15), (62, 1), (62, 6), (62, 31))


def exact_two_point_pivots(blocks, high_count):
    """Enumerate Binomial(n, k/n) bootstrap counts for a 0/1 sample.

    Translation and positive scaling leave the studentized pivot unchanged.
    The formula is independent of production resampling/studentization code.
    """
    if type(blocks) is not int or type(high_count) is not int or not 0 < high_count < blocks:
        raise ValueError("integer block count and nonconstant two-point sample required")
    probability = high_count/blocks
    pivots, masses = [], []
    for count in range(blocks+1):
        mean = count/blocks
        error = math.sqrt(mean*(1-mean)/(blocks-1))
        pivot = (mean-probability)/error if error else (-math.inf if mean < probability else math.inf)
        pivots.append(pivot)
        masses.append(math.comb(blocks, count)*probability**count*(1-probability)**(blocks-count))
    masses = np.asarray(masses)
    masses /= masses.sum()
    return np.asarray(pivots), masses


def exact_quantiles(blocks, high_count, probabilities):
    pivots, masses = exact_two_point_pivots(blocks, high_count)
    probabilities = np.asarray(probabilities, dtype=float)
    if not np.isfinite(probabilities).all() or np.any((probabilities <= 0) | (probabilities >= 1)):
        raise ValueError("quantile probabilities strictly inside (0,1) required")
    cdf = np.cumsum(masses)
    cdf[-1] = 1.
    return pivots[np.searchsorted(cdf, probabilities, side="left")]


def fixture_rows(blocks, high_count, delta_m):
    """Five directed contrasts sharing a full-model anchor; fixed five seeds."""
    config = InferenceConfig(delta_m=delta_m)
    config.validate(len(PRIMARY_FAMILY))
    exact_two_point_pivots(blocks, high_count)
    high = np.arange(blocks) < high_count
    p = high_count/blocks
    standardized = (high.astype(float)-p)/math.sqrt(p*(1-p))
    amplitudes = np.array([1., .75, 1.5, .5, 1.25])
    shifts = np.array([-2., -1., 0., 1., 2.])*delta_m
    differences = shifts+2*delta_m*standardized[:, None]*amplitudes
    rows = []
    for block in range(blocks):
        for seed in SEEDS:
            scores = {"all-terrain": 1000*delta_m}
            for column, (_, control) in enumerate(PRIMARY_FAMILY.values()):
                scores[control] = scores["all-terrain"]-differences[block, column]
            for name, score in scores.items():
                rows.append({"origin_id": f"fixture-origin-{block}", "independent_block_id": f"fixture-block-{block}",
                             "configuration": name, "seed": seed, "score_m": float(score), "status": "success"})
    # Use the actual floating-point fixture ledger, as the inference engine does.
    _, _, values, _ = inference.paired_blocks(rows)
    block_means = values.mean(axis=1)
    return rows, block_means.mean(axis=0), block_means.std(axis=0, ddof=1)/math.sqrt(blocks)


def check(*, delta_m, repetitions=32, cases=CASES):
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("positive integer repetition count required")
    if not cases or len(set(cases)) != len(cases):
        raise ValueError("nonempty unique oracle cases required")
    config = InferenceConfig(delta_m=delta_m)
    config.validate(len(PRIMARY_FAMILY))
    result = {"schema_version": "pirc17-exact-bootstrap-tail-check-v1", "certified": False,
        "scope": "conditional bootstrap quantile Monte Carlo check on two-point software fixtures only; not population coverage or research data",
        "config": vars(config), "repetitions_per_case": repetitions, "cases": [],
        "final_eval_label_prediction_metric_reads": 0}
    tail = config.alpha/(2*len(PRIMARY_FAMILY))
    tolerance = delta_m*config.tail_tolerance_fraction_of_delta
    for blocks, high_count in cases:
        rows, estimates, errors = fixture_rows(blocks, high_count, delta_m)
        quantiles = exact_quantiles(blocks, high_count, [tail, 1-tail])
        exact = np.stack((estimates-quantiles[1]*errors, estimates-quantiles[0]*errors))
        comparisons = {name: {"tail_check_pass_count": 0, "reference_exceedance_count": 0,
                             "silent_reference_exceedance_count": 0, "reported_nonfinite_count": 0,
                             "finite_endpoint_comparisons": 0, "maximum_finite_endpoint_error_m": None}
                       for name in PRIMARY_FAMILY}
        for repetition in range(repetitions):
            current = replace(config, bootstrap_seed=config.bootstrap_seed+2*repetition,
                              mc_check_seed=config.mc_check_seed+2*repetition)
            report = infer(rows, config=current, mechanism_passed={name: True for name in PRIMARY_FAMILY})
            for column, name in enumerate(PRIMARY_FAMILY):
                actual = report["results"][name]
                expected = exact[:, column]
                endpoints = actual["simultaneous_interval_m"]
                finite = all(v is not None for v in endpoints) and np.isfinite(expected).all()
                if finite:
                    error = float(np.max(np.abs(np.asarray(endpoints)-expected)))
                    exceedance = error > tolerance
                    comparisons[name]["finite_endpoint_comparisons"] += 1
                    previous = comparisons[name]["maximum_finite_endpoint_error_m"]
                    comparisons[name]["maximum_finite_endpoint_error_m"] = error if previous is None else max(previous, error)
                else:
                    # Unbounded exact tails must never be certified by a finite band.
                    exceedance = all(v is not None for v in endpoints) and not np.isfinite(expected).all()
                stable = actual["tail_check_passed"]
                comparisons[name]["reported_nonfinite_count"] += int(not actual["interval_finite"])
                comparisons[name]["tail_check_pass_count"] += int(stable)
                comparisons[name]["reference_exceedance_count"] += int(exceedance)
                comparisons[name]["silent_reference_exceedance_count"] += int(stable and exceedance)
        result["cases"].append({"blocks": blocks, "high_count": high_count,
            "exact_tail_pivots": [float(v) if np.isfinite(v) else None for v in quantiles],
            "exact_interval_finite": bool(np.isfinite(exact).all()), "comparisons": comparisons})
    result["silent_reference_exceedances"] = sum(c["silent_reference_exceedance_count"]
        for case in result["cases"] for c in case["comparisons"].values())
    result["source_sha256"] = {"inference.py": _hash(Path(inference.__file__)),
                              "bootstrap_tail_check.py": _hash(Path(__file__))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta-m", type=float, required=True)
    parser.add_argument("--repetitions", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    result = check(delta_m=args.delta_m, repetitions=args.repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output), "silent_reference_exceedances": result["silent_reference_exceedances"],
                      "cases": len(result["cases"]), "certified": False}))


if __name__ == "__main__":
    main()
