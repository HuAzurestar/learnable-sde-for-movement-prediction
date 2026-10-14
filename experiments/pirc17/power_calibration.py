"""Prospective power inputs from complete, paired development runs only.

The observed effect is not substituted for an assumed alternative. This tool
estimates the SD of independent-block differences and exposes uncertainty in
that planning input through explicit SD sensitivity scenarios.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .inference import PRIMARY_FAMILY, SEEDS, paired_blocks, planning_power
from .numerical_check import audit
from .precision_check import load_ledger
from .qualification import _hash


def calibrate(ledgers, *, step_seconds, particles, delta_m, planning_block_counts,
              comparisons=tuple(PRIMARY_FAMILY), family_size=5, target_power=.8):
    if not ledgers or any(not rows for rows in ledgers) or not comparisons or len(set(comparisons)) != len(comparisons):
        raise ValueError("development ledgers and unique registered comparisons required")
    if any(name not in PRIMARY_FAMILY for name in comparisons) or family_size != len(PRIMARY_FAMILY):
        raise ValueError("primary family remains the five registered comparisons")
    if (not np.isfinite(step_seconds) or step_seconds <= 0 or type(particles) is not int or particles < 3
            or not planning_block_counts or any(type(n) is not int or n < 2 for n in planning_block_counts)):
        raise ValueError("explicit positive step, particles and prospective block counts required")
    family = {name: PRIMARY_FAMILY[name] for name in comparisons}
    required_configurations = {v for pair in family.values() for v in pair}
    identity_fields = ("fit_sha256", "development_identity", "rollout_version", "physical_history_step_seconds",
                       "brownian_driver", "scoring_grid", "time_weights", "entropy_grid_m", "map_backend",
                       "map_source_sha256")
    initial = ledgers[0][0]
    selected, seen, expected_origins, source_seeds = [], set(), set(), set()
    simulation_paths = {}
    score_identity = None
    source_counts = []
    for rows in ledgers:
        result = audit(rows, tolerance_m=delta_m/4)
        if result["failures"]:
            raise ValueError("failed development workloads cannot be dropped from power calibration")
        header = rows[0]
        for name in identity_fields:
            if name not in header or header[name] != initial.get(name):
                raise ValueError("development calibration mixes protocol identities")
        source = {name: header["source_sha256"][name] for name in (
            "rollout.py", "brownian.py", "dynamics.py", "checkpoints.py", "development.py",
            "features.py", "metrics.py", "configurations.py")}
        if score_identity is not None and score_identity != source:
            raise ValueError("development calibration mixes prediction or score source")
        score_identity = source
        if step_seconds not in header["max_steps_seconds"] or particles not in header["particle_counts"]:
            raise ValueError("every input ledger must contain the chosen numerical settings")
        expected_origins.update(header["sample_ids"])
        source_seeds.update(header["seeds"])
        source_counts.append(result["attempted_runs"])
        for row in rows[1:-1]:
            if (row["max_step_seconds"] != step_seconds or row["particles"] != particles
                    or row["configuration"] not in required_configurations):
                continue
            key = (row["sample_id"], row["configuration"], row["seed"])
            if key in seen:
                raise ValueError("duplicate development workload across input ledgers")
            seen.add(key)
            stream = (row["sample_id"], row["seed"])
            identity = (row["brownian_identity"], row["actual_horizons_seconds"])
            if stream in simulation_paths and simulation_paths[stream] != identity:
                raise ValueError("paired calibration changes simulation paths or scoring times")
            simulation_paths[stream] = identity
            selected.append({"origin_id": row["sample_id"], "independent_block_id": row["independent_block_id"],
                "configuration": row["configuration"], "seed": row["seed"], "status": row["status"],
                "score_m": row["scores"]["time_weighted_energy_score_m"]})
    if source_seeds != set(SEEDS):
        raise ValueError("power calibration requires all five fixed training seeds")
    if {row["origin_id"] for row in selected} != expected_origins:
        raise ValueError("an input origin is missing from the selected comparisons")
    blocks, names, differences, origin_counts = paired_blocks(selected, family)
    if len(blocks) < 2:
        raise ValueError("at least two independent blocks required to estimate paired SD")
    block_means = differences.mean(axis=1)
    paired_sd = block_means.std(axis=0, ddof=1)
    reports = {}
    for index, name in enumerate(names):
        reports[name] = {"paired_block_sd_m": float(paired_sd[index]),
            "observed_development_improvement_m": float(-block_means[:, index].mean()),
            "seed_mean_improvement_m": (-differences[:, :, index].mean(axis=0)).tolist(),
            "planning_scenarios": [
                {"planning_blocks": count, "sd_multiplier": multiplier,
                 **planning_power(paired_sd_m=float(paired_sd[index])*multiplier, blocks=count,
                     true_improvement_m=2*delta_m, delta_m=delta_m, family_size=family_size, target=target_power)}
                for count in planning_block_counts for multiplier in (1., 2.)]}
    return {"schema_version": "pirc17-development-paired-power-v1", "certified": False,
        "scope": "registered development subset only; no final efficacy or numerical qualification",
        "source_identity": {name: initial[name] for name in identity_fields},
        "prediction_source_sha256": score_identity, "independent_block_ids": blocks,
        "source_attempted_runs": source_counts, "source_independent_blocks": len(blocks),
        "source_origin_count": len(expected_origins), "origin_count_by_block": origin_counts,
        "registered_seeds": list(SEEDS), "numerical_settings": {"step_seconds": step_seconds, "particles": particles},
        "primary_family_size": family_size, "missing_primary_comparisons": sorted(set(PRIMARY_FAMILY)-set(comparisons)),
        "assumed_alternative": "2 * preregistered practical margin, not observed development effect",
        "sd_uncertainty": "1x and 2x observed paired SD are sensitivity scenarios, not confidence bounds",
        "small_sample_caution": "few development blocks give an imprecise SD; neither normal planning nor 30 blocks guarantees power",
        "results": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, action="append", required=True)
    parser.add_argument("--ledger-sha256", action="append", required=True)
    parser.add_argument("--step-seconds", type=float, required=True)
    parser.add_argument("--particles", type=int, required=True)
    parser.add_argument("--delta-m", type=float, required=True)
    parser.add_argument("--planning-blocks", type=int, nargs="+", required=True)
    parser.add_argument("--comparisons", nargs="+", choices=tuple(PRIMARY_FAMILY), default=list(PRIMARY_FAMILY))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.ledger) != len(args.ledger_sha256) or args.output.exists():
        parser.error("one hash per ledger and a new output path are required")
    result = calibrate([load_ledger(p, h) for p, h in zip(args.ledger, args.ledger_sha256)],
        step_seconds=args.step_seconds, particles=args.particles, delta_m=args.delta_m,
        planning_block_counts=args.planning_blocks, comparisons=args.comparisons)
    result["input_ledger_sha256"] = args.ledger_sha256
    result["calibration_source_sha256"] = {name: _hash(Path(__file__).with_name(name))
        for name in ("power_calibration.py", "inference.py", "numerical_check.py", "precision_check.py")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output), "source_independent_blocks": result["source_independent_blocks"],
                      "missing_primary_comparisons": result["missing_primary_comparisons"], "certified": False}))


if __name__ == "__main__":
    main()
