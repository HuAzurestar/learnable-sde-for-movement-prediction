"""Describe existing radial regions on the already bound 690-row method grid.

No forecasting, fitting, rescoring, bootstrap or significance testing. A missing
or invalid row fails the entire projection; no successful-row intersections.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics

from .protocol_core import canonical, digest, read_json, unpack


SUBJECTS = ("arm-01/full", "arm-04/gmm_kernel", "arm-06/dt300")
SEEDS = tuple(range(20260814, 20260819))
LEVELS = (.5, .8, .9, .95)
HORIZONS = (60, 300, 900, 1800)
STAGE_SHA = "8900f3dee68d5fc9cafd5ced2e2394503432d2493251601c0ba9cb771cbefba1"


def validate_regions(row):
    scores = row["scores"]
    if scores["particle_count"] != 512 or len(scores["by_time"]) != 4:
        raise ValueError("original four-time 512-particle scores required")
    for target, time in zip(HORIZONS, scores["by_time"]):
        elapsed = time["elapsed_seconds"]
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or abs(elapsed - target) > 30:
            raise ValueError("original target time outside frozen tolerance")
        region = time["region"]
        if region["region"] != "ensemble_mean_radial_quantile_disk":
            raise ValueError("original marginal radial disk required")
        levels = region["levels"]
        if [v["level"] for v in levels] != list(LEVELS):
            raise ValueError("all four original coverage levels required")
        for entry in levels:
            radius, area = entry["radius_m"], entry["area_m2"]
            if (type(radius) not in (int, float) or type(area) not in (int, float)
                    or not math.isfinite(radius) or not math.isfinite(area)
                    or radius < 0 or area < 0 or type(entry["covered"]) is not bool
                    or not math.isclose(area, math.pi * radius**2, rel_tol=1e-12, abs_tol=1e-8)):
                raise ValueError("invalid saved radius/area/containment")
            mass = entry["empirical_mass"]
            if type(mass) not in (int, float) or not math.isfinite(mass) or not 0 <= mass <= 1:
                raise ValueError("invalid saved empirical particle mass")
        if (any(a["radius_m"] > b["radius_m"] for a, b in zip(levels, levels[1:]))
                or any(a["covered"] and not b["covered"] for a, b in zip(levels, levels[1:]))):
            raise ValueError("saved radial disks are not nested")


def distribution(values):
    """Descriptive linear quartiles over exactly 46 recording-block means."""
    if len(values) != 46 or any(not math.isfinite(value) for value in values):
        raise ValueError("46 finite block values required")
    q1, median, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return {"minimum": min(values), "q25": q1, "median": median,
            "q75": q3, "maximum": max(values)}


def summarize(rows, *, include_blocks=False):
    expected = {(subject, rank, seed) for subject in SUBJECTS
                for rank in range(46) for seed in SEEDS}
    indexed = {}
    for row in rows:
        axes = row["configuration"], row["origin_rank"], row["seed"]
        if axes not in expected or axes in indexed:
            raise ValueError("duplicate or unregistered calibration row")
        if (row["status"] != "success" or row["matrix"] != "NEX326-methods"
                or row["origin_mode"] != "causal_prefix" or row["partition"] != "final_eval"
                or row["scientific"] is not True):
            raise ValueError("failed or wrong-population calibration row")
        validate_regions(row)
        indexed[axes] = row
    if set(indexed) != expected:
        raise ValueError("complete original 690-row grid required")
    result = []
    for subject in SUBJECTS:
        profile = []
        for time_index, horizon in enumerate(HORIZONS):
            levels = []
            for level_index, level in enumerate(LEVELS):
                block_coverage, block_area, block_radius = [], [], []
                block_records = []
                for rank in range(46):
                    entries = [indexed[subject, rank, seed]["scores"]["by_time"][time_index]
                               ["region"]["levels"][level_index] for seed in SEEDS]
                    block_coverage.append(statistics.mean(v["covered"] for v in entries))
                    block_area.append(statistics.mean(v["area_m2"] for v in entries))
                    block_radius.append(statistics.mean(v["radius_m"] for v in entries))
                    if include_blocks:
                        block_records.append({
                            "block_number": rank + 1,
                            "covered_seed_count": sum(v["covered"] for v in entries),
                            "seed_coverage_fraction": block_coverage[-1],
                            "mean_disk_area_km2": block_area[-1] / 1e6,
                            "mean_disk_radius_m": block_radius[-1],
                        })
                levels.append({"nominal_level": level,
                               "empirical_coverage": statistics.mean(block_coverage),
                               "mean_disk_area_km2": statistics.mean(block_area) / 1e6,
                               "mean_disk_radius_m": statistics.mean(block_radius)})
                if include_blocks:
                    counts = [v["covered_seed_count"] for v in block_records]
                    levels[-1].update({
                        "blocks": block_records,
                        "coverage_distribution": distribution(block_coverage),
                        "area_km2_distribution": distribution([v / 1e6 for v in block_area]),
                        "radius_m_distribution": distribution(block_radius),
                        "blocks_by_covered_seed_count": [counts.count(i) for i in range(6)],
                    })
            profile.append({"nominal_horizon_seconds": horizon, "levels": levels})
        result.append({"configuration": subject, "forecast_rows": 230,
                       "independent_blocks": 46, "horizons": profile})
    return result


def project(cache, bindings_path, bindings_sha256, stage_path, *, include_blocks=False):
    bindings = read_json(bindings_path, expected_file_sha256=bindings_sha256)
    stage = read_json(stage_path, expected_file_sha256=STAGE_SHA)
    if cache.name != stage["cache_sha256"] or len(bindings["score_sources"]) != 690:
        raise ValueError("original cache and bound 690-row input required")
    origins = {r["rank"]: r for r in bindings["origins"]}
    if set(origins) != set(range(46)):
        raise ValueError("original 46-origin population required")
    rows = []
    targets_by_rank = {}
    for work_id, value_sha in bindings["score_sources"].items():
        matches = []
        for path in (cache / "rows" / work_id).glob("*.json"):
            value = unpack(read_json(path))
            if digest(value) != value_sha:
                continue
            scope = value["scope"]
            if (value["schema_version"] != "pirc17-checkpoint-scoring-v1-row"
                    or scope["cache_sha256"] != cache.name or path.stem != digest(scope)):
                raise ValueError("saved score scope differs")
            matches.append(value["row"])
        if len(matches) != 1:
            raise ValueError("missing/ambiguous original bound score; no partial projection")
        row = matches[0]
        origin = origins[row["origin_rank"]]
        if (row["forecast_work_id"] != work_id or row["sample_id"] != origin["sample_id"]
                or row["independent_block_id"] != origin["independent_block_id"]
                or row["origin_id"] != digest(["pirc17-formal-origin-stream-v1", origin["sample_id"], "causal_prefix"])):
            raise ValueError("original score population identity differs")
        rank = row["origin_rank"]
        if targets_by_rank.setdefault(rank, row["target_sha256"]) != row["target_sha256"]:
            raise ValueError("compared configurations/seeds do not share the same original targets")
        rows.append(row)
    profiles = summarize(rows, include_blocks=include_blocks)
    for profile in profiles:
        subject = profile["configuration"]
        original = next(v for v in stage["configs"] if v["configuration"] == subject)
        values = [r for r in rows if r["configuration"] == subject]
        for actual, saved in ((statistics.mean(r["score_m"] for r in values), "weighted_es_m"),
                              (statistics.mean(r["scores"]["fde_m"] for r in values), "fde_m")):
            if not math.isclose(actual, original[saved], rel_tol=1e-12, abs_tol=1e-8):
                raise ValueError("bound scores differ from original published stage means")
        if any(not math.isclose(time["levels"][2]["empirical_coverage"], old, abs_tol=1e-12)
               for time, old in zip(profile["horizons"], original["coverage_90_by_time"])):
            raise ValueError("90-percent coverage differs from original stage")
    result = {
        "schema_version": ("pirc17-saved-calibration-block-description-v1" if include_blocks
                           else "pirc17-saved-calibration-description-v1"),
        "input_bindings_sha256": bindings_sha256,
        "cached_score_records_sha256": digest(bindings["score_sources"]),
        "original_stage_sha256": STAGE_SHA, "cache_sha256": cache.name,
        "complete_cached_rows": len(rows), "independent_blocks": 46,
        "forecast_rng_seeds": 5, "profiles": profiles,
        "scope": {
            "new_fits": 0, "new_predictions": 0, "new_particle_scores": 0,
            "new_tests_or_bootstrap": 0, "independent_output_audit": False,
            "private_routes_paths_or_origin_ids_exported": False,
            "description": "Post-outcome descriptive profiles for previously displayed Full/GMM/dt300. Equal block weights after within-block seed averaging. Marginal ensemble-mean radial disks, not highest-density or simultaneous path regions. Area alone is not a quality ranking; final qualification pending.",
        },
    }
    if include_blocks:
        result["block_distribution_scope"] = {
            "block_labels": "Anonymous original rank plus one; no recording hashes, targets, coordinates or paths.",
            "aggregation": "Within-block means over exactly five original forecast seeds; equal weights over 46 recording blocks.",
            "quantiles": "Linear interpolation at (n-1)*p, p=0.25,0.5,0.75; inclusive sample quartiles, not confidence intervals.",
            "coverage_resolution": 0.2,
            "all_four_levels_and_horizons_retained": True,
            "all_twenty_eight_models_described": False,
            "seed_count_is_independent_target_count": False,
            "public_route_case_permission_verified": False,
        }
    canonical(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "bindings", "stage", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--bindings-sha256", required=True)
    parser.add_argument("--include-blocks", action="store_true",
                        help="Retain anonymous per-block coverage/size descriptions; preserve original aggregate mode by default.")
    args = parser.parse_args()
    result = project(args.cache, args.bindings, args.bindings_sha256, args.stage,
                     include_blocks=args.include_blocks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print("Described 690 existing scores at four horizons/four levels; zero fits, forecasts, scores or new inference tests.")


if __name__ == "__main__":
    main()
