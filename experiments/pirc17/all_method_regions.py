"""Complete 28-method disk-size descriptions from immutable saved scores only.

No particles, targets, fitting, forecasting, rescoring or inference are run.
Missing/invalid rows reject the complete projection rather than a subset.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import math
import statistics

from .calibration_description import HORIZONS, LEVELS, SEEDS, STAGE_SHA, validate_regions
from .checkpoint_preview import read_cached_row
from .checkpoint_resume import load
from .protocol_core import canonical, digest, read_json

BINDINGS_SHA = "64ec590c296689ee17f4a8ae4f6e5bec5253c509d8f3fb09e6d3eafc4f836277"
CALIBRATION_SHA = "3d79b46f7cf368547c4e9773c4f843fce4646044eaaec27032994bb982c062f3"
SUBJECTS = (
    "arm-01/full", "arm-02/pointwise", "arm-03/single_gaussian", "arm-04/gmm_kernel",
    "arm-05/explicit_decomp", "arm-06/dt30", "arm-06/dt60", "arm-06/dt120",
    "arm-06/dt300", "arm-06/dt600", "arm-07/full", "arm-08/qmle", "arm-09/mixed",
    "arm-09/pure_es", "arm-10/d2_mc", "arm-10/d2_closed", "arm-11/full",
    "arm-12/scratch", "arm-14/reptile", "arm-15/drift_only", "arm-15/two_step",
    "arm-16/full", "arm-18/full", "arm-19/em", "arm-19/euler", "arm-20/full",
    "arm-21/mc", "arm-21/crn",
)


def summarize(rows):
    expected = {(subject, rank, seed) for subject in SUBJECTS
                for rank in range(46) for seed in SEEDS}
    indexed = {}
    for row in rows:
        key = row["configuration"], row["origin_rank"], row["seed"]
        if key not in expected or key in indexed:
            raise ValueError("duplicate or unregistered region row")
        if (row["status"] != "success" or row["matrix"] != "NEX326-methods"
                or row["origin_mode"] != "causal_prefix" or row["partition"] != "final_eval"
                or row["scientific"] is not True):
            raise ValueError("failed or wrong-population region row")
        validate_regions(row)
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("all original 6440 rows required; no successful subset")
    profiles = []
    for subject in SUBJECTS:
        horizons = []
        for t, horizon in enumerate(HORIZONS):
            levels = []
            for l, probability in enumerate(LEVELS):
                block_coverage, block_area, block_radius = [], [], []
                for rank in range(46):
                    entries = [indexed[subject, rank, seed]["scores"]["by_time"][t]
                               ["region"]["levels"][l] for seed in SEEDS]
                    block_coverage.append(statistics.mean(x["covered"] for x in entries))
                    block_area.append(statistics.mean(x["area_m2"] for x in entries))
                    block_radius.append(statistics.mean(x["radius_m"] for x in entries))
                levels.append(dict(nominal_level=probability,
                    empirical_coverage=statistics.mean(block_coverage),
                    mean_disk_area_km2=statistics.mean(block_area)/1e6,
                    mean_disk_radius_m=statistics.mean(block_radius)))
            horizons.append(dict(nominal_horizon_seconds=horizon, levels=levels))
        profiles.append(dict(configuration=subject, forecast_rows=230,
                             independent_blocks=46, horizons=horizons))
    return profiles


def agree(actual, expected, name):
    if (type(actual) not in (int, float) or type(expected) not in (int, float)
            or not math.isfinite(actual) or not math.isfinite(expected)
            or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-8)):
        raise ValueError("saved original stage differs: " + name)


def project(root, cache, stage_path, bindings_path, calibration_path):
    stage = read_json(stage_path, expected_file_sha256=STAGE_SHA)
    bindings = read_json(bindings_path, expected_file_sha256=BINDINGS_SHA)
    old_calibration = read_json(calibration_path, expected_file_sha256=CALIBRATION_SHA)
    if (stage["primary_method_rows"] != 6440 or stage["independent_blocks"] != 46
            or stage["forecast_seeds"] != list(SEEDS)
            or stage["independent_raw_output_audit_completed"] is not False
            or stage["scientific_claim_authorized"] is not False
            or cache.name != stage["cache_sha256"]):
        raise ValueError("original complete preliminary method population required")
    originals = {x["configuration"]: x for x in stage["configs"]}
    if len(stage["configs"]) != 28 or set(originals) != set(SUBJECTS):
        raise ValueError("all 28 original identities required")
    for row in originals.values():
        if (row["matrix"] != "NEX326-methods" or row["origin_mode"] != "causal_prefix"
                or row["counts"] != {"success": 230} or row["expected"] != 230
                or row["missing"] != 0 or row["status"] != "computed"):
            raise ValueError("no failed or incomplete original primary profile")
    origins = {r["rank"]: r for r in bindings["origins"]}
    if len(bindings["origins"]) != 46 or set(origins) != set(range(46)):
        raise ValueError("original 46 bound origins required")
    settings, imported = load(root)
    progress = read_json(root/"progress.json")
    index = read_json(cache/"progress.json")
    if index["cache_sha256"] != cache.name or settings["settings_id"] != stage["settings_id"]:
        raise ValueError("original settings/cache identity differs")
    works = [w for w in settings["workloads"] if w["kind"] == "scientific_forecast"
             and w["matrix"] == "NEX326-methods" and w["origin_mode"] == "causal_prefix"]
    if len(works) != 6440:
        raise ValueError("original complete primary work inventory required")
    rows, versions, targets = [], [], {}
    for work in works:
        wid = work["work_id"]
        entry = index["rows"].get(wid)
        if entry is None or entry["status"] != "success":
            raise ValueError("missing or failed original saved score")
        binding = None
        if wid in imported:
            m = imported[wid]["manifest"]
            binding = dict(path=m["artifact_path"], content_sha256=m["artifact_sha256"])
        for state in (progress["failures"], progress["completed"]):
            item = state.get(wid)
            if isinstance(item, dict) and item.get("artifact_path"):
                binding = dict(path=item["artifact_path"], content_sha256=item["artifact_sha256"])
        source = digest(dict(binding=binding, failure=progress["failures"].get(wid)))
        row, value = read_cached_row(cache, work, entry, source)
        origin = origins[row["origin_rank"]]
        if (row["sample_id"] != origin["sample_id"]
                or row["independent_block_id"] != origin["independent_block_id"]
                or row["origin_id"] != digest(["pirc17-formal-origin-stream-v1", origin["sample_id"], "causal_prefix"])):
            raise ValueError("original origin/population identity differs")
        if targets.setdefault(row["origin_rank"], row["target_sha256"]) != row["target_sha256"]:
            raise ValueError("all configurations/seeds must use the same bound targets")
        version = digest(value)
        if wid in bindings["score_sources"] and version != bindings["score_sources"][wid]:
            raise ValueError("previous three-model score version changed")
        rows.append(row)
        versions.append((wid, version))
    profiles = summarize(rows)
    for profile in profiles:
        subject = profile["configuration"]
        original = originals[subject]
        selected = [r for r in rows if r["configuration"] == subject]
        for field, value in (("weighted_es_m", statistics.mean(r["score_m"] for r in selected)),
                             ("ade_m", statistics.mean(r["scores"]["ade_grid_mean_m"] for r in selected)),
                             ("fde_m", statistics.mean(r["scores"]["fde_m"] for r in selected))):
            agree(value, original[field], field)
        for t, horizon in enumerate(profile["horizons"]):
            agree(statistics.mean(r["scores"]["by_time"][t]["energy_score_m"] for r in selected),
                  original["es_by_time_m"][t], "horizon ES")
            agree(horizon["levels"][2]["empirical_coverage"],
                  original["coverage_90_by_time"][t], "90-percent coverage")
    by_subject = {p["configuration"]: p for p in profiles}
    for old in old_calibration["profiles"]:
        if by_subject[old["configuration"]] != old:
            raise ValueError("previous complete three-model region description changed")
    return dict(schema_version="pirc17-saved-all-method-regions-v1",
        original_stage_sha256=STAGE_SHA, original_three_model_calibration_sha256=CALIBRATION_SHA,
        input_bindings_sha256=BINDINGS_SHA, cached_score_records_sha256=digest(sorted(versions)),
        cache_sha256=cache.name, settings_id=settings["settings_id"],
        complete_cached_rows=6440, recording_hash_blocks=46, forecast_seed_ids=list(SEEDS),
        profiles=profiles, scope=dict(all_twenty_eight_models_described=True,
            all_four_levels_and_horizons_retained=True, original_three_model_profiles_exact=True,
            averaging="five-seed block means, then equal means over46recording-hash blocks;230scores per configuration are not230 independent targets",
            region="saved marginal ensemble-mean radial quantile disks;not highest-density or simultaneous path regions",
            mean_area_computed_as_area_of_mean_radius=False,
            size_alone_is_quality_ranking=False, conditional_calibration_established=False,
            all_prediction_array_target_clocks_certified=False, physical_utc_certified=False,
            new_fits=0, new_forecasts=0, new_particle_scores=0, new_tests_or_bootstrap=0,
            forecast_arrays_opened=False, independent_saved_output_audit_completed=False,
            scientific_claim_authorized=False, public_route_publication_permission_verified=False,
            original_final_sixteen_figure_inventory_replaced=False,
            private_routes_paths_or_origin_ids_exported=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "cache", "stage", "bindings", "calibration", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.root.resolve()):
        raise ValueError("descriptive output must not modify the live runtime")
    result = project(args.root, args.cache, args.stage, args.bindings, args.calibration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(canonical(result)+b"\n")
    print("Described all6440 saved scores,28 configurations,4 times,4 levels;zero new experiments or output audit.")


if __name__ == "__main__":
    main()
