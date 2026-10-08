"""Describe original terrain fitting weights/time support, never refit.

Only clock and identity columns of the selected development segments are
read; two adjacent recorded ticks per transition are used. No positions,
rate residuals, maps, forecasts or final numeric rows are decoded. Saved Q
spectra describe the recorded parameters, not new estimates.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .checkpoint_resume import load
from .feature_design_description import INVENTORY_SHA, ORIGINAL_CONTEXT_FILE_SHA, ROLES
from .protocol_core import canonical, digest, file_hash, read_json, under, unpack


SOURCES = ("experiments/pirc17/development.py", "experiments/pirc17/dynamics.py",
           "experiments/pirc17/direct_linear.py")
METADATA_COLUMNS = ["dataset_version", "file_id", "segment_id", "split",
                    "independent_block_id", "point_index", "absolute_epoch_ns"]


def first_interval_ns(sample, rows, dataset_id):
    """Exactly adjacent saved origin/first-future ticks, never a nominal tau."""
    if sample["split"] not in ROLES:
        raise ValueError("development-only clock rows required")
    a, b = sample["history_end"], sample["target_start"]
    if type(a) is not int or type(b) is not int or a < 0 or b != a + 1:
        raise ValueError("adjacent origin/first-future indices required")
    if (type(sample["target_end"]) is not int or sample["target_end"] < b
            or len(rows) != sample["target_end"] + 1
            or any(type(row["point_index"]) is not int for row in rows)
            or len({row["point_index"] for row in rows}) != len(rows)):
        raise ValueError("missing or duplicate original segment metadata row")
    # Sample bounds are segment offsets; point_index is the ORIGINAL FILE
    # row identity. These are not equal after the first segment of a file.
    ordered = sorted(rows, key=lambda row: row["point_index"])
    ticks = []
    for row in (ordered[a], ordered[b]):
        for key, expected in (("dataset_version", dataset_id), ("split", sample["split"]),
                              ("file_id", sample["file_id"]), ("segment_id", sample["segment_id"]),
                              ("independent_block_id", sample["independent_block_id"])):
            if row[key] != expected:
                raise ValueError("original clock row identity mismatch")
        if type(row["absolute_epoch_ns"]) is not int:
            raise ValueError("integer recorded clock ticks required")
        ticks.append(row["absolute_epoch_ns"])
    delta = ticks[1] - ticks[0]
    if not 0 < delta <= 60_000_000_000:
        raise ValueError("original development first interval outside positive 60-second support")
    return delta


def describe_intervals(deltas_ns):
    if not deltas_ns or any(type(x) is not int or not 0 < x <= 60_000_000_000 for x in deltas_ns):
        raise ValueError("positive recorded integer-nanosecond development intervals required")
    dt = np.asarray(deltas_ns, dtype=float) / 1e9
    q = np.quantile(dt, [.25, .5, .75], method="linear")
    edges = [0, 1, 5, 10, 30, 60]
    bins = {f"({a},{b}]": int(np.count_nonzero((dt > a) & (dt <= b)))
            for a, b in zip(edges, edges[1:])}
    assert sum(bins.values()) == len(dt)
    return {"transitions": len(dt), "minimum_seconds": float(dt.min()),
            "q25_seconds": float(q[0]), "median_seconds": float(q[1]),
            "q75_seconds": float(q[2]), "maximum_seconds": float(dt.max()),
            "mean_seconds": float(dt.mean()), "sum_seconds": float(dt.sum()),
            "equal_drift_row_weight": 1 / len(dt),
            "rate_outer_product_Q_coefficient_minimum_seconds": float(dt.min() / len(dt)),
            "rate_outer_product_Q_coefficient_maximum_seconds": float(dt.max() / len(dt)),
            "interval_bin_counts_seconds": bins}


def saved_q_description(model):
    q = np.asarray(model["diffusion_covariance_m2_per_s"], dtype=float)
    if q.shape != (2, 2) or not np.isfinite(q).all():
        raise ValueError("finite two-dimensional saved Q required")
    asymmetry = float(np.max(np.abs(q - q.T)))
    if asymmetry > 1e-12:
        raise ValueError("saved Q asymmetry exceeds the original absolute tolerance")
    eigenvalues = np.linalg.eigvalsh(q)
    if eigenvalues[0] < -1e-10:
        raise ValueError("saved Q is materially indefinite under original terrain rule")
    return {"configuration": model["configuration"],
            "saved_eigenvalues_m2_per_s": eigenvalues.tolist(),
            "negative_eigenvalue_rejection_threshold_m2_per_s": -1e-10,
            "symmetry_absolute_tolerance_m2_per_s": 1e-12,
            "saved_asymmetry_m2_per_s": asymmetry,
            "negative_roundoff_clamp_required_for_L": bool(eigenvalues[0] < 0),
            "positive_floor_or_jitter_added_by_terrain_diffusion_estimator": False}


def project(runtime, original_context, inventory_path, fit_path, fit_sha, feature_path, feature_sha):
    settings, _ = load(runtime)
    original = unpack(read_json(original_context, expected_file_sha256=ORIGINAL_CONTEXT_FILE_SHA))
    inputs = unpack(original["input_identity"])
    current = unpack(read_json(settings["context"]), expected_sha256=settings["context_sha256"])
    if inputs["terrain_development_identity"] != unpack(current["input_identity"])["terrain_development_identity"]:
        raise ValueError("original and resumed development population differ")
    bundle = unpack(read_json(settings["bundle"]))
    protocol, execution = unpack(bundle["protocol"]), unpack(bundle["execution"])
    if original["protocol_sha256"] != bundle["protocol"]["sha256"]:
        raise ValueError("original fitting protocol differs")
    repository = Path(__file__).resolve().parents[2]
    sources = {}
    for name in SOURCES:
        expected = protocol["source_sha256"]["PSDE-SDE/" + name]
        if file_hash(repository / name) != expected or execution["source_sha256"]["PSDE-SDE/" + name] != expected:
            raise ValueError("original development/weighting/diffusion source differs")
        sources[name] = expected
    inventory = read_json(inventory_path, expected_file_sha256=INVENTORY_SHA)
    base = read_json(fit_path, expected_file_sha256=fit_sha)
    feature = read_json(feature_path, expected_file_sha256=feature_sha)
    if (feature["original_input_identity_sha256"] != original["input_identity"]["sha256"]
            or feature["original_context_file_sha256"] != ORIGINAL_CONTEXT_FILE_SHA
            or base["inventory_sha256"] != INVENTORY_SHA
            or base["matrix_sha256"] != original["matrix_sha256"]):
        raise ValueError("earlier original fitting/design projection bindings differ")
    terrain = []
    for binding in inventory["models"]:
        if binding["matrix"] != "terrain":
            continue
        p = unpack(read_json(binding["original_model_record_path"],
                             expected_file_sha256=binding["original_model_record_file_sha256"]),
                   expected_sha256=binding["original_model_record_content_sha256"])
        model = p["artifact"]["model"]
        if (p["input_sha256"] != original["input_identity"]["sha256"]
                or model["configuration"] != binding["configuration"]
                or model["sha256"] != binding["parameter_identity"]
                or model["training_identity"] != digest({"inputs": p["input_sha256"],
                                                        "configuration": binding["configuration"]})
                or model["train_transition_count"] != ROLES["train"]
                or model["validation_transition_count"] != ROLES["validation"]):
            raise ValueError("original terrain fit/input/time population mismatch")
        terrain.append({**saved_q_description(model),
                        "original_record_sha256": binding["original_model_record_file_sha256"]})
    terrain.sort(key=lambda row: row["configuration"])
    if len(terrain) != 10 or len(base["terrain"]) != 10:
        raise ValueError("complete original ten-terrain fit population required")
    for row, old in zip(terrain, base["terrain"]):
        if (row["configuration"] != old["configuration"]
                or row["original_record_sha256"] != old["original_record_sha256"]
                or row["saved_eigenvalues_m2_per_s"] != old["diffusion_eigenvalues_m2_per_s"]):
            raise ValueError("original saved Q spectrum changed")
    binding = protocol["dataset_inputs"]
    release = Path(settings["input_paths"]["release"])
    snapshot = Path(settings["input_paths"]["snapshot"])
    manifest = read_json(snapshot / "manifest.json", expected_file_sha256=binding["snapshot"]["manifest_sha256"])
    samples_path = release / "samples.jsonl"
    if file_hash(samples_path) != binding["release_artifact_sha256"]["samples.jsonl"]:
        raise ValueError("original sample clock-index manifest differs")
    development = inputs["terrain_development_identity"]
    wanted = {sid for role in ROLES for sid in development["sample_ids"][role]}
    if len(wanted) != sum(ROLES.values()):
        raise ValueError("complete disjoint development windows required")
    samples = {}
    with samples_path.open(encoding="utf-8") as stream:
        for line in stream:
            s = json.loads(line)
            if s["sample_id"] in wanted:
                if s["sample_id"] in samples:
                    raise ValueError("duplicate original development sample")
                samples[s["sample_id"]] = s
    if set(samples) != wanted:
        raise ValueError("original development sample missing")
    groups = defaultdict(list)
    for role, expected in ROLES.items():
        if len(development["sample_ids"][role]) != expected:
            raise ValueError("original development role count changed")
        for sid in development["sample_ids"][role]:
            s = samples[sid]
            if s["split"] != role:
                raise ValueError("no final-evaluation rows allowed")
            groups[role, s["file_id"]].append(s)
    entries = {(e["split"], e["file_id"]): e for e in manifest["files"]}
    intervals, selected_sources = {}, {}
    for (role, file_id), group in sorted(groups.items()):
        entry = entries[role, file_id]
        bound = development["sources"][role + ":" + file_id]
        if entry["sha256"] != bound["feature_sha256"] or entry["condition_sha256"] != bound["condition_sha256"]:
            raise ValueError("original development clock source binding differs")
        path = under(snapshot, entry["path"])
        if file_hash(path) != entry["sha256"] or pq.ParquetFile(path).metadata.num_rows != entry["row_count"]:
            raise ValueError("selected development clock file differs")
        table = pq.read_table(path, columns=METADATA_COLUMNS,
                              filters=[("split", "=", role),
                                       ("segment_id", "in", sorted({s["segment_id"] for s in group}))])
        segments = defaultdict(list)
        for row in table.to_pylist():
            segments[row["segment_id"]].append(row)
        for s in group:
            intervals[s["sample_id"]] = first_interval_ns(s, segments[s["segment_id"]], binding["dataset_id"])
        selected_sources[role + ":" + file_id] = entry["sha256"]
    if (len(selected_sources) != feature["selected_feature_files"]
            or digest(selected_sources) != feature["selected_sources_sha256"]
            or set(intervals) != wanted):
        raise ValueError("original design/clock source population differs")
    result = {
        "schema_version": "pirc17-original-terrain-fit-scope-description-v1",
        "original_context_file_sha256": ORIGINAL_CONTEXT_FILE_SHA,
        "original_input_identity_sha256": original["input_identity"]["sha256"],
        "inventory_sha256": INVENTORY_SHA, "base_fit_projection_sha256": fit_sha,
        "base_feature_projection_sha256": feature_sha,
        "selected_feature_files": len(selected_sources), "selected_sources_sha256": digest(selected_sources),
        "source_sha256": sources,
        "roles": {role: describe_intervals([intervals[sid] for sid in development["sample_ids"][role]]) for role in ROLES},
        "terrain": terrain,
        "weighting": {"drift": "equal transition rows on displacement rates; no dt, inverse-noise, block or participant weighting",
                      "diffusion": "Q = (1/n) sum_i dt_i r_i r_i^T, r_i = dx_i/dt_i - fitted_drift_i",
                      "diffusion_centered": False, "diffusion_degrees_of_freedom_correction": False,
                      "joint_drift_diffusion_MLE_or_unbiased_estimator_claimed": False},
        "scope": {"new_fits_forecasts_particle_scores_resampling_or_map_queries": 0,
                  "coordinates_rate_residuals_or_final_eval_numeric_rows_decoded": False,
                  "Q_reestimated_or_saved_parameters_changed": False,
                  "private_clock_ticks_sample_identifiers_or_source_paths_exported": False,
                  "physical_source_clock_provenance_certified": False,
                  "whole_snapshot_revalidated": False,
                  "independent_saved_forecast_audit_completed": False,
                  "all_review_items_or_paper_complete": False},
    }
    canonical(result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "original-context", "inventory", "fit-projection", "feature-projection", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("fit-projection-sha256", "feature-projection-sha256"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    result = project(args.runtime, args.original_context, args.inventory, args.fit_projection,
                     args.fit_projection_sha256, args.feature_projection, args.feature_projection_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print(f"Described original 404/81 development intervals and ten saved Q matrices; no fitting or forecasts.")


if __name__ == "__main__":
    main()
