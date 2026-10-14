"""Describe pinned clock policy, reported reconciliation and three saved footers."""
from __future__ import annotations

import argparse
import ast
from itertools import islice
from pathlib import Path

import pyarrow.parquet as pq

from .method_algorithm_description import INVENTORY_SHA
from .protocol_core import canonical, decode, file_hash, read_json, under, unpack

COUNTS = ("source_condition_points", "aligned_refined_points", "total_condition_points_not_aligned",
          "duplicate_timestamp_intervals", "segments_with_duplicate_timestamps",
          "backward_timestamp_intervals_causing_split", "long_gap_timestamp_intervals_causing_split")


def clock_literals(source):
    values = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"SOLAR_POLICY", "SOLAR_REFERENCE", "NANOSECONDS"}:
                    values[target.id] = ast.literal_eval(node.value)
    if set(values) != {"SOLAR_POLICY", "SOLAR_REFERENCE", "NANOSECONDS"}:
        raise ValueError("complete literal solar-clock policy required")
    if values["NANOSECONDS"] != 1_000_000_000:
        raise ValueError("original nanosecond unit required")
    return values


def reconciliation_description(dataset):
    result = {name: dataset["reconciliation"][name] for name in COUNTS}
    if any(type(value) is not int or value < 0 for value in result.values()):
        raise ValueError("nonnegative integer reported reconciliation counts required")
    if result["source_condition_points"] - result["aligned_refined_points"] != result["total_condition_points_not_aligned"]:
        raise ValueError("reported source/aligned denominator differs")
    return result


def footer_description(field, rows):
    if field.name != "t" or type(rows) is not int or rows < 0:
        raise ValueError("clock field and nonnegative footer row count required")
    return {"clock_type": str(field.type), "clock_timezone": getattr(field.type, "tz", None),
            "physical_parquet_rows": rows}


def project(release, data_root, bundle_path, inventory_path):
    bundle = unpack(read_json(bundle_path))
    protocol, execution = unpack(bundle["protocol"]), unpack(bundle["execution"])
    if execution["protocol_sha256"] != bundle["protocol"]["sha256"]:
        raise ValueError("clock protocol/execution binding differs")
    binding = protocol["dataset_inputs"]
    release = Path(release)
    dataset = read_json(release / "dataset.json", expected_file_sha256=binding["dataset_file_sha256"])
    if dataset["dataset_id"] != binding["dataset_id"]:
        raise ValueError("original release clock identity required")
    root = Path(__file__).resolve().parents[2]
    sources = {"DSDE-SDE/trajectory/cohort_release.py": root.parent / "DSDE-SDE/trajectory/cohort_release.py"}
    for name in ("experiments/nex326/pirc20_adapter.py", "experiments/pirc17/method_inputs.py",
                 "experiments/pirc17/formal_inputs.py", "experiments/pirc17/method_rollout.py"):
        sources["PSDE-SDE/" + name] = root / name
    hashes = {}
    for name, path in sources.items():
        expected = execution["source_sha256"][name]
        if (file_hash(path) != expected
                or name != "PSDE-SDE/experiments/pirc17/formal_inputs.py"
                and protocol["source_sha256"][name] != expected):
            raise ValueError("frozen clock conversion source differs")
        hashes[name] = expected
    constants = clock_literals((root / "experiments/pirc17/method_inputs.py").read_text(encoding="utf-8"))
    manifest = release / "condition_file_manifest.jsonl"
    manifest_sha = binding["release_artifact_sha256"][manifest.name]
    if file_hash(manifest) != manifest_sha or dataset["artifacts"][manifest.name]["sha256"] != manifest_sha:
        raise ValueError("original condition-file metadata differs")
    footers = []
    with manifest.open("rb") as stream:
        for ordinal, raw in enumerate(islice(stream, 3)):
            row = decode(raw)
            path = under(Path(data_root) / "cond_slices", row["relative_path"])
            if file_hash(path) != row["sha256"]:
                raise ValueError("selected original footer source bytes differ")
            saved = pq.ParquetFile(path)
            footers.append({"selection_ordinal": ordinal, "source_file_sha256": row["sha256"],
                            **footer_description(saved.schema_arrow.field("t"), saved.metadata.num_rows)})
    if len(footers) != 3:
        raise ValueError("exact fixed first-three-footer description required")
    inventory = read_json(inventory_path, expected_file_sha256=INVENTORY_SHA)
    fit_ids, methods = set(), []
    for row in inventory["models"]:
        if row["matrix"] != "NEX326-methods":
            continue
        fit = unpack(read_json(row["original_model_record_path"], expected_file_sha256=row["original_model_record_file_sha256"]),
                     expected_sha256=row["original_model_record_content_sha256"])
        training, model = fit["artifact"]["training"], fit["artifact"]["model"]
        if (fit["fit_identity"] != row["fit_identity"] or fit["parameter_identity"] != row["parameter_identity"]
                or fit["protocol_sha256"] != bundle["protocol"]["sha256"]
                or fit["matrix_sha256"] != inventory["matrix_sha256"]
                or fit["fit_identity"] in fit_ids
                or training["input_sha256"] != binding["development"]["method_input_sha256"]
                or training["solar_policy"] != constants["SOLAR_POLICY"]):
            raise ValueError("original saved fit clock population differs")
        fit_ids.add(fit["fit_identity"])
        methods.append({"representative_slot": row["representative_slot"],
                        "condition_names": model["condition_names"], "solar_policy": training["solar_policy"],
                        "fit_record_sha256": row["original_model_record_file_sha256"]})
    if len(methods) != 16:
        raise ValueError("complete original sixteen saved method fits required")
    result = {"schema_version": "pirc17-clock-description-v1",
        "dataset_sha256": binding["dataset_file_sha256"], "condition_manifest_sha256": manifest_sha,
        "protocol_sha256": bundle["protocol"]["sha256"], "inventory_sha256": INVENTORY_SHA,
        "source_sha256": hashes, "solar_constants": constants,
        "source_binding_scope": "formal_inputs.py is execution-bound; other four clock sources match both protocol and execution",
        "release_alignment_semantics": dataset["alignment_semantics"],
        "release_segment_semantics": dataset["segment_semantics"],
        "reported_release_reconciliation": reconciliation_description(dataset),
        "condition_clock_footers": footers, "saved_method_clock_policies": sorted(methods, key=lambda x: x["representative_slot"]),
        "footer_selection": "First three files in the original hash-bound condition manifest; no outcome/role filtering; not representative or full-cohort audit",
        "scope": {"new_fits": 0, "new_forecasts": 0, "new_particle_scores": 0,
            "coordinate_speed_or_timestamp_values_decoded": False, "selected_source_file_bytes_hashed": True,
            "reported_reconciliation_recomputed_from_points": False, "full_cohort_footer_audit": False,
            "utc_or_sensor_clock_source_certified": False, "stored_historical_solar_generator_parity_certified": False,
            "physical_horizon_or_daylight_effect_certified": False, "independent_saved_forecast_audit": False,
            "private_file_segment_sample_or_recording_ids_exported": False}}
    canonical(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "data-root", "bundle", "inventory", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = project(args.release, args.data_root, args.bundle, args.inventory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print("Described frozen clock sources and three existing footers; no fits or forecasts.")


if __name__ == "__main__":
    main()
