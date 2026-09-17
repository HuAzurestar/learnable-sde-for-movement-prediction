"""Run the bounded PIRC-21 baseline/full-feature integration pilot.

This is an execution/readiness check, not a feature-selection sweep or a source of
paper conclusions.  It runs one frozen NEX326 arm with a deterministic small cohort
while retaining one registered representation for every PIRC-21 factor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from .pirc20_adapter import load_pirc20_nex326_cohort
from .pirc20_runtime import PIRC20NEX326Runner
from .pirc21_adapter import FeatureSelection
from .pirc21_runtime import PIRC21FeatureRuntime, PIRC21NEX326Runner
from .specification import SPEC_PATH, load_experiment_spec


PILOT_SCHEMA_VERSION = "pirc21-runtime-pilot-v1"
PILOT_CONFIGURATION_IDS = ("no_pirc21_features", "all_registered_factors")
PILOT_VARIANTS_BY_FACTOR = {
    "dem_elevation": "elevation.absolute",
    "dem_local_elevation": "elevation.local",
    "dem_surface": "surface.orientation",
    "worldcover": "worldcover.grouped",
    "overture_road": "road.distance_log1p",
    "overture_path": "path.distance_log1p",
    "overture_rail": "rail.distance_log1p",
    "hydrorivers_river": "river.distance_log1p",
    "jrc_surface_water": "jrc.distance_log1p",
    "overture_navigable_water": "navigable_water.distance_log1p",
    "osm_ridge": "ridge.distance_log1p",
    "osm_cliff": "cliff.distance_log1p",
    "historical_motion": "history.direction",
}


class PIRC21PilotError(ValueError):
    """The bounded runtime pilot cannot produce an auditable receipt."""


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def build_pilot_receipt(
    *,
    dataset_id: str,
    snapshot_id: str,
    runtime_identity_sha256: str,
    transition_count: int,
    records: Mapping[str, Mapping[str, object]],
    cohort_fingerprint: str,
    maximum_segments_per_role: int,
) -> dict[str, object]:
    """Build a strict receipt accepted by the DSDE completion audit."""

    if set(records) != set(PILOT_CONFIGURATION_IDS):
        raise PIRC21PilotError("both registered pilot configurations are required")
    if any(record.get("run_status") != "succeeded" for record in records.values()):
        raise PIRC21PilotError("every pilot configuration must complete successfully")
    if len(runtime_identity_sha256) != 64 or transition_count <= 0:
        raise PIRC21PilotError(
            "runtime identity and positive transition count are required"
        )
    return {
        "schema_version": PILOT_SCHEMA_VERSION,
        "scientific_role": "bounded_integration_pilot_not_feature_selection",
        "dataset_id": dataset_id,
        "snapshot_id": snapshot_id,
        "runtime_identity_sha256": runtime_identity_sha256,
        "transition_count": transition_count,
        "selected_factor_ids": sorted(PILOT_VARIANTS_BY_FACTOR),
        "selected_variant_ids": list(PILOT_VARIANTS_BY_FACTOR.values()),
        "configuration_ids": list(PILOT_CONFIGURATION_IDS),
        "cohort_fingerprint": cohort_fingerprint,
        "maximum_segments_per_role": maximum_segments_per_role,
        "run_records": {
            configuration_id: {
                "run_id": record["run_id"],
                "result_id": record["result_id"],
                "run_status": record["run_status"],
                "metrics": record["metrics"],
            }
            for configuration_id, record in records.items()
        },
        "receipt_identity_sha256": _canonical_hash(
            {
                "dataset_id": dataset_id,
                "snapshot_id": snapshot_id,
                "runtime_identity_sha256": runtime_identity_sha256,
                "transition_count": transition_count,
                "configuration_ids": PILOT_CONFIGURATION_IDS,
                "record_result_ids": {
                    key: value["result_id"] for key, value in records.items()
                },
            }
        ),
        "paper_conclusions_ready": False,
        "paper_conclusions_reason": (
            "bounded integration pilot; final feature selection and outcome "
            "interpretation are outside PIRC-21"
        ),
    }


def run_pilot(
    *,
    cohort_path: Path,
    trajectory_path: Path,
    condition_root: Path,
    snapshot_root: Path,
    output_root: Path,
    final_eval_unlock: str,
    maximum_segments_per_role: int,
    prediction_samples: int,
    spec_path: Path = SPEC_PATH,
) -> dict[str, object]:
    if maximum_segments_per_role <= 0 or prediction_samples <= 0:
        raise PIRC21PilotError("pilot bounds must be positive")
    if output_root.exists() and any(output_root.iterdir()):
        raise PIRC21PilotError("pilot output directory must be absent or empty")
    output_root.mkdir(parents=True, exist_ok=True)
    limits = {
        role: maximum_segments_per_role
        for role in ("train", "validation", "adapt", "evaluation")
    }
    cohort = load_pirc20_nex326_cohort(
        cohort_path,
        trajectory_path,
        condition_root,
        final_eval_unlock=final_eval_unlock,
        maximum_segments_per_role=limits,
    )
    spec = load_experiment_spec(spec_path)
    arm = next(item for item in spec.arms if item.arm_id == 1)
    subconfig = next(item for item in arm.subconfigs if item["subconfig_id"] == "full")

    baseline_runner = PIRC20NEX326Runner(
        spec,
        cohort,
        output_root / "no_pirc21_features",
        n_samples=prediction_samples,
    )
    baseline_record = baseline_runner.run_one(arm, subconfig)

    selection = FeatureSelection(
        variant_ids=tuple(PILOT_VARIANTS_BY_FACTOR.values()),
        include_validity_indicators=True,
    )
    feature_runtime = PIRC21FeatureRuntime(snapshot_root, selection)
    feature_runner = PIRC21NEX326Runner(
        spec,
        cohort,
        output_root / "all_registered_factors",
        n_samples=prediction_samples,
        feature_runtime=feature_runtime,
    )
    feature_record = feature_runner.run_one(arm, subconfig)
    records = {
        "no_pirc21_features": baseline_record,
        "all_registered_factors": feature_record,
    }
    transition_count = sum(
        max(0, len(segment.time) - 1)
        for segment in feature_runner.cohort.splits["train"]
    )
    receipt = build_pilot_receipt(
        dataset_id=feature_runtime.adapter.dataset_id,
        snapshot_id=str(feature_runtime.adapter.manifest["snapshot_id"]),
        runtime_identity_sha256=str(
            feature_runtime.identity_record["runtime_identity_sha256"]
        ),
        transition_count=transition_count,
        records=records,
        cohort_fingerprint=cohort.fingerprint,
        maximum_segments_per_role=maximum_segments_per_role,
    )
    _write_json(output_root / "receipt.json", receipt)
    _write_json(
        output_root / "comparison.json",
        {
            "schema_version": "pirc21-runtime-pilot-comparison-v1",
            "scientific_role": "diagnostic_only_no_conclusion",
            "configurations": {key: value["metrics"] for key, value in records.items()},
        },
    )
    metric_names = sorted(
        set(baseline_record["metrics"]) & set(feature_record["metrics"])  # type: ignore[arg-type]
    )
    with (output_root / "comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("metric", *PILOT_CONFIGURATION_IDS))
        for metric in metric_names:
            writer.writerow(
                (
                    metric,
                    baseline_record["metrics"][metric],  # type: ignore[index]
                    feature_record["metrics"][metric],  # type: ignore[index]
                )
            )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--condition-root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-eval-unlock", required=True)
    parser.add_argument("--maximum-segments-per-role", type=int, default=8)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    args = parser.parse_args(argv)
    receipt = run_pilot(
        cohort_path=args.cohort,
        trajectory_path=args.trajectory,
        condition_root=args.condition_root,
        snapshot_root=args.snapshot,
        output_root=args.output,
        final_eval_unlock=args.final_eval_unlock,
        maximum_segments_per_role=args.maximum_segments_per_role,
        prediction_samples=args.samples,
        spec_path=args.spec,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PILOT_CONFIGURATION_IDS",
    "PILOT_SCHEMA_VERSION",
    "PILOT_VARIANTS_BY_FACTOR",
    "PIRC21PilotError",
    "build_pilot_receipt",
    "main",
    "run_pilot",
]
