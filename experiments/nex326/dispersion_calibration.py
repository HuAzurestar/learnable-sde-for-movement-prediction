"""Validation-only dispersion calibration for frozen PIRC-20 candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence

import numpy as np

from .cohort import select_segments
from .model import ModelState
from .pirc20_adapter import load_pirc20_nex326_cohort
from .pirc20_runtime import predict_segments_without_endpoint_start
from .runner import Prediction, _stable_seed, compute_metrics, validate_run_record
from .specification import load_experiment_spec


class DispersionCalibrationError(ValueError):
    """Calibration inputs are incomplete, inconsistent, or integrity-invalid."""


TARGET_COVERAGE = 0.9
CALIBRATION_VERSION = "pirc20-validation-dispersion-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DispersionCalibrationError(f"{label} cannot be read") from exc
    if not isinstance(payload, dict):
        raise DispersionCalibrationError(f"{label} must contain a JSON object")
    return payload


def _verify_reference(root: Path, reference: Mapping[str, object], label: str) -> Path:
    path = (root / str(reference.get("path", ""))).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise DispersionCalibrationError(f"{label} escapes its run root") from exc
    if (
        not path.is_file()
        or path.stat().st_size != reference.get("size_bytes")
        or _sha256(path) != reference.get("sha256")
    ):
        raise DispersionCalibrationError(f"{label} fails integrity verification")
    return path


def _model(payload: Mapping[str, object]) -> ModelState:
    return ModelState(
        model_kind=str(payload["model_kind"]),
        condition_names=tuple(str(name) for name in payload["condition_names"]),
        weights=tuple(np.asarray(value, dtype=float) for value in payload["weights"]),
        covariances=tuple(
            np.asarray(value, dtype=float) for value in payload["covariances"]
        ),
        mode_probabilities=np.asarray(payload["mode_probabilities"], dtype=float),
        training_sample_count=int(payload["training_sample_count"]),
        validation_objective_before=float(payload["validation_objective_before"]),
        validation_objective_after=float(payload["validation_objective_after"]),
        covariance_scale=float(payload["covariance_scale"]),
        meta_task_count=int(payload.get("meta_task_count", 0)),
        adaptation_parameter_delta=float(payload.get("adaptation_parameter_delta", 0.0)),
        estimator_method=str(payload.get("estimator_method", "unassigned")),
        estimator_optimization_scope=str(
            payload.get("estimator_optimization_scope", "unassigned")
        ),
        estimator_drift_fraction=float(payload.get("estimator_drift_fraction", 0.0)),
        estimator_candidate_count=int(payload.get("estimator_candidate_count", 0)),
        transfer_method=str(payload.get("transfer_method", "unassigned")),
        finetune_method=str(payload.get("finetune_method", "unassigned")),
        meta_algorithm=str(payload.get("meta_algorithm", "none")),
        meta_outer_epochs=int(payload.get("meta_outer_epochs", 0)),
        meta_inner_steps=int(payload.get("meta_inner_steps", 0)),
        meta_inner_rate=float(payload.get("meta_inner_rate", 0.0)),
        meta_inner_objective_before=float(
            payload.get("meta_inner_objective_before", 0.0)
        ),
        meta_inner_objective_after=float(payload.get("meta_inner_objective_after", 0.0)),
    )


def _predictions(payload: Mapping[str, object]) -> tuple[Prediction, ...]:
    rows = payload.get("predictions")
    if not isinstance(rows, list):
        raise DispersionCalibrationError("prediction artifact has no predictions")
    return tuple(
        Prediction(
            segment_id=str(row["segment_id"]),
            target=np.asarray(row["target"], dtype=float),
            samples=np.asarray(row["samples"], dtype=float),
            prior_mean=(
                np.asarray(row["endpoint_prior_mean"], dtype=float)
                if row.get("endpoint_prior_mean") is not None
                else None
            ),
            prior_source=row.get("endpoint_prior_source"),
            unbridged_prior_distance=None,
            bridged_prior_distance=None,
            bridge_path=None,
            bridge_diagnostics=row.get("bridge_diagnostics"),
        )
        for row in rows
    )


def calibration_factor(
    predictions: Sequence[Prediction], *, target_coverage: float = TARGET_COVERAGE
) -> float:
    """Return the smallest empirical scale attaining the requested HDR coverage."""

    if not predictions or not 0.0 < target_coverage < 1.0:
        raise DispersionCalibrationError("calibration inputs are empty or invalid")
    ratios: list[float] = []
    for prediction in predictions:
        center = prediction.samples.mean(axis=0)
        radii = np.linalg.norm(prediction.samples - center, axis=1)
        hdr_radius = float(np.quantile(radii, target_coverage))
        if not np.isfinite(hdr_radius) or hdr_radius <= 0.0:
            raise DispersionCalibrationError("validation forecast has zero HDR radius")
        target_radius = float(np.linalg.norm(prediction.target - center))
        ratios.append(target_radius / hdr_radius)
    return float(np.quantile(np.asarray(ratios), target_coverage, method="higher"))


def scale_predictions(
    predictions: Sequence[Prediction], factor: float
) -> tuple[Prediction, ...]:
    if not np.isfinite(factor) or factor <= 0.0:
        raise DispersionCalibrationError("dispersion factor must be positive and finite")
    scaled: list[Prediction] = []
    for prediction in predictions:
        center = prediction.samples.mean(axis=0)
        samples = center + factor * (prediction.samples - center)
        scaled.append(
            Prediction(
                segment_id=prediction.segment_id,
                target=prediction.target.copy(),
                samples=samples,
                prior_mean=prediction.prior_mean,
                prior_source=prediction.prior_source,
                unbridged_prior_distance=prediction.unbridged_prior_distance,
                bridged_prior_distance=prediction.bridged_prior_distance,
                bridge_path=prediction.bridge_path,
                bridge_diagnostics=prediction.bridge_diagnostics,
            )
        )
    return tuple(scaled)


def _record(root: Path, arm_id: int, subconfig_id: str) -> tuple[dict[str, object], Path, Path]:
    manifest = _load(root / "manifest.json", "run manifest")
    for relative in manifest.get("run_records", []):
        record_file = (root / str(relative)).resolve()
        record = _load(record_file, "RunRecord")
        if int(record.get("arm_id", -1)) != arm_id or record.get("subconfig_id") != subconfig_id:
            continue
        validate_run_record(record)
        artifacts = record["artifacts"]
        checkpoint = _verify_reference(root, artifacts["checkpoint"], "checkpoint")
        predictions = _verify_reference(root, artifacts["predictions"], "predictions")
        return record, checkpoint, predictions
    raise DispersionCalibrationError(f"run root lacks {arm_id}/{subconfig_id}")


def _configs(selection: Mapping[str, object]) -> list[tuple[int, str]]:
    configs: set[tuple[int, str]] = set()
    for candidate in selection["candidates"]:
        configs.add((int(candidate["arm_id"]), str(candidate["subconfig_id"])))
        configs.add(
            (
                int(candidate["reference_arm_id"]),
                str(candidate["reference_subconfig_id"]),
            )
        )
    return sorted(configs)


def _spec_execution(arm_id: int, subconfig_id: str):
    spec = load_experiment_spec()
    for arm, subconfig in spec.executions:
        if arm.arm_id == arm_id and subconfig["subconfig_id"] == subconfig_id:
            return spec, arm, subconfig
    raise DispersionCalibrationError(f"unknown frozen execution {arm_id}/{subconfig_id}")


def run_dispersion_calibration(
    cohort_path: Path | str,
    trajectory_path: Path | str,
    condition_root: Path | str,
    candidate_selection_path: Path | str,
    run_roots: Sequence[Path | str],
    output_path: Path | str,
) -> dict[str, object]:
    """Fit validation scales first, then evaluate already-authorized predictions."""

    selection_file = Path(candidate_selection_path).resolve()
    selection = _load(selection_file, "candidate selection")
    if selection.get("schema_version") != "pirc20-candidate-selection-v1":
        raise DispersionCalibrationError("unsupported candidate selection")
    roots = tuple(Path(root).resolve() for root in run_roots)
    if len(roots) < 2 or len(roots) != len(set(roots)):
        raise DispersionCalibrationError("at least two distinct seed roots are required")

    # No unlock is supplied: this materialization cannot read final_eval source rows.
    validation_cohort = load_pirc20_nex326_cohort(
        cohort_path, trajectory_path, condition_root, final_eval_unlock=None
    )
    configs = _configs(selection)
    calibrated: dict[tuple[int, int, str], dict[str, object]] = {}
    source_records: list[dict[str, object]] = []
    for root in roots:
        seed_values: set[int] = set()
        for arm_id, subconfig_id in configs:
            record, checkpoint_file, evaluation_file = _record(
                root, arm_id, subconfig_id
            )
            replicate_seed = int(record["replicate_seed"])
            seed_values.add(replicate_seed)
            spec, arm, subconfig = _spec_execution(arm_id, subconfig_id)
            config = {**spec.full_components, **subconfig.get("components", {})}
            validation_segments = select_segments(
                validation_cohort.splits["validation"],
                dt_seconds=float(config["dt_seconds"]),
            )
            model = _model(_load(checkpoint_file, "checkpoint"))
            prediction_count = int(record["runtime"]["effective_prediction_samples"])
            validation_predictions = predict_segments_without_endpoint_start(
                model,
                validation_segments,
                config,
                seed=_stable_seed(replicate_seed, arm.arm_id, subconfig_id),
                n_samples=prediction_count,
            )
            factor = calibration_factor(validation_predictions)
            validation_before = compute_metrics(validation_predictions, "d2_mc")
            validation_after = compute_metrics(
                scale_predictions(validation_predictions, factor), "d2_mc"
            )

            # Evaluation targets are opened only after the validation factor is fixed.
            evaluation_predictions = _predictions(_load(evaluation_file, "predictions"))
            evaluation_before = compute_metrics(
                evaluation_predictions, str(config.get("score", "d2_mc"))
            )
            for metric, recorded in record["metrics"].items():
                if metric == "n_evaluation_segments":
                    continue
                if not np.isclose(float(evaluation_before[metric]), float(recorded), rtol=1e-12, atol=1e-12):
                    raise DispersionCalibrationError("stored evaluation metrics do not reproduce")
            evaluation_after = compute_metrics(
                scale_predictions(evaluation_predictions, factor),
                str(config.get("score", "d2_mc")),
            )
            calibrated[(replicate_seed, arm_id, subconfig_id)] = {
                "replicate_seed": replicate_seed,
                "arm_id": arm_id,
                "subconfig_id": subconfig_id,
                "validation_dispersion_factor": factor,
                "validation_metrics_before": validation_before,
                "validation_metrics_after": validation_after,
                "evaluation_metrics_before": evaluation_before,
                "evaluation_metrics_after": evaluation_after,
            }
            source_records.append(
                {
                    "replicate_seed": replicate_seed,
                    "arm_id": arm_id,
                    "subconfig_id": subconfig_id,
                    "run_record_sha256": _sha256(
                        root / f"{record['run_id']}/run_record.json"
                    ),
                    "checkpoint_sha256": _sha256(checkpoint_file),
                    "predictions_sha256": _sha256(evaluation_file),
                }
            )
        if len(seed_values) != 1:
            raise DispersionCalibrationError("one run root contains multiple replicate seeds")

    seeds = sorted({key[0] for key in calibrated})
    comparisons: list[dict[str, object]] = []
    for candidate in selection["candidates"]:
        arm_id = int(candidate["arm_id"])
        subconfig_id = str(candidate["subconfig_id"])
        reference_arm = int(candidate["reference_arm_id"])
        reference_subconfig = str(candidate["reference_subconfig_id"])
        per_seed: list[dict[str, object]] = []
        for seed in seeds:
            candidate_metrics = calibrated[(seed, arm_id, subconfig_id)][
                "evaluation_metrics_after"
            ]
            reference_metrics = calibrated[(seed, reference_arm, reference_subconfig)][
                "evaluation_metrics_after"
            ]
            deltas = {
                "energy_score_d2": float(candidate_metrics["energy_score_d2"])
                - float(reference_metrics["energy_score_d2"]),
                "hdr90_abs_error_from_90": abs(
                    float(candidate_metrics["hdr90_coverage"]) - TARGET_COVERAGE
                )
                - abs(float(reference_metrics["hdr90_coverage"]) - TARGET_COVERAGE),
                "cep50_error": float(candidate_metrics["cep50_error"])
                - float(reference_metrics["cep50_error"]),
            }
            per_seed.append({"replicate_seed": seed, "deltas": deltas})
        comparisons.append(
            {
                "arm_id": arm_id,
                "subconfig_id": subconfig_id,
                "reference_arm_id": reference_arm,
                "reference_subconfig_id": reference_subconfig,
                "per_seed": per_seed,
                "mean_deltas": {
                    metric: mean(row["deltas"][metric] for row in per_seed)
                    for metric in (
                        "energy_score_d2",
                        "hdr90_abs_error_from_90",
                        "cep50_error",
                    )
                },
            }
        )

    result: dict[str, object] = {
        "schema_version": "pirc20-dispersion-calibration-v1",
        "task_id": "PIRC-20",
        "experiment_id": "NEX326",
        "calibration_version": CALIBRATION_VERSION,
        "scientific_status": "exploratory_calibration_not_confirmation",
        "fit_split": "validation",
        "evaluation_split": "already_authorized_final_eval",
        "target_coverage": TARGET_COVERAGE,
        "candidate_selection_sha256": _sha256(selection_file),
        "replicate_seeds": seeds,
        "calibrated_configurations": [
            calibrated[key] for key in sorted(calibrated)
        ],
        "comparisons": comparisons,
        "integrity": {
            "implementation_sha256": _sha256(Path(__file__)),
            "source_records": sorted(
                source_records,
                key=lambda row: (
                    int(row["replicate_seed"]),
                    int(row["arm_id"]),
                    str(row["subconfig_id"]),
                ),
            ),
        },
        "privacy": {
            "contains_sample_ids": False,
            "contains_file_ids": False,
            "contains_coordinates": False,
            "contains_timestamps": False,
            "contains_row_level_predictions": False,
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise DispersionCalibrationError(f"stale staging file exists: {temporary}")
    try:
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--condition-root", type=Path, required=True)
    parser.add_argument("--candidate-selection", type=Path, required=True)
    parser.add_argument("--run-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_dispersion_calibration(
        args.cohort,
        args.trajectory,
        args.condition_root,
        args.candidate_selection,
        args.run_roots,
        args.output,
    )
    print(json.dumps({"replicate_seeds": result["replicate_seeds"], "comparisons": result["comparisons"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DispersionCalibrationError",
    "calibration_factor",
    "run_dispersion_calibration",
    "scale_predictions",
]
