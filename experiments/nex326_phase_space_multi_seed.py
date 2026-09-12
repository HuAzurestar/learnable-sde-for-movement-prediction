"""Run and summarize explicit replicate seeds for the NEX326 4D benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import mean, median, stdev
from typing import Mapping, Sequence

from experiments.nex326.cohort import load_cohort
from experiments.nex326.phase_space import (
    REPORT_SCHEMA_VERSION,
    SPEC_PATH,
    PhaseSpaceError,
    load_phase_space_spec,
    write_phase_space_report,
)
from experiments.nex326.spatial_conditions import DSDERasterConditionResolver


class PhaseSpaceReplicateError(ValueError):
    """A phase-space replicate batch is incomplete or ambiguous."""


UNCERTAINTY_PROTOCOL_PATH = Path(__file__).with_name("nex326") / (
    "phase_space_uncertainty_protocol.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    values = tuple(seeds)
    if len(values) < 2 or len(values) != len(set(values)):
        raise PhaseSpaceReplicateError("replicate seeds must contain at least two unique values")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in values):
        raise PhaseSpaceReplicateError("replicate seeds must be integers in [0, 2**32)")
    return values


def run_phase_space_replicates(
    cohort_path: Path | str,
    output_root: Path | str,
    seeds: Sequence[int],
    *,
    spec_path: Path | str = SPEC_PATH,
    n_samples: int = 64,
    condition_root: Path | str | None = None,
    srtm_root: Path | str | None = None,
) -> dict[str, object]:
    """Write one report per seed and a compact descriptive manifest."""
    selected_seeds = _validated_seeds(seeds)
    if n_samples < 2:
        raise PhaseSpaceReplicateError("prediction samples must be at least two")
    if (condition_root is None) != (srtm_root is None):
        raise PhaseSpaceReplicateError(
            "condition_root and srtm_root must be supplied together"
        )
    cohort_file = Path(cohort_path)
    destination = Path(output_root)
    manifest_path = destination / "phase_space_multi_seed_manifest.json"
    if not cohort_file.is_file():
        raise PhaseSpaceReplicateError("cohort file does not exist")
    if manifest_path.exists() or any((destination / f"seed-{seed}").exists() for seed in selected_seeds):
        raise PhaseSpaceReplicateError("one or more phase-space outputs already exist")
    spec = load_phase_space_spec(spec_path)
    condition_names = tuple(spec["condition_contract"]["registered_condition_names"])
    cohort = load_cohort(cohort_file)
    destination.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    reports: list[dict[str, object]] = []
    for seed in selected_seeds:
        relative = Path(f"seed-{seed}") / "phase_space_report.json"
        report_path = destination / relative
        resolver = (
            DSDERasterConditionResolver(
                cohort, condition_root, srtm_root, names=condition_names
            )
            if condition_root is not None
            else None
        )
        report = write_phase_space_report(
            cohort_file,
            report_path,
            spec_path=spec_path,
            n_samples=n_samples,
            seed=seed,
            condition_resolver=resolver,
        )
        reports.append(report)
        entries.append(
            {
                "seed": seed,
                "report": {
                    "path": relative.as_posix(),
                    "sha256": _sha256(report_path),
                    "size_bytes": report_path.stat().st_size,
                },
            }
        )
    identities = {
        str(report["implementation"]["source_bundle_sha256"])
        for report in reports
    }
    fingerprints = {str(report["dataset"]["fingerprint"]) for report in reports}
    if len(identities) != 1 or len(fingerprints) != 1:
        raise PhaseSpaceReplicateError("replicates do not share one implementation and cohort")
    condition_fields = {
        json.dumps(report.get("condition_field"), sort_keys=True)
        for report in reports
    }
    if len(condition_fields) != 1:
        raise PhaseSpaceReplicateError("replicates do not share one condition field identity")
    metric_names = tuple(str(name) for name in spec["metrics"])
    summary: dict[str, object] = {}
    for metric in metric_names:
        values = [float(report["metrics"][metric]) for report in reports]
        summary[metric] = {
            "mean": mean(values),
            "sample_std": stdev(values),
            "min": min(values),
            "max": max(values),
        }
    manifest: dict[str, object] = {
        "schema_version": "nex326-phase-space-multi-seed-manifest-v1",
        "benchmark_id": spec["benchmark_id"],
        "scientific_role": "supplemental_exploratory_benchmark",
        "assessment": "not_assessed",
        "replicate_seeds": list(selected_seeds),
        "replicate_count": len(selected_seeds),
        "prediction_samples_per_segment": n_samples,
        "cohort": {
            **reports[0]["dataset"],
            "source_file": cohort_file.name,
            "sha256": _sha256(cohort_file),
        },
        "implementation": reports[0]["implementation"],
        "state_contract": reports[0]["state_contract"],
        "condition_contract": reports[0]["condition_contract"],
        "condition_field": reports[0]["condition_field"],
        "registered_protocol": spec["protocol"],
        "metric_summary": summary,
        "reports": entries,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def write_phase_space_receipt(
    manifest_path: Path | str,
    output_path: Path | str,
) -> dict[str, object]:
    """Verify a replicate manifest and publish only compact, non-trajectory evidence."""
    manifest_file = Path(manifest_path)
    destination = Path(output_path)
    if destination.exists():
        raise PhaseSpaceReplicateError(f"output already exists: {destination}")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "nex326-phase-space-multi-seed-manifest-v1":
        raise PhaseSpaceReplicateError("unsupported phase-space replicate manifest")
    root = manifest_file.parent.resolve()
    verified: list[dict[str, object]] = []
    loaded_reports: list[dict[str, object]] = []
    seen_paths: set[Path] = set()
    for entry in manifest.get("reports", []):
        if not isinstance(entry, Mapping) or not isinstance(entry.get("report"), Mapping):
            raise PhaseSpaceReplicateError("manifest report entry is invalid")
        reference = entry["report"]
        path = (root / str(reference["path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise PhaseSpaceReplicateError("report escapes the manifest root") from exc
        if path in seen_paths:
            raise PhaseSpaceReplicateError("manifest repeats a phase-space report")
        seen_paths.add(path)
        if (
            not path.is_file()
            or path.stat().st_size != reference.get("size_bytes")
            or _sha256(path) != reference.get("sha256")
        ):
            raise PhaseSpaceReplicateError("phase-space report integrity check failed")
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("schema_version") != REPORT_SCHEMA_VERSION:
            raise PhaseSpaceReplicateError("phase-space report schema mismatch")
        if report.get("benchmark_id") != manifest.get("benchmark_id"):
            raise PhaseSpaceReplicateError("phase-space report benchmark mismatch")
        if report.get("protocol", {}).get("executed_seed") != entry.get("seed"):
            raise PhaseSpaceReplicateError("phase-space report seed mismatch")
        if (
            report.get("dataset", {}).get("fingerprint")
            != manifest.get("cohort", {}).get("fingerprint")
            or report.get("implementation", {}).get("source_bundle_sha256")
            != manifest.get("implementation", {}).get("source_bundle_sha256")
            or report.get("condition_field") != manifest.get("condition_field")
        ):
            raise PhaseSpaceReplicateError("phase-space report identity mismatch")
        loaded_reports.append(report)
        verified.append({"seed": entry["seed"], **dict(reference)})
    if len(verified) != manifest.get("replicate_count"):
        raise PhaseSpaceReplicateError("phase-space report count mismatch")
    if [item["seed"] for item in verified] != manifest.get("replicate_seeds"):
        raise PhaseSpaceReplicateError("phase-space replicate seed list mismatch")
    recomputed_summary: dict[str, object] = {}
    for metric in manifest.get("metric_summary", {}):
        values = [float(report["metrics"][metric]) for report in loaded_reports]
        recomputed_summary[metric] = {
            "mean": mean(values),
            "sample_std": stdev(values),
            "min": min(values),
            "max": max(values),
        }
    if recomputed_summary != manifest.get("metric_summary"):
        raise PhaseSpaceReplicateError("phase-space metric summary mismatch")
    receipt = {
        "schema_version": "nex326-phase-space-dsde-receipt-v1",
        "benchmark_id": manifest["benchmark_id"],
        "scientific_role": manifest["scientific_role"],
        "assessment": manifest["assessment"],
        "replicate_seeds": manifest["replicate_seeds"],
        "replicate_count": manifest["replicate_count"],
        "prediction_samples_per_segment": manifest["prediction_samples_per_segment"],
        "cohort": manifest["cohort"],
        "implementation": manifest["implementation"],
        "state_contract": manifest["state_contract"],
        "condition_contract": manifest["condition_contract"],
        "condition_field": manifest.get("condition_field"),
        "registered_protocol": manifest.get("registered_protocol"),
        "metric_summary": manifest["metric_summary"],
        "integrity": {
            "manifest_sha256": _sha256(manifest_file),
            "reports": verified,
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def _verified_manifest_reports(
    manifest_file: Path,
    manifest: Mapping[str, object],
) -> dict[int, dict[str, object]]:
    root = manifest_file.parent.resolve()
    reports: dict[int, dict[str, object]] = {}
    for entry in manifest.get("reports", []):
        if not isinstance(entry, Mapping) or not isinstance(entry.get("report"), Mapping):
            raise PhaseSpaceReplicateError("manifest report entry is invalid")
        reference = entry["report"]
        path = (root / str(reference["path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise PhaseSpaceReplicateError("report escapes the manifest root") from error
        seed = entry.get("seed")
        if not isinstance(seed, int) or seed in reports:
            raise PhaseSpaceReplicateError("manifest seed identity is invalid")
        if (
            not path.is_file()
            or path.stat().st_size != reference.get("size_bytes")
            or _sha256(path) != reference.get("sha256")
        ):
            raise PhaseSpaceReplicateError("phase-space report integrity check failed")
        report = json.loads(path.read_text(encoding="utf-8"))
        if (
            report.get("schema_version") != REPORT_SCHEMA_VERSION
            or report.get("benchmark_id") != manifest.get("benchmark_id")
            or report.get("protocol", {}).get("executed_seed") != seed
            or report.get("dataset", {}).get("fingerprint")
            != manifest.get("cohort", {}).get("fingerprint")
            or report.get("implementation", {}).get("source_bundle_sha256")
            != manifest.get("implementation", {}).get("source_bundle_sha256")
            or report.get("condition_field") != manifest.get("condition_field")
        ):
            raise PhaseSpaceReplicateError("phase-space report identity mismatch")
        reports[seed] = report
    if list(reports) != manifest.get("replicate_seeds"):
        raise PhaseSpaceReplicateError("phase-space replicate seed list mismatch")
    return reports


def write_phase_space_contrast(
    baseline_manifest_path: Path | str,
    candidate_manifest_path: Path | str,
    output_path: Path | str,
) -> dict[str, object]:
    """Write a paired-seed terrain contrast without promoting exploratory evidence."""
    baseline_file = Path(baseline_manifest_path)
    candidate_file = Path(candidate_manifest_path)
    destination = Path(output_path)
    if destination.exists():
        raise PhaseSpaceReplicateError(f"output already exists: {destination}")
    baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_file.read_text(encoding="utf-8"))
    for manifest in (baseline, candidate):
        if manifest.get("schema_version") != "nex326-phase-space-multi-seed-manifest-v1":
            raise PhaseSpaceReplicateError("unsupported phase-space replicate manifest")
    identity_fields = (
        "replicate_seeds",
        "replicate_count",
        "prediction_samples_per_segment",
        "state_contract",
    )
    if any(baseline.get(field) != candidate.get(field) for field in identity_fields):
        raise PhaseSpaceReplicateError("contrast manifests do not share one protocol")
    if baseline.get("cohort", {}).get("fingerprint") != candidate.get("cohort", {}).get(
        "fingerprint"
    ):
        raise PhaseSpaceReplicateError("contrast manifests do not share one cohort")
    baseline_reports = _verified_manifest_reports(baseline_file, baseline)
    candidate_reports = _verified_manifest_reports(candidate_file, candidate)
    metric_names = (
        "position_energy_score_d2",
        "position_hdr90_coverage",
        "position_cep50_error",
        "velocity_endpoint_rmse",
    )
    paired: list[dict[str, object]] = []
    for seed in baseline["replicate_seeds"]:
        baseline_metrics = baseline_reports[seed]["metrics"]
        candidate_metrics = candidate_reports[seed]["metrics"]
        paired.append(
            {
                "seed": seed,
                "candidate_minus_baseline": {
                    metric: float(candidate_metrics[metric])
                    - float(baseline_metrics[metric])
                    for metric in metric_names
                },
            }
        )
    deltas = {
        metric: [row["candidate_minus_baseline"][metric] for row in paired]
        for metric in metric_names
    }
    primary = deltas["position_energy_score_d2"]
    primary_better_count = sum(value < 0.0 for value in primary)
    required_better_count = len(primary) // 2 + 1
    conclusion = (
        "observed_primary_metric_gain"
        if mean(primary) < 0.0 and primary_better_count >= required_better_count
        else "no_observed_primary_metric_gain"
    )
    registered_primary = candidate.get("registered_protocol", {}).get(
        "primary_comparator"
    )
    contrast = {
        "schema_version": "nex326-phase-space-paired-contrast-v1",
        "scientific_role": "supplemental_exploratory_contrast_not_final_evidence",
        "baseline_benchmark_id": baseline["benchmark_id"],
        "candidate_benchmark_id": candidate["benchmark_id"],
        "cohort_fingerprint": baseline["cohort"]["fingerprint"],
        "replicate_seeds": baseline["replicate_seeds"],
        "prediction_samples_per_segment": baseline["prediction_samples_per_segment"],
        "primary_metric": "position_energy_score_d2",
        "comparator_role": (
            "registered_primary"
            if registered_primary == baseline["benchmark_id"]
            else "secondary_or_unregistered"
        ),
        "metric_direction": {
            "position_energy_score_d2": "lower_is_better",
            "position_hdr90_coverage": "descriptive_calibration_rate",
            "position_cep50_error": "lower_is_better",
            "velocity_endpoint_rmse": "lower_is_better",
        },
        "paired_deltas": paired,
        "delta_summary": {
            metric: {
                "mean": mean(values),
                "candidate_better_seed_count": sum(value < 0.0 for value in values),
                "seed_count": len(values),
            }
            for metric, values in deltas.items()
        },
        "conclusion": conclusion,
        "exploratory_gain_rule": {
            "mean_candidate_minus_baseline_below_zero": mean(primary) < 0.0,
            "candidate_better_seed_count": primary_better_count,
            "required_better_seed_count": required_better_count,
        },
        "interpretation": (
            "The terrain field is operational, but this paired pilot does not by itself "
            "establish that terrain conditioning improves trajectory prediction."
        ),
        "integrity": {
            "baseline_manifest_sha256": _sha256(baseline_file),
            "candidate_manifest_sha256": _sha256(candidate_file),
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(contrast, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return contrast


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered or not 0.0 <= probability <= 1.0:
        raise PhaseSpaceReplicateError("bootstrap percentile input is invalid")
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _segment_metric_delta(
    baseline_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    indices: Sequence[int],
    metric: str,
) -> float:
    if metric == "position_energy_score_d2":
        return mean(
            float(candidate_rows[index][metric])
            - float(baseline_rows[index][metric])
            for index in indices
        )
    if metric == "position_cep50_error":
        source = "position_endpoint_error"
        return median(float(candidate_rows[index][source]) for index in indices) - median(
            float(baseline_rows[index][source]) for index in indices
        )
    if metric == "velocity_endpoint_rmse":
        source = "velocity_endpoint_error"
        candidate = math.sqrt(
            mean(float(candidate_rows[index][source]) ** 2 for index in indices)
        )
        baseline = math.sqrt(
            mean(float(baseline_rows[index][source]) ** 2 for index in indices)
        )
        return candidate - baseline
    raise PhaseSpaceReplicateError(f"unsupported segment bootstrap metric: {metric}")


def write_phase_space_segment_bootstrap(
    baseline_manifest_path: Path | str,
    candidate_manifest_path: Path | str,
    contrast_path: Path | str,
    output_path: Path | str,
    *,
    protocol_path: Path | str = UNCERTAINTY_PROTOCOL_PATH,
) -> dict[str, object]:
    """Bootstrap paired evaluation segments while averaging sampling-seed noise."""
    baseline_file = Path(baseline_manifest_path)
    candidate_file = Path(candidate_manifest_path)
    contrast_file = Path(contrast_path)
    protocol_file = Path(protocol_path)
    destination = Path(output_path)
    if destination.exists():
        raise PhaseSpaceReplicateError(f"output already exists: {destination}")
    baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_file.read_text(encoding="utf-8"))
    contrast = json.loads(contrast_file.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "nex326-phase-space-uncertainty-protocol-v1":
        raise PhaseSpaceReplicateError("unsupported phase-space uncertainty protocol")
    if contrast.get("schema_version") != "nex326-phase-space-paired-contrast-v1":
        raise PhaseSpaceReplicateError("unsupported phase-space contrast")
    target = protocol.get("target_contrast", {})
    if (
        baseline.get("benchmark_id") != target.get("baseline_benchmark_id")
        or candidate.get("benchmark_id") != target.get("candidate_benchmark_id")
        or contrast.get("baseline_benchmark_id") != baseline.get("benchmark_id")
        or contrast.get("candidate_benchmark_id") != candidate.get("benchmark_id")
    ):
        raise PhaseSpaceReplicateError("uncertainty target does not match the contrast")
    integrity = contrast.get("integrity", {})
    if (
        integrity.get("baseline_manifest_sha256") != _sha256(baseline_file)
        or integrity.get("candidate_manifest_sha256") != _sha256(candidate_file)
    ):
        raise PhaseSpaceReplicateError("contrast manifest binding is invalid")
    if (
        baseline.get("replicate_seeds") != candidate.get("replicate_seeds")
        or baseline.get("cohort", {}).get("fingerprint")
        != candidate.get("cohort", {}).get("fingerprint")
    ):
        raise PhaseSpaceReplicateError("bootstrap manifests do not share one protocol")
    baseline_reports = _verified_manifest_reports(baseline_file, baseline)
    candidate_reports = _verified_manifest_reports(candidate_file, candidate)
    seeds = tuple(int(seed) for seed in baseline["replicate_seeds"])
    baseline_rows: dict[int, list[Mapping[str, object]]] = {}
    candidate_rows: dict[int, list[Mapping[str, object]]] = {}
    segment_ids: tuple[str, ...] | None = None
    for seed in seeds:
        baseline_per_segment = baseline_reports[seed].get("per_segment")
        candidate_per_segment = candidate_reports[seed].get("per_segment")
        if not isinstance(baseline_per_segment, list) or not isinstance(
            candidate_per_segment, list
        ):
            raise PhaseSpaceReplicateError("bootstrap reports lack per-segment metrics")
        baseline_ids = tuple(str(row.get("segment_id")) for row in baseline_per_segment)
        candidate_ids = tuple(str(row.get("segment_id")) for row in candidate_per_segment)
        if baseline_ids != candidate_ids or len(baseline_ids) != len(set(baseline_ids)):
            raise PhaseSpaceReplicateError("bootstrap segment pairing is invalid")
        if segment_ids is None:
            segment_ids = baseline_ids
        elif baseline_ids != segment_ids:
            raise PhaseSpaceReplicateError("bootstrap segment order differs across seeds")
        baseline_rows[seed] = baseline_per_segment
        candidate_rows[seed] = candidate_per_segment
    if not segment_ids:
        raise PhaseSpaceReplicateError("bootstrap requires evaluation segments")
    iterations = int(protocol.get("bootstrap_iterations", 0))
    bootstrap_seed = int(protocol.get("bootstrap_seed", -1))
    confidence_level = float(protocol.get("confidence_level", 0.0))
    metrics = tuple(str(metric) for metric in protocol.get("metrics", []))
    if (
        iterations < 100
        or bootstrap_seed < 0
        or not 0.0 < confidence_level < 1.0
        or metrics
        != (
            "position_energy_score_d2",
            "position_cep50_error",
            "velocity_endpoint_rmse",
        )
    ):
        raise PhaseSpaceReplicateError("uncertainty protocol parameters are invalid")

    full_indices = tuple(range(len(segment_ids)))
    point_estimates = {
        metric: mean(
            _segment_metric_delta(
                baseline_rows[seed], candidate_rows[seed], full_indices, metric
            )
            for seed in seeds
        )
        for metric in metrics
    }
    for metric in metrics:
        expected = float(contrast["delta_summary"][metric]["mean"])
        if not math.isclose(point_estimates[metric], expected, rel_tol=1e-12, abs_tol=1e-12):
            raise PhaseSpaceReplicateError(
                f"segment-derived {metric} point estimate disagrees with contrast"
            )

    generator = random.Random(bootstrap_seed)
    draws = {metric: [] for metric in metrics}
    for _ in range(iterations):
        indices = tuple(generator.randrange(len(segment_ids)) for _ in segment_ids)
        for metric in metrics:
            draws[metric].append(
                mean(
                    _segment_metric_delta(
                        baseline_rows[seed], candidate_rows[seed], indices, metric
                    )
                    for seed in seeds
                )
            )
    tail = (1.0 - confidence_level) / 2.0
    uncertainty = {}
    for metric in metrics:
        low = _percentile(draws[metric], tail)
        high = _percentile(draws[metric], 1.0 - tail)
        uncertainty[metric] = {
            "candidate_minus_baseline": point_estimates[metric],
            "ci_low": low,
            "ci_high": high,
            "interval_excludes_zero": high < 0.0 or low > 0.0,
            "bootstrap_fraction_below_zero": sum(
                value < 0.0 for value in draws[metric]
            )
            / iterations,
        }
    energy_per_segment = [
        mean(
            float(candidate_rows[seed][index]["position_energy_score_d2"])
            - float(baseline_rows[seed][index]["position_energy_score_d2"])
            for seed in seeds
        )
        for index in full_indices
    ]
    result = {
        "schema_version": "nex326-phase-space-segment-bootstrap-v1",
        "analysis_id": protocol["analysis_id"],
        "scientific_role": "supplemental_exploratory_uncertainty_not_final_evidence",
        "assessment": "not_assessed",
        "baseline_benchmark_id": baseline["benchmark_id"],
        "candidate_benchmark_id": candidate["benchmark_id"],
        "cohort_fingerprint": baseline["cohort"]["fingerprint"],
        "replicate_seeds": list(seeds),
        "prediction_samples_per_segment": baseline[
            "prediction_samples_per_segment"
        ],
        "bootstrap_unit": "paired_evaluation_segment",
        "evaluation_segment_count": len(segment_ids),
        "bootstrap_iterations": iterations,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "sampling_seed_handling": protocol["sampling_seed_handling"],
        "uncertainty_scope": "heldout_segment_sampling_only",
        "uncertainty": uncertainty,
        "energy_segment_direction": {
            "candidate_better_count": sum(value < 0.0 for value in energy_per_segment),
            "candidate_worse_count": sum(value > 0.0 for value in energy_per_segment),
            "median_candidate_minus_baseline": median(energy_per_segment),
        },
        "coverage_uncertainty": {
            "status": "not_computable_from_compact_reports",
            "reason": "legacy per-segment rows do not retain the HDR90 inclusion indicator",
        },
        "interpretation_limit": (
            "Prediction-sampling seeds share one fitted cohort and evaluation set; "
            "the interval does not quantify training-data or dataset-version uncertainty."
        ),
        "integrity": {
            "protocol_sha256": _sha256(protocol_file),
            "baseline_manifest_sha256": _sha256(baseline_file),
            "candidate_manifest_sha256": _sha256(candidate_file),
            "contrast_sha256": _sha256(contrast_file),
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--condition-root", type=Path)
    parser.add_argument("--srtm-root", type=Path)
    args = parser.parse_args(argv)
    manifest = run_phase_space_replicates(
        args.cohort,
        args.output,
        args.seeds,
        spec_path=args.spec,
        n_samples=args.samples,
        condition_root=args.condition_root,
        srtm_root=args.srtm_root,
    )
    if args.receipt is not None:
        write_phase_space_receipt(
            args.output / "phase_space_multi_seed_manifest.json", args.receipt
        )
    print(
        json.dumps(
            {
                "benchmark_id": manifest["benchmark_id"],
                "replicate_seeds": manifest["replicate_seeds"],
                "metric_summary": manifest["metric_summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PhaseSpaceReplicateError",
    "run_phase_space_replicates",
    "write_phase_space_contrast",
    "write_phase_space_receipt",
]
