"""Run and summarize explicit replicate seeds for the NEX326 4D benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
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
    cohort = load_cohort(cohort_file)
    destination.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    reports: list[dict[str, object]] = []
    for seed in selected_seeds:
        relative = Path(f"seed-{seed}") / "phase_space_report.json"
        report_path = destination / relative
        resolver = (
            DSDERasterConditionResolver(cohort, condition_root, srtm_root)
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
    conclusion = (
        "observed_primary_metric_gain"
        if mean(primary) < 0.0
        else "no_observed_primary_metric_gain"
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
