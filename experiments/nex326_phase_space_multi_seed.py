"""Run and summarize explicit replicate seeds for the NEX326 4D benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Mapping, Sequence

from experiments.nex326.phase_space import (
    REPORT_SCHEMA_VERSION,
    SPEC_PATH,
    PhaseSpaceError,
    load_phase_space_spec,
    write_phase_space_report,
)


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
) -> dict[str, object]:
    """Write one report per seed and a compact descriptive manifest."""
    selected_seeds = _validated_seeds(seeds)
    if n_samples < 2:
        raise PhaseSpaceReplicateError("prediction samples must be at least two")
    cohort_file = Path(cohort_path)
    destination = Path(output_root)
    manifest_path = destination / "phase_space_multi_seed_manifest.json"
    if not cohort_file.is_file():
        raise PhaseSpaceReplicateError("cohort file does not exist")
    if manifest_path.exists() or any((destination / f"seed-{seed}").exists() for seed in selected_seeds):
        raise PhaseSpaceReplicateError("one or more phase-space outputs already exist")
    spec = load_phase_space_spec(spec_path)
    destination.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    reports: list[dict[str, object]] = []
    for seed in selected_seeds:
        relative = Path(f"seed-{seed}") / "phase_space_report.json"
        report_path = destination / relative
        report = write_phase_space_report(
            cohort_file,
            report_path,
            spec_path=spec_path,
            n_samples=n_samples,
            seed=seed,
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    manifest = run_phase_space_replicates(
        args.cohort,
        args.output,
        args.seeds,
        spec_path=args.spec,
        n_samples=args.samples,
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
    "write_phase_space_receipt",
]
