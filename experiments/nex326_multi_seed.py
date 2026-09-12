"""Run a registered NEX326 cohort under multiple explicit replicate seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from experiments.nex326.runner import NEX326Runner
from experiments.nex326.specification import SPEC_PATH, load_experiment_spec


class MultiSeedError(ValueError):
    """The requested replicate batch is ambiguous or unsafe to execute."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_replicate_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(seeds)
    if len(normalized) < 2:
        raise MultiSeedError("a multi-seed batch requires at least two replicate seeds")
    if len(normalized) != len(set(normalized)):
        raise MultiSeedError("replicate seeds must be unique")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32
        for seed in normalized
    ):
        raise MultiSeedError("replicate seeds must be integers in [0, 2**32)")
    return normalized


def run_replicates(
    cohort_path: Path | str,
    output_root: Path | str,
    seeds: Sequence[int],
    *,
    spec_path: Path | str = SPEC_PATH,
    n_samples: int = 64,
    strict_environment: bool = False,
    condition_root: Path | str | None = None,
    srtm_root: Path | str | None = None,
) -> dict[str, object]:
    """Run complete 36-execution replicas and write one hash-bound batch manifest."""
    replicate_seeds = validate_replicate_seeds(seeds)
    cohort_file = Path(cohort_path)
    if not cohort_file.is_file():
        raise MultiSeedError("cohort file does not exist")
    if (condition_root is None) != (srtm_root is None):
        raise MultiSeedError("condition_root and srtm_root must be supplied together")
    destination = Path(output_root)
    manifest_path = destination / "multi_seed_manifest.json"
    if manifest_path.exists():
        raise MultiSeedError(f"multi-seed manifest already exists: {manifest_path}")
    existing_roots = [destination / f"seed-{seed}" for seed in replicate_seeds]
    if any(path.exists() for path in existing_roots):
        raise MultiSeedError("one or more replicate output directories already exist")
    destination.mkdir(parents=True, exist_ok=True)
    spec = load_experiment_spec(spec_path)
    entries: list[dict[str, object]] = []
    dataset_fingerprints: set[str] = set()
    implementation_identities: dict[str, dict[str, object]] = {}
    for seed in replicate_seeds:
        relative_root = Path(f"seed-{seed}")
        records_root = destination / relative_root
        runner = NEX326Runner.from_paths(
            cohort_file,
            records_root,
            spec_path=spec_path,
            n_samples=n_samples,
            replicate_seed=seed,
            strict_environment=strict_environment,
            condition_root=condition_root,
            srtm_root=srtm_root,
        )
        records = runner.run_all()
        dataset_fingerprints.update(str(record["dataset"]["fingerprint"]) for record in records)
        for record in records:
            implementation = dict(record["implementation"])
            implementation_identities[
                str(implementation["execution_identity_sha256"])
            ] = implementation
        run_manifest = records_root / "manifest.json"
        status = Counter(str(record["run_status"]) for record in records)
        entries.append(
            {
                "replicate_seed": seed,
                "records_root": relative_root.as_posix(),
                "record_count": len(records),
                "run_status": dict(status),
                "manifest": {
                    "path": (relative_root / "manifest.json").as_posix(),
                    "sha256": _sha256(run_manifest),
                    "size_bytes": run_manifest.stat().st_size,
                },
            }
        )
    if len(dataset_fingerprints) != 1:
        raise MultiSeedError("replicates did not use one identical cohort fingerprint")
    if len(implementation_identities) != 1:
        raise MultiSeedError("replicates did not use one identical execution identity")
    payload: dict[str, object] = {
        "schema_version": "nex326-multi-seed-manifest-v1",
        "experiment_id": spec.experiment_id,
        "spec_version": spec.spec_version,
        "scientific_status": "replicate_runs_not_combined_verdict",
        "protocol_seed": spec.seed,
        "replicate_seeds": list(replicate_seeds),
        "replicate_count": len(replicate_seeds),
        "requested_prediction_samples": n_samples,
        "implementation": next(iter(implementation_identities.values())),
        "executions_per_replicate": len(spec.executions),
        "total_executions": len(spec.executions) * len(replicate_seeds),
        "cohort": {
            "source_file": cohort_file.name,
            "sha256": _sha256(cohort_file),
            "fingerprint": next(iter(dataset_fingerprints)),
        },
        "replicates": entries,
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    parser.add_argument(
        "--condition-root",
        type=Path,
        help="DSDE condition-slice root used to recover registered local projections",
    )
    parser.add_argument(
        "--srtm-root",
        type=Path,
        help="SRTM HGT root for leakage-safe Arm 17 terrain lookup",
    )
    parser.add_argument(
        "--strict-environment",
        action="store_true",
        help="fail unless Python and package versions match environment.lock.json",
    )
    args = parser.parse_args(argv)
    manifest = run_replicates(
        args.cohort,
        args.output,
        args.seeds,
        spec_path=args.spec,
        n_samples=args.samples,
        strict_environment=args.strict_environment,
        condition_root=args.condition_root,
        srtm_root=args.srtm_root,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["MultiSeedError", "run_replicates", "validate_replicate_seeds"]
