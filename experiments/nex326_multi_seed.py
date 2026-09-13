"""Run a registered NEX326 cohort under multiple explicit replicate seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from experiments.nex326.runner import NEX326Runner
from experiments.nex326.scoped_runner import run_approved_scope
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
    pirc20_trajectory: Path | str | None = None,
    pirc20_condition_root: Path | str | None = None,
    pirc20_geolife_conditions: Path | str | None = None,
    final_eval_unlock: str | None = None,
    scope_policy: Path | str | None = None,
) -> dict[str, object]:
    """Run replicate batches and write one hash-bound batch manifest."""
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
    runner_type = NEX326Runner
    cohort_header = json.loads(cohort_file.read_text(encoding="utf-8"))
    if cohort_header.get("schema_version") == "pirc20-cohort-v1":
        if pirc20_trajectory is None:
            raise MultiSeedError("a PIRC-20 cohort requires its trajectory source")
        from experiments.nex326.pirc20_runtime import PIRC20NEX326Runner

        runner_type = PIRC20NEX326Runner
        release_header_path = cohort_file.parent / "dataset.json"
        release_header = json.loads(release_header_path.read_text(encoding="utf-8"))
        if release_header.get("schema_version") == "pirc20-geolife-release-v1":
            if pirc20_condition_root is not None:
                raise MultiSeedError(
                    "a GeoLife confirmation cohort does not accept a condition root"
                )
            if pirc20_geolife_conditions is None:
                raise MultiSeedError(
                    "a GeoLife confirmation cohort requires its solar condition receipt"
                )
            from experiments.nex326.geolife_confirmation_adapter import (
                load_geolife_confirmation_cohort,
            )

            shared_cohort = load_geolife_confirmation_cohort(
                cohort_file,
                pirc20_trajectory,
                pirc20_geolife_conditions,
                final_eval_unlock=final_eval_unlock,
            )
        else:
            if pirc20_geolife_conditions is not None:
                raise MultiSeedError(
                    "GeoLife conditions require a GeoLife confirmation cohort"
                )
            if pirc20_condition_root is None:
                raise MultiSeedError(
                    "an OSM-derived PIRC-20 cohort requires its condition root"
                )
            from experiments.nex326.pirc20_adapter import load_pirc20_nex326_cohort

            shared_cohort = load_pirc20_nex326_cohort(
                cohort_file,
                pirc20_trajectory,
                pirc20_condition_root,
                final_eval_unlock=final_eval_unlock,
            )
        shared_resolver = None
        if condition_root is not None and srtm_root is not None:
            from experiments.nex326.spatial_conditions import DSDERasterConditionResolver

            shared_resolver = DSDERasterConditionResolver(
                shared_cohort, condition_root, srtm_root
            )
    else:
        if (
            pirc20_trajectory is not None
            or pirc20_condition_root is not None
            or pirc20_geolife_conditions is not None
            or final_eval_unlock is not None
        ):
            raise MultiSeedError("PIRC-20 source options require a PIRC-20 cohort")
        shared = NEX326Runner.from_paths(
            cohort_file,
            destination / ".cohort-validation",
            spec_path=spec_path,
            n_samples=n_samples,
            replicate_seed=replicate_seeds[0],
            strict_environment=strict_environment,
            condition_root=condition_root,
            srtm_root=srtm_root,
        )
        shared_cohort = shared.cohort
        shared_resolver = shared.condition_resolver
    entries: list[dict[str, object]] = []
    dataset_fingerprints: set[str] = set()
    implementation_identities: dict[str, dict[str, object]] = {}
    for seed in replicate_seeds:
        relative_root = Path(f"seed-{seed}")
        records_root = destination / relative_root
        runner = runner_type(
            spec,
            shared_cohort,
            records_root,
            n_samples=n_samples,
            replicate_seed=seed,
            strict_environment=strict_environment,
            condition_resolver=shared_resolver,
        )
        records = (
            run_approved_scope(runner, scope_policy)
            if scope_policy is not None
            else runner.run_all()
        )
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
    executions_per_replicate = {int(entry["record_count"]) for entry in entries}
    if len(executions_per_replicate) != 1:
        raise MultiSeedError("replicates did not execute one identical slot count")
    execution_count = next(iter(executions_per_replicate))
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
        "executions_per_replicate": execution_count,
        "total_executions": execution_count * len(replicate_seeds),
        "cohort": {
            "source_file": cohort_file.name,
            "sha256": _sha256(cohort_file),
            "fingerprint": next(iter(dataset_fingerprints)),
        },
        "replicates": entries,
    }
    if scope_policy is not None:
        policy_file = Path(scope_policy).resolve()
        payload["scope_policy"] = {
            "source_file": policy_file.name,
            "sha256": _sha256(policy_file),
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
    parser.add_argument("--pirc20-trajectory", type=Path)
    parser.add_argument(
        "--pirc20-condition-root",
        type=Path,
        help="condition root required by OSM-derived PIRC-20 releases; omit for GeoLife",
    )
    parser.add_argument(
        "--pirc20-geolife-conditions",
        type=Path,
        help="hash-bound receipt.json for reconstructed GeoLife solar elevation",
    )
    parser.add_argument(
        "--scope-policy",
        type=Path,
        help="approved exclusion policy applied identically to every replicate",
    )
    parser.add_argument(
        "--final-eval-unlock",
        help="exact PIRC-20 cohort ID acknowledgement required to read final_eval",
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
        pirc20_trajectory=args.pirc20_trajectory,
        pirc20_condition_root=args.pirc20_condition_root,
        pirc20_geolife_conditions=args.pirc20_geolife_conditions,
        final_eval_unlock=args.final_eval_unlock,
        scope_policy=args.scope_policy,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["MultiSeedError", "run_replicates", "validate_replicate_seeds"]
