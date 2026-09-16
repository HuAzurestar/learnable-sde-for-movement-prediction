"""CLI for rebuilding the complete NEX326 22-arm process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.nex326.runner import NEX326Runner
from experiments.nex326.specification import SPEC_PATH, load_experiment_spec


ROOT = Path(__file__).resolve().parent
DEFAULT_FIXTURE = ROOT / "nex326" / "fixtures" / "registered_cohort.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument(
        "--implementation-fixture",
        action="store_true",
        help="explicitly run the registered synthetic implementation fixture",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
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
        "--pirc20-trajectory",
        type=Path,
        help="hash-bound trajectory parquet used by a PIRC-20 release",
    )
    parser.add_argument(
        "--pirc20-condition-root",
        type=Path,
        help="condition-slice root bound by the PIRC-20 file manifest",
    )
    parser.add_argument(
        "--pirc20-geolife-conditions",
        type=Path,
        help="hash-bound receipt.json for reconstructed GeoLife solar elevation",
    )
    parser.add_argument(
        "--final-eval-unlock",
        help="exact PIRC-20 cohort ID acknowledgement required to read final_eval",
    )
    parser.add_argument(
        "--scope-policy",
        type=Path,
        help="approved exclusion policy; runs only its required execution slots",
    )
    parser.add_argument(
        "--replicate-seed",
        type=int,
        help="prediction/training replicate seed; defaults to the frozen protocol seed",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--strict-environment",
        action="store_true",
        help="fail unless Python and package versions match environment.lock.json",
    )
    args = parser.parse_args(argv)

    spec = load_experiment_spec(args.spec)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "experiment_id": spec.experiment_id,
                    "spec_version": spec.spec_version,
                    "numbered_arms": len(spec.arms),
                    "executions": len(spec.executions),
                    "status": "valid",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.cohort is not None and args.implementation_fixture:
        parser.error("choose --cohort or --implementation-fixture, not both")
    if args.cohort is None and not args.implementation_fixture:
        parser.error(
            "a scientific --cohort is required; use --implementation-fixture only for contract validation"
        )
    if (args.condition_root is None) != (args.srtm_root is None):
        parser.error("--condition-root and --srtm-root must be supplied together")
    cohort_path = DEFAULT_FIXTURE if args.implementation_fixture else args.cohort
    cohort_header = json.loads(Path(cohort_path).read_text(encoding="utf-8"))
    if cohort_header.get("schema_version") == "pirc20-cohort-v1":
        if args.pirc20_trajectory is None:
            parser.error("a PIRC-20 cohort requires --pirc20-trajectory")
        from experiments.nex326.pirc20_runtime import PIRC20NEX326Runner

        release_header = json.loads(
            (Path(cohort_path).parent / "dataset.json").read_text(encoding="utf-8")
        )
        if release_header.get("schema_version") == "pirc20-geolife-release-v1":
            if args.pirc20_condition_root is not None:
                parser.error("GeoLife confirmation does not accept --pirc20-condition-root")
            if args.pirc20_geolife_conditions is None:
                parser.error("GeoLife confirmation requires --pirc20-geolife-conditions")
            from experiments.nex326.geolife_confirmation_adapter import (
                load_geolife_confirmation_cohort,
            )

            cohort = load_geolife_confirmation_cohort(
                cohort_path,
                args.pirc20_trajectory,
                args.pirc20_geolife_conditions,
                final_eval_unlock=args.final_eval_unlock,
            )
        else:
            if args.pirc20_geolife_conditions is not None:
                parser.error("GeoLife conditions require a GeoLife confirmation cohort")
            if args.pirc20_condition_root is None:
                parser.error("OSM-derived PIRC-20 requires --pirc20-condition-root")
            from experiments.nex326.pirc20_adapter import load_pirc20_nex326_cohort

            cohort = load_pirc20_nex326_cohort(
                cohort_path,
                args.pirc20_trajectory,
                args.pirc20_condition_root,
                final_eval_unlock=args.final_eval_unlock,
            )
        condition_resolver = None
        if args.condition_root is not None and args.srtm_root is not None:
            from experiments.nex326.spatial_conditions import DSDERasterConditionResolver

            condition_resolver = DSDERasterConditionResolver(
                cohort, args.condition_root, args.srtm_root
            )
        runner = PIRC20NEX326Runner(
            spec,
            cohort,
            args.output,
            n_samples=args.samples,
            replicate_seed=args.replicate_seed,
            strict_environment=args.strict_environment,
            condition_resolver=condition_resolver,
        )
    else:
        if (
            args.pirc20_trajectory is not None
            or args.pirc20_condition_root is not None
            or args.pirc20_geolife_conditions is not None
            or args.final_eval_unlock is not None
        ):
            parser.error("PIRC-20 source options require a PIRC-20 cohort")
        runner = NEX326Runner.from_paths(
            cohort_path,
            args.output,
            spec_path=args.spec,
            n_samples=args.samples,
            replicate_seed=args.replicate_seed,
            strict_environment=args.strict_environment,
            condition_root=args.condition_root,
            srtm_root=args.srtm_root,
        )
    if args.scope_policy is None:
        records = runner.run_all()
    else:
        from experiments.nex326.scoped_runner import run_approved_scope

        records = run_approved_scope(runner, args.scope_policy)
    print(
        json.dumps(
            {
                "experiment_id": spec.experiment_id,
                "spec_version": spec.spec_version,
                "record_count": len(records),
                "arm_count": len({record["arm_id"] for record in records}),
                "protocol_seed": spec.seed,
                "replicate_seed": runner.replicate_seed,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
