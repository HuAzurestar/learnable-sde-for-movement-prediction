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
    records = runner.run_all()
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
