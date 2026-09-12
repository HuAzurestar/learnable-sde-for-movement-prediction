"""Run the supplemental NEX326 four-dimensional phase-space benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from experiments.nex326.cohort import load_cohort
from experiments.nex326.phase_space import SPEC_PATH, write_phase_space_report
from experiments.nex326.spatial_conditions import DSDERasterConditionResolver


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--condition-root", type=Path)
    parser.add_argument("--srtm-root", type=Path)
    args = parser.parse_args(argv)
    if (args.condition_root is None) != (args.srtm_root is None):
        parser.error("--condition-root and --srtm-root must be supplied together")
    resolver = (
        DSDERasterConditionResolver(
            load_cohort(args.cohort), args.condition_root, args.srtm_root
        )
        if args.condition_root is not None
        else None
    )
    report = write_phase_space_report(
        args.cohort,
        args.output,
        spec_path=args.spec,
        n_samples=args.samples,
        seed=args.seed,
        condition_resolver=resolver,
    )
    print(
        json.dumps(
            {
                "benchmark_id": report["benchmark_id"],
                "scientific_role": report["scientific_role"],
                "dataset": report["dataset"]["dataset_id"],
                "metrics": report["metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
