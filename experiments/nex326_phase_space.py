"""Run the supplemental NEX326 four-dimensional phase-space benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from experiments.nex326.phase_space import SPEC_PATH, write_phase_space_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    report = write_phase_space_report(
        args.cohort,
        args.output,
        spec_path=args.spec,
        n_samples=args.samples,
        seed=args.seed,
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
