"""Build a paired-seed contrast between two NEX326 phase-space manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from experiments.nex326_phase_space_multi_seed import write_phase_space_contrast


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    contrast = write_phase_space_contrast(
        args.baseline, args.candidate, args.output
    )
    print(
        json.dumps(
            {
                "baseline": contrast["baseline_benchmark_id"],
                "candidate": contrast["candidate_benchmark_id"],
                "conclusion": contrast["conclusion"],
                "delta_summary": contrast["delta_summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
