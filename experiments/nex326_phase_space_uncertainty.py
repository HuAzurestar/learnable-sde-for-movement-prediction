"""Bootstrap paired NEX326 phase-space evaluation segments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from experiments.nex326_phase_space_multi_seed import (
    UNCERTAINTY_PROTOCOL_PATH,
    write_phase_space_segment_bootstrap,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--contrast", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=UNCERTAINTY_PROTOCOL_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = write_phase_space_segment_bootstrap(
        args.baseline,
        args.candidate,
        args.contrast,
        args.output,
        protocol_path=args.protocol,
    )
    print(
        json.dumps(
            {
                "analysis_id": result["analysis_id"],
                "evaluation_segment_count": result["evaluation_segment_count"],
                "uncertainty": result["uncertainty"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
