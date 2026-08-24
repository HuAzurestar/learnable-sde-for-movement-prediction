"""Train a configured model through the application service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from application.experiment import ExperimentApplication
from application.synthetic import make_synthetic_em_data
from config import Config
from data.source import TrajectorySource
from data.loader import to_phase_space_1d
from estimation.em import SegmentEMData


def _real_data(split: str, max_segments: int | None, seed: int) -> SegmentEMData:
    segments = TrajectorySource(
        split=split,
        max_segments=max_segments,
        seed=seed,
    ).load()
    return SegmentEMData(
        tuple(to_phase_space_1d(segment) for segment in segments),
        tuple(float(segment.dt) for segment in segments),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="learnable-sde train")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--smoke", action="store_true", help="使用确定性合成双模式数据")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-segments", type=int)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)

    config = Config.from_yaml(args.config)
    app = ExperimentApplication.from_config(config)
    if args.smoke:
        data, _ = make_synthetic_em_data(seed=config.seed)
    else:
        data = _real_data(args.split, args.max_segments, config.seed)
    run = app.train(data)
    if args.checkpoint is not None:
        app.save_checkpoint(args.checkpoint, {"fit": run.fit.to_dict()})
    report = {
        "command": "train",
        "model": type(run.model).__name__,
        "fit": run.fit.to_dict(),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
