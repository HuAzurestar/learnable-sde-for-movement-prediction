"""Load a checkpoint and generate a forecast through the configured engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from application.experiment import ExperimentApplication
from config import Config
from domain import ForecastRequest, ModelContext
from infrastructure import JsonArtifactStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="learnable-sde predict")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--x0", type=float, nargs="+", required=True)
    parser.add_argument("--horizons", type=float, nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--regime", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    config = Config.from_yaml(args.config)
    app = ExperimentApplication.from_config(config)
    metadata = app.load_checkpoint(args.checkpoint)
    forecast = app.forecast(
        ForecastRequest(
            initial_state=torch.tensor(args.x0, dtype=app.runtime.dtype),
            horizons=torch.tensor(args.horizons, dtype=app.runtime.dtype),
            n_samples=args.samples,
            context=ModelContext(regime=args.regime),
        )
    )
    report = {
        "command": "predict",
        "engine": forecast.metadata.get("engine"),
        "sample_shape": list(forecast.samples.shape),
        "final_mean": forecast.mean[-1].tolist() if forecast.mean is not None else None,
        "final_covariance": forecast.covariance.tolist() if forecast.covariance is not None else None,
        "checkpoint_metadata": dict(metadata),
    }
    if args.output is not None:
        JsonArtifactStore().write(
            {**report, "samples": forecast.samples.tolist()},
            args.output,
        )
        report["output"] = str(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
