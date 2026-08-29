# Learnable SDE for Movement Prediction

An object-oriented research framework for training, evaluating, and running
learnable stochastic differential equation models.

The repository separates domain objects, model capabilities, estimators,
inference engines, evaluation rules, application services, and infrastructure
adapters. It includes a deterministic synthetic workflow, so installation and
the public test suite do not require the private trajectory dataset.

> Project status: research software. The segment-constant model, EM estimator,
> exact Gaussian inference, split-step inference, CRN inference, checkpoint
> round trip, and synthetic CLI workflow are implemented and tested. Components
> explicitly described as experimental or incomplete fail fast instead of
> silently falling back to another algorithm.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

## Quick start

Train a small model on deterministic synthetic data:

```bash
learnable-sde train --config config.yaml --smoke \
  --checkpoint .local/checkpoints/smoke.pt
```

Generate a forecast from that checkpoint:

```bash
learnable-sde predict --config config.yaml \
  --checkpoint .local/checkpoints/smoke.pt \
  --x0 0 0 --horizons 60 120 --samples 100 --regime 0 \
  --output .local/outputs/forecast.json
```

The module form is equivalent when the console script is not installed:

```bash
python -m cli train --config config.yaml --smoke
python -m cli ablate --matrix
python -m cli ablate --verify
```

## Using local data

No research data, derived trajectory samples, or trained checkpoints are
included. Configure local paths with environment variables:

```bash
LEARNABLE_SDE_DATA_ROOT=/path/to/trajectory-data
LEARNABLE_SDE_COND_ROOT=/path/to/condition-data
LEARNABLE_SDE_CHECKPOINT=/path/to/checkpoint-directory
```

On PowerShell:

```powershell
$env:LEARNABLE_SDE_DATA_ROOT = "D:\datasets\trajectory-data"
```

When no environment variable is set, local-only paths under `.local/` are
used. That directory is ignored by Git. See [DATA.md](DATA.md) for the expected
schema, split contract, and data-release boundary.

## Architecture

The public execution path is:

```text
CLI -> ExperimentApplication -> registry -> model / estimator / inference
                                      -> checkpoint and artifact adapters
Forecast -> Evaluator -> scoring rules
```

Core interfaces:

- `SDEModel`: a standard `torch.nn.Module` with explicit model context.
- `Estimator.fit(model, data, context) -> FitResult`.
- `InferenceEngine.forecast(model, request, context) -> Forecast`.
- `Evaluator`: applies one canonical scoring-rule implementation.
- `ExperimentApplication`: owns component assembly and use-case orchestration.

See [DESIGN.md](DESIGN.md) for dependency rules, capabilities, interfaces, and
the remaining migration boundary.

## Repository layout

```text
application/      use-case orchestration and runtime ownership
cli/              train, predict, and ablate commands
data/             data-source interfaces, loaders, paths, and validation
domain/           shared types, results, requests, and errors
estimation/       estimator interface and implementations
evaluation/       evaluators and scoring rules
inference/        inference interface and implementations
infrastructure/   checkpoint and artifact adapters
models/           SDE model interface and implementations
tests/            unit, characterization, and integration tests
```

## Development

```bash
python -m pytest -q
python -m experiments.smoke_test
```

Contribution workflow, commit message format, and pull-request requirements are
documented in [CONTRIBUTING.md](CONTRIBUTING.md). Known numerical and migration
limitations are tracked in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

For the purpose, checks, triggers, and release assets of GitHub Actions, see
[CI_CD.md](CI_CD.md).

## Related formalization

The Lean proofs are maintained in
[learnable-sde-theory-to-predict-movement](https://github.com/HuAzurestar/learnable-sde-theory-to-predict-movement).
Keeping the formal development separate gives it an independent toolchain, CI
workflow, and release history.

## Citation and license

Citation metadata is provided in [CITATION.cff](CITATION.cff). No software
license has been granted yet. Until the rights holder adds a `LICENSE` file,
the source is available for inspection but no permission to copy, modify, or
redistribute it is implied. See [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).
