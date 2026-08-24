# Changelog

All notable public changes are recorded here. The format follows Keep a
Changelog, and releases use semantic versioning while the public API stabilizes.

## [Unreleased]

### Changed

- Replaced workstation-specific dataset defaults with ignored `.local/` paths
  and `LEARNABLE_SDE_*` environment-variable overrides.
- Added public release, security, data, contribution, citation, and CI metadata.

### Removed

- Removed a machine-specific legacy experiment runner and generated verification
  report from the public source tree.

## [0.2.0] - 2026-08-24

### Added

- `SDEModel` as a standard `torch.nn.Module` with explicit model context and
  capability interfaces.
- Generic estimator, inference engine, evaluator, registry, checkpoint adapter,
  and experiment application contracts.
- Train, predict, and ablation CLI commands.
- Deterministic synthetic train/checkpoint/predict workflow.
- Architecture, application, data validation, and characterization tests.

### Changed

- Migrated the segment-constant model and EM estimator to the public OOP
  contracts while preserving the established numerical gates.
