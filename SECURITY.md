# Security policy

## Supported versions

Security fixes are applied to the current `main` branch. Research snapshots and
unreleased experiment branches are not supported.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting feature for issues involving code
execution, unsafe checkpoint loading, dependency compromise, or accidental data
exposure. Do not open a public issue containing a proof of concept, private
dataset location, credential, or personal information.

For non-sensitive correctness bugs, open a normal GitHub issue with the smallest
synthetic reproducer possible.

## Checkpoint warning

PyTorch checkpoints are serialized artifacts and must be treated as untrusted
input. Only load checkpoints obtained from a trusted source and verify their
published checksum. This repository does not distribute trained checkpoints.
