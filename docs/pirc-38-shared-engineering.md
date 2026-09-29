# PIRC-38 shared engineering (draft PR)

This PR is the integration branch for PIRC-38. It is opened before implementation so the shared runtime work can converge through one reviewed PR. This document is a scope marker, not evidence that the execution gate or acceptance has passed.

## Scope

- W01-W04: audit reusable PIRC-19–22 artifacts, then add durable experiment identity, query history, shared budget/process supervision, and the existing affine-runner adapter.
- Shared W06-W09: declared recovery contracts, comparison/evidence records, read-only results UI, and a reproducible engineering handoff.
- Use fixture plugins for shared V07/V12 onboarding checks. PIRC-26/27/28 own method-specific algorithms, recovery state, scientific evidence, and papers.

## Convergence checklist

- [ ] PIRC-38 GATE-START passes with current refs, permitted data scope, and upstream artifact identities recorded in MPA.
- [ ] W01-W04 implementation and tests pass against a pinned commit.
- [ ] Shared W06-W09 implementation and fixture-based V06-V12 checks pass against a pinned commit.
- [ ] Clean-process query/recovery and read-only result trace are reproducible without protected final-evaluation data.
- [ ] Review and required checks pass; the PR is merged before PIRC-26/27/28 begin their concurrent implementation gates.

The authoritative decisions and task dependencies live in MPA `project/PIRC-38`. Do not treat this draft PR's creation as issue completion.
