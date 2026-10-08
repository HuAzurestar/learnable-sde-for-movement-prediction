# PIRC-17 review snapshot — not final scientific acceptance

This public review snapshot projects the existing implementation at
`40855e678a0802947bc590cbe24584b85bf72e49` onto public `main`
`00098969df964d3f1799bc1d4278bf919c2ff9a8`. It retains the public base's
independent engineering work. The live experiment branch is unchanged.

The snapshot intentionally does not include private implementation history,
raw trajectories, row-level prediction arrays, fitted models, maps, logs or
workstation mounts. Historical cost-probe path fields are redacted to a
relative placeholder; their historic hashes remain historic, **not newly
validated executable-plan bindings**. Recovery-path defaults are local
placeholders configurable through `PIRC17_DATA_REPOSITORY`,
`PIRC17_FEATURE_PRODUCER` and `PIRC17_RECOVERY_ROOT`. None of these tools was
executed for this review. Do not use this snapshot to restart the live study.

## Existing experiment status

Metadata observation: 2026-10-08 10:22 Hong Kong time. All 11,020 original
scientific workloads have terminal dispositions: 11,015 success and 5 failure,
with no runnable scientific forecasts. The existing common scorer completed
11,368 rows across 58 groups. Original auxiliary workloads: 169/261 success,
92 remaining, no auxiliary failures at this observation. Runtime/replay
processing continues independently of this PR.

The five failures are retained. Four affected comparison families must remain
unavailable; selecting only their successful subset is not an accepted effect
estimate. Final analysis, the single independent saved-output audit, aggregate
cards/tables/figures and final manuscript integration remain pending.

## Review limits

This PR is for source and existing aggregate evidence review. Software tests
do not establish scientific validity. No new predictions, fits, scores,
resampling, experiments or map queries were launched for the publication.
No acceptance, authorship verification, public route permission, deployment,
merge or finished-paper release is implied.

## Test environment

The test extra now declares the resource-sampling and document/figure packages
used by existing tests. CI still runs the original complete pytest command;
no missing test or evidence is treated as a pass. Dependency repair is not
a replay of the scientific study, and no live runtime source is changed.
