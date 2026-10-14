# PIRC-17 simple checkpoint continuation

Use `checkpoint_resume`, **not** the legacy `formal_entrypoint run` restoration
pipeline, for the saved partial experiment. The user authorized this replacement
on 2026-10-04. Scientific settings, fitted parameters and existing outputs do
not change. Original ledgers remain read-only.

The old partial/resource `formal_entrypoint run` is now retired in code, not
just discouraged here. It returns a checkpoint-resume instruction before any
launch claim, full restoration, historical budget replay or controller is
created. Old readers remain available for archived evidence. Resume never
requires another prepare/test/review/approval chain or a full result audit.

Once a checkpoint has been initialized:

```powershell
python -m experiments.pirc17.checkpoint_resume status --directory <checkpoint-directory>
python -m experiments.pirc17.checkpoint_resume run --directory <checkpoint-directory> --maximum-items 20
python -m experiments.pirc17.checkpoint_resume stop --directory <checkpoint-directory>
```

Omit `--maximum-items` to continue runnable forecasts within the retained
cumulative phase/total budgets. `stop` interrupts the current owned worker;
previously committed results stay complete. No other process is terminated.

The user explicitly superseded time-budget stops on 2026-10-04:

```powershell
python -u -m experiments.pirc17.checkpoint_resume run --directory <checkpoint-directory> --ignore-time-budgets --report-seconds 600
```

This disables startup/per-item/phase/total elapsed-time cutoffs, without
resetting cost records, adding experiments, changing scientific settings or
silently rerunning successes/failures. Explicit stop, dead-process detection
and kill-on-supervisor-death ownership remain. Fixed machine-wide RAM/disk
thresholds are advisory ONLY: observe at startup and scheduled reports, never
abort a running forecast for a transient reading or poll RAM every 50ms.
Actual allocation or write failures still surface normally and retain the last
checkpoint. Reports separate successful, failed and runnable science, so terminal
interruptions do not receive a fictitious remaining-compute ETA. Reports give actual counts and
rolling measured estimates separately for methods and terrain; a phase with
no measured samples has no fabricated ETA.

Restart reads one completion index and one progress snapshot, reconstructs the
saved causal context and 26 fitted models once, and starts missing items. It
does not regenerate old result wrappers, reopen every saved prediction, scan
snapshot Parquet files or run a second parent-side domain admission. The online
encoder checks only its two metadata files; it uses the original transformation
kernels and cannot load observed snapshot rows. Actual online map assets remain
lazy and checked when queried.

Progress is replaced atomically after each result. Only the interrupted item's
atomic reply may need reconciliation. A partial output is **not** a successful
checkpoint. Failed/interrupted work and generated-attempt debits remain visible;
there is no silent retry, successful rerun or reset of consumed time. A dead
writer marker is reclaimed immediately using PID plus creation time. A live
duplicate invocation is rejected immediately, never waited on. There are no
profiling hooks, monitor threads or blocking restart locks.

Worker requests are immutable `sessions/<uuid>/requests/<sequence>.json`, read
once each. The supervisor never replaces a request still being read by the
worker. Atomic progress/report replacements tolerate Windows read-sharing
errors with at most 200ms of IO retries; permanent permission errors surface
normally and preserve the prior checkpoint. No unbounded lock wait is added.

A single retained Windows Job provides stop/kill-on-supervisor-death protection.
This is a process-tree kill switch, not a restoration/admission gate. Without
the explicit time-budget override, startup is bounded (default 120 seconds)
and each forecast keeps its original per-item limit. With the override, none
of those elapsed-time limits stops prediction. Costs are still recorded.

`init` is a **one-time** metadata import, not a command to repeat at restart. It
requires the saved bundle, already produced import manifest, and an explicit
charge/closure note for the stopped deadlocked run. It cannot overwrite an
existing checkpoint. The new orchestration sources are recorded separately;
old sealed scientific records are not rewritten or falsely declared to have
passed the retired full bootstrap gate.

Current runner scope: scientific forecasts, same-grid references, inertial
paths, forecast replays and runtime cold/warm trials. Common scoring, mechanism
tests, final reanalysis/export/statistics and manuscript/final human acceptance
remain downstream work; this runner is not a declaration that PIRC-17 is done.

## Incremental offline scoring

Run the separate private scorer while predictions continue:

```powershell
python -u -m experiments.pirc17.checkpoint_scoring score --directory <checkpoint-directory> --follow --report-seconds 600
```

It loads saved targets/models once, then refreshes only the progress index.
Only terminal forecasts are scored; unfinished ones are not frozen as missing.
Original metrics, four targets, time weights and full denominators are unchanged.
Rows are cached under `offline-scores/<code-and-input-identity>/rows/`; restart
reuses them without recomputing their arrays. Private targets/coordinates/cache
files must not be exported as public aggregates. No new fits, forecasts or map
queries are performed. Set numerical-library thread counts to one in the launch
environment when running beside the prediction worker.

The fixed scope is 11368 score rows and 58 complete common-score blocks of
196 dependencies each. A block is published only after all dependencies are
terminal and all row caches are present, including failures/capacity dispositions.
After all blocks exist, `checkpoint_scoring analyze --directory ...` uses the
original mechanism/paired-statistic kernels. Its new checkpoint metric receipts
explicitly do NOT pretend the retired legacy admission gate passed. Cache reuse
is NOT independent raw-output replay; final reanalysis/runtime/public export,
evidence cards, manuscript/PDF, review and human acceptance remain required.

For a lower-overhead follower of the SAME cached rows, use the separate
`checkpoint_score_follow` scheduler instead of launching a second scorer:

```powershell
python -u -m experiments.pirc17.checkpoint_score_follow --directory <checkpoint-directory> --follow --report-seconds 600
```

Only one scorer may own the score cache at a time. When replacing the old
follower, stop only that identified scoring process; never the prediction
runner. Its completed atomic rows/blocks survive pickup. The four original
cache-bound source files and cache identity are unchanged. Pickup checks row
bindings/files once; subsequent polls compare only in-memory source bindings,
then score new/changed terminal results using the original arithmetic. Complete
blocks still contain all 196 original dependencies and validate original cached
row artifacts; missing rows are not treated as zero or frozen as unavailable.
No repeated old-row scope hashing/file checks occur while waiting. Polling is
10 seconds, not a promise of faster predictions; actual CPU/throughput must be
measured. This does not substitute for final independent saved-output replay.

For the versioned inference pointer and final independent replay:

```powershell
python -u -m experiments.pirc17.checkpoint_audit analysis --directory <checkpoint-directory>
python -u -m experiments.pirc17.checkpoint_audit audit --directory <checkpoint-directory> --report-seconds 600
```

Analysis requires all 58 complete score blocks. Final audit additionally requires
terminal dispositions for the full original forecast inventory; it never silently
shrinks the fixed 11659-work matrix. It reopens actual saved paths, recomputes each
original score once, and reuses only its OWN freshly computed rows for mechanism
and paired-statistic replay. Producer cache integrity is not numeric proof. The
audit also checks the saved runtime/replay evidence, original model owners and
full disposition inventory. Interruptions remain failures rather than invented
successful predictions. This is final scientific checking, not renewed input
qualification/full-ledger restoration or a forecast-restart prerequisite. Public
aggregate export, cards, complete manuscript/PDF, review and actual final human
acceptance are still separate deliverables.

Focused regression command (synthetic data, including real Windows Job tests):

```powershell
python -m pytest tests/test_pirc17_checkpoint_resume.py tests/test_pirc17_paid_validation.py tests/test_pirc17_features.py -q
```

## Public aggregates after final audit

After all original predictions/runtime/replay dispositions, all 58 common-score
blocks, analysis and the independent fresh-output audit have completed:

```powershell
python -u -m experiments.pirc17.checkpoint_export --directory <checkpoint-directory>
```

This loads the existing saved context/models once, then reads final versioned
analysis/audit pointers and the 58 audited score files. It does not reopen
original trajectory/map assets or forecast-array files, or recompute metrics.
It keeps all 11368 score rows, the full
11659-work disposition inventory, 36 method slots (including eight exemptions),
ten terrain configurations and original uncertainty/verdict rules. Original
origin/block identifiers become sorted ordinal labels. Nested metric projection
drops coordinates/targets/model parameters/host paths; failure reasons are hashed
rather than copying free-form private errors. Public bootstrap/Holm/verdict
arithmetic must reproduce the audited analysis exactly. An uncompleted/cached-only
audit or changed source index cannot be silently published as complete.

The new private `export.json` pointer names an immutable public aggregate under
`public-aggregates/`. No remote upload/PR, claim authorization or human acceptance
is performed. The standalone public replay command remains compatible:

```powershell
python -m experiments.pirc17.formal_export --input <public-json> --sha256 <pinned-content-sha256> --output-directory <replay-directory>
```

The runner-cost snapshot explicitly excludes separate scoring/analysis/audit
costs and export completion, so it must not be advertised as total project cost.
All this is final result delivery, not a prediction-restart prerequisite. Evidence
cards, complete editable manuscript/PDF, review and actual human acceptance still
remain after aggregate export.

## Local downstream review command

`checkpoint_review` connects the original completed-result producers and the
declared TSDE public artifact consumers. It never starts/stops/retries predictions
or changes the scientific settings. Generate the metadata-only evidence catalog
with `evidence_catalog` and retain its returned content SHA256, then run:

```powershell
python -u -m experiments.pirc17.checkpoint_review --directory <checkpoint-directory> --score-progress <existing-score-cache>/progress.json --catalog <public-catalog-json> --catalog-sha256 <catalog-content-sha256> --output-directory <local-review-output> --tsde-directory <TSDE-root> --follow --report-seconds 600
```

Optional `--follow` observes only prediction/scoring indexes at 60-second
intervals, loading metadata once; it holds no writer marker and reads no maps,
model arrays or old prediction arrays while waiting. It requires all original
forecast kinds including runtime/replay, and all 11368 rows/58 score blocks.
Interruptions remain terminal failures, not successful predictions. Without
`--follow`, unfinished dependencies fail immediately without affecting producers.

After dependencies finish, the command runs original analysis, one fresh final
saved-output audit, public export, review cards, sixteen CSVs and sixteen PNG/PDF
figures. Existing analysis/audit pointers are reused, not automatically replayed;
the original export validates the reused audit against current sources. Stale or
corrupt artifacts stop this downstream command instead of retrying experiments
or replacing evidence. Rendering reuses its ordinary missing-only outputs.
No upload, TEST-02 qualification, claim permission, complete manuscript/PDF or
human acceptance is granted. `--software-fixture` labels synthetic figures only.
