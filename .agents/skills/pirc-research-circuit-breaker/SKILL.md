---
name: pirc-research-circuit-breaker
description: Register, run, resume, or inspect shared SDE research jobs through the local budget and exposure ledger. Use for research execution in this repository, not for ordinary source edits or synthetic unit tests.
---

# Shared research execution

Read `docs/pirc-38-shared-engineering.md` for the implemented commands and plugin
interfaces. The compatibility identifier is `pirc25-contract-v1`; the shared
store lives at `<runtime_root>/pirc25/`. Do not create a new store or rename an
arm to obtain fresh budget. Every participating worktree uses the same explicit
absolute root outside Git and the existing store ID.

Before execution, bind the immutable study spec, code/data/protocol hashes,
registered cells, required plugin capabilities and the user's authorized data
scope. Missing prerequisites stop that run. A synthetic fixture grant does not
authorize a real dataset, pilot, formal test, or research extension. Read real
blocks only through `EvaluationExposureLedger`; preserve prior exposures and
do not describe previously exposed or unknown-history blocks as blind tests.

Use `SharedRunner`/`ResearchSupervisor`, including for child-method plugins.
Do not launch unmetered workers beside them. Smoke, pilot and job caps are
900/1800/7200 seconds; each registered model/method/objective arm shares 86400
slot-seconds across seeds, horizons, retries, resumes and resource classes.
The durable queue fixes local resource slots and rotates between arms. CPU
work is not free. The watchdog deadline has no extra termination grace period.

After timeout, unknown worker cost, corrupt history or an exhausted arm, stop
that arm. Preserve failed attempts, reservations, logs and partial evidence.
First prove an old worker tree stopped before explicit unknown-cost recovery;
that recovery charges the full reservation and keeps the arm closed. Do not
edit authoritative files or infer budget from SQLite. Quarantining a corrupt
tail preserves its bytes and holds all budgets for manual reconciliation.
Budget extensions require a new explicit user-approved addendum, not retries.

For recovery, inspect the plugin's declared exact/chunk/restart-only level.
Validate checkpoint content and all compatibility bindings; checkpoint state
never rolls back budget. Use a new attempt linked to its parent. At most two
explicit retries are supported; scientific failure, OOM or nonfinite output
does not authorize configuration changes or an automatic retry.

Export the full registered matrix, including failed/missing cells, through the
authorized evidence interface. TSDE aggregates independent blocks and returns
immutable JSON/CSV/evidence-index hashes. Preserve those identities in reports;
multiple seeds on one block do not create independent scientific replication.
The read-only UI cannot authorize training or broaden disclosure permission.
