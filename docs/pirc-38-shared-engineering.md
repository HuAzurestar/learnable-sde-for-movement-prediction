# Shared engineering runtime and evidence

PR #22 integrates the shared runtime. The contract remains `pirc25-contract-v1`;
changing the implementing issue does not create a new ledger or reset a research
arm. Child packages retain ownership of scientific algorithms and qualification.

`show` is a registration-metadata command restricted to `study-*`, `run-*` and
`artifact-*` manifests. It never displays frozen bundles or arbitrary manifest
kinds. Use `export --authorization-id ...` or the authorized read-only service
to disclose results; an earlier export does not authorize a later disclosure.

Budget refusal is recorded atomically before returning an error. If other
attempts temporarily reserve the available capacity, the rejected attempt ends
as `PREFLIGHT_FAILED` with retryable `BUDGET_BUSY`; no worker starts and no other
reservation is released. After settlement, an explicit child attempt can retry
under the existing retry limit. A closed arm or insufficient remaining lifetime
budget yields terminal `BUDGET_EXHAUSTED`, not an untracked active attempt.

### Frozen test-read evidence

`test` and `final-eval` protocol reads require both `preregistration_hash` and
`history_hash`, resolved from immutable manifests. `history_status="known"`
alone is never evidence. Use `PreregistrationGate.register_preregistration`
and `import_history` with the expected canonical content hashes before reading.
The Python interfaces live in `application.research_preregistration`.

A `pirc25-preregistration-v1` document contains `study_ids`,
`protocol_bindings`, `primary_metrics`, `selection_rule`, `stopping_rule`,
`comparisons`, and an explicit `test_mode` (`blind` or `exploratory`). Compute
each protocol binding with `protocol_binding(protocol)`: it hashes the protocol
except its evidence pointers and obsolete history-status hint, avoiding circular
hashes while binding all data identities and semantic fields. A joint plan can
freeze multiple study/protocol bindings before any included data is exposed.

A `pirc25-exposure-history-v1` document records its `source`, the imported
`source_evidence` object, and `source_evidence_hash`. Its `records` bind
dataset/release/source-block/content SHA to `unexposed`, `exposed`, or `unknown`.
The importer is responsible for external source truth; the runtime verifies
content, scope and ledger consistency, not off-platform activity. Keep these
records outside Git. Missing, unknown or mismatched history refuses test reads.
Blind mode also rejects previous imported exposure and controlled reads before
the plan's authoritative publication sequence, even across renamed studies or
windows. Partial/failed reads count as possible exposure. Known exposed data can
be read under an explicitly exploratory plan and matching authorization, never
silently reclassified as blind. These plan and history hashes are recorded in
each read event. This read gate is separate from package qualification and does
not itself grant formal-comparison eligibility.

## Independent reproduction

From this repository, with the matching paper repository checked out:

```console
python -m experiments.pirc25.reproduce --tsde-root ../TSDE-SDE
```

The command creates a temporary root outside Git. Independent CLI processes
register a synthetic affine study, execute a cell, export an incomplete matrix,
generate a TSDE evidence package, retain a failed attempt, retry without resetting
its cost, import complete evidence, rebuild the index and reuse completed cells.
The receipt binds both Git HEADs, source/protocol/fixture/aggregate hashes and
command exit codes. Two seeds over the same block remain one independent block.
No private trajectory, protected evaluation data or actual research pilot is read.

Optional browser verification needs Playwright 1.55 and Microsoft Edge. Use an
isolated environment for browser dependencies (older Playwright 1.45 may crash
on current Edge download teardown):

```console
python -m experiments.pirc25.reproduce --tsde-root ../TSDE-SDE --browser
```

It checks comparison rendering and that CSV download returns the exact TSDE
bytes. Screenshots and receipts stay in the temporary root.

## Store and CLI

Choose an absolute local directory outside every Git worktree. Use `--root` or
`SDE_RUNTIME_ROOT`, and supply the original store ID on every command. There is
no implicit empty ledger. Windows accepts an absolute local drive path.

```console
python -m experiments.pirc25 --root /absolute/runtime --store-id research init
python -m experiments.pirc25 --root /absolute/runtime --store-id research fixture
python -m experiments.pirc25 --root /absolute/runtime --store-id research run affine-fixture
python -m experiments.pirc25 --root /absolute/runtime --store-id research rebuild-index
python -m experiments.pirc25 --root /absolute/runtime --store-id research list
python -m experiments.pirc25 --root /absolute/runtime --store-id research budget affine-4
```

Use subcommand `--help` for registration, explicit retry, authorization, export,
import and recovery. Fixtures support the existing single-axis (`--dimensions 1`)
and four-state (`--dimensions 4`) affine chains. Frozen upstream inputs remain
read-only; missing optional terrain only blocks dependent inputs.

| Location under `pirc25/` | Purpose |
| --- | --- |
| `store.json` | Fixed store/schema/root identity |
| `manifests/` | Immutable study, run, protocol, grant and evidence bindings |
| `events/`, `head.json` | Authoritative checksummed lifecycle/budget/exposure history |
| `artifacts/` | Content-hashed results/checkpoints/evidence and private attempt staging |
| `research.sqlite3` | Disposable relational query projection, never budget authority |

Use one root across research packages/worktrees. A backup must include identity,
events, manifests and referenced artifacts from a quiescent store. A SQLite file
alone is not a backup. Relocation needs explicit migration; changing the recorded
root is rejected. Network shares and distributed writers are outside this contract.

## Durability and budget

OS locks serialize mutations. Content is flushed and atomically published before
success. Terminal attempts cannot change; retries append linked attempts and
successful cells are reused. Registration is idempotent; changed content under an
existing identity is rejected. Arm families cannot be renamed for a fresh budget.

SQLite uses foreign keys, WAL, bounded busy timeouts and a watermark. Index rebuild
first validates authority; corrupt SQLite bytes are preserved in quarantine.
Broken authoritative history fails closed. Explicit `quarantine-tail --reason ...`
preserves invalid bytes and the old head, records recovery, and holds all budgets
for external reconciliation. It does not invent lost cost or reset a budget.

`BudgetSpec` enforces smoke ≤900 s, pilot ≤1800 s, job ≤7200 s and arm cumulative
≤86400 slot-seconds. Reservations/settlements are idempotent and cannot oversubscribe
the arm. Deadline/lost-worker outcomes close the affected arm. Unknown cost retains
the full reservation; `recover-unknown` records interruption. At most two explicit
retries are supported. No timer launches a new study.

For possibly launched work, recovery requires `--stop-evidence-hash` identifying
the operator's verified whole-tree stop evidence. A live or uninspectable
recorded PID (or POSIX process group) vetoes release even with that attestation.
Missing launch metadata also requires explicit evidence, never an assumed stop.
Recovery records the attestation, full-cost settlement and terminal attempt
under one lock. It does not terminate arbitrary PIDs from metadata. On Windows,
the operator's evidence must cover the Job Object's descendants, not just the
wrapper PID. A changed authority snapshot causes a query to return INDEX_STALE;
retry the list instead of continuing with a cursor from mixed versions.

`ResearchQueue` persists reservations, resource classes and claims. The default
store has one CPU slot and no GPU slot; initialize `ResearchQueue(store,
slots={"cpu": N, "gpu": M})` before its first use to fix a different local layout.
The resource manifest cannot be silently changed. FIFO is preserved within an
arm, with the oldest other arm preferred at each next claim. Unsettled claims
remain occupied across restart: reconcile the old process tree before recovery.
The managed worker environment limits OpenMP/MKL/OpenBLAS threads to one.

Agent execution guidance lives in
`.agents/skills/pirc-research-circuit-breaker/SKILL.md`; it preserves the same
authorization, store identity, budget and recovery boundaries for child methods.

Windows workers use kill-on-close job objects; POSIX uses process groups. An
independent wrapper enforces its own deadline and parent liveness. The supervisor
checks heartbeat, requests checkpointing at 80%, terminates the tree, charges
elapsed time and records a terminal state. Worker logs remain outside Git.

## Authorization and plugins

`EvaluationExposureLedger` checks immutable protocol/grant scope before opening
data. Grants bind study, protocol where applicable, blocks, purposes, visibility,
evidence hash and expiry. Only train blocks can fit preprocessing; legacy adapt
roles need explicit mapping. Final-eval requires a frozen protocol, known history
and explicit test authorization. Allowed/denied access and read start are logged
before reads; interrupted reads remain potentially exposed. Preview does not imply
export permission. This cannot detect unrelated tools bypassing controlled entry.

Grant publication is an explicit local-owner `authorize` operation, never a web
route. `CapabilityRegistry` routes exact transition, generic rollout, switching,
coupled-level and rare-event capabilities. Plugins supply command builders to
`SharedRunner` and retain the same supervisor. `accept_model`,
`accept_propagation` and `accept_switching` validate `pirc25-package-v1` sidecars:
schema/kind, hashes, state `[x,y,vx,vy]`, SI units, capability and qualification.
Formal comparisons require qualification/preregistration evidence. Nonlinear
propagation needs its qualified frozen model; oracle fixtures need no neural model.
The legacy single-axis adapter explicitly keeps `[x,vx]` and legacy state-norm units.

`SharedRecovery` validates parent/run/cell identity, code/data/feature/selection/
model/objective/protocol hashes, method state, data position and actual RNG state.
Exact/chunk recovery requires a registered method command; chunk recovery requires
a completed boundary. Affine adapters declare restart-only. Child packages own
their optimizer/Brownian/sampling state and qualification. Checkpoints never
restore budget balances.

## Frozen evidence and read-only UI

PSDE exports the complete expected-cell matrix through an export grant. TSDE owns
`scripts/pirc25/aggregate.py`: it validates hashes/units/protocols, retains missing
and failed cells, averages seeds within blocks, explicitly excludes incomplete
blocks and performs paired bootstrap only with sufficient independent blocks.
Formal mode requires a frozen comparison plan and qualified inputs.

TSDE atomically publishes `aggregate.json`, `metrics.csv`, `PaperEvidenceIndex.json`
and a hash manifest. PSDE imports these exact files, bound to its exported matrix.
The UI displays the frozen aggregate and downloads the original CSV; it does not
implement another statistics calculation. Raw trajectories never enter TSDE.

After publishing a preview/export grant, open the printed local session link:

```console
python -m experiments.pirc25 --root /absolute/runtime --store-id research serve --authorization-id my-grant
```

The service binds only `127.0.0.1`. Session tokens, Host/Origin checks and artifact-ID
resolution protect reads; tokens are removed from browser history after loading.
Mutation verbs are rejected. Matrix, attempts, failures, comparison values,
per-case results, provenance and CSV exports are available. Unstarted cells show
MISSING. Artifact scripts are never rendered and no remote training controls exist.

Limits: default 50 objects/page, maximum 200; responses/previews ≤2 MiB; forecast
previews ≤64 trajectories and ≤512 points each. Oversized content is explicitly
rejected. Cursors bind filters and the data watermark. Empty, incomplete,
unauthorized and corrupt states are distinguishable.

## Verification and delivery

```console
python -m pytest -q
python scripts/check_public_release.py
```

Acceptance also binds the independent reproduction and matching TSDE tests.
Required review and CI must pass before merging the relevant PRs. A draft PR,
fixture run or closed-but-unmerged PR is not completion. Authoritative decisions
and task/ref records remain in MPA `project/PIRC-38`.
