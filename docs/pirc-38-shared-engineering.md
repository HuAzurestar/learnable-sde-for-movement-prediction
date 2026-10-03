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

Within one PID/thread-owned verified physical phase, imported history may share
only content-bound, schema-validated identity/status facts. Exact coverage and
the conservative same-content or same-dataset/source-block veto use sorted keys;
every original guard still rereads and hashes the actual manifests and rechecks
current grants and expiry. New history publication, uncertain writes, recovery
and the physical phase boundary discard these facts. Fact publication is all or
discard: publish the completion marker only after all veto scopes are merged,
and clear derived facts on interruption so the next guard rebuilds them. This
reduces repeated record predicates and structural validation, not the remaining
manifest enumeration, canonical hashing or complete physical-prefix scans; it
is not proof of the full parent read-check complexity contract.

### Execution admission and formal evidence

`experiments.pirc25.upstream.UpstreamSnapshot` and
`resolve_snapshot(manifest, root=..., accepted_versions=...)` implement the
versioned metadata takeover API. A `pirc25-upstream-snapshot-v1` definition has
`inputs`, `studies` with explicit `pirc22_cutover`, and `cells` with frozen
`upstream_ids`. Each input records issue/acceptance commit/code SHA, artifact
ID/raw SHA256/physical byte size/schema, data/split/fold/feature/selection hashes
(or an explicit non-applicability reason), metadata-only license and frozen
scalar schema/status checks. `source_branch` is display only, not identity.
The external acceptance catalog must come from independently reconciled
operator evidence; matching it does not authenticate a human decision.

The resolver streams actual raw bytes through regular lexical-root handles,
then syntax-checks JSON while projecting only schema/status/identity headers.
It does not materialize unused large values or use a new arbitrary file-size
cap. Exact raw byte identity is explicit in this new schema, not a replacement
for the existing canonical-value public bindings. A changed checkout encoding
needs its explicit accepted byte binding, not an inferred latest branch.
Inputs stay pinned through the final batch boundary. `MISSING_ARTIFACT`,
`IDENTITY_MISMATCH` and `UNACCEPTED_VERSION` reject only dependent cells, without
fallback; PIRC-22 adoption requires the same immutable zero-final-eval selection.
`snapshot.register(store, ...)` persists immutable normalized definition and
content-addressed validation receipt, including the actual validation time.
These are metadata only, never data grants or proof that a future read is safe.

Actual registered-plugin admission requires `upstream_snapshot_hash`,
`upstream_acceptance_hash` and an explicit absolute lexical `upstream_root`.
The qualification package binds both hashes. Register the normalized snapshot
definition and a `pirc25-upstream-acceptance-v1` operator catalog before freezing
the plan. Its `source`, complete `entries`, `source_evidence.entries` and
`source_evidence_hash` are retained; `register_acceptance_catalog` validates
that content binding, not the honesty of off-platform human attestations.
For formal or held-out input, preregistration `upstream_bindings[study_id]`
binds `snapshot_hash`, `acceptance_catalog_hash` and explicit `pirc22_cutover`.
Both upstream records must have been published before preregistration.
Snapshot cells identify exact complete registered cell hashes and study roles;
there is no implicit empty/no-terrain cutover or missing-cell fallback.

`AdmissionGate` freshly resolves only the selected cell's dependencies before
qualification/data reads or worker startup, ignoring older ready receipts.
It publishes content-addressed validation plus `UPSTREAM_VALIDATION` and typed
`UPSTREAM_REFUSED` metadata. Its immutable admission contains the definition,
catalog and exact validation hash/time. Independent cells keep their own
outcome; missing inputs do not alter dependency sets or scientific routes.
Snapshot/catalog `visibility` defaults to `restricted`, not public metadata;
explicit narrower input labels also propagate. Both registered study lineage
and detached admission evidence contribute to result/export/query visibility.
Metadata-only licensing does not permit a synthetic/public-only grant to
disclose restricted source evidence; normal fresh purpose/expiry/scope checks
still apply. An explicit source-wide label is inherited only when the input
has no narrower label. Whole catalogs retained in evidence are conservatively
covered too, not silently disclosed because this cell uses fewer inputs.
Explicit legacy public `upstream_hash`/`upstream_ids` remain additional frozen
checks when requested, never a substitute for the required new snapshot.

This API/runtime integration does not by itself close the parent takeover
contract. Complete real accepted PIRC-19–22 field/license reconciliation,
independent Paper and UI rejection integration remain required. Do not treat
the historical `audit_inputs` binding as that complete receipt or label this
standalone API validation as final engineering/scientific acceptance.

`SharedRunner` and `SharedRecovery.resume` use `AdmissionGate` before invoking
a command builder. A registered capability alone is not permission. The source
tree code hash must match the registered spec. Reused legacy successes without
an admission event fail closed. Built-in affine fixtures must match the complete
frozen synthetic recipe and pass their public upstream audit; they receive only
fixture qualification.

Other plugins declare an immutable spec `admission` object with `mode`
(`fixture`, `pilot`, `formal`), `protocol_id`, `authorization_id`, `package_hash`,
required snapshot/catalog references and metadata root described above,
optional legacy `upstream_ids`/`upstream_hash`, and a role-appropriate
read `purpose`. The grant covers study/protocol/block/visibility, `execute` and
the read purpose, remains unexpired and supplies an absolute `data_root`.
`data_binding(protocol)` binds source identities/content hashes/roles; the ledger
checks actual input bytes and records the attempt/run before a plugin is invoked.
Stage caps are fixture 15 minutes, pilot 30 minutes, formal 2 hours, all subject
to the cumulative arm cap.

Execution packages retain the child kind/state/unit/capability/recovery contract
and additionally bind `study_id`, `visibility`, `plugin_hash=plugin_binding(plugin)`,
`command_hash=command_binding(builder)`, a JSON `payload` covered by `output_hash`,
and `upstream_hash`. Code/data/input/protocol hashes must match the spec.
Recovery additionally binds the actual adapter's `recovery_command_hash`.
The standalone package validator does not itself grant execution permission.

Formal admission requires a blind frozen plan, reserved-test input and a
registered `qualification-HASH` report. Its `pirc25-qualification-v1` schema
records passed status, `package_binding(package)` (excluding the circular
qualification hash), code/preregistration hashes and `checks`. Each
preregistered `qualification_checks` name links a canonical JSON qualification
artifact containing `check_id`, passed `outcome`, package binding and code/
preregistration hashes. The runtime authorizes and checks these artifacts; it
does not independently prove an operator-imported report's scientific truth.

Frozen models also declare `model_protocol_id`, checked against their own
source-study/data/preregistration bindings, not the consumer's protocol.
Cross-study frozen models require `model_authorization_id` with the consumer
study in the source grant's `consumer_study_ids`. Evaluation permission does not
grant export permission. Export rechecks qualification artifact access, including
foreign-model evidence; weights/evidence are not automatically public.
Aggregate/CSV/index artifacts inherit restricted package or qualification
attachment visibility even when every result cell is synthetic. Missing package
visibility is conservatively restricted, never an implicit synthetic grant.

The immutable `pirc25-admission-v1` receipt binds spec/cell, actual attempt/run,
plugin/command, input read events and upstream/package/qualification evidence.
The supervisor attaches its hash to the validated result; a worker cannot promote
fixture qualification or substitute another receipt. Export verifies the receipt
against the successful attempt. TSDE formal aggregation checks internal hashes,
scopes, freeze/read order, qualification artifacts and result identities, not
just `qualified` text. The expected bundle hash is the transport trust boundary,
not a signature proving arbitrary external claims. Legacy fixture evidence can
still be inspected, but legacy qualified results without admission cannot be
exported or formally aggregated. Register new versioned inputs rather than
rewriting an immutable legacy success.

Complete evidence export assembles source results and all-attempt costs inside
one lock-held verified read scope. It rejects changed data/reservation/recovery
snapshots, freshly checks the main and consumed foreign-model export grants
before and after private immutable bundle publication, and checks the earliest
expiry again after final physical validation. All consumed grant manifests are
rehashed after the successful permission journals; a journal-time grant change
is durably denied before publication or return. Read/disclosure journals remain
durable and do not invalidate an otherwise unchanged export. The CLI also
reauthenticates its frozen bundle after target staging fsync and before atomic
rename. A refused target publication removes only its own staging file; an
existing target and already authorized private bundle are preserved. External
export publication is atomic and no-clobber: Windows uses non-overwriting
rename, POSIX uses a hard link from the fully flushed same-directory staging
file. Unsupported filesystem operations fail closed, never fall back to
overwriting replace. A conflicting target is rejected; identical content is
compared through one size-checked regular file handle with a read bounded by
the expected bytes plus one, then export authority is rechecked after that I/O.
Idempotent retries also sync the directory. Ordinary locked authority-store
writes retain their existing replace semantics.
Staging cleanup requires successful exclusive creation by the current writer;
a preexisting staging-name collision is retained, not deleted as failed output.
Successful moves relinquish the staging pathname before directory sync, so a
later writer reusing the freed name cannot be deleted by the earlier cleanup.

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

### Immutable authorization versions

New grants may declare a nonempty string `version`. The immutable primary key
is `(authorization_id, version)`: multiple versions can coexist, and changing
the contents of an existing version is refused. Select a specific grant with
`store.authorization(id, version="v2")`, provider/query/managed comparison/case
`authorization_version="v2"`, or CLI `--authorization-version v2` on export,
compare, render-case and serve. An admission spec freezes `authorization_version`
and, for a foreign model, `model_authorization_version` alongside the respective
IDs. Read events retain the selected version and complete grant hash; offline
paper admission checks the same binding.

Omitting the selector addresses only the original unversioned grant. Legacy
manifests and receipts retain their original bytes and hashes. A missing explicit
version is refused, without falling back to a legacy grant or another version.
Publishing a new version does not revoke earlier versions: each retains its
own scope and expiry. Local UI sessions stay bound to the version selected when
the service was started; request parameters cannot replace that selection.

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

Rebuilding writes disposable SQLite schema2, with unique exposure event IDs and
a `(block_id, sequence)` primary-key index. Schema1 registration-list queries
remain supported; exposure queries require an explicit rebuild. From the same
local root, use `python -B -m experiments.pirc25 --root <runtime-root> --store-id <store-id> audit-exposures --block <block-id> --limit 50`.
`--after <sequence>`, `--before <sequence>` and `--watermark <watermark>` page
the frozen sequence range; `--include-checks` includes allow/deny checks as well
as started/completed/failed reads. Missing, corrupt or lagging projections return
`INDEX_STALE`; a cursor from a different rebuilt version returns `CURSOR_STALE`.

This command displays local audit metadata only, never trajectory/result bytes
or an access token. Scope comes from declared block/source-block IDs and real
hash-verified artifact manifests, not a grant. Selected original events and scope
are rechecked against physical authority; the SQLite projection is not permission.
An empty page does not certify an untouched block. Content aliases, external
exposure history and blind-test authorization still use the preregistration/data
gate. A SQLite range-query plan does not prove the complete O(B log E) read-check
contract, including those aliases and fresh physical verification.

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

Raw artifact and data-provider reads use shared regular opened handles, reject
late path replacement and POSIX symlink/FIFO races before bytes, and verify file
identity and the authorized root again after the complete read. Artifact bytes
retain their frozen-size-plus-one bound. Legacy providers first verify their
frozen digest with at most 1 MiB per read and constant auxiliary memory; a valid
byte-return read then materializes through the same handle and rechecks its hash.
There is no new blanket data-size cap. Execution admission uses the same streamed
verification and durable consumer receipt without materializing unused inputs.
Native `scripts/check_research_source_reads.py` tests the actual APIs and real
Path/descriptor reads on Windows and Linux; it is not full platform acceptance.

Authoritative manifest/event loads also use regular handles, preserving the
admitted file size and rejecting redirected roots or replaced files before
reading. Runtime code and paper source identities hash UTF-8 incrementally in
64 KiB chunks with the existing universal-newline normalization, including CRLF
and multibyte sequences split between chunks. Paper's existing 16 MiB per-source
admission quota remains; it is not imposed on runtime code or other data. Code
identity roots retain their declared lexical paths so directory links cannot
silently establish a new trusted tree. Native source checks use actual Git when
testing paper identity; missing Git is an explicit skip, not substituted proof.
Runtime source enumeration rejects directory links rather than silently omitting
modules that Python can import through them. Ordinary nested source trees retain
the previous path/content hash mapping and platform-specific extension matching.

Raw artifact and data-provider reads rehash the immutable grant and its current
scope after allowed/start/completed journals, before opening bytes and before
returning them. Provider checks retain role, fit scope and frozen test/history
bindings; raw reads retain conservative source visibility. Successful reads keep
the original three-event journal rather than writing redundant allowed records.
The actual outer scope rehashes each read's authority after final physical
event-chain verification, even for nested reads. It shares only the freshly
verified event prefix; grants and manifests are never permission-cache entries.
All pure expiry checks follow all guards' I/O, without additional allowed writes
or a per-guard physical chain scan. Recovery
preparation rechecks resume authority after its checkpoint source/save validation,
before returning method state; this never creates an attempt or resets costs.

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

The actual recovery command handoff revalidates the checkpoint's resume grant
after retry/resume/admission I/O, before passing method state to the registered
builder. A distinct data/execution grant cannot replace this restore permission.
Denial uses failed-preflight settlement with zero new worker charge, retains the
parent's spent cost and creates neither a resume-state file nor a native worker.

## Frozen evidence and read-only UI

PSDE exports the complete expected-cell matrix through an export grant. TSDE owns
`scripts/pirc25/aggregate.py`: it validates hashes/units/protocols, retains missing
and failed cells, averages seeds within blocks, explicitly excludes incomplete
blocks and performs paired bootstrap only with sufficient independent blocks.
Formal mode requires a frozen comparison plan and qualified inputs.

TSDE atomically publishes `aggregate.json`, `metrics.csv`, `PaperEvidenceIndex.json`
and a hash manifest. PSDE imports these exact files, bound to its exported matrix.
Incoming package reads keep the original 64 MiB core-file, 1 MiB computation
receipt and 2 MiB figure/index quotas. Regular opened handles are checked against
the inside-root path and the admitted file identity; actual reads are bounded by
that size plus one byte. Growth, replacement or in-place modification is rejected
before parsing/import, rather than allocating the entire changed file first.
Owner-side success transitions and duplicate artifact registration use the same
frozen-size regular-file/hash integrity primitive as authorized artifact reads;
damaged bytes cannot be allocated without that bound or produce a new success.
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

Every public query assembles its response inside one verified, lock-held read
scope. Preview and export remain separate purposes: entry permission is checked
against the immutable grant and source lineage, each actual artifact read keeps
its own authorization/read journal, and final disclosure rechecks the actual
grant, including a fresh manifest rehash after the allowed journal is flushed.
This post-journal check does not create another allowed write; denials remain
durable. Current expiry is checked after permission-journal writes and after the
last physical chain verification, before returning metadata or result bytes.
A failed journal, changed grant or expired permission cannot yield a successful
response. Genuine manifest/attempt/budget/recovery-hold changes during assembly
reject a mixed snapshot; disclosure journals alone do not invalidate pagination.
No permission or arm-balance cache is shared between requests.

Study disclosure also registers a current-grant rehash at the actual outer
scope completion, after the uncached physical validation. This applies even to
metadata-only pages, all-missing matrix exports and the external publication
guard, where there is no raw artifact read. Every consumed grant's pure expiry
guard follows all final authority I/O. A denied export retains the already
authorized private bundle but does not permit external publication or disclosure.

Limits: default 50 objects/page, maximum 200; responses/previews ≤2 MiB; forecast
previews ≤64 trajectories and ≤512 points each. Oversized content is explicitly
rejected. Cursors bind filters and the data watermark. Empty, incomplete,
unauthorized and corrupt states are distinguishable.

Run queries accept exact-match `model`, `version` (registered spec hash), `horizon`,
`seed`, `trainer`, and `predictor` in addition to arm/state. Model is the registered
arm's model family, trainer is its explicit trainer ID (or study trainer ID), and
predictor is the cell's plugin ID; missing declarations remain unknown. Responses
include the selectors and comparison dimensions, including unstarted cells.
All filters bind pagination cursors. Budget reservations, settlements and arm
closure also invalidate a cursor, so pages cannot silently mix charge snapshots.

Each run includes cumulative arm balance in `slot-ms` and per-run reservation/
settlement sources with event hash/sequence. `cost_basis` distinguishes active
reservation, measured monotonic cost, and unknown conservatively charged cost;
unknown elapsed time remains null. Balances cover the whole registered arm, not
only the currently filtered horizon. Source records are limited to this study/run.

The page exposes these exact filters, run budget/source tables, comparison strata
and explicit paired interval availability. Comparison horizon selection only
selects frozen summary rows; it does not recalculate scores or independent sample
counts. Each metric uses its own visual scale. Case horizon selection requires an
explicit `forecast.horizons` grid; absent grids are reported as unavailable, not
invented. The existing single-axis fixture now exports its actual requested grid.

Managed comparison previews and SVG exports consume exact budget-worker frozen
files selected by the bound figure index. The browser verifies byte size and SHA
before displaying a Blob image (never inline artifact markup). Horizon changes
cannot replace the current image with a late previous response. Export performs a
new permission-checked artifact read and downloads the original bytes/filename,
not cached preview bytes or a locally regenerated figure. Frozen provenance refers
to the settled computation receipt; that receipt and its measured cost are shown
separately and remain downloadable. Preview permission alone cannot export.
Legacy packages without frozen figures retain their tables/CSV and explicitly
report unavailable figures rather than silently regenerating them.

Case previews and exports also consume saved worker SVGs, never browser-generated
plots. Prepare them offline from an already-authorized saved result:

```console
python -m experiments.pirc25 --root /absolute/runtime --store-id research render-case RESULT_ARTIFACT_ID --authorization-id my-grant --seconds 7200
```

This job uses the same supervisor, 2H job/24H original-source-arm ledger, immutable
attempts and measured settlement proof as other shared jobs. Its source input is
bounded to 2 MiB; the frozen path/layout policy allows at most 64 paths and 512
points each. Joint output/operation quotas are checked before reserving or
starting a worker. Four-state saved paths have separate position and velocity
panels with their original units. No new predictions or metric calculations occur.
Missing saved paths explicitly return unavailable without launching any job;
per-segment scores are never converted into invented sample paths.

GET `/api/cases/{result_artifact_id}` returns the authorized saved result and,
when present, a verified frozen figure index and computation receipt. Queries
never launch jobs. The index uses `horizon_index` (zero-based saved-grid ordinal,
or null for all), with the original numeric horizon stored separately. It binds
the source artifact/spec/cell/protocol, exact preview policy, worker request and
settled source-arm cost. Reopening the store can reuse an intact successful job
without resetting or charging the budget again. The browser verifies index,
size/MIME/SHA, uses only Blob images and performs a fresh export-purpose read
for every download. Legacy results without a saved graph retain their values
but show the figure as unavailable. Preview-only permission never authorizes
export. Frozen CSV remains the original TSDE byte stream. Run budgets are
current ledger views, not retrospectively attached costs to an older aggregate.

Result and aggregate manifest downloads use `?manifest=1` on the authorized
artifact-ID endpoint. They require export permission, verify the source artifact,
and return registered artifact metadata plus result/aggregate binding references;
they do not attach forecast trajectories. Filtered empty lists are distinguished
from an unregistered matrix. UI errors retain the diagnostic code alongside safe
details, including corrupt artifacts and oversized previews.

Separately, new evidence bundles freeze each cell's latest reservation/settlement
events for **all** its attempts, including failures and retries. TSDE validates
event hashes and identities and summarizes these costs within the same comparison
stratum. The comparison page, CSV, evidence index and exported comparison figure
use those frozen costs. `charged_ms` is settled ledger charge; `reserved_ms` is
still-held capacity; `measured_ms` remains null if any elapsed cost is unknown,
pending or missing. Unknown-cost settlement retains the conservative reservation
charge. Missing legacy cost records mean unavailable, never free execution.
Costs include unscored/incomplete blocks, whereas quality metrics use complete
blocks; the page labels this different denominator explicitly. Later settlement
produces a new bundle version; previously published cost evidence is unchanged.

An explicit synthetic browser check is available with an existing isolated
Playwright environment and Edge: `python -B -m tests.browser_research_ui`. It needs
the sibling TSDE checkout, uses temporary Git-external fixtures, and checks filter
interactions, budget provenance, horizon selection, source-bound figure downloads,
exact CSV bytes and preview-only export denial. It is not scientific validation.

## Verification and delivery

### Managed statistical comparison

The frozen comparison plan may include a complete `pirc25-adjudication-spec-v1`
policy and its canonical hash. It declares the primary estimator/unit/direction,
weighted strata, minimum seeds and paired blocks, explicit seed pairing,
confidence/resampling/multiplicity, practical threshold, quality gates and fixed
failure/attempt/stopping rules. A missing decision produces
`NEEDS_PREREGISTRATION`, not an implicit default or a scientific gain.

After importing a TSDE input package, run from the runtime repository:

```console
python -B -m experiments.pirc25 --root RUNTIME_DIR --store-id STORE_ID compare STUDY_ID INPUT_AGGREGATE_HASH --authorization-id GRANT_ID --paper-root TSDE_ROOT --seconds 7200 --output NEW_PACKAGE_DIR
```

The original-study grant must authorize export of the complete source matrix.
The runtime retains the selected frozen input version, checks current source
disclosure, bounds registered-size operations and bytes, and launches the TSDE
kernel inside the existing supervisor/process-tree boundary. The derived
computation study copies the original reference arm's identity/family and charges
its existing cumulative 24-hour budget. It does not add a cell to the scientific
matrix. The allocation is explicit; no new caller-selected arm resets costs.
`--max-operations` can reduce the fixed 20-million operation cap. `--formal`
requires qualified source evidence and its exact preregistration chain; fixtures
remain engineering-only.

The job freezes a full decision, same-version CSV/index and `computation_ref`.
After the entire worker tree stops and measured cost settles, an immutable
`ComputationReceipt.json` binds the actual result and reservation/start/settlement
events. Aggregate/CSV/index reference that receipt without a circular cost hash.
Imports verify the exact managed worker result; rehashing a changed verdict or
table is insufficient. Older descriptive-only four-file packages retain their
original bytes. Statistical computation cost is separate from frozen experimental
costs and is not multiplied by the number of comparisons.

Successful reuse validates actual worker bytes and provenance without another
attempt/charge. Failed retries require `--parent-attempt` and `--reason`; a closed
arm remains closed. Publication interrupted after settlement resumes publication
of the same successful attempt. A timeout returns a nonzero CLI exit status.
The UI displays the full fixed adjudication family independently of horizon
display filters and exports its frozen decision and receipt with figures.

These interfaces do not establish final acceptance. General plugin resource
planning, acknowledged soft checkpoints, continuous-versus-resumed recovery and
final paired-SHA review/merge evidence remain separate required audit items.

Development UI renderer checks execute the actual JavaScript with Node.js
(`node` on PATH); browser checks use the separately installed Playwright/Edge
environment. A missing tool is a verification-environment failure, not a product
pass. Partial preregistration remains diagnostic and renders unavailable fields,
never a chosen default. Index rebuilding verifies one locked event chain and each
manifest hash without a whole-chain reread per manifest; it remains a disposable
projection, not an authority or permission source.

### Comparison dimensions

Evidence export preserves each registered cell and its `comparison_dimensions`.
The standard cell axes `horizon`, `horizons`, `region`, `scenario`, `initialization`,
`prediction_origin`, `context_profile`, and `time_grid` are automatically included.
Additional scientific strata must be explicitly registered in the cell's
`comparison_dimensions` mapping; conflicting duplicate axis values are rejected.
`arm_id`, `block_id`, and `seed` remain identities, not strata. A different horizon
does not create a new budget arm or reset its cumulative charges.

TSDE groups by exact dimensions plus arm, averages seeds within each complete
block, and compares arms only within the same stratum. Missing arms or incomplete
blocks yield no paired score, not a cross-horizon fallback. Each arm summary and
comparison carries `stratum_id` and `comparison_dimensions`; CSV includes both,
and evidence claims reference only contributing complete-block attempts in that
stratum. Consumers must use `(arm_id, stratum_id)`, not `arm_id` alone, as the
summary key. Legacy dimensionless bundles remain a single `{}` stratum; discarded
dimensions cannot be reconstructed from an old export, so re-export its registered
study when dimensions are needed. Existing frozen packages are never rewritten.

```console
python -m pytest -q
python scripts/check_public_release.py
```

Acceptance also binds the independent reproduction and matching TSDE tests.
Required review and CI must pass before merging the relevant PRs. A draft PR,
fixture run or closed-but-unmerged PR is not completion. Authoritative decisions
and task/ref records remain in MPA `project/PIRC-38`.
