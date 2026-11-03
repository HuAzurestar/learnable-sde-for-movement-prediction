DEV50 Original205 source-bound constructor bindings

This is executable source-only preparation. builder.py uses stdlib AST/hash/JSON
operations to produce invocation-spec.json; it never imports or executes original
product modules. The 29 invocations are a typed deferred graph, not executed
ResearchStore registrations, and artifact metadata is explicitly predicted.

From repository root:
  python -X utf8 qualification_preparation/dev50/builder.py --source-root . --output plan.json
  python -X utf8 qualification_preparation/dev50/check_builder.py
The output path is caller-owned; do not overwrite an accepted frozen candidate.
--mode live always raises BindingError. No runtime executor is supplied.

Graph interpretation
- operation=construct calls the class named by callable and yields its instance;
  the pinned __init__ supplies the statically checked argument shape. operation=call
  invokes the named method/function on receiver when present. No invocation runs here.
- $result refers to a prior invocation result; path selects fields. $external names
  an unresolved mandatory input. $fixture_bytes is a pinned artificial byte file.
- $expression is deferred field/item selection or original sorted compact UTF8
  digest. Never hash unresolved expressions as an actual registration hash.
- $dataclass is a BudgetSpec field binding, statically checked against source
  annotations/default presence; defaults are never evaluated. Exact source_arguments
  accompanies the binding_shape, whose None means a default exists, not its value.
- prepare_callback supplies the exact six original arguments and returns
  (package_path, verify, deadline). Its output/deadline only exist after RUNNING.
  command_builder and original prepare lambda remain required in-process inputs.
- owner is a context manager, not an ordinary one-shot registration. All inside-owner
  operations stay inside its OS lock/capability lifetime, through terminal exit.

Concrete outputs
The 22 unchanged DEV49 drafts and hashes are included. Seven artificial population
byte files produce exact predicted artifact_id=SHA256(bytes), complete original
artifact metadata (including media_type), patched inventory/geometry/grant shapes,
and a config/study/execution template. Null dev-6 stays CONTRACT_REJECTED. All eleven
sources, ten original arm IDs, and 32 distinct cells remain; selected source is
train-0/A-S. Original M0/M2 identities and exact registered family tuples are never
invented: original_study remains mandatory. The study's experiment/comparison/feature/
selection fields also require that original spec. No canonical study hash is claimed
until those fields resolve and the original constructor returns.

Authority and retained history
The original policy is resolved, never replaced with a replenished account.
Original config, source grants, policy, qualification, B1 decision and billing
receipt are mandatory external objects/refs. Source v2 bindings call original
rebind_grant, preserving scope and expiry; inert_grants_metadata is only a fixture
shape and must not be supplied as real authority. Original _validate additionally
requires unchanged numeric/source inventory/geometry, accurate code/qualification,
original pending-cost and stop evidence. Generated artificial inventory cannot be
substituted for historical inventory. A separately admitted artificial setup would
need actual matching original registrations, not a fabricated history.

previous_execution_ref and retained_prefix are distinct: former is the actual
activation predecessor (or actual legacy_ref); latter is read by original
owner_session from the immediately previous real event inside the lock. No synthetic
chain is generated. live_owner_capability is process/thread-local, cannot be passed
as JSON, and is installed only by original owner_session. All live modes fail closed,
even when JSON looks complete. Draft passed/user fields never become qualification
or a user decision. No independent quality result or runtime admission is asserted.

Original contracts preserved
Supervisor reserves/admission/queue/claim once; opens charged stage before protected
parent reads; original native containment/fuse/GO checks remain. prepare returns the
real tuple. Whole-tree stop precedes postcheck(), finite bounded result parsing,
postcheck(result), original artifact publication, unique known SETTLE and terminal
attempt. Source route has no extra result_validator or partial validator. Terminal
saves raw finite report bytes before t-star, hashes those exact bytes, retains
settlement_pending=True, appends original receipt and verifies physical transaction
exit. Minimal205 is not207/full209. Event map is source pinned; full event validation,
native OS/filesystem/queue/exposure bodies are delegated transitive contracts, not
new runtime proof. DEV48 static54/conservative80 and dynamic56 dossier remain static;
17-file code identity is not closure and Torch/numpy precede source parsing.

Validation and limits
10 stdlib DEV checks cover 650 source bytes/blob identities, fixture tampering,
7 artifact hashes, matrix completeness, reference order, API/dataclass failure,
determinism and live rejection. They do not execute retained tests or constructors.
All 650 original tracked bytes are unchanged. 11 frozen inputs and 66 additional
bounded UTF8 pages were individually read with SHA/BEGIN/END before semantic use;
semantic-notes.txt records exact scope. Full manifests/AST are mechanical comparison,
not a full semantic read of 650 files. Source reading, identity, method deviations,
ZIP/bundle/diff/source manifests and actual commands are in the delivery dossier.

Still MISSING/UNPROVEN: real matching store/study/grant/policy/qualification/decision/
billing/activation history, non-genesis predecessor, live capability/resources,
native imports/WindowsJob/fuse/stop, paid reads/live clocks/settlement/finite terminal/
verified exit and original returned Supervisor result. Missing historical gridtest,
f220/full651, original205/207/209/SOL022 UNKNOWN, all FAIL/UNKNOWN, costs/grants/balance/
stops remain. Preserve pilot1800s, per-arm training120s, stage120s, Nov03 23:59HKT,
finaltestclosed, user-only COMPLETION and separate release. No push/PR/merge/deploy.