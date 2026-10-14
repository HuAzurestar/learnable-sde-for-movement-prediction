# Concrete formal runtime boundary

`formal_environment.ConfiguredRuntime` configures the frozen CPU counts and
checks effective Torch intra/inter-op, NumPy OpenBLAS and Numba values. It
preloads the scientific readers/worker and selected map backend **without
constructing a data/map provider**. Two empty in-memory DuckDB connections
verify that the registered 2/4-thread settings take effect. They are not raw
input reads or observations of historical database connections.

The fingerprint includes the actual interpreter/base interpreter, dependency
versions and module entry-file hashes, and all thread pools discovered by
`threadpoolctl`, with their DLL bytes. It is deliberately not described as a
hash of every installed package file or every operating-system thread. Paths
and private environment fingerprints belong in local artifacts, not the public
aggregate export.

From the PSDE repository root, using the registered research interpreter and
the sibling DSDE repository on `PYTHONPATH`:

```text
python -m experiments.pirc17.formal_entrypoint environment --output-directory LOCAL_METADATA_DIRECTORY
```

This writes one immutable environment record. It reads no research inputs,
runs no fit/forecast, and grants no final-eval access. Repeating into a directory
already containing that exact record does not overwrite it.

## Prepare and run

`prepare` records the complete 11,659-item matrix, current source catalog,
actual effective environment, fixed input locators and one unused ledger:

```text
python -m experiments.pirc17.formal_entrypoint prepare --output-directory BUNDLE_DIRECTORY --ledger-directory UNIQUE_LEDGER --release RELEASE_DIRECTORY --snapshot SNAPSHOT_DIRECTORY --data-root DATA_ROOT --trajectory-path TRAJECTORY_FILE --development-eligibility-path DEVELOPMENT_ELIGIBILITY_FILE
```

It does not open those research inputs or create the ledger. The returned
bundle and execution hashes must be the exact identities tested, reviewed
and presented at human ACCEPT-01. Any subsequent source change invalidates
this execution seal. A newly prepared bundle is not a test pass or approval.

Only after that actual human decision, using its pinned receipt:

```text
python -m experiments.pirc17.formal_entrypoint run --bundle BUNDLE_FILE --bundle-sha256 BUNDLE_CONTENT_SHA256 --approval HUMAN_RECEIPT --approval-sha256 HUMAN_RECEIPT_CONTENT_SHA256 --test TEST01_RECEIPT --review REVIEW01_RECEIPT
```

The same research interpreter, PSDE working directory and sibling DSDE
`PYTHONPATH` are required. The public command connects the real controller,
runtime/scientific callbacks and retained worker, with no alternate handler
or validator option. CLI options cannot reduce or enlarge the matrix.

Before full validation it exclusively creates the deterministic sibling
`UNIQUE_LEDGER.launch`, recording the pinned bundle, receipt locators and
module-entry monotonic clock. It never deletes this claim. A failed or
interrupted claim, even an empty one from an interrupted write, cannot be
automatically retried. Missing required files are denied before claiming.
Before startup transfers to a controller span, the original 3,600-second
input-phase allowance is conservatively retained, not claimed as observed
compute. A saved failure also retains known elapsed time and any overrun.
Once transferred, the first controller span accounts for the same prefix;
do not add the launch reservation again or reset its directory.

## Worker and controller integration

`formal_entrypoint worker` is an internal command for the existing real
`OwnedSession`; its required session arguments are supplied by that owner.
Direct invocation cannot bypass native Job membership. A bundle must bind the
complete protocol, matrix, execution source catalog and runtime. The runtime
also pins one canonical ledger directory, the source working directory, all
five input locators and the fixed dependency phase order. Input locators are
not opened during metadata binding; guarded readers verify their actual
identities after human authorization.

There are no CLI options to change N, h, H, seeds, origins, matrix size or
thread counts, and no resume/reset/retry or approval-creation option. The
worker compares its actual command and ledger contract with the bundle,
checks the real approval/test/review receipts, verifies the effective local
environment, and only then constructs the retained `FormalWorker`. Runtime
checks surround each work call without repairing runtime drift. Startup and
first-work-per-phase observations are saved inside the real session directory.
The independent closed-output reader rereads the actual binding/start/phase
files, checks their scope and bytes, and rejects missing/changed proofs even
if an earlier controller observation claimed success.

`RuntimeCallbacks` connects the real `ScientificResults` to the controller.
Heavy environment/domain initialization is lazy, under the first input work
reservation. Each result binds actual worker runtime evidence into its saved
artifact-verification details. Full matrix traversal is cached at startup,
not repeated for each forecast to find phase-first work IDs.

`Controller.run_all(..., started_ns=...)` can carry the concrete launcher's
monotonic entry time into its first phase. The already elapsed startup prefix
plus the normal five-second control credit is debited from the **original
input-phase budget**. It is not a new phase/budget or a per-work surcharge.
An already expired phase admits no worker and retains observed overrun. Later
phases retain their original disjoint work/control clocks. This interface
does not itself supervise a caller that never reaches the controller.

The public launcher publishes its terminal receipt after actual owned-tree
closure, then `Controller.finish_terminal` records the outer result/receipt
work in the last used phase. Unused prepaid credit is never refunded. The
terminal event permanently prevents further work admission; crossing the
old credit records the excess and halts, including a slow final append.
The receipt's `candidate_complete` is not final accounting: the subsequent
ledger terminal event must bind its content/file-byte digest, with no halted,
pending, unknown, failed or unattempted required work. A missing terminal
event/receipt is incomplete, not success. These clocks measure module-entry
through controlled work and terminal publication; they do not claim to
measure Python interpreter bootstrap or OS teardown after the command exits.

## Acceptance boundary

No actual complete runtime/execution bundle has yet been sealed for human
acceptance. Software fixtures and their temporary bundles are not TEST-01,
REVIEW-01, human ACCEPT-01, a real formal run, full numerical qualification or
paper completion. Do not run the legacy data entrypoints as a substitute.
