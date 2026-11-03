# DEV54 source-only original205 driver

This directory defines a future artificial minimal205 invocation. DEV54 does not
admit native execution. Default/explain/preflight return exit 2 and report
SOURCE_ONLY_NATIVE_UNEXECUTED. No baseline Python module is imported by those
modes or by the new selfchecks. No native/scientific PASS is claimed.

## Entrypoints

From an isolated clone, `python -X utf8 qualification_preparation/dev54/driver.py`
explains the interface. `--mode preflight --runtime-root <fresh absolute path>`
checks the 704 baseline bytes and prospective root without creating it. A CLI
`--mode execute` always refuses: flags, JSON receipts and expiry fields cannot
provide admission.

Future separately authorized execution uses an independently reviewed trusted
host in a clean process, whose working directory is the verified repository:

```python
# FUTURE ONLY, NOT RUN BY DEV54
# Load driver.py by absolute path, then:
result = driver.main(
    ['--mode', 'execute', '--runtime-root', exact_fresh_absolute_root],
    admission=trusted_host_context_factory,
)
original_supervisor_result = result['supervisor_run']
```

The host context factory is a Python function, not deserialized user data. It
must independently verify the actual authorization, exact source/hash/full argv,
interpreter/environment/resources, parent-directory ownership/ACL, exclusive
root, and requested original205 scope. It receives that concrete request before
root creation or native import. Merely supplying a no-op factory is not an
approval. This repository defines no approving host or native CLI bypass. The
explain output reports the absolute driver, SHA256, future cwd and full argv;
these identify a proposed action, not permission to perform it. Fixed expiry is
2026-11-03 23:59 HKT; expired permission fails closed and is never renewed.

## Source route and ownership

`verify_source` pins baseline.json by SHA256 and checks all 704 original paths
against baseline 7f426dfa4fe0f0666ff9cec69301a81f6acb183c. It rejects changed bytes,
hardlinked files and redirected/reparse directories. The future root must not
exist, must have an existing parent and must be outside every Git ancestor.
Existing/live/uncertain roots are never adopted or replayed. The admitted action
creates the new root and an exclusive owner marker, holds a real OS byte lock
through original terminal and physical owner exit, and retains all artifacts on
success or failure. This lock supplements the original store and owner locks.

`load_native` is reachable only from `execute_after_admission`, after the host
context and root claim. It obtains actual classes/functions/unbound methods
from verified original modules, checks module origins, and proves owner_session
and GridClock.check share the original `_owner` threading.local. Its 21-provider
set equals the unchanged graph/API contract. It loads the original DEV52 Adapter
programmatic entry, never its disabled CLI. No constructor/callback is replaced.

`artificial_baseline` is an unexecuted recipe using genuine original APIs:

1. Explicitly initialize a new artificial ResearchStore and publish seven exact
   fixture artifacts through original artifact constructors.
2. Register the original artificial study, ten unchanged account family tuples
   including M0/M2 IDs, 32 cells, exact old config, immutable 1800s/120s policy,
   v1 source/internal grants, artificial guard receipts, and metadata files.
3. Read back original objects/grants and derive the genuine legacy_ref. Receipts
   marked fixture_only exercise guard shapes only. Their `passed`/user fields
   are not evidence of real user approval or independent qualification.
4. Hand those actual objects/absolute metadata paths to the unchanged 29-node
   Adapter. It creates the new study/v2 rebindings/config/activation before owner
   entry; derives actual meter, capability, immediate non-genesis predecessor,
   activation binding and original prepare callback in process.
5. The selected source cell is train-0/A-S, train_steps=0, inputs={}. Register one
   actual attempt; build the original source-worker argv from that attempt. The
   original supervisor owns package preparation and the contained compute child.

Legacy remaining/account/expiry values bound a conservative request before
owner entry, at most 120s. The unchanged original grid reserve/STAGE_OPEN guards
check actual remaining_ns after owner overhead and narrow the same absolute
fuse. The request cannot renew or replenish anything. An overrun/refusal remains
an error, not success. Runtime viability and time/resource margins are UNKNOWN.

The Adapter and originals retain stop-before-verify, finite publication before
unique known SETTLE, exact raw report save before t-star, immutable cost accounting
and physical owner exit. `save_report_bytes` uses original finite encode/atomic
write. This driver returns the Adapter mapping unchanged, preserving the actual
original supervisor result; it does not promote failures or uncertain settlement
to PASS. No substitute lifecycle/controller or custom result validator is used.

## Development evidence and handoff

`python -X utf8 qualification_preparation/dev54/selftest.py` exercises only new
stdlib code with new synthetic bindings, temporary filesystem guards and AST/JSON
inspection. It never imports/executes baseline Adapter/builder/product/native
modules, original callbacks/tests or SOL022; it never calls the artificial
baseline recipe or admitted execution body. A real Windows byte lock is tested
on an ordinary temporary marker, not a research store. Synthetic provider objects
prove mapping mechanics, not native qualification.

The local task graph is: qualified full source reads -> this DEV implementation
and synthetic checks -> frozen noreply commit/recovery bundle -> sole collector
reconciles actual lineage/delivery -> independent TEST29 and REVIEW15 in separate
contexts. Native minimal205, full207/full209/32-cell controller and original SOL022
verification each remain separately scoped future work. This DEV does not launch
those tasks, make formal PM edits, push a PR, merge, deploy or approve completion.

Historical TEST15 fractional FAIL, missing grid test, f220, full651 and all prior
FAIL/UNKNOWN results are preserved. Actual costs/grants/balances/stops remain
unmodified. Final test remains closed; pilot 1800s, per-arm training/stage 120s,
fixed Nov03 expiry, user-only COMPLETION and separate release authority remain.
