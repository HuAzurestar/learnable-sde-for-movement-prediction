# DEV55 concrete source host

`host.py` implements the trusted in-process admission function used by unchanged
DEV54 `execute_after_admission(repository,runtime_root,admission,argv)`.
The executable CLI exits 2. There is no CLI, JSON, manifest, boolean or function
shape that approves execution. No original/native body was run in this DEV.

## Future caller contract (separate native authorization required)

The trusted embedding control plane supplies a `Frozen` object, its actual live
`authority` service, an actual trusted owner SID and an evidence list, then calls
`future_entry`. This is a trusted Python embedding API, not an untrusted RPC.
The `authority` implementation is independently reviewed and byte-pinned by
`authority_code`, must originate in that exact file, and be included in the full
closure. A no-op implementation violates the caller contract and is not authority.
This host cannot certify the business legitimacy of an arbitrary implementation
or protect against arbitrary code already executing as its trusted process/user.

`authority.hold(request,reference=...,limits=...)` must open a live control-plane
lease for the exact action. Its context owns the existing real user authorization,
current quality/read prerequisites, permitted artificial-only scope, unchanged
cost/grant/balance state, enforceable remaining resource/stop plan and revocation.
It must raise on missing/expired/denied evidence. `lease.checkpoint(phase)` must
revalidate before preflight and immediately before native entry. The enclosing
context stays active through the driver and original physical owner exit; it
must not suppress errors. No provider/grant is installed or fabricated here.
A separately admitted caller must integrate the real existing authority service;
this source-only deliverable intentionally supplies no approving fake service.

Freeze all file tuples as `(absolute_path,size,sha256)`: all710 baseline files,
all new host sources/data, complete Python/stdlib/native-package dependency
closure, actual loaded PE/DLL images, interpreter and reviewed authority service.
Freeze exact loaded module-name/origin inventory, installed distribution metadata,
ACL reader executable identity, all runtime ancestor ACLs, source repository,
future fresh nonGit root, resource floors and decision locator. The locator and
all tuples are identity data only. Unknown values cannot be used to construct a
usable live freeze. Preload only reviewed stdlib/driver definitions in the clean
future process before its final freeze; no product/native imports are allowed
before admission. Freeze and admission must use the same unchanged process.

`future_entry` verifies and loads the pinned stdlib driver definitions, then
calls its unchanged entry. `make_admission` is also available for a host which
already loaded that exact driver. The returned object is a Python function as
required by driver54. Before yielding it enforces exact canonical request equality
(including entire outer argv/cwd/fixed expiry/native=true/science=false), process
and thread identity, one-use semantics, byte identities, clean module origins,
complete source set, loaded native image inclusion, exact distribution metadata,
fresh root, readonly Windows ancestor owner/DACL/reparse/link checks and observed
memory/disk floors. Missing or unrecognized facts reject. ACL checks use a
conservative SID/ACE allowlist; they do not modify ACLs. Only the trusted owner,
SYSTEM and Administrators may write. Source/dependency ancestors receive the
same check. DEV56 additionally checks the immediate runtime parent's inheritable
ACEs, including inherit-only OI/CI writer ACEs, before the unchanged driver mkdir.
The policy conservatively rejects untrusted inheritable writers (including
OI-only/NP combinations) without simulating a complete Windows token AccessCheck.
Each pinned file's own DACL is checked, including interpreter, ACL reader and
provider implementation; all their ancestors are checked for replacement routes.
Trusting these OS principals and the interpreter/control plane is
explicit. These observations do not lock bytes or ACLs for the future. The existing
trusted embedding/authority hold must preserve the entire closure, descriptors,
ancestor identities and root parent from freeze through physical owner exit,
including against modifications by trusted principals and outstanding write
handles. Its existing review/admission must establish that sustained protection;
this host does not certify it from a hash, single-link check or DACL snapshot.
No remote service or new authorization mechanism is required by this correction.
Unknown ACE forms and unfamiliar read masks are rejected conservatively.

All rejection/body errors append scoped evidence and propagate. Original body
errors retain identity even if authority context exit raises. The host creates
no runtime root, journal, store, grant, ledger or budget. Driver54 creates its
fresh root only after admission and owns root marker/byte lock and the unchanged
original execution. Actual nested worker argv includes an original registered
attempt ID unavailable at host preflight; it is derived by unchanged driver54
inside original constructors, never fabricated by this host. This host freezes
the entire outer invocation and all nested command-building source.

## Development verification and actual feasibility

Run only `python -B qualification_preparation/dev55/selftest.py`. These are DEV
synthetic tests using an explicitly synthetic authority/probe via private
`_admission`, never the future entry or original imports. They cover exact request
fields, lifecycle/order, denial/revocation, one-use, mutation, errors, link/root/
Git/ACL rejection, closure provider pin rejection and disabled CLI. Passing these
checks is not a native execution test or authority proof.

Actual readonly metadata evidence is in the handoff artifact. Current Anaconda
python.exe has multiple hardlinks; the strict interpreter guard rejects it.
The workspace ACL also grants write access to another SID, so this location
fails the host's trust condition. No ACLs or installed software were changed.
Observed free memory does not establish runtime headroom; time/resource margins
remain UNKNOWN. A future separately authorized execution needs a qualified
interpreter/dependency closure, owned restrictive parent and actual live lease.
Source-only delivery can be reviewed without supplying those missing live facts.

Original native imports/constructors, registration, grid/owner capability,
remaining balances, stage admission, containment/fuse/stop, publication, unique
known SETTLE and physical exit must be measured only by separately admitted
DEV30/original205 execution. No PASS follows from metadata or synthetic tests.
G1 runtime coverage, R1 unmeasured margin and H1 contract-violating trust scenario
retain their classifications; no existing business defect or waiver is asserted.
Original SOL022 exact823bytes/3controls22assertions, TEST15 fractional FAIL,
missingf220/full651/gridtest/other necessaryFAILUNKNOWN, actual costs/grants/
balances/stops, fixed Nov03 23:59HKT, finaltestclosed, pilot1800s/per-arm120s/
stage120s, user-only COMPLETION and separate release remain unchanged.

## DEV56 ACL disposition and bounded development evidence

The previous check skipped IO on existing parents, while the unchanged driver
creates children using inherited security. A synthetic OICIIO untrusted writer
was accepted on the parent yet effective in the modeled child. Existing files
also need their own DACL checked: a safe parent and unchanged byte hash do not
establish that file's protection. These are source-backed conditional gaps;
no actual runtime exploit or permissive local admission was reproduced.
Current interpreter hardlink/workspace ACL refusals remain counterevidence.

Run the same selftest command: 36 DEV synthetic methods, including prospective
inheritance variants, safe propagation, unknown flags and environment wiring for
source/interpreter/reader file DACL rejection. These are not independent QA or
native tests. TEST30/REVIEW16 still target original8b; a new candidate needs its
own separately admitted TEST31/REVIEW17. No future_entry or original body runs.

Windows semantics used (read completely for this bounded investigation):
- https://learn.microsoft.com/en-us/windows/win32/secauthz/ace-inheritance-rules
- https://learn.microsoft.com/en-us/windows/win32/fileio/file-security-and-access-rights
