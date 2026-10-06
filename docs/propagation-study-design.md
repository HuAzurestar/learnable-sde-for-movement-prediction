# Frozen synthetic propagation design preparation

`experiments.pirc27.design.freeze_design` compiles an explicit
`PropagationStudyDesign` into immutable `PropagationStudyManifest` bytes.
This is preparation, not preregistration, authorization, numerical evaluation or
scientific qualification. It creates no ledger, grants, reservations or workers.

The design fixes models and their Gaussian initial distributions, methods and
allocations/proposals/grid/chunk/recovery settings, endpoint functional geometry,
positive horizons, seeds, origin/history cutoff, protocol/input/context/selection
hashes, primary metrics and a fixed stopping rule. Four-state SI/local Cartesian
synthetic packages are required; real, missing or differently dimensioned models
are not silently replaced. Current package implementation identities are checked.
Initial covariance requiring projection is rejected, not repaired.

Axes are immutable, unique and bounded. The candidate product must be at most
10,000 before any Cartesian expansion. A separate conservative 32 MiB manifest
quota is checked before expansion, and the final encoded size is checked again.
Each runnable method configuration is validated against its shared typed resource
contract before matrix construction, including the finest MLMC grid and total
fine/coarse work. No automatic sample allocation or experiment expansion occurs.

Comparators use the same package, initial distribution, endpoint definition,
origin/cutoff, horizon and seed. A common coupling root derives from those inputs,
not method, chunk size or allocation. This declares the pairing identity; it does
not assert identical paths across distinct solvers or independent MLMC levels.
One registered cell has one horizon. This is not first-passage inference.

Arms must be supplied explicitly for exactly the model-family/method/objective
triples. Horizon, seed, region threshold, study ID, package version and numerical
configuration never create extra budget identities. Multiple regions within one
endpoint-probability objective share the same arm. The 86,400 slot-second cap is
unchanged. This compiler neither discovers prior identities nor replaces the
shared store's cross-study identity/conflict and closed-arm checks. An operator
must reuse the existing bindings; renamed identities are not permission to reset
budgets.

The complete expected matrix retains `NOT_IMPLEMENTED` PDE rows,
`INELIGIBLE` mixture rows lacking explicit frozen settings, and
`INELIGIBLE` affine-only methods on nonlinear packages. These
have reasons, requests and immutable shared non-execution declarations, not
executable bindings. `PLANNED` means only
that the engineering adapter supports the declared input/configuration. It does
not mean scientific eligibility or a passing reference/pilot qualification.
The additive-noise reversible-Heun path candidate now has a versioned engineering
implementation and is `PLANNED` for affine and synthetic nonlinear inputs,
including chunk recovery. Its auxiliary mode/stability limitations still need
formal qualification; the compiler does not label it scientifically eligible.

The bounded mixture candidate requires an explicit immutable `MixtureSettings`
and chunk recovery. Its full per-cell policy binds request/model/current source,
component cap, merge/prune thresholds and state scales, discarded-mass/work/job
caps. Its operation proxy and candidate/live workspaces are explicitly planned.
No missing threshold is filled by a scientific default. Configured rows can be
`PLANNED`, but formal or held-out execution is refused until a dedicated mixture
qualifier exists. See [the algorithm and error boundary](bounded-mixture.md).

`manifest()` returns detached data. `study_spec(expected_hash=...)` also verifies
the current runtime source and returns **all** cells, including declared
non-executable rows. `SharedRunner` records those as `PREFLIGHT_FAILED` with their
immutable reason/error code; repetition reuses that refusal, not numerical
results. No worker, reservation or retry is permitted for such a row. Both direct
execution resolution, owner admission and direct shared budget reservation reject
a non-execution marker, including
malformed declarations or declarations combined with an executable binding.
This records a planned lack of capability, not a qualification decision or
scientific failure. Existing evidence export retains each row/reason/history.
The ordinary frozen cost report retains missing measurements as unknown; the
original budget ledger has no reservation/charge for an unlaunched refusal.

Endpoint identity/geometry, initialization and origin are explicit comparison
dimensions, so regions sharing a budget arm do not merge statistical strata or
produce duplicate block/seed identities. Horizon remains a separate dimension;
seed is still not an independent block.

Runnable cells have shared resource bindings, but the complete draft intentionally
has no `runtime_binding` or `admission`.
Operators must supply an existing absolute Git-external root/store and obtain
normal shared grants, preregistration, upstream/qualification/pilot decisions and
budget admission before execution. Supplying digest-shaped metadata alone is
not evidence of those checks, and this draft is not an approved study.
Shared runtime admission supports an explicit complete per-cell package table
for heterogeneous affine/nonlinear or restart/chunk matrices. Under
`admission.cell_packages`, schema `pirc25-cell-packages-v1` fixes `bindings` with
each executable `cell_hash` and its registered `package_hash`. Only optional
model-source `model_authorization_id`, `model_authorization_version` and
`model_protocol_id` may differ by cell. Every runnable row is covered exactly
once; unavailable, foreign, duplicate and missing rows are refused. A table
cannot coexist with a default package or default model-source references. No
fallback selection, implicit package/grant creation or permission widening occurs.

Mode, execution grant, protocol, data and upstream snapshot remain common frozen
study authority. Normal plugin/payload/qualification and budget checks still
apply. Owner receipts bind the selected cell/package, entry hash and whole table
hash; run/resume admission, result validation, verified reuse and source evidence
export check that binding. Source visibility includes all mapped packages, model
lineage and qualification artifacts, even for unexecuted cells. Legacy explicit
single-package studies remain supported. Tables and matrices are bounded to
10,000 entries, table metadata to 4 MiB, lineage to 10,000 visits and 32 levels.

The paired paper offline admission validator independently checks the same
table/selection and selected model-source references without importing runtime
code or opening research roots. The CI paired-checkout ref fixes its exact
version. Cross-repository synthetic formal-admission controls verify genuine
shared receipts and resealed substitution denials; standalone paper aggregation
remains descriptive and never creates managed computation proof. Engineering
exports do not qualify any method/model or permit research without the existing
shared store and grants.

`tests/test_propagation_study_design.py` covers immutability, deterministic content
hashing, exact owner preflight compatibility, paired inputs, stable budget arms,
cap-before-expansion, invalid metadata/configuration, retained missing methods,
source freshness and tampered resource bindings without executing research.
`tests/test_research_declared_disposition.py` checks terminal/concurrent refusal,
malformed declarations and direct owner-plan bypasses.
`tests/test_propagation_matrix_dispositions.py` exports the actual registered
matrix and exercises the unchanged paired paper aggregator: all region/method/
horizon/seed rows and reasons survive, with zero successful independent blocks.
Those disposable engineering exports are not formal research evidence.
`tests/test_research_admission_selection.py` checks exact coverage, immutable
authority, substituted selection refusal, conservative source visibility,
resume-command admission, and a two-plugin shared worker/reuse/cost/export
fixture. `tests/test_cell_admission_recovery.py` also exercises the real shared
MLMC soft-save/ACK and reopened linked restore with a frozen per-cell table,
checking original/resumed selection and original-arm accounting.
`tests/test_cell_admission_paper_chain.py` exercises the independent paper CLI
on shared synthetic formal receipts, with and without a foreign frozen-model
source. These engineering attestations are not actual research qualification.
