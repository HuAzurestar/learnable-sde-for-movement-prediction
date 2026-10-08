# Frozen synthetic propagation design preparation

## Explicit affine cubature bridge and numerical pilot

Affine designs can select the restart-only `affine-cubature` adapter without
changing the frozen dynamics package into a nonlinear model. It uses the existing
eight equal-weight points, Euler pushforward and additive diffusion. The bounded
registry retains four SI states, an at-least-eight-row workspace and `8*steps`
point-update work. This is discrete Gaussian closure, not a continuous PDE solver.
Existing published unavailable rows are never rewritten under a new source hash.

`affine-cubature-qualification` adds a separate frozen per-request
`CubatureQualificationPolicy` and explicit pilot/qualification role. Inside the
charged shared worker, independent rational continuous-affine and finite-Euler
Gaussian certificates bound the actual functional implementation error and time
bias. Thresholds, scales, operation and job caps are fixed before execution.
Its resource proxy includes both output/replay point updates and the certifier
operation allowance; reference work is not a free compiler/preflight calculation.
Original model/method/objective arms and cumulative budgets are retained.

Failed numerical checks remain saved evidence without automatic refinement or
retry. Nonlinear packages cannot inherit the affine proof; strict PSD refusal is
unchanged. Pilot analysis is explicitly not scientific/formal qualification:
ordinary output keeps its unknown reference/model/closure error fields, and
formal or test/final-eval admission is refused before grants or protected bytes.
Dedicated owner consumption and independent paper qualification checks are still
required before formal cubature studies, as are legitimate source preparation
and immutable preregistration. No interface here grants data access or launches
an actual study automatically.

## Two-dimensional motion and file-based preparation

This study retains position and velocity in a plane: `[x, y, vx, vy]`, with
motion-space dimension 2 and state dimension 4. It does not add independent
vertical position/velocity or arbitrary state-dimension scaling. Terrain and
other conditioning features are not additional dynamical states. Only
comparisons requiring completed admissible terrain outputs wait for the
upstream terrain study; independent no-terrain affine/synthetic preparation
can continue. This scope clarification does not qualify any synthetic or real
model, admit a study or establish a scientific result.

`python -m experiments.pirc27 prepare` reloads a saved compiler manifest,
checks its expected canonical hash and current source, reconstructs its
bounded models/configurations/arms, and recompiles the entire matrix. A
matching digest alone is insufficient: every generated request, resource
binding and unavailable row must match the compiler output. Changed budgets,
state dimensions, comparison strata or test visibility are refused, even
when the edited document has been rehashed.

```console
python -m experiments.pirc27 prepare frozen-design.json --expected-hash DESIGN_HASH --runtime-config /absolute/runtime/runtime.json --expected-runtime-config-hash CONNECTION_FILE_SHA256
```

The explicitly selected connection file uses
`pirc25-local-runtime-config-v1`, includes this study's participant, and declares
the existing Git-external root and original store ID. Its exact file bytes are
hash-bound. The command reads only that configuration and existing `store.json`
identity; it does not instantiate `ResearchStore`, acquire a ledger writer lock,
create directories, register/authorize studies, check balances, read trajectory
payloads, or run numerical research. Input files are bounded opened regular
files, with duplicate/nonfinite JSON fields rejected.

Standard output is a `pirc27-prepared-study-v1` envelope containing the bound
`study_spec`, its canonical hash, design/configuration/identity hashes, explicit
2D/4-state semantics and `NOT_REGISTERED_NOT_ADMITTED`. Registration consumes
the nested `study_spec`, not this envelope. Shared grants, immutable protocol,
package qualifications, budget admission and supervised execution must still
be established using the existing shared interfaces. Connection metadata is
never a data-read grant, evidence of unused budget, or a ledger-health check.
The command does not initialize a missing store or substitute fixture stores.
No workstation-specific root or private data is committed in this example.

`tests/test_propagation_preparation.py` exercises affine/nonlinear and labelled
full-matrix roundtrips, bounded CLI output and failures, retained unavailable
rows, byte/hash/schema/dimension/identity refusals and unchanged disposable
runtime files. These are engineering controls, not formal research evidence.

## Compiler contract

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
Initial mean/covariance have their immutable four-state finite-scalar shape
checked before tensor conversion. Endpoint geometry, method count/grid/chunk
fields and all arm identity fields are checked before dataclass copying or
request-ID serialization. Nested or cyclic malformed fields are rejected as
contract errors, not recursively copied. These early checks preserve the
existing request constraints; covariance symmetry/PSD and method-specific work
checks still apply and are not replaced by structural validation.
Each runnable method configuration is validated against its shared typed resource
contract before matrix construction, including the finest MLMC grid and total
fine/coarse work. No automatic sample allocation or experiment expansion occurs.

Comparators use the same package, initial distribution, endpoint definition,
origin/cutoff, horizon and seed. A common coupling root derives from those inputs,
not method, chunk size or allocation. This declares the pairing identity; it does
not assert identical paths across distinct solvers or independent MLMC levels.
One registered cell has one horizon. This is not first-passage inference.

Multiple grid/path-count configurations of one method can be frozen together
using `StudyMethod.configuration_id`. When any method supplies that field,
every method entry must supply a valid explicit label, and each
`(method, configuration_id)` pair must be unique. Repeating the same numerical
configuration under another label is refused. There is no automatic Cartesian
grid/count search: supply the exact configurations before exposure. The bounded
method axis and 10,000-cell/32 MiB limits include all of them.

Labels are frozen comparison strata, not method-family or budget identities.
Matching labels across methods declare the intended configuration comparison;
the actual grid, allocation and proposal remain fully request-bound and need
not be identical across different algorithms. The cell's comparison dimensions
include `numerical_configuration`, so distinct configurations under one arm
cannot collapse into duplicate block/seed rows in runtime export or the
independent paper aggregate. The common coupling root remains based on shared
physical/request inputs; it does not promise nested Brownian increments between
arbitrary grids. Original family arms cover every configuration and horizon.
Legacy single-configuration designs without labels keep their original method
manifest/request-ID shape and comparison dimensions, without an inferred label.

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

Model scale and initial-law axes can likewise use `StudyModel.configuration_id`:
all model entries must be explicitly labelled when any is labelled, and each
`(family_id, configuration_id)` pair must be unique. Each entry still supplies
its actual immutable package, mean and covariance; labels do not generate or
rescale a model. Repeating the same package and numerically identical initial
distribution under a new label is refused, including int/float/signed-zero
initial-value aliases. This is not a theorem that different package hashes or
labels imply statistically independent models or different physical laws.

Rows bind the explicit `model_configuration` comparison stratum, while retaining
the original model family as `block_id` and the original model/method/objective
arm identity. Configurations, initializations and seeds are **not independent
blocks** and cannot create a new 24-hour budget. Numerical-method strata remain
separate. Unlabelled legacy manifests and request/coupling identities are
unchanged; all missing/refused rows and matrix/byte/work limits are preserved.
Current packages still have exactly four physical states; length/diffusion scale
variants or reversible auxiliary variables do not implement arbitrary physical
dimensions. No compilation initializes a research root, creates an authorization,
launches a worker or changes scientific qualification.

`tests/test_propagation_model_configuration_axis.py` checks same-family affine
and nonlinear scale/initial-distribution matrices with numerical configurations,
original block/arm identity, legacy shapes, bounded label/duplicate/cell-cap
refusals, closed-arm non-reset and actual authorized missing/refused exports
through the independent paired Paper CLI with zero successful independent blocks.

`tests/test_propagation_study_design.py` covers immutability, deterministic content
hashing, exact owner preflight compatibility, paired inputs, stable budget arms,
cap-before-expansion, invalid metadata/configuration, retained missing methods,
source freshness and tampered resource bindings without executing research.
`tests/test_study_structure_limits.py` checks primitive refusal before copying,
numeric allocation or package evaluation, including malformed later arm fields
and cyclic values using bounded engineering inputs.
`tests/test_research_declared_disposition.py` checks terminal/concurrent refusal,
malformed declarations and direct owner-plan bypasses.
`tests/test_propagation_matrix_dispositions.py` exports the actual registered
matrix and exercises the unchanged paired paper aggregator: all region/method/
horizon/seed rows and reasons survive, with zero successful independent blocks.
Those disposable engineering exports are not formal research evidence.

## Frozen source blocks and one preselected input instance

`PropagationStudyDesign.input_cases` optionally binds up to64 immutable
`StudyInputCase` declarations and one `StudyInputPolicy`. Each declaration
retains dataset/release, original source-block/content digest, block/instance
identity, exact four-state initial law and origin/history cutoff. The complete
policy binds initialization/origin procedures and the study's selection hash.
Only `one-preselected-instance-per-source-block` is currently supported: several
windows of one source are refused, including release/dataset renames caught by
the shared conservative source-identity rule. No extra independent units or
budget arms are created by seeds, configurations or instances.

With explicit cases, the case supplies each request's actual initial law and
origin (model/default design initializations remain legacy defaults only).
Comparison strata explicitly use the frozen common initialization/origin policy,
not an unannounced deletion of case identity. Dataset/release/source kind/split
remain population strata; exact input values remain in the immutable case and
request. Method coupling and request identity bind the complete case/policy.
The original model/method/objective arm and86400s cumulative cap are unchanged.

The complete model×method×functional×horizon×seed×case matrix is bounded before
copying/expansion, including a larger conservative byte allowance for cases.
Owner validation checks case/request/dimensions/selection identity; actual
shared admission additionally matches the original protocol source before grant,
package or data access. Reload reconstructs every case, policy and matrix row,
not just a rehashed declaration. Empty optional case fields are omitted for
legacy manifest/request compatibility.

These are `declared-only-not-scientific` inputs: metadata and matching hashes
do not establish independence, lawful access, causal past-only initialization,
unknown exposure history or numerical qualification. `private-past-prefix`
declarations are restricted, explicit INELIGIBLE rows until a real admitted
past-only adapter exists. They are never relabeled synthetic or executed by the
current synthetic adapter; method-specific unavailability such as PDE is retained.
`synthetic-recipe` declarations still require ordinary registration, grants,
preregistration and method-specific owner qualification before research.

## Frozen horizon-specific grids

`StudyMethod.horizon_steps` optionally fixes an immutable tuple of
`(horizon_seconds, base_steps)` pairs. It must cover the complete registered
horizon axis in that exact order; its first step count must equal `steps`.
Every count is an integer in1..8192. An empty tuple preserves the legacy
constant-grid shape and request identity; no table is synthesized or inferred.
For example, `steps=4, horizon_steps=((1.0,4),(10.0,40))` explicitly binds two
grids. This example is not a qualified timestep recommendation.

All methods, including declared unavailable rows, retain their horizon's actual
request grid. Owner validation consumes that request; MLMC uses it as the base
grid and binds the finest level in shared resource metadata. Each horizon's
configuration, level/work quota and resource binding is checked before the full
seed matrix is expanded. An admissible first horizon cannot hide an over-quota
later grid. Implicit and explicit constant tables cannot create duplicate
numerical configurations under different labels. Original family arms and24h
budgets remain unchanged; a different grid is not an independent evidence block.

Reload reconstructs the complete table and recompiles every request/resource
row. Rehashing a changed table or cell does not make it valid compiler output.
There is no automatic grid refinement, numerical qualification, registration,
worker, new grant, budget extension or alteration of a published candidate.
Horizon-appropriate stability/error qualification still has to precede research.
`tests/test_propagation_horizon_grid.py` covers affine/nonlinear owner bindings,
all runnable/unavailable methods, pairing, early malformed/later-overwork
refusals, aliases, reload integrity and closed-arm non-reset.
`tests/test_research_admission_selection.py` checks exact coverage, immutable
authority, substituted selection refusal, conservative source visibility,
resume-command admission, and a two-plugin shared worker/reuse/cost/export
fixture. `tests/test_cell_admission_recovery.py` also exercises the real shared
MLMC soft-save/ACK and reopened linked restore with a frozen per-cell table,
checking original/resumed selection and original-arm accounting.
`tests/test_cell_admission_paper_chain.py` exercises the independent paper CLI
on shared synthetic formal receipts, with and without a foreign frozen-model
source. These engineering attestations are not actual research qualification.
