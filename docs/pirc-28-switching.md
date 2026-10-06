# PIRC-28 switching research

The first-round model has K=1 or K=2, state `[x,y,vx,vy]` in SI units, and a
time-homogeneous continuous-time Markov chain. Each mode has affine acceleration
and constant velocity diffusion. Labels `state-1/state-2` have no behavior names.

`models.switching.SwitchingSDEModel` follows the existing `SDEModel` drift and
diffusion interfaces with an explicit `ModelContext.regime`. It intentionally
does not implement the joint `ExactTransitionProvider` capability. The
`conditional_affine_transition(mode, dt)` kernel is exact for a fixed-mode
segment and reuses the existing matrix-exponential covariance implementation.
The CTMC endpoint matrix alone is not an exact switching motion likelihood.

`experiments.pirc28.synthetic.ScenarioSpec` freezes the model, irregular observation
schedule, observation matrix/noise, mode-conditioned initial state distribution,
path count, seed, label map and jump quota. `generate_scenario(spec,
check_budget=...)` samples exponential dwell times and all jumps, propagating
continuous motion separately in each segment. The caller's poll must connect to
shared supervisor control in production. This module creates no budget ledger,
runtime root or implicit study authorization.

The returned `ObservationBatch` contains only times, observations, H/R and scenario
identity. Mode/state truth, jump paths and dwell statistics are separate
`SyntheticTruth` evaluator inputs. Role-specific PCG64 streams keep observation
noise from changing the latent path. Per-path seed identities support replay;
the current adapter declares `restart-only`, not unimplemented exact recovery.
Expected-jump preflight, actual jump limits, finite checks and conservative tensor
quotas refuse incomplete paths. A refused path is not a successful truncated one.

The scientific design's conditional state-path formulation is motivated by
[Michelot & Blackwell (2019)](https://doi.org/10.1111/2041-210X.13154).
Our generator and planned approximate filter are separate implementations; no
claim is made that the paper validates our inference candidate.

Run the numerical qualification fixtures with:

```console
python -m pytest tests/test_pirc28_synthetic.py -q
```

These fixtures establish generator semantics only. Identifiability, fitted
parameter recovery, predictive benefit and real-data eligibility require the
separate registered protocol, inference qualification and shared runtime gate.
Research execution must use the same explicit external runtime root/store identity
as PIRC-26/27. Do not initialize a fresh per-worktree store to refresh arm budgets.
Private or reserved data is never an input to these generator fixtures.

## Unlabeled inference and forecasting

`inference.switching.filter_switching` propagates every source/destination mode
pair on substeps with `max(-qii) * dt <= 0.05`, updates observations and moment
matches all incoming Gaussians. It retains one Gaussian per mode. The returned
objective is explicitly the approximate split-step observation likelihood.
`smooth_switching` adds moment-based backward diagnostics. Filtering posterior
hashes include only their observed prefix; smoothing is refused as a forecast
origin. K=1 reduces to the ordinary affine Kalman filter at observation times.

`estimation.switching.fit_switching` uses these moments for a bounded generalized
EM proposal and line search. Conditional CTMC bridge moments include repeat
jumps, including paths that return to their initial mode. Accepted objective
values, rejected candidates, all initializations, occupancy, collapse and stopping
reason remain available. Initialization selection uses a separate selection batch.
No approximate objective is relabeled as the exact switching likelihood.
Numerically identical mode dynamics are flagged even when occupancy is balanced;
the tolerance is frozen in the inference specification. This flag is a numerical
diagnostic, not a replacement for a registered scientific identifiability gate.

`forecast_switching` first samples a mode and then that mode's conditional state,
preserving their correlation. Future paths are event-driven; mode marginals,
at-least-one-jump probability and different-endpoint-mode probability are reported
separately. A mean trajectory is a summary, not smooth switching dynamics.

## Shared execution and recovery

`experiments.pirc28.runtime` declares the `pirc28-switching` versioned CPU adapter,
its resource plan and numerical-tolerance recovery plugin. The worker refuses
execution without shared supervisor control. It validates the frozen recipe,
source tree, store identity, resource binding and owner-created checkpoint bytes.
No new ledger or mutable arm budget is implemented by PIRC-28.
Resource counts separately bound live fit histories and all test forecast paths;
completed training histories are released after recording their diagnostics.

Numerical recovery restores accepted EM iteration/model/objective identities.
Generation, selection and forecast role streams are deterministically recomputed;
their work remains charged to the original arm in a new linked attempt. This is
not exact mid-segment Brownian recovery. A changed recipe, seed, observations,
initialization, objective or model family refuses recovery. Shared arm closure
cannot be reopened from a checkpoint.

The CLI runs only an already registered study in an existing external store:

```console
python -m experiments.pirc28 --root C:/absolute/shared/runtime --store-id ORIGINAL_ID --stage smoke --seconds 900 run REGISTERED_STUDY
python -m experiments.pirc28 --root C:/absolute/shared/runtime --store-id ORIGINAL_ID --stage pilot --seconds 1800 resume ATTEMPT_ID CHECKPOINT_ID --authorization-id EXISTING_GRANT
```

These are placeholders, not a configured production root or a grant. Registration,
frozen scientific thresholds, qualification and source/data access evidence must
precede a research run. The CLI has no `init` command or unbudgeted fit route.
Production synthetic admission binds the canonical recipe's physical content hash
in a `synthetic-pirc28` source protocol. It does not permit reading a real data
source through this synthetic adapter. Formal metric names and definitions must
match the independently frozen comparison plan. A tiny fixture source protocol is
accepted only with explicit fixture admission and produces fixture qualification.

The frozen recipe separates generated train/selection/test streams and matches
K=1/K=2 test blocks and seeds. Truth is only used by the synthetic evaluator.
Mode Brier diagnostics are permutation-invariant; K=1 does not fabricate a
two-mode score. Results retain all initialization diagnostics and shared attempt
lineage. The synthetic pipeline does not claim identifiability qualification.

```console
python -m pytest tests/test_pirc28_synthetic.py tests/test_pirc28_inference.py tests/test_pirc28_runtime.py -q
```

## Synthetic identifiability qualification

`experiments.pirc28.qualification.phase_scenarios` expands an explicitly supplied,
bounded Cartesian matrix over mode separation, mean stationary sojourn time,
observation interval, position noise standard deviation, trajectory length and
stationary mode occupancy. It does not run simulations or create a ledger.
Separation scales affine drift contrasts; `dwell_time_s` is not the CTMC
relaxation time. Each generated scenario preserves its actual schedule/model
hash, including the six axis values. Oversized matrices and invalid axes refuse
before numerical execution.

`IdentifiabilitySpec` freezes independent replicate IDs, all unit-specific
parameter scales/recovery thresholds, occupancy and collapse criteria, effective
replicate minimum, recovery fraction, calibration bins, marginal coverage level,
bootstrap seed/draw count, selection-only initialization choice and shared-arm
stop rule. Unfilled thresholds are `NEEDS_PREREGISTRATION`, not an implicit pass.
No default scientific success thresholds are chosen from observed results.

Explicit scenario roles use `pirc28-identifiability-spec-v2` and are part of the
same immutable recipe/definition/preregistration hashes. They cover exactly
every registered scenario, with at least one `required-qualified` point. Legacy
v1 definitions round-trip unchanged and still require every scenario to pass;
they are not silently upgraded or assigned new roles after outcomes.

| Preregistered role | Scope requirement |
| --- | --- |
| required-qualified | This point must pass recovery/collapse/calibration qualification. |
| required-unidentifiable | Complete successful paired fits and complete metrics must yield UNIDENTIFIABLE. Runtime failures are not successful negative controls. |
| phase-diagram-only | Retain the measured terminal outcome without claiming it is an applicable range. Missing cells and deadline/budget failures still block the aggregate. |

The v2 gate preserves every scenario's axes, state, replicate denominators and
failure/collapse/metric disclosure. A passed required range plus expected
negative controls can qualify even with unidentifiable descriptive points.
An unexpectedly passing negative control is CONTROL_GATE_FAILED; failures,
inapplicability or insufficient complete pairs cannot establish that control.
Only an overall PASSED gate emits `applicable_scenarios`/`applicable_ranges`,
and these contain exact preregistered required points, not an interpolated or
Cartesian bounding box. `passed_required_scenarios` remains visible when a
control or another prerequisite fails. An exploratory success cannot substitute
for a failing required point. Scope changes require a new pre-fit frozen
definition/preregistration; they cannot relabel existing evidence into a pass.

An optional canonical `identifiability_spec` inside the frozen synthetic recipe
adds evaluator-only diagnostics to each result. Parameter matching enumerates
the two label maps with unit-specific normalized errors; Q is relabeled on both
axes. Diffusion recovery compares `L L^T`, since a noise rotation changes L but
not its model law. Scenario truth label maps are applied consistently to model,
mode probabilities and continuous dwell. Parameter/mode recovery is explicitly
unavailable for K=1 versus K=2 truth, not reported as artificial zero error.

Mode Brier/ECE and continuous-dwell occupancy error accompany coordinate-marginal
equal-tail forecast coverage. The latter is not a joint credible-region claim.
Two binary proper scores separately evaluate at least one actual future jump
and a different endpoint mode; evaluator jump paths include return jumps. All
such statistics are within one independently generated replicate. Paths, points,
horizons and initialization seeds within a cell are not extra independent units.

`qualify(spec, rows, aggregate_hash, check_budget=...)` requires unique registered
(scenario, independent replicate, K) rows, matched K=1/K=2 input hashes, and
failure-inclusive denominators. One input block cannot be counted under several
replicate IDs. Results distinguish missing/incomplete, budget exhausted,
unidentifiable and passing scenarios, and retain their applicable axis ranges.
Missing/nonfinite metrics or label/identity conflicts cannot pass. Recovery
fractions use Wilson binomial intervals, including the all-success boundary;
successful-fit metric means and paired energy differences use a bounded
replicate-level percentile bootstrap. Those conditional intervals are not used
to hide failed fits. Polls occur between bounded bootstrap batches.
Deadline failures remain TIMEOUT in the raw rows/cost lineage but make their
scenario gate BUDGET_EXHAUSTED, including legacy all-required definitions; they
cannot be tolerated as ordinary fit failures when enough other replicates pass.

K=2 collapse and successful-fit diagnostic intervals retain observed K=2 outcomes
even if their K=1 reference fails. Paired scores and the recovery gate still
require successful pairs. Observed collapse is conditional on successful K=2
fits (unavailable when there are none); its separate registered-denominator
lower bound and unknown replicate IDs prevent missing results from becoming
observed zeros. The collapse threshold retains the all-registered-replicate
denominator; it is not silently retuned to the successful-fit denominator.

Raw evaluator rows and a passing mathematical report never authorize real-data
access. Their source, immutable artifacts, attempts, admission and qualification
bindings must still be validated through the shared store. Neither cell outputs
nor this module promote `qualified_identifiability`; the pipeline retains false.
No scientific matrix has been executed merely by passing these software tests:

```console
python -m pytest tests/test_pirc28_qualification.py -q
```

## Shared-store qualification evidence

`experiments.pirc28.evidence.collect_qualification` consumes existing registered
studies, an exact `IdentifiabilitySpec`, evaluate-purpose grants and a shared
budget poll. It creates no store, grant, run, reservation or fresh arm identity.
It uses existing authorized artifact reads and verified-success reuse, including
the original source/attempt/admission/worker/cost provenance. The shared APIs
append their usual disclosure/read/reuse audit receipts. Production interval
computation must be called under shared supervisor control; this library is not
an unbudgeted research CLI.

Scientific studies additionally bind a `phase_matrix_hash` pointing to the
canonical `pirc28-phase-matrix-v1` produced by `phase_manifest(base, axes)`.
The collector reconstructs the scenarios from the actual frozen baseline and
requires the qualified axis ranges and physical model/schedule/noise to match.
Only assigning arbitrary axis labels to a recipe cannot establish qualification.
Each cell freezes `replicate_id`, `fit_K` and the paired seed. All registered
scenarios/replicates/K values must be covered; independent seeds cannot be reused
as additional replicates. The two fixed K arm identities remain cumulative
across scenarios and studies, and cannot refresh their 24-hour balances.

The source protocol physically hashes the canonical recipe as `synthetic-pirc28`.
The pre-fit preregistration freezes `pirc28_identifiability_spec_hash`,
`pirc28_phase_matrix_hash`, the actual K=1/K=2 proper-score comparison and
`result_selection_rule=one-row-per-independent-replicate; immutable-success-else-latest-attempt`.
Plans published after any source fit was admitted cannot qualify it. Synthetic
pilot admission can establish this prerequisite gate; requiring a previously
qualified formal blind-test study would create a circular gate. Formal studies
retain all the existing blind-test admission requirements.

An immutable successful outcome is one row; otherwise the latest attempt gives
the row's terminal state. All retry histories and charged/outstanding costs stay
in the lineage, including failures before a later success. Live or unstarted
cells remain uncompleted; arm closure without a run is budget exhaustion, not a
successful zero. One pass indexes source events/attempts instead of comparing
every trajectory with every other trajectory. Per-result, total-input and final
aggregate byte quotas refuse oversized collections rather than truncate rows.
Each scoped source-run index hashes a complete immutable study once and reuses
that hash only for its own cell identities. Snapshot/run hashes are unchanged;
there is no cross-call permission, ledger or outcome cache.

Grant checks also protect empty/failure-only metadata, and are repeated after
extra source integrity and bootstrap work before disclosure. Complete source
visibility/block scopes must be authorized. Fixture collection is opt-in and
always `FIXTURE_ONLY`, regardless of a mathematical pass. A verified scientific
synthetic gate may set `qualified_identifiability`, but `real_data_authorized`
remains false: no result or aggregate grants access to private or reserved data.

```console
python -m pytest tests/test_pirc28_evidence.py -q
```

These tests use disposable stores and explicit synthetic operator attestations,
including a managed pilot-mode control. They are software integration checks,
not the production phase diagram, a crash-at-any-instruction proof or a claimed
accepted scientific result.

## Numerical initialization failures

Registered initializers are attempted in their frozen order. Numerical fit and
selection failures (`INFERENCE_INAPPLICABLE`, `NONFINITE`, or bounded numerical
library errors) remain explicit diagnostics; failed candidates have null, never
zero, selection scores. Only finite successful selection objectives can win,
with the same stable tie rule. Contract, identity, authorization, supervisor
budget/deadline and unknown errors are not downgraded into numerical failures.
Successful cell artifacts retain every declared fit/selection candidate, and
the shared collector verifies this roster and winner before exposing concise
initializer diagnostics in cost lineage. Initializers are not new replicates.

Pipeline checkpoint v2 retains failed initializers and their last accepted EM
state. Recovery validates model, recipe/seed, observation, prior, inference and
accepted-progress identities before skipping a retained failed initializer; it
does not reset its initialization or iteration budget. Legacy v1 checkpoint
shape remains readable, subject to the existing shared source/receipt binding.
All-initializer failures raise a typed observation-only NumericalCellFailure
with the complete history, not fabricated successful metrics. Whole-cell
generation, test inference, forecast and evaluation failures retain their stage
and bounded numerical cause; all-selection failures retain successful fits and
null failed selection scores.

The switching adapter opts into a separate managed failure validator. Its worker
writes a bounded canonical `pirc25-failure-v1` result and exits 86; this is never
a successful result or an unbudgeted side-channel artifact. The supervisor stops
and confirms the whole process tree before validating recipe, generated streams,
initializer history and ordinary admission bindings. Only then may it publish a
role=`failure` artifact on the original study/block and attach its manifest hash
to a FAILED attempt. Legacy adapters without this validator do not interpret
exit 86 as typed evidence. Deadline, checkpoint, containment and arm-fuse authority
still take precedence; even delayed owner validation cannot publish after timeout.

Failure provenance verification checks immutable admission/code, artifact bytes,
actual worker launch/tree stop and original-arm settlement. It explicitly does
not grant successful reuse. The collector reads failed artifacts under existing
evaluate-purpose grants and cumulative byte quotas, retaining each diagnostic
and retry cost even before a later success. Failed rows have no fabricated metrics,
occupancy or collapse observations; initializations and retries remain one
independent cell. Retrying requires an explicit parent and reason and cannot
refresh an arm balance. All tests use disposable stores, not scientific evidence.

## Immutable scientific study definitions

`experiments.pirc28.study.SyntheticStudyDefinition` owns canonical immutable
bytes, not externally mutable nested dictionaries. It freezes the reconstructible
physical phase matrix, numeric v2 qualification/roles, one recipe per scenario,
globally independent replicate seeds, distinct K=2 initializations, actual EM
iterations, common physical forecast origin/horizons and runtime code identity.
K=1/K=2 share the same source seed for a replicate. Recipe indices must implement
the declared physical horizons across observation intervals. Every cell/recipe
gets the existing bounded execution declaration before registration.

Identical-mode controls are checked against actual A, B, a and L L^T; zero drift
separation alone is insufficient when mode diffusion covariances differ. A
required-qualified point cannot have identical observation laws. At least one
true identical-mode control is required. Missing numeric thresholds, duplicated
source seeds/initializations, changed physical recipes/horizons and new method
family aliases are refused. Arm IDs are explicit original bindings, while
algorithm/objective/model families and the 24-hour arm cap remain fixed across
all scenarios, studies, initializations and retries.

`stage_definition(store, definition)` publishes only phase/spec/definition
metadata into an existing shared store before any source attempt. The CLI has
an equivalent `stage-definition PATH` command. Neither path initializes a store,
registers studies, creates grants or reserves budget. `proposals(binding)` returns
detached, unregistered source metadata; it deliberately lacks a pirc25 schema,
protocol/data binding and admission, so it is not an executable study.

An operator supplies actual source protocols, packages, authorizations and
external-history/upstream attestations through the existing shared workflow.
`preregistration_fields(protocols)` binds the entire definition hash, all source
study IDs, canonical synthetic recipe bytes, replicate blocks and paired primary
score comparison; operator evidence is not fabricated by this helper.
`register_definition_studies(store, definition, admissions)` then validates the
complete original pilot/fit input matrix and registered preregistrations before
registering source studies. It creates no attempts, grants or new arm families.
Ordinary SharedRunner admission still validates actual package/source/access and
budget authority before any numerical execution.

The shared collector verifies the complete definition against every frozen
source recipe, seed, initialization, arm and resource binding, and verifies its
publication precedes source admissions. An unbound mathematical PASSED report
becomes NEEDS_PREREGISTRATION with missing pirc28_study_definition_hash, not
scientific qualification. Incomplete legacy diagnostic/pilot reports remain
observable without being upgraded. No report grants real-data access.

The independent supervisor output validator repeats the bounded master-metadata
checks against its frozen, currently authorized source studies: the report's
`study_definition_hash` and strictly boolean `study_definition_bound` must match
the actual complete definition, source projections and publication order. A
self-consistent worker report/hash cannot claim scientific qualification without
an actual master binding. This check also runs on successful-result reuse before
publication; it neither fits models nor reruns numerical aggregation outside the
managed budget. The final disclosure-expiry check remains the last valid-path I/O.

```console
python -m experiments.pirc28 --root SHARED_ROOT --store-id ORIGINAL_STORE_ID stage-definition definition.json
python -m pytest tests/test_pirc28_study.py -q
```

Temporary tests contain explicit synthetic operator attestations, including one
actual managed source fit. They are not scientific independent replicates or
the canonical production matrix. No numeric scientific defaults are inferred
from those fixtures.

## Supervisor-budgeted qualification computation

`experiments.pirc28.computation.QualificationRunner` runs collection and bounded
bootstrap computation in the existing shared OS supervisor. It copies the
original K=2 arm verbatim into a derived computation study: no new method family,
24-hour balance, motion capability, store or data grant is introduced. Stage
budgets must match the immutable source stage (fixture/smoke, pilot/pilot,
formal/job). The canonical definition and the complete source study, attempt,
code, authorization and resource-plan identities are frozen before the job.
Concurrent source outcome changes reject collection/publication as INDEX_STALE;
unrelated computation attempts do not invalidate the scoped source snapshot.

At 80% the worker can save a seed-deterministic full-restart marker. This is not
partial interval recovery. An explicit failed-parent retry and reason restart
the same frozen input and retain all charges on the original arm. The native
deadline/tree stop and cumulative-arm fuse remain shared supervisor authority.
Source-read bytes, output/request bytes, bootstrap elements and batch size are
bounded; these allocation declarations are not a claim of a hard RSS limit.

Success produces an immutable result artifact and
`pirc28-qualification-computation-receipt-v1` binding request/report/source hashes,
actual reserve/launch/tree-stop/settlement provenance and charged slot-ms with
the original arm ID. Receipt publication can resume after a crash without
another worker reservation. Successful reuse revalidates source attempts,
current grants, code, artifact and cost provenance; it does not rerun bootstrap.
Current source visibility and block scopes are checked before disclosure and
after publication. No external file export or authorization is created.
Reports retain source-arm balances observed during computation, which may
include this job's outstanding reservation; the final receipt records its
actual settlement, not that temporary reserved amount.

Only definitions already published as `pirc28-identifiability-spec-HASH` and
existing evaluate-purpose grants can be used by the CLI:

```console
python -m experiments.pirc28 --root SHARED_ROOT --store-id STORE_ID --stage pilot --seconds 1800 qualify DEFINITION_HASH --study-grant STUDY_ID=AUTHORIZATION_ID
python -m pytest tests/test_pirc28_computation.py -q
```

Fixture use additionally requires `--allow-fixture` and a smoke budget; it can
only publish FIXTURE_ONLY. Even a synthetic scientific pass does not grant
real-data access or replace the remaining preregistration, applicable-range,
study execution, paper, review and acceptance obligations.

## Explicit six-axis scientific plan

`experiments/pirc28/plans/identifiability-v1.json` is the first explicit numeric
scientific plan, not a software fixture, observed result or operator approval.
It is prepared before scientific fitting; no thresholds or ranges were chosen
from a production recovery result. Its source-bound compiler reads at most
64 KiB of unique-key finite JSON and expands the complete immutable definition.
Every inference, sampling, initialization, threshold, scale and scope field is
explicit; missing fields are refused instead of taking fixture defaults.

| Axis | Frozen levels |
| --- | --- |
| Affine drift-contrast multiplier | 0, 2 |
| Stationary mean sojourn (s) | 0.5, 2 |
| Position observation interval (s) | 0.25, 0.5 |
| Isotropic position-noise standard deviation (m) | 0.1, 0.5 |
| Trajectory length (s) | 8, 16 |
| Stationary synthetic mode-0 occupancy | 0.2, 0.5 |

All 64 Cartesian points have 16 independent source replicates and paired K=1/K=2
cells: 2,048 cells, not 2,048 independent replicates for one point. Each replicate
has 3 training, 2 selection and 8 test paths; within-path points, paths in a cell,
64 forecast samples and the two K=2 initializations are not CI sampling units.
Seeds derive from the complete plan hash, physical scenario and replicate ID,
with collision refusal rather than silent reseeding. K arms share source streams;
train/selection/test retain their existing independent role streams. Initializers
are fixed across every scenario and never adapted to generated truth.

The required qualification points are contrast 2, interval 0.25 s, noise 0.1 m,
length 16 s and occupancy 0.5, at both 0.5 s and 2 s mean sojourns. All 32 zero-
contrast points are required-unidentifiable controls. They have identical A, B,
a and diffusion covariance plus the same mode-conditional initial law; zero
drift contrast alone would not establish that control. The other 30 points are
phase-diagram-only, retained regardless of outcome. These are exact predeclared
points, not a post-fit selector or an interpolated applicable box. Every control
requires complete successful pairs: numerical failure cannot establish it.

Physical truth has common A=-0.04 I s^-2, B=-0.2 I s^-1 and L=0.2 I m s^-3/2;
baseline a=(-0.8,0.4)/(0.8,-0.4) m s^-2 is scaled only by the contrast axis.
The initial state has zero mean and covariance diag(0.25,0.25,1,1) in its
coordinate units. Position alone is observed; velocity and mode truth remain
evaluator-only. All fitted models use the same [0,5] s^-1 rate family. The plan
declares 12 GEM iterations, two distinct K=2 initializers, max step 0.125 s,
rate-step product <=0.05 and 1,600 substeps. The compiler checks the quota at the
maximum allowed fitted rate on every physical grid, not just at initial rates.
All cells fit the existing 20-million-element/512-MiB declaration limits. This
does not claim an RSS sandbox or guaranteed elapsed time. Shared per-job and
24-hour original-arm caps still govern all fits, scenarios, retries and aggregates.

| Recovery/calibration quantity | Maximum per-replicate value |
| --- | --- |
| Q RMSE (s^-1) | 0.25 |
| A RMSE (s^-2) | 0.01 |
| B RMSE (s^-1) | 0.05 |
| a RMSE (m s^-2) | 0.3 |
| L L^T RMSE (m^2 s^-3) | 0.01 |
| Two-component mode Brier | 0.3 |
| Mode ECE / continuous-dwell occupancy error | 0.1 / 0.1 |
| Absolute coordinate-marginal coverage error, nominal 0.9 | 0.1 |
| Any-jump / different-endpoint binary Brier | 0.3 / 0.3 |

These are operational scientific criteria, not universal identifiability
constants. A pre-data analytical review tightened the initial A/B/Q/covariance
criteria: zero A would have RMSE 0.0282843 s^-2, zero B 0.1414214 s^-1,
zero covariance 0.0282843 m^2 s^-3 and zero Q at the slow required point
0.5 s^-1. The initial thresholds admitted those individual null estimates.
No scientific fitting or recovery outcomes informed the correction. The compiler
now refuses any required-point nonzero parameter threshold at or above its
zero-estimate error. Truly zero generating coefficients are valid null truths
and are not refused. Parameter matching scales are unchanged: Q=1 s^-1, A=0.04 s^-2,
B=0.2 s^-1, a=1 m s^-2 and diffusion covariance=0.04 m^2 s^-3; it compares
L L^T, not the noise-rotation-dependent L. Two-label matching is unchanged.
A balanced uninformative two-component mode forecast has Brier risk 0.5.
For a future Bernoulli event with probability p, even an oracle forecast has
expected Brier p(1-p), up to 0.25. The 0.3 event gates therefore allow inherent
future randomness; they must not be misreported as improvement over a constant
0.5 binary forecast. Rate recovery and the remaining calibration gates are
separate requirements. Coverage is coordinate-marginal, never joint-region.

Each point needs at least 14 effective paired replicates, minimum posterior
occupancy 0.05 and maximum registered-denominator collapse fraction 0.125.
The 95% Wilson lower bound on the failure-inclusive joint recovery/calibration
fraction must be >=0.7: with 16 registered replicates, 16 or 15 successes can
meet that bound, while 14 cannot. The compiler refuses a count/confidence/gate
combination that cannot pass even with all successes. Mean-metric and paired
energy intervals use 2,000 bounded bootstrap draws, seed 2026100701, sampling
independent replicates. These are pointwise intervals, not simultaneous
64-point coverage or a familywise error-rate claim. No negative/unidentifiable
outcome is hidden, and no failure is assigned fabricated metrics.

Read-only preparation requires no runtime or grants:

```console
python -m experiments.pirc28.plan experiments/pirc28/plans/identifiability-v1.json
```

The summary reports definition/plan/current-source hashes, role/count/byte
checks and an analytical all-success Wilson bound. It always reports zero
scientific groups executed, prepared_only=true, registered_studies=false and
qualified_identifiability=real_data_authorized=false. It is not data exposure,
scientific qualification or frozen operator preregistration in the shared store.

After the canonical root and original store identity are supplied, metadata can
be staged explicitly, without creating a store, study, grant or reservation:

```console
python -m experiments.pirc28 --root SHARED_ROOT --store-id ORIGINAL_STORE_ID stage-plan experiments/pirc28/plans/identifiability-v1.json
python -m pytest tests/test_pirc28_plan.py -q
```

Source-bound definitions change when runtime source changes. Stage the final
tested source before source admission and fitting; a later code/plan change
requires its own explicit preregistration and does not reset original arm costs.
Actual operator protocols, packages, grants and pre-fit full-master publication
remain mandatory. This plan does not replace known-state/K=1 numerical controls,
scientific recovery, approximation/calibration verification, qualified real-data
comparison, paper, independent review, human acceptance or merged delivery.

## Frozen approximation contract and current execution stop

The numeric plan now uses `pirc28-scientific-plan-v2`, master definition v2 and
recipe v2. The filename remains `identifiability-v1.json` for the same research
plan; its schema and content hashes change. Every recipe carries the complete
same `pirc28-approximation-spec-v1` as its master, transitively binding protocol
bytes, preregistration/master hash, source hash, seeds and admitted recipe. An
old schema cannot silently drop this requirement. Legacy v1 recipes remain
usable software diagnostics, never sufficient scientific approximation proof.

All registered K=2 source cells must check the selection-only chosen fit on
every held-out path's observation prefix through the frozen 2-second forecast
origin. Future observations, true modes and true velocities are not reference
filter inputs. Within-prefix points, paths, particles, batches and initializers
do not become independent scientific replicates or tune the selected fit.

| Error (maximum over the registered prefix paths/observation times) | Step-halving limit | Event-driven reference limit |
| --- | --- | --- |
| Mode posterior total variation | 0.02 | 0.1 |
| Marginal position-mean maximum absolute difference (m) | 0.02 | 0.1 |
| Marginal velocity-mean maximum absolute difference (m/s) | 0.05 | 0.2 |
| Absolute log-likelihood difference per observation | 0.01 | 0.05 |

These are pre-data operational accuracy tolerances, not established error
bounds or universal identification constants. Step halving reduces BOTH
`max_step_seconds` and `max_rate_step` by two; changing the time limit alone
can leave the active rate-controlled grid unchanged. The fine prefix quota is
400 substeps, checked at the whole permitted [0,5] s^-1 fitted rate family on
every physical grid. Only the 2-second prefix is checked here, not silently the
full 16-second trajectory. The fine specification retains other fit policies.
Coarse/fine agreement alone does not validate moment matching or true likelihood.

The registered reference candidate is a conditional-affine, event-driven
bootstrap particle filter with 2048 particles, eight independent MC batches,
at most 32 jumps per particle/observation interval and minimum ESS fraction
0.1. Its confidence setting is 0.95; MC half-widths must be at most one quarter
of the corresponding reference tolerance. Insufficient MC precision must fail,
not cause adaptive extra batches, particle counts, selective batch deletion or
unregistered retries. Conditional path propagation is not an exact joint
switching likelihood or an exact posterior. The future implementation must
specify and validate the MC uncertainty calculation and compare prefix marginal
posteriors/likelihoods, not just unconditioned forward moments. The known-mode
affine K=1/Kalman boundary has absolute and relative tolerances 1e-8. All required
checks and failures must remain disclosed; failure state is
`INFERENCE_INAPPLICABLE`, never a fabricated successful diagnostic.

This is a pre-data design revision, not an outcome-conditioned change. The prior
512-particle/four-batch candidate had a balanced-IID population-SD nominal
Student half-width benchmark 0.0351614, exceeding its unchanged 0.025 mode
precision limit. Eight batches of 2048 particles give a benchmark below 0.01.
This necessary sanity check does NOT establish actual weighted/resampled batch
variance, bias, simultaneous coverage or later-observation precision. The
compiler now refuses a reference whose balanced-IID benchmark already exceeds
its precision gate; every actual batch still must pass the frozen checks.

The reduced jump quota keeps worst-case work within the SAME 50-million-segment
core ceiling while increasing independent reference samples. At the registered
rate upper bound 5 s^-1 and interval at most 0.5 s, uniformisation dominates
actual K=2 event counts by Poisson(2.5). For quota J, the pre-data Chernoff bound
on exceeding it uses m=J+1 and exp(-lambda+m*(1+log(lambda)-log(m))) when
m>lambda, otherwise 1. Summing over every prefix interval, particle, batch,
held-out path and registered K=2 cell gives a union bound without assuming
independence. This is an application of the
[Poisson Chernoff derivation](https://www.cs.cmu.edu/~harchol/Probability/slides/Chpt18.pdf),
not an empirical failure-rate estimate or permission to truncate paths. An
actual quota exceedance still fails and remains disclosed. Positive-rate tail
underflow is floored rather than falsely reported as exact zero; only an
absorbing rate has zero event risk. Schedule-only preparation reports these
design/work bounds without generating data, fitting, admitting resources or
creating a store. Plan changes alter full-master/seed identities and require
new pre-fit registration, never a new original-arm budget identity.

CURRENT LIMITATION: the immutable contract, fine-grid metadata preflight,
fail-closed publication boundary and bounded numerical core below are implemented.
The managed numerical executor, its registered allocation plan and independent
evidence/provenance adapter are NOT implemented.
V2 execution therefore refuses with `NEEDS_IMPLEMENTATION` before generation
or fitting. This software gap is not published as a numerical-failure artifact.
Preparation/staging is metadata only and admits no reference computation or
additional memory. Existing 20-million-element/512-MiB caps remain unchanged.
The future diagnostic executor must charge all work, checkpoints, retries and
publication to the original K=2 arm under the same shared supervisor/store,
without creating a reference method/arm or extra budget authority.

Every shared qualification now discloses `approximation_qualification` as
pending and `qualified_identifiability=false`. An otherwise mathematical
`PASSED` production report becomes `NEEDS_APPROXIMATION_QUALIFICATION`, with
no applicable ranges. Incomplete, exhausted, unidentifiable and control-failed
conclusions retain their actual states. Fixture reports stay `FIXTURE_ONLY`.
The owner independently derives the pending contract from actual source
recipes; neither a forged passing worker status nor a recomputed report hash
can promote it. Implementing a numerical checker alone will not lift this
stop: actual managed cost, source, input, selected-fit and approximation evidence
must be independently bound before scientific qualification is enabled.

## Observation-prefix numerical core (not an execution authorization)

### Registered additive allocation envelope

Plugin version 1.1.0 retains the 24 EM tensors, original forecast workspace and
unchanged shared hard limits. `recipe_allocation` adds a conservative live pool
for generated/retained observations, synthetic states/modes/jump/dwell buffers,
bounded fit/selection/checkpoint numeric metadata, retained per-path forecasts
and their jump/mode histories/copies, and the complete K=2 reference core plan.
No phase reuse or early disposal is assumed to avoid accounting for simultaneous
allocations. The K=1 arm and legacy recipes reserve source/forecast storage but
do not invent a reference run or arm. Reference work and coarse/fine quotas are
preflighted at every allowed fitted rate, not a favorable initializer.

The real observation count remains the required `inputs.observations` field.
The separate required `inputs.workspace_vectors` rounds the added pool upward
to four-float64 storage vectors. The unchanged shared axis vocabulary uses
its `observations` count to index this explicitly named storage pool; this count
is not claimed to be the real observation count. It does not multiply large
particle or forecast counts into each EM history, redefine Gaussian modes, add
resource dimensions, relax quotas or create new original-method budget keys.
The generic registered planner sums ALL declared tensors and rejects combined
over-allocation even when each individual tensor/core would fit. The actual
worker independently derives both input fields/config from the frozen recipe
before opening its control channel or doing numerical work; recomputing a forged
generic resource hash does not bypass this comparison.

All 128 K/scenario recipe families in the full reference master fit the additive
declaration: worst combined allocation is 19,698,300 float64-equivalent numeric
slots within the unchanged 20,000,000 ceiling. This is a declared tensor/storage
bound, not observed process RSS, complete Python/interpreter heap accounting or
an execution admission. Budget/cost/timeouts, source/input/selected-fit/seed
provenance, independent numerical evidence and actual accuracy still need their
managed path. V2 source execution remains `NEEDS_IMPLEMENTATION`. New component
and whole-source identities require refreshed final-source pre-fit registration,
not a new store, grant, original-arm budget identity or successful-artifact reuse.

`experiments.pirc28.approximation_checks.check_prefix_approximation` now computes
the checks from observation-only prefixes, a fixed selected-model input, the
same filtering prior, frozen inference/approximation specifications, an explicit
reference seed and caller-supplied budget poll. It never fits or selects a model,
opens a store, reserves budget, creates an arm, accesses truth or reads future
observations. Every held-out prefix path is retained. Its worst-rate grid,
conservative sequential-allocation and worst-case event-segment preflight refuse
oversized work; this is a core plan, NOT shared resource admission or accounting
for concurrent source-fit/forecast allocations. The future adapter must register
and verify the whole live allocation/work plan under the existing caps.

The reference conditions the finite initial Gaussian mode mixture analytically:
each positive-prior component receives its conditional Kalman update, exact
initial evidence normalizes prior times likelihood, and initial mode/state-mean
summaries use the full posterior mixture. Zero-prior components are not updated.
It then draws IID initial modes from that posterior, not the unconditioned prior;
their Gaussian states already include the initial observation. It does not
resample these fresh uniform-weight particles before the first propagation.
Subsequent observation-weighted particles use multinomial resampling before
the next interval. Analytic initial conditioning removes unnecessary initial
sampling noise but cannot guarantee later precision. This initialization is
also consistent with the importance-distribution derivation in
[Doucet, Godsill and Andrieu, 2000](https://www.stats.ox.ac.uk/~doucet/doucet_godsill_andrieu_sequentialmontecarloforbayesfiltering.pdf).

Autonomous CTMC jump paths contain zero or multiple events per interval.
Each particle carries a Gaussian
continuous mean/covariance: fixed-mode affine kernels compose over actual jump
segments, and a direct conditional Kalman update supplies observation weights.
It uses independent initial/event/resampling streams per seed/path/batch and
the resampling schedule above. It keeps compact prefix summaries,
not all jump/particle histories. The conditional continuous marginalisation
follows the structure described by [Doucet et al., UAI 2000](https://arxiv.org/abs/1301.3853);
this particular CTMC implementation is ours, not a claim that the paper supplies
this code or establishes the chosen tolerances. It is not endpoint-only affine
propagation, drift averaging, a true-mode input or a smoothing reference.

All frozen independent batches execute exactly once, without deleting failed
batches or adding particles/MC runs when precision is poor. Batch-mean Student-t
half-widths are nominal pointwise normal-approximation diagnostics, not exact
finite-sample or simultaneous coverage and not bounds on finite-particle
self-normalisation/posterior bias. The integer-df critical value uses bounded
numerical inversion of the [Student density](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3664.htm),
without a new SciPy dependency. The likelihood comparison logs the mean of
positive likelihood estimates using stable rescaling, rather than averaging
their logs. A nonpositive likelihood interval lower bound fails precision; it
does not become a zero error. The transformed log estimate still has finite-MC
bias. The distinction between likelihood estimators and log-noise assumptions is
also explicit in [Doucet et al., Biometrika 2015](https://www.stats.ox.ac.uk/~doucet/DoucetPittDeligiannidisKohn_efficientimplementationMCMCunbiasedlikelihoodestimator.pdf).
None of these papers proves our core's scientific accuracy or MC bias bound.

Checks compare the coarse filter to the fine filter and to reference prefix
mode/marginal-state means and per-observation log likelihood. Reference errors
plus their nominal half-widths must meet the frozen discrepancy limits; widths
must separately meet the quarter-tolerance precision rule, and every batch must
meet ESS. These are not full-distribution distances or a covariance error bound.
Every model mode additionally becomes a fixed-mode K=1 control whose means,
covariances and likelihood are checked against a direct full-interval Kalman
implementation using Cholesky scoring/covariance subtraction rather than the
IMM measurement update's slogdet/Joseph implementation. Kernel analytic controls
remain separate software evidence.

The result records spec/model/prior/inference/prefix hashes, seed, path identities,
all error/MC/ESS/jump/boundary diagnostics and resource estimates. Compact
coarse/fine/reference observation-time summaries and the complete indexed batch
summary/seed roster are included, so a future verifier does not have to trust
a hash of unpublished numbers or a claimed MC width without its batch inputs.
This is still not independently verified managed evidence. Any numerical
gate failure remains `INFERENCE_INAPPLICABLE`; timeouts and other caller budget
errors propagate without retry or a partial pass. `NUMERICAL_CHECKS_PASSED` means
only these core comparisons met their limits, never scientific qualification:
`managed_evidence_bound=qualified_identifiability=real_data_authorized=false`.
The V2 source worker still refuses `NEEDS_IMPLEMENTATION`; independently verified
managed selected-fit/input/source/cost/authority evidence and actual scientific
recovery/calibration are still outstanding. Code-source changes require fresh
pre-fit master/preregistration binding without resetting the original arm costs.

```console
python -m pytest tests/test_pirc28_approximation_checks.py -q
```
