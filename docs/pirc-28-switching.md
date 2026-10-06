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
