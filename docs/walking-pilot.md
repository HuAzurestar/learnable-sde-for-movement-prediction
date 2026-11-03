# Walking pilot v1 (PIRC-26 DEV-13 / DEV-14)

This is an exploratory ten-arm position-observation pilot. Historical O1 and
observed-state O2 APIs and scientific assertions retain their previous meaning.
This change implements DEV and synthetic self-tests; independent qualification
and controlled scientific execution are separate tasks.

## DEV-14 controlled consumers and cumulative limits

`ResearchStore.read_artifact` and `verify_artifact_read` enforce an explicit
`artifact_ids` whitelist whenever present. Pilot-constrained grants also require
`consumer=` at the read and final disclosure checks. Historical nonpilot grants
without consumer constraints retain their existing semantics.

Machine-readable `consumer_scope` has schema `walking-pilot-consumers-v1`, an
exact `pilot_id`, `entrypoint: walking-pilot-v1`, `worker_stages`, and a boolean
`controller`. Source grants allow only `[source]` and `controller: false`;
internal grants allow `[prepare, train, predict]` and `controller: true`.
Keep the original `consumer_arm_ids`, source inventory hash, original study,
artifact/block whitelists, purposes, restricted visibility, expiry, and false
test/public-export flags. Natural-language consumer prose alone fails closed.

A worker consumer supplies exactly `attempt_id`, `run_id`, and `entrypoint`.
The store resolves the active registered attempt, run, cell, pilot policy and
config; source reads additionally match the actual registered source cell.
The controller supplies exactly `controller: true`, `pilot_id`, `entrypoint`;
its explicit internal scope is checked against the registered config/policy and
all ten permitted arms. It cannot consume original old-study sources directly.
All internal wrapper calls pass this context and revalidate after JSON parsing.

Emit reviewable v2 templates without touching ResearchStore:

```powershell
python scripts/build_walking_pilot_grants.py --source-grants SOURCE_GRANTS_V1 --internal-grant INTERNAL_RECEIPT_V1 --pilot-id pirc26-walking-20261027-v1 --output-directory OWNER_TEMPLATES
```

The owner registers the two source grants and internal grant at exact `v2`,
preserving the original prose in `consumer_scope_description` where applicable.
The source document is marked `TEMPLATE_ONLY_NOT_REGISTERED`; its old receipt
time/head are labeled `previous_v1_*`. After actual registration, record its
new `registered_at`/head, then rebuild with `scripts/build_walking_pilot_config.py`
using that v2 source document. Set `artifact_grant` to the exact internal v2
reference, fill the new independent qualification reference and original-account
continuation reference, and recompute policy `config_hash` and study code/data
hashes. This DEV does not register or independently qualify these records.

Budget reservations must match the registered cell's pilot, study, arm and
stage. `source` charges `prepare`; training, including monitoring/retries, always
charges `train`. Omitting/changing pilot or phase, or using an ordinary job,
fails before reservation. Both original accounts and eight new arms consume the
same 1800-second pilot global limit and 120-second cumulative training limits.
A continuation declaring `pilot_study_id` cannot fund another study or nonpilot
work. Unknown charges and whole-tree stop requirements retain conservative
closure; known training timeouts retain usable partial checkpoints.

## Model and numerical contract

| Arm | Parameters | Internal velocity | Continuous guarantee under stated support |
| --- | ---: | --- | --- |
| M0-UO2 | 10 | physical v | no speed guarantee |
| M2-UO2 | 124 | physical v | no speed guarantee |
| A-S | 10 | physical v | second moment control, no hard cap with constant noise |
| A-N | 124 | physical v | same as A-S |
| B-S | 10 | latent z | physical speed < V for finite latent state |
| B-N | 124 | latent z | same as B-S |
| C-T-S | 10 | physical v | closed speed ball and C-T pathwise radial envelope |
| C-T-N | 124 | physical v | same as C-T-S |
| C-R-S | 10 | physical v | closed speed ball; expectation energy bound |
| C-R-N | 124 | physical v | same as C-R-S; no C-T pathwise envelope |

Every arm consumes the same four physical local features `[p_x,p_y,v_x,v_y]`.
Simple affine drives have ten parameters. Neural drives add a 4→16→2 tanh MLP
with Xavier hidden weights, zero biases/output, and a Frobenius bound of one
on each evaluated weight matrix. Turning off the neural residual gives its
own simple family. This is not an assertion of equivalence to historical O1.
The actual M0/M2 affine families remain unconstrained; M2's residual is bounded.

U=V=2.5 m/s is the supplied, pre-development walking assumption, not a universal
human maximum or a threshold estimated from development outcomes. The config
records its source and freeze time. Unknown movement mode is not verified
walking. B requires every initial particle strictly inside its ball; C requires
every particle in the closed ball. Invalid particles are rejected, never clipped,
resampled or replaced with another source. A can exceed U under nonzero noise.

M uses Euler–Maruyama. A uses the exact frozen-target isotropic OU step (omega=0).
B uses an OU latent step and a physical velocity map. C uses frozen-target
exponential recovery and an exact random rotation, with positive trapezoid
weights for position. C-R adds a genuinely radial step. With
`x=|v|²/U²`, `eta=rho*(1-x)`, its radial Ito subflow is
`dv=-eta²*v*dt/2+eta*v*dB`. Thus `dx=2*rho*x*(1-x)*dB`, and
`dy=2*rho²*tanh(y/2)*dt+2*rho*dB` for `y=logit(x)`.
The implemented step is algebraically
`v_new=v*sqrt(exp(delta)/(1-x+x*exp(delta)))`,
`delta=2*rho²*(2*x-1)*h+2*rho*dB`. Sign-dependent negative exponentials
avoid overflow. It includes the limiting Jacobians at zero and the sphere;
extreme nonfinite computations are rejected. This is a consistent splitting,
not an exact solution of the full state-dependent driven SDE. Continuous Ito
energy identities, discrete first/second-moment generators, boundary absorption
and gradients are checked separately. No empirical bound check is called a proof.

Native timestamps are retained. Integration grids include requested observations
and shorten steps at those timestamps. The 0.05 grid refines the 0.1 grid;
coarse Brownian increments sum the corresponding fine increments. Coupling is
within each arm's noise dimension, including meaningful coupling of each
family's simple/neural streams. No cross-dimension identification is asserted.
Reports contain actual h/h2 endpoint and score differences with MC thresholds.

## Data, preparation and fitting

The primary input is the supplied source inventory, grant receipt and geometry
mapping, each with its byte SHA256. Source preparation reads old artifacts only
through `ResearchStore.read_artifact`, using **their original study** and exact
new grant version (`resume` for train, `evaluate` for development). It records
consumer lineage and ordinary exposure events. It checks the artifact whitelist,
role, bytes, original study, pilot consumer, expiry, source inventory and disabled
final-test/public export flags. It never reads checkpoint weights from these
populations. Old normalizers, backward velocities and conditions are ignored.

Training pooled segments are mapped back to four independent sources. Segments
are never joined across a boundary. Per source, the first legal native training
origin with native 1/3-second targets is selected. The five-second past window
needs at least three distinct observation times and five seconds of available
history. Centered linear regression gives an approximate joint position/velocity
distribution, represented by eight symmetric particles. Rp is the train residual
observation covariance. Non-overlapping train prefixes provide velocity increments;
their propagated observation covariance is subtracted. QV and physical diffusion
trace matching weight original sources equally, rather than the 60 pooled segments.
Missing calibration/native targets are explicit failures, not interpolated truth.

All model noise/rates and normalization are fixed before development scoring.
Zero estimated QV is allowed and reported; a zero-speed C calibration with no
physical noise match is not applicable. One incompatible training particle
rejects that arm's training population; development sources are rejected separately.
Coverage and common-source intersections stay visible. Initial particles are an
approximate prefix regression model, not an identified instantaneous velocity.

Training uses 16 paths, equal source/horizon position observation Energy U-score,
Adam 1e-3, at most 100 updates, seed 2601. A fixed, independent train monitor
(seed 2602) includes the initial candidate and runs every ten updates/final update.
Development is never used for checkpoint selection. The selected finite candidate,
best update, completed updates and stop reason are emitted atomically. Initial
candidate protection does not imply convergence. Checkpoints are finite versioned
JSON, not pickle, and retain model config, normalization and selected weights.

Development keeps all seven original candidates, including the originally
CONTRACT_REJECTED source without an artifact. At most the first valid causal
prefix per source is used; each 10/30/60-second native target has its own row.
Prediction uses 64 paths and seed 2603. It reports Energy score, four 16-path group
MC SE, mean ADE/FDE at native observations, coordinate 90% coverage/width, speed
exceedance, B saturation and actual paired h/h2 differences. Failure never changes
the scoring population to surviving paths. There is no new final-test consumer.

## Controlled entrypoint and registration handoff

Build a reviewable local config (metadata only; no registration or science):

```powershell
python scripts/build_walking_pilot_config.py --inventory SOURCE-INVENTORY.json --grants SOURCE-GRANTS.json --geometry GEOMETRY-METADATA.json --runtime-root E:/Worktable/SDE-Research-Runtime --store-id pirc25-shared-research-v1 --output walking-pilot.config.json
python -m application.walking_pilot plan --config walking-pilot.config.json
```

The helper generates an example and registration template. Before freezing,
the authorized maintainer supplies an independent qualification reference and a
**new pilot study** internal artifact grant permitting evaluate/resume on all
eleven lineage blocks. This is separate from old-study source grants. Do not
reuse an old source grant as an internal result grant. Preserve the historical
M0/M2 model/method/objective family tuples when registering the pilot study, to
reuse their original budget IDs; new display names do not grant new budgets.

The externally maintained immutable records are:

1. `walking-config-<pilot_id>`: the completed config. Recompute its `digest` after
   filling references. All source metadata files must remain at their explicit
   hashed locations accessible to owner and workers.
   Fill `continuation_approval` with the exact new owner continuation reference
   registered on both original M0/M2 accounts; positive balance and source grants
   do not replace an expired execution approval. The preflight checks the current
   receipt/reference and its live dates, without registering a continuation.
   The generated `preparation_arm=A-S` funds common preparation from a new pilot
   account under the same global allowance. An unavailable M0/M2 continuation
   produces an account-not-ready row without blocking the remaining arms.
2. `walking-pilot-<pilot_id>`: template policy, `config_hash=digest(config)`,
   user authorization manifest ID/hash, exact ten arm IDs, 1800000 global ms,
   120000 training ms, walking, final_test=false, and original expiry.
3. `study-<study_id>`: normal ResearchStore preregistration with exact generated
   32 cells, `code_hash=code_identity()`, `data_hash=digest(config['sources'])`,
   and `protocol_hash=digest({'inventory': inventory_sha, 'geometry': geometry_sha})`.
   This lineage protocol identifies already registered artifacts, not raw files.
4. The independent qualification manifest referenced by config, with matching
   code_hash, `scope=independent-walking-pilot`, `status=passed`. DEV must not
   create this claim. Register the required new-study internal grant separately.

Then the assigned scientific executor can run in the foreground:

```powershell
python -m application.walking_pilot run --config walking-pilot.config.json --report walking-pilot.status.json
```

This CLI does not grant permission or register a budget/study. It creates only
registered attempts and runs them through the existing ResearchSupervisor,
ResearchQueue and BudgetLedger. It fails closed on source/config/code/grant or
qualification mismatches and still writes ten status rows. The config's old CSV
raw-source route is an explicit optional transport using EvaluationExposureLedger;
it is not used for the supplied original artifact inputs.

## Costs, termination and reporting

The same shared event chain records pilot sublimits and original arm charges.
Every training attempt, monitor and retry shares its arm's cumulative 120 seconds;
prepare, calibration, prediction, h/h2 and controller science overhead share the
1800 seconds. No per-arm prediction cap is invented. Training stage startup and
model reconstruction conservatively consume its training allocation too. New arm
IDs bound to a walking policy cannot reserve nonpilot jobs. An unresolved pilot
reservation blocks another job; unknown costs remain fully reserved/charged.
Known pilot timeout does not close the historical M0/M2 account; unknown worker
cost still closes it for reconciliation. Historical nonpilot timeout semantics
are unchanged. The policy is not a new 24-hour allowance for eight arms.

A persistent OS lock serializes the pilot controller. A funded absolute deadline
also caps queue waits and worker compute. The shared supervisor stops only its
owned native process tree and confirms stop before settlement. Windows Job
Objects impose a 2GiB whole-tree memory limit. Scientific CLI use currently fails
closed on other operating systems; it does not claim a portable memory sandbox.
All workers use one compute thread. There is no private detached runner.

Training atomically writes progress after each completed update. The supervisor
can publish the last complete JSON checkpoint **after confirmed native stop**,
even when the worker times out. The timeout verdict and real cost remain intact.
Prediction similarly retains completed sources. With remaining global allowance,
an exhausted training arm can still predict from its selected checkpoint and
other arms can continue. No extra time is requested automatically.

The local report keeps the ten-arm execution/coverage matrix (210 source/horizon
status cells), costs, progress and artifact references. Metrics and selected
weights remain restricted ResearchStore artifacts; normal grant reads are required.
The restricted summary includes common sources per horizon and their three-horizon
intersection. It makes no confirmatory winner claim. The first implementation
reports paired per-source metrics; additional inferential aggregation is out of
scope. Snapshot generation and durable metadata writes can consume the remainder
or overrun while blocked in I/O; those measured costs are retained, further
compute is refused, and the overrun is not called an extended allocation.
The run never opens a browser, reads final-test data or merges/deploys code.

Original-account continuation also supports a protected-PID read-only fallback.
When native OpenProcess/GetProcessTimes returns access denied, it first retains
the existing complete-snapshot absence check, then may issue two fixed CIM queries
for that exact uint32 PID. Both queries must report the same identity and aware
UTC creation time more than two seconds after the old recorded launch. Each query
has a five-second deadline and 2048-byte read cap. The command contains no injected
PID/script text, uses no profile or visible window, and never invokes a method on
the target process. Only the locally started query helper is stopped/reaped on
its own timeout. Denial, unknown identity, multiple rows, wrong PID, inconsistent,
old, malformed or future births remain a veto. This proves identity reuse only;
the existing settled/terminal/whole-tree-stop requirements remain mandatory.
Microsoft documents CreationDate as read-only process start time in
[Win32_Process](https://learn.microsoft.com/en-us/windows/win32/cimwin32prov/win32-process).

## Validation

Use the DEV tests in `tests/test_walking_pilot*.py` plus relevant existing dynamics,
training, budget, source authorization, artifact boundary and supervisor tests.
The test fixtures use only newly created artificial trajectories and stores.
They are neither real source exposure nor an independent qualification receipt.
No real scientific payload or training has been consumed by DEV-13.
