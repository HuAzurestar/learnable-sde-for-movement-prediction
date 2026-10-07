# Endpoint propagation methods

The affine implementation compares exact Gaussian reference, Euler path MC,
additive-noise stochastic Heun and reversible Heun, Euler Gaussian moment propagation, coupled Euler
MLMC and unnormalized velocity-drift importance sampling. These kernels support
four-state frozen synthetic affine packages and Gaussian initial distributions.
They do not consume a trained nonlinear package, perform mode learning or solve
a PDE. Separate synthetic nonlinear adapters support the declared constant-noise
stress packages; these are not qualified trained models.

`domain.propagation.PropagationRequest` is an immutable per-cell endpoint request:
one registered horizon, initial distribution, model content identity, origin and
history cutoff, functional/version, half-space geometry/boundary, seed/coupling
identity, arm and bounded numerical configuration. The first functionals are
endpoint `x` expectation and endpoint half-space probability. Multi-horizon studies
register one request per horizon under the same model/method/objective arm.

`FunctionalResult` retains estimator identity, original signed estimate, interval
kind, sample count, method status and a separate numerical error budget. Time bias
is identified for this affine slice by comparing the discretized Gaussian target
against the continuous float64 reference. Reference roundoff bounds and real-model
error remain unknown; they are not filled with zero or added to unrelated errors.
Affine Gaussian closure has zero additional closure error under its explicit
assumptions. A nonlinear extension needs a separate qualification.

Path generation streams bounded chunks without storing all Brownian paths.
Per-sample SeedSequence streams and fixed arithmetic reduction order preserve
endpoint paths when changing chunk size or resuming an uncommitted sample range.
For MLMC, each coarse increment is the sum of two adjacent increments from its
paired fine path, and each level uses an independent stream. The signed telescoping
estimate and its interval are retained without probability clipping.
`allocate_mlmc` proposes bounded allocations from supplied independent pilot
variance/cost measurements; it starts no additional work or budget reservation.

## Explicit managed MLMC pilot

`mlmc-pilot` is an explicit chunk-adapter execution configuration, not a new
method family, budget arm, automatic matrix expansion, or production estimator.
Register a separate immutable pilot request/cell under the original `mlmc`
method-family arm, with `execution_role: pilot`, admission mode `pilot`, and
`BudgetSpec(..., category="pilot")`. Its independent sampling phase is fixed to
2; ordinary MLMC uses phase 1. A different production seed and explicit coupling
identity must be frozen in the policy before the pilot. The upstream study role
retains its existing primary/secondary meaning, independently of execution role.

The cell's `mlmc_pilot_policy` is the full serialized
`domain.mlmc_pilot.MLMCPilotPolicy`: pilot request hash, production seed/coupling,
sampling and bias tolerances, maximum correction variance ratio and measured cost
ratio, minimum level samples, maximum samples and total fine/coarse work. There
are no default research thresholds. Register at least three fixed levels;
allocation, finest-grid, sample and total-work caps are checked before exposure.
Only train/validation protocol blocks are permitted. Test/final-eval input and
qualification artifact reads are refused at the admission boundary, even if the
caller bypasses the numerical command builder. The existing 1800-second pilot
stage cap and original cumulative arm budget remain authoritative.

Each completed chunk measures monotonic compute nanoseconds, including path and
functional/statistic computation but excluding checkpoint callback and owner ACK.
Per-level compute totals are saved with completed numerical statistics, then
accumulated only for remaining samples on resume. These hardware-dependent cost
measurements are not ledger charges or restorable budget balances. The shared
owner independently charges whole-worker slot occupancy, including startup and
save/ACK. Numerical results replay within the frozen environment; timing values,
empirical cost ratios, recommendations and resulting output hashes need not.

`inference.mlmc_pilot.analyze_mlmc_pilot` reports correction-level variance ratios
(level zero is not a correction), measured cost ratios, separate signed affine
time bias and reference uncertainty, and explicit rejection reasons. A bounded
sampling-only allocation may be proposed when empirical sampling/rate checks
pass; unknown bias/reference components remain unknown and cannot grant
qualification. Every current adapter's pilot analysis remains `UNQUALIFIED` /
`ENGINEERING_ONLY`, and the shared result remains `fixture`. Production requires
separate registration and qualification; no proposal executes or adds levels.
The allocation principle follows [Giles's primary MLMC research and software
references](https://people.maths.ox.ac.uk/gilesm/mlmc.html); this implementation
is a bounded engineering slice, not a certified convergence theorem or research
comparison.

Engineering tests:

```console
python -m pytest tests/test_mlmc_pilot.py tests/test_mlmc_pilot_shared.py -q
```

## Other methods and shared recovery

`reversible-heun` is a distinct path-MC configuration, not an alias for ordinary
Heun. It specializes Algorithm 1 of [Kidger et al., *Efficient and Accurate
Gradients for Neural SDEs*](https://arxiv.org/abs/2105.13493) to autonomous drift
and constant additive diffusion. In this supported slice the Itô and
Stratonovich conventions coincide. The extended pair `(y, z)` starts with
`z=y` and advances as

```text
z_next = 2*y - z + h*f(z) + L*dW
y_next = y + h/2*(f(z) + f(z_next)) + L*dW
```

The retained auxiliary drift needs one new drift evaluation per step after
initialization. The *extended pair*, not `y` alone, reverses algebraically using
`-h` and the same negated Brownian increment. Numerical unit tests check that
property to floating-point tolerance; no adjoint/backpropagation system or
cross-platform bitwise reversal is provided. Coarse and fine paths retain their
own auxiliary states and use actual sums of adjacent Brownian increments.

For affine dynamics, an eight-dimensional augmented Gaussian recurrence tracks
the physical and auxiliary states with their **fully correlated** initial
covariance. Only its four-dimensional physical marginal is returned. The affine
time-bias component compares that marginal with the continuous float64 reference;
reference roundoff and real-model errors remain unknown. Nonlinear time bias is
also unknown. Physical state order and units stay unchanged; the resource
contract explicitly reserves bounded auxiliary/cached-drift workspace. Adapter
version `1.1.0` reflects the changed configuration and allocation contract.

The paper establishes strong order one for constant diffusion under its
regularity assumptions; this is not evidence that a particular frozen grid or
long horizon meets the requested error. The method is not A-stable. Affine
results expose the extended transition's spectral radius, including growth of
parasitic auxiliary modes. Nonfinite paths, drifts or Gaussian moments fail;
there is no step expansion, PSD projection or stability promotion. Finite
outputs remain engineering fixtures until independent method/reference
qualification. No multiplicative-noise Itô model is supported.

`tests/test_integrator_convergence.py` adds independent manufactured-law
controls for Euler, additive Heun and reversible Heun. Fixed grids 8/16/32
check the damped-oscillator endpoint expectation against its scalar closed form
and integrated-Brownian half-space probability against closed-form moments/CDF,
not against the production continuous moment solver. These slices distinguish
Euler's first-order weak functional bias from the Heun variants' second-order
weak bias. They do not establish an order for arbitrary region functionals.

Nonzero-noise strong controls call the actual paired endpoint sampler at
horizons 1 and 10, reconstruct its documented per-sample entropy contract
independently, and verify every fine/coarse physical endpoint against weighted
increments and actual adjacent fine-increment sums. For `dx=v dt, dv=sigma dW`,
the exact endpoint conditional on the increments includes the independent
Gaussian Brownian-bridge residual with variance `T*h^2/12`. The ideal-law
position mean-square error is `sigma^2*T*h^2/3` for Euler and
`sigma^2*T*h^2/12` for both Heun variants on this nilpotent drift. The finite
PRNG ensembles check those meanings and first-order strong RMS refinement;
weak second order must not be reported as strong second order. Velocity alone
being exact does not mean the four-state path has zero error.

These bounded, disposable unit controls neither create a research store nor
authorize a study. Different resolution ensembles are not independent study
blocks. Cross-platform roundoff/PRNG coverage, nonlinear weak/strong rates,
long-horizon stability, complete method/physical-axis qualification and actual
registered error-cost studies remain separate requirements. The original
sampling kernels, recovery protocol, grants and cumulative budgets are unchanged.

Recovery occurs only after *complete sample chunks*. There is no partial path
or auxiliary trajectory to serialize at that boundary: completed statistics
are saved and remaining independent sample streams reconstruct both initial
states. The solver identity is checkpoint-bound, so a regular-Heun state cannot
be used as a reversible-Heun continuation. Real affine and nonlinear worker
tests exercise the unchanged80% soft-save/ACK, reopened linked restore and
cumulative original-arm charging.

Importance sampling changes drift only by `L @ u` and retains the original
diffusion. Its discrete path likelihood ratio is accumulated in log space:
`log w = -sum(u @ dW) - 0.5 * ||u||^2 * horizon`. The estimator is unnormalized,
with weight/ESS/proposal diagnostics; it is not a self-normalized estimate.
No weighted hits or very low ESS remain explicit method failure/uncertainty states.
Zero-hit ordinary MC has an exact one-sided 95% binomial upper bound and unknown
sampling SE; IS and MLMC do not reuse that independent Bernoulli interval.
Other intervals are explicitly normal approximations, not certified tail coverage.

`experiments.pirc27.plugin.propagation_plugin()` supplies one versioned registered
execution adapter with typed configuration and an allocation plan. Workers use
`application.propagation_execution` through `SharedRunner`; the normal admission,
upstream snapshot/catalog, grant, code/input hashes, stage cap, measured settlement
and arm closure remain mandatory. The immutable study spec must supply
`runtime_binding = {"root": <absolute existing runtime root>, "store_id": <ID>}`.
The worker never initializes a store, supplies authorization or resets budgets.

Both adapters remain fixture-only. The original adapter declares `restart-only`.
`propagation_plugin(recovery=True)` registers the separate
`affine-propagation-chunk` adapter for Euler/Heun/reversible-Heun MC, MLMC and importance sampling.
Register `propagation_recovery_plugin()` with the shared recovery registry and
bind its command in the admitted package before running. Analytic exact and
Gaussian methods retain the original restart-only adapter; they have no streaming
sample accumulator to resume.

Chunk state retains sample position, Welford moments/hits, MLMC per-level signed
statistics, or IS log-weight/event moments and maximum weight. It binds the
request, allocation, proposal, phase, NumPy/PCG64 version and per-sample RNG root.
At a completed chunk, there is no partially consumed Brownian stream: the next
sample stream is reconstructed from the stored immutable root and sample/level
identity. Changing chunk size or the numerical environment refuses resume.
No uncommitted partial chunk or arbitrary internal time step is restorable.
Identical results are tested within one environment and frozen configuration,
not promised bit-for-bit across different libraries/platforms.

The chunk adapter explicitly counts total fine/coarse path-step updates for its
progress/resource plan (maximum 1,000,000), separately from the finest temporal
grid (maximum 8192). Checkpoints contain no budget balance; normal shared resume
creates a linked attempt and charges both attempts to the original arm. The
worker polls at completed chunks, not within a long chunk. If that boundary or
owner acknowledgement cannot be reached before the existing hard deadline, the
normal timeout/closed-arm rule applies; no grace period is added.

Computational
completion retains an estimator's unresolved/low-ESS diagnostics as evidence;
it does not promote the estimate to scientific qualification. Standalone study
registration, formal qualification,
nonlinear stress extension, benchmark/paper/UI evidence adaptation and independent
research acceptance remain subsequent work. The raw numerical functions are for
unit qualification; actual research must run through the shared supervisor.

Validation:

```console
python -m pytest tests/test_propagation_oracles.py tests/test_propagation_methods.py tests/test_propagation_recovery.py tests/test_propagation_shared_adapter.py -q
python -m pytest tests/test_reversible_heun.py -q
```

Synthetic integration tests create disposable test stores outside Git, run actual
managed worker processes, verify admission and settlement, re-open completed
results without charging again, and refuse missing grants, excessive plans and
closed arms. They are engineering tests, not a scientific benchmark or a new
research ledger.
