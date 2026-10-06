# Endpoint propagation methods

The affine implementation compares exact Gaussian reference, Euler path MC,
additive-noise stochastic Heun, Euler Gaussian moment propagation, coupled Euler
MLMC and unnormalized velocity-drift importance sampling. These kernels support
four-state frozen synthetic affine packages and Gaussian initial distributions.
They do not consume a trained nonlinear package, implement reversible Heun,
perform mode learning or solve a PDE.

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

This first adapter is fixture-only and declares `restart-only`. Computational
completion retains an estimator's unresolved/low-ESS diagnostics as evidence;
it does not promote the estimate to scientific qualification. Exact method-state
checkpoint/restore, standalone study registration, formal qualification,
nonlinear stress extension, benchmark/paper/UI evidence adaptation and independent
research acceptance remain subsequent work. The raw numerical functions are for
unit qualification; actual research must run through the shared supervisor.

Validation:

```console
python -m pytest tests/test_propagation_oracles.py tests/test_propagation_methods.py tests/test_propagation_shared_adapter.py -q
```

Synthetic integration tests create disposable test stores outside Git, run actual
managed worker processes, verify admission and settlement, re-open completed
results without charging again, and refuse missing grants, excessive plans and
closed arms. They are engineering tests, not a scientific benchmark or a new
research ledger.
