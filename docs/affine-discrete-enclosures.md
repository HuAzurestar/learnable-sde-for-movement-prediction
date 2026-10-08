# Bounded affine finite-grid and signed time-bias certificates

`inference.affine_discrete_reference` extends the independently recomputed
[continuous reference enclosures](affine-reference-enclosures.md) with explicit
Euler, ordinary additive Heun and constant-noise reversible Heun grids. This is
an engineering proof ingredient, **not** a method qualification, a research run,
an operator grant, or an automatically selected method/grid.

## Frozen target and numerical scope

The APIs require `solver` and integer `steps` explicitly, with no defaults.
Allowed solvers are `euler`, `heun`, `reversible-heun`; steps are 1 through 8192.
For an MLMC correction/telescoping estimate, callers must specify the registered
finest target grid rather than silently using the request's base `steps`.
The complete request hash and the separately explicit target grid are both
bound. A certificate alone does not prove that an actual worker used that grid.

Inputs are exact binary rational values of the declared Python floats. The
mathematical step size is **exact horizon divided by steps**. The target is the
affine recurrence with independent centered Gaussian increments of covariance
`h I`. This is different from claiming a bound on every NumPy floating-point
sampler operation: implementation roundoff remains unknown and separately
requires qualification. Parameter/model uncertainty, nonlinear closure,
first-passage events, correlated path input, proposal likelihood roundoff and
sampling uncertainty are not covered.

## One-step laws

Write the frozen drift as `A y + b`, noise as `L dW`, and `h = horizon/steps`.
Euler has `F = I + h A`, `c = h b`, `B = L`. Ordinary additive Heun has:

```
F = I + h A + h² A² / 2
c = h b + h² A b / 2
B = (I + h A / 2) L
```

Its one-step covariance is `h B B'`. Reversible Heun is not ordinary Heun with
a label: it propagates the joint state `[physical y, auxiliary z]`:

```
F = [[I + h A, h² A² / 2], [2 I, h A - I]]
c = [h b + h² A b / 2, h b]
B = [(I + h A / 2) L, L]
```

The initial joint mean is `[m,m]`, and the covariance consists of four identical
blocks `C`, because `z0=y0`, not two independent Gaussian initial states. The
reported physical output is only the first four coordinates. The eight-state
transition certificate documents four auxiliary coordinates, **not** a new
eight-dimensional frozen model. Derivation substitutes the affine drift into
the actual two-state step in [the propagation methods](propagation-methods.md).

## Bounded semigroup composition

An affine Gaussian transition `(F,c,Q)` composed after `(G,d,R)` is exactly:

```
(F G, c + F d, Q + F R F')
```

Binary powering needs at most 26 such compositions for the allowed grid. It
does not iterate all 8192 path steps or allocate all independent path noises.
Each arithmetic operation is rounded outward on the same fixed dyadic grid as
the continuous enclosure. Covariances are never projected, clipped, jittered
or repaired; initial covariance must satisfy the exact symmetry and all
principal-minor tests. Growth that exceeds fixed integer/intermediate/operation
quotas is refused, not retried with more steps or an expanded resource profile.
Wide intervals or unresolved variance/boundary statuses are not eligibility.
No stability or asymptotic convergence theorem follows merely from a bound.

The certificate parser bounds canonical JSON size, depth and nodes. The whole
production Python source hash, model hash, request hash, grid, algorithm profile,
moment and functional intervals, composition/operation counts are recorded.
`verify_affine_discrete` recomputes the complete canonical certificate; a content
hash resealed around invented intervals does not pass verification.

## Signed time bias

`bound_affine_time_bias` separately computes the discrete target law and the
continuous law and subtracts their functional intervals outward. Its sign is:

```
E[finite-grid endpoint functional] - E[continuous endpoint functional]
```

It records both component certificate hashes, exact grid/source/request/model
bindings, component operation counts and their sum. Both components must share
one source hash. Each component has its own fixed 200000-operation quota; the
reported total may exceed one component's quota. This is explicit, not a shared
worker budget declaration or permission to run beyond an arm. No worker or
ledger API is called. Actual owner cost/timeout/resource accounting remains a
separate preregistered responsibility.

`UNRESOLVED_COMPONENT` preserves an unresolved component status. Signed bounds
are never clipped to `[0,1]`, even for probabilities. An interval containing zero
does not prove zero bias; identical floating estimates do not prove it either.
`verify_affine_time_bias` recomputes both components and their subtraction, not
just their recorded hashes. Continuous, discrete and bias certificate schemas
cannot be substituted for each other at these verifier entry points.

## Counterchecks and remaining qualification work

Tests expand each path into formal independent noise coefficients using exact
rationals, independently of the production interval matrix-powering recurrence.
They check all physical means and covariances for odd/even grids, correlated
diffusion, nonzero drift and uncertain initial state. Polynomial integrated
Brownian laws check maximum grids and the sign of Euler's acceleration bias.
Centered Gaussian and deterministic open/closed halfspaces test boundary scope.
Changed request/grid/source, resealed false intervals/qualification flags,
out-of-range grids, strict PSD failure and explosive grids are refused.

These certificates remain `scientific_qualification: false`. There is no
automatic use by a sampler, owner admission, frozen method package, pilot or
formal result. A future explicit qualification contract must establish the
request/model/grid/functional binding, tolerances, stability, implementation
roundoff and measured cost before protected test exposure. Nonlinear methods,
mixture closures, real studies, paper/review/acceptance and merged delivery are
not completed by these proof ingredients.
