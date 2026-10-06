# Bounded affine reference enclosures

`inference.affine_reference.bound_affine_reference(package, request)` encloses
the moments and requested endpoint functional of the **declared** four-state
constant-affine Gaussian law. Its immutable certificate binds the model package,
complete request, current runtime source and fixed algorithm. This is a building
block for numerical reference qualification, not a qualification decision,
preregistration, permission, fitted-model validation or research result.

It supports endpoint `x` expectation and open/closed endpoint half-space
probability. It does not support nonlinear dynamics, arbitrary region integrals,
first passage, PDEs, model/parameter uncertainty or a different state dimension.
No input model is fitted, projected, jittered or silently replaced. Actual
research must use the shared supervisor and existing store/grants; these raw
functions are for bounded numerical engineering checks.

## Arithmetic and explicit quotas

Input integers/floats denote their exact rational values, including the binary
representation of a Python float. Interval endpoints round outward onto the
fixed grid `2^-160`. Addition, subtraction, all four signed multiplication
corners and division by an interval excluding zero retain containment. Integer
square roots bound roots without floating-point rounding. Exact algebraic zeros
may be skipped; there is no small-value cutoff.

The fixed profile permits 4096-bit grid numerators, 16384-bit rational
intermediates, 200,000 interval-producing operations, 32 matrix Taylor terms,
at most 60 semigroup doublings, 256 CDF terms and 40 arctan terms. Initial-PSD
checks also have a fixed four-dimensional principal-minor enumeration. These
counts are algorithm bounds, not FLOPs or ledger charges. The certificate is
at most128 KiB, with bounded JSON nodes/depth/strings. There is no caller option
or retry that increases precision, terms, dimensions or quotas. A failed bound
raises an explicit validation/numerical error; actual supervisor failure and
resource settlement remain the owner's responsibility.

Initial covariance must be **exactly symmetric and PSD** as a rational matrix.
All principal minors are checked; roundoff-sized negative eigenvalues accepted
by a separate float64 countercheck cannot establish a Gaussian reference law.
This implementation never projects that covariance.

## Matrix-exponential enclosure

For the stated `dX=(A X+b)dt+L dW`, form the nine-dimensional block generator

```text
K = [ A   L L^T   b ]
    [ 0   -A^T    0 ]
    [ 0    0      0 ]
```

Scale the actual boxed short-step matrix until its infinity norm is at most
`1/4`, within the fixed doubling cap. The finite Taylor polynomial is evaluated
with outward arithmetic. For `u >= ||M||`, the remaining series is bounded by

```text
r = u^(N+1)/(N+1)! / (1-u/(N+2)), N=32.
```

This follows by bounding each matrix-power norm and the ratios after the first
omitted term. An entry's absolute remainder is no greater than that norm bound.
No positive-length walk in the matrix's nonzero graph proves that entry of all
positive powers is zero, so the exponential entry is exactly the identity entry.
If the computed interval for `M^N/N!` is exactly zero everywhere, all subsequent
powers are also zero. These are algebraic proofs, not PSD repair or heuristic
truncation. Otherwise the explicit remainder is added, never inferred from
agreement with a floating implementation.

The upper-left block is `F`, the final column is `c`, and the upper-middle block
`E` gives `Q=E F^T`. Long intervals compose `F^2`, `c+F c` and `Q+F Q F^T` using
outward arithmetic; the exponentially growing inverse block is not squared.
Initial Gaussian moments propagate through `F m+c` and `F C F^T+Q`. Returned
covariance intervals need not themselves be PSD or symmetric: they enclose the
true matrix and are not replacement Gaussian parameters. Exactly zero initial
and diffusion covariance proves an exactly deterministic law independently of
the generic exponential bounds.

## Endpoint probabilities

The half-space normal and threshold are scaled by the same positive **exact
rational** magnitude before boxing. This preserves geometry without avoidable
large-normal loss of precision. If projected variance has a strictly positive
lower bound, project/standardize the moment intervals and use CDF monotonicity.

`pi=16 arctan(1/5)-4 arctan(1/239)` is enclosed with alternating-series
remainders, not a rounded machine constant. The identity follows from tangent
addition (`tan(4 arctan(1/5))=120/119`) and the principal-angle range. The
[NIST arctan expansion](https://dlmf.nist.gov/4.24.E3) supplies its series.
For `|z|<=8`, integrate the Gaussian density's alternating power series with
256 terms; the first omitted term bounds the decreasing remaining tail. This
is the [NIST error-function expansion](https://dlmf.nist.gov/7.6.E1) under the
normal-CDF change of variables. All arithmetic error is retained as well.

For positive `z>8`, the Gaussian tail is bounded above by
`exp(-z^2/2)/(z sqrt(2*pi))`; its lower bound is zero. The inequality follows by
replacing the integrand with `(t/z) exp(-t^2/2)` for `t>=z`, consistent with
[NIST's Mills-ratio bounds](https://dlmf.nist.gov/7.8). Its exponential is bounded
with the same Taylor argument and scalar squaring. Beyond32 use the bound at32
and monotonicity, so extreme arguments do not create an unbounded computation.
Negative arguments use symmetry. Intersecting these bounds with the proved
probability range `[0,1]` is not clipping a signed sampling estimator.

When variance is exactly zero, the open/closed boundary is decided only when
the mean enclosure proves the corresponding relation. A boundary not resolved
at fixed precision returns `[0,1]` with `UNRESOLVED_DEGENERATE_BOUNDARY`.
Variance containing zero returns `[0,1]` with `UNRESOLVED_VARIANCE`. A wholly
negative variance enclosure contradicts the assumed PSD law and fails. Rare
tails below the grid remain bounded above by a positive grid value; this is not
a relative-tail-accuracy claim. Wide bounds do not pass a tolerance merely
because the status says `BOUNDED`.

## Evidence verification and remaining qualification work

The certificate serializes rational numerator/denominator strings, not rounded
display values. `manifest()` is detached and structurally bounded. Accessing
`functional_bounds()` alone does **not** authenticate an untrusted certificate.
`verify_affine_reference` recomputes the complete canonical certificate against
the actual frozen request/model/source. Resealing invented intervals, changing
the request or source, or promoting a qualification flag cannot pass replay.
These guarantees assume the trusted frozen Python implementation; this is not
a process sandbox or off-platform attestation.

`reference_error_bound` first verifies, then bounds the absolute difference
between a finite numerical approximation and **both** functional endpoints.
The final float is rounded outward. Zero is possible only for a proved exact
functional identity, such as half-probability under a centered nondegenerate
Gaussian projection; numerical agreement alone does not justify zero.

Certificates always carry `scientific_qualification: false`. Existing ordinary
propagation outputs still retain an unknown reference component: this new
calculation is not silently inserted into every worker's unregistered workload.
Production/reference qualification must explicitly bind the certificate scope,
method/grid/functional checks, measured cost and preregistered tolerances to
owner-controlled evidence **before** test/final-eval exposure. Separate sampling,
time-bias, closure and model-error components remain separate. A continuous
reference enclosure alone cannot qualify a finite-grid method, nonlinear model
or formal scientific comparison.

Engineering validation:

```console
python -m pytest tests/test_affine_reference.py -q
```

Independent checks include exact rational integrated-Brownian transition and
initial-moment polynomials, scalar high-precision exponentials, all four affine
cases at short/long horizons, geometry scaling, proven and unresolved degenerate
boundaries, positive rare upper bounds, strict covariance refusal, quota/parser
failures and tampered/resealed-certificate rejection. Decimal/float64 comparisons
are counterchecks, not the proof behind the enclosures or formal research evidence.
