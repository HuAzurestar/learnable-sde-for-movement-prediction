# Budgeted affine analytic qualification evidence

`affine-propagation-qualification` is an explicit restart-only execution adapter
for **actual numerical qualification work** in the existing shared supervisor.
It supports continuous affine `exact` and affine Euler-Gaussian `gaussian`
endpoint functionals. It does not qualify path Monte Carlo, MLMC, importance
sampling, nonlinear closures, real fitted models or an entire method family.

## Frozen policy and the positive numerical path

`AffineQualificationPolicy` binds the complete request, model package, current
production Python source, method and all thresholds. There are no runtime
threshold defaults or inferred grids. All widths/error/growth/job thresholds
and four unit-bearing state scales must be strictly positive and finite;
operations are bounded by400001 and job seconds by1800. These are upper limits,
not permission to create or extend an arm. The Gaussian target is the exact
declared horizon/registered steps Euler law.

The worker computes the actual analytic functional, checks it against another
execution of the same frozen analytic estimator to refuse substitution, and
generates the [continuous enclosure](affine-reference-enclosures.md) and,
for Gaussian propagation, [finite-grid enclosure](affine-discrete-enclosures.md).
The enclosure, not agreement of floating computations, supplies error bounds.
The following checks have actual values rather than an imported `passed` flag:

- resolved component intervals;
- continuous reference functional interval width;
- absolute error of the actual float functional versus its declared target law;
- signed and absolute mathematical finite-grid time bias;
- whole-horizon transition growth in the registered scaled infinity norm;
- fixed arithmetic operation count across distinct components.

The functional roundoff bound covers the returned deterministic analytic
functional for this request, including its algorithm/implementation deviation
from its declared continuous or grid target. It is not an operation-by-operation
sampler roundoff proof. The time bias is not sampling error. Scaled growth means
`max_i sum_j |F_ij| scale_j/scale_i`, evaluated outward for continuous and grid
transitions over the declared horizon. State scales have units `[m,m,m/s,m/s]`.
This request-level growth check is **not** a uniform-time stability theorem or
an asymptotic convergence-rate claim.

Each numerical certificate has the existing fixed200000-operation quota;
the analysis also records/checks the total. No expansion, automatic retry,
extra grid, changed tolerance, PSD projection or numerical jitter is permitted.
Sampling is explicitly `NOT_APPLICABLE` for these deterministic analytic
functionals. Model error remains `NOT_IDENTIFIABLE`, not zero. Numerical failure
returns `FAILED` analysis with every failed check and the original policy hash;
computational completion may still be `SUCCEEDED` so negative evidence survives.

## Admission, supervision and resources

The adapter requires an explicit `execution_role: qualification` and admission
mode `pilot`. Before qualification artifacts or input exposure, owner admission
requires a train/validation block. Test/final-eval input is refused, including
direct calls that bypass the run wrapper. A pilot budget is mandatory and its
job cap cannot exceed the frozen policy. No store, grant, model or budget arm is
created by the implementation; the qualification cell must retain its registered
model/method/objective family and existing shared cumulative arm.

Policy/config/binding preflight is lightweight. The actual interval calculations
run inside the worker after reservation, under the existing hard watchdog. Their
time is therefore part of the confirmed worker-tree charge, not free numerical
work before the supervisor's deadline starts. Resource declarations retain four
physical coordinates and add two pooled32MiB allowances for bounded Fraction
objects/intermediates and certificate serialization. These are conservative
workspace declarations, not allocated scientific tensors or a peak-RSS claim;
dependency/import baseline memory remains separate. Result size remains512KiB.

The result binds policy/source/request/model, component certificates and hashes,
all check values, signed bias, growth and arithmetic counts. The immutable owner
admission and settlement record the actual pilot budget, input role, resources,
worker launch, elapsed/charged cost and terminal artifact provenance. An analysis
alone has `cost_status: OWNER_SETTLEMENT_REQUIRED`; it does not invent measured
cost from its arithmetic count or its configured job maximum.

## Formal qualification remains a separate gate

The analysis and result deliberately remain `scientific_qualification: false`
and `qualification: fixture`. Positive numerical checks are not a formal grant.
Next, formal owner admission must accept only an actual successful, settled,
source-bound qualification attempt and its exact frozen policy/thresholds before
protected input exposure, additionally proving cost/resource and preregistration
bindings. Imported generic report hashes or a worker mode/flag are insufficient.
Future result/paper readers must preserve this chain; no descriptive result is
promoted by this adapter. Other methods require their own actual qualification.

Tests cover the positive numerical checks, each failed threshold, strict policy
binding, explicit non-float resources, real shared workers with positive/negative
numeric evidence and actual pilot settlement, and pre-read refusal for the wrong
budget or held-out input. Disposable engineering stores/grants are test controls,
not real study permission or scientific results.
