# Bounded affine spatial probability calibration kernel

The numerical building block now has an explicit charged pilot adapter, not a
formal-study admission API. Do not call the kernel from the design compiler or
preflight to obtain free reference work. Settled source ownership, frozen geometry
consumers, preregistration and independent Paper verification remain required
before using its output in research.

The motion model stays two-dimensional with four SI states `[x,y,vx,vy]`.
Calibration supports the declared constant-affine Gaussian endpoint law only,
not nonlinear closure, fitted-model accuracy, first passage or terrain science.
Only position normals `[nx,ny,0,0]` are accepted, so the region threshold has
units of metres. A mixed position/velocity halfspace needs another explicit
unit contract; it is not silently treated as a spatial region.

## Frozen inputs and fixed work

`AffineHalfspaceCalibrationPolicy` binds the complete zero-threshold template
request, immutable model and current runtime source. It requires explicit target
probability, relative probability-error and interval-width controls, operation
cap and pilot job cap. Scalar shape is validated before copying/hash encoding;
cyclic or nested control values are rejected. Template origin/history cutoff,
initial law, horizon, normal and closed-boundary changes invalidate the policy.

The bounded kernel currently supports nominal probabilities between `1e-8` and
`0.1`, including `0.01`, `0.0001` and `0.000001`. This is an implementation
domain, not an observed-model qualification or a scientifically accepted error
tolerance. The caller must choose and freeze its controls before execution.

1. Bound affine moments with the existing rational reference implementation.
2. Project along the positive-scaled spatial normal. Require a strictly positive
   projected variance; do not repair covariance or degenerate geometry.
3. Bracket the positive normal-tail quantile on `[0,8]` with exactly 48 certified
   bisections at the existing fixed precision. Reject unresolved comparisons;
   no adaptive precision, floating inverse CDF or quota expansion.
4. Take the normalized interval's rational midpoint, multiply by the exact
   positive scale under the original temporary-bit cap, and convert that
   candidate to a finite floating threshold. Do not use the coarse physical
   interval midpoint when the scale is smaller than the dyadic grid.
5. Independently recompute the bounded probability for that actual rounded
   threshold. Compare exact rational error/width ratios with the frozen limits.

Moment computation, projection/quantile search and actual-threshold reference
share a single operation counter capped at at most the original `200000`.
Exhaustion refuses before executing another boxed operation. Each reference's
certificate still records its own operation count and retains the canonical
public recomputation format. Analysis byte quota remains the original reference
quota; policy pilot cap is at most the original 1800 seconds, requiring original
worker supervision rather than a timer in this kernel.

Analysis v2 retains the bounded normalized ideal-threshold interval and exact
original scale as well as outward physical bounds. Positive normal scaling does
not change geometry eligibility merely because the physical interval is coarse;
genuine float underflow/representation failure still fails the unchanged controls.
The bounded ideal-threshold interval is not a claim that the rounded float lies
inside it: its actual probability is certified separately. Failed numerical
controls retain the candidate and probability certificate. Nonpositive or
unresolved variance retains moment evidence without promoting a threshold.
Fixed-quota/unresolved-comparison exceptions require ordinary owner failure
recording; they never authorize retry with larger controls.

## Output is not permission

`experiments.pirc27.calibration_plugin` registers the restart-only
`affine-halfspace-calibration` adapter through the existing propagation worker.
It requires an explicit frozen policy, pilot admission, train/validation inputs,
`execution_role: probability-calibration` and the original exact/halfspace
reference arm. Held-out/formal input is refused before grant lookup or protected
reads. The existing pre-reserve budget guard requires the pilot category and
the frozen job cap; there is no second timer, queue, ledger or research launch.

Its resource declaration includes the entire frozen operation allowance and
the conservative 64 MiB rational/serialization pools, under unchanged global
hard caps. It honestly declares zero sampled paths and four physical SI states.
The shared resource-plan `steps` count denotes counted arithmetic here, not an
integration-grid length; ordinary propagation keeps its original grid limit.
No generic estimate or discarded float-reference metric is computed. The pilot
result's metric is actual `reference_arithmetic_operations`, in operations,
not an accuracy score or a successful scientific qualification.

Numerical `FAILED` analyses remain computationally successful saved artifacts;
operation exhaustion is an ordinary charged failed attempt, never an automatic
retry or quota expansion. Existing successful-attempt reuse verifies original
admission/output/native-cost provenance without starting another worker. These
producer controls do not prove cross-method geometry reuse or full research
cost accounting; the dedicated owner/consumer/export/Paper chain is still needed.

## Settled source preparation owner

`prepare_probability_calibration` consumes an explicitly selected source pointer
under current evaluate/consumer permission. It verifies the original admitted
synthetic train/validation pilot, canonical artifact and current source code,
native tree-stop confirmation, reservation/settlement/completion ordering and
positive charge within the frozen pilot cap. Disclosure is physically verified
again after extra source I/O. Version selection is exact; no grant is inferred.

The owner checks bounded saved certificates, aggregate operation counts, fixed
quantile bracket, normalized candidate, actual rounded threshold and separated
relative probability diagnostics. It executes no matrix exponential, CDF,
square root, inverse-CDF search or estimator. All that work remains charged in
the source worker. Tiny-normal physical envelopes stay conservatively coarse;
candidate selection uses the saved normalized midpoint with exact scale.

The physical geometry key includes the frozen model/law/origin/cutoff/horizon,
normal/threshold/open-or-closed rule/functional version/coordinates/SI units
and causal input identity. It excludes request ID, algorithm seed/grid/path
count/chunk/tolerance/coupling/arm. For explicit source cases the owner uses the
original validated input binding. Otherwise only the declared fixed synthetic
initial law is identified, never observed-prefix causality or independent
trajectories. Private-prefix source preparation is not implemented here.

The preparation proof remains `NOT_ADMITTED` with both qualification flags false.
`OWNER_SETTLED` refers only to its proven original calibration source cost; one
source-cost identity is retained for future deduplication, not a new ledger or
proof that an aggregate currently counts it correctly. Target method admission,
frozen threshold matrix/preregistration, cross-method reuse, authorized export
and independent Paper validation still require their dedicated integration.

Every result retains `scientific_qualification: false`,
`method_qualification: false`, `admission_status: NOT_ADMITTED` and
`cost_status: OWNER_SETTLEMENT_REQUIRED`, even when the numerical status is
`PASSED`. A hash alone is not evidence of source execution or settlement.

Next integration must bind settled native events/costs/source caps and authority, freeze exact
model/input/horizon geometry before heldout access, preserve method-qualification
requirements at the new threshold, refuse missing/altered proof before reading
generic attachments or protected inputs, and close reuse/export/Paper checks.
Sharing geometry across methods must not multiply source cost or create new
independent samples, budget arms or fitting authority. Unit tests do not close
this chain, a calibrated study matrix or the full measured scientific delivery.
