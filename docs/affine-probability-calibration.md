# Bounded affine spatial probability calibration kernel

This is a numerical building block, not a registered research producer or a
formal-study admission API. Do not call it from the design compiler or preflight
to obtain free reference work. Shared producer, settled source ownership,
frozen geometry consumers, preregistration and independent Paper verification
remain required before using its output in research.

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

Every result retains `scientific_qualification: false`,
`method_qualification: false`, `admission_status: NOT_ADMITTED` and
`cost_status: OWNER_SETTLEMENT_REQUIRED`, even when the numerical status is
`PASSED`. A hash alone is not evidence of source execution or settlement.

Next integration must run all numerical work inside an original charged pilot
worker, bind settled native events/costs/source caps and authority, freeze exact
model/input/horizon geometry before heldout access, preserve method-qualification
requirements at the new threshold, refuse missing/altered proof before reading
generic attachments or protected inputs, and close reuse/export/Paper checks.
Sharing geometry across methods must not multiply source cost or create new
independent samples, budget arms or fitting authority. Unit tests do not close
this chain, a calibrated study matrix or the full measured scientific delivery.
