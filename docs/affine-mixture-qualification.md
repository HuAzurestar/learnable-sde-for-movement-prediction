# Dedicated affine mixture functional evidence

`qualify_affine_mixture` evaluates the actual frozen bounded mixture once and
generates continuous affine and frozen Euler-grid reference enclosures. It uses
the low-level bounded reference arithmetic, not an analytic/Gaussian/MLMC pass
report. All thresholds are explicit in `MixtureQualificationPolicy`, bound to
the complete request, model, current source and mixture policy. Nothing is
selected, relaxed or expanded after a failed check.

For the exact rational value of the produced floating-point functional, its
maximum distance to each endpoint of the saved continuous interval is a bound
on that particular output's absolute error against the declared affine law.
Distance to the Euler-grid expectation encloses the retained approximation's
combined branching, closure, pruning/normalization and implementation error.
It does not separately identify closure and roundoff. Both individual entries
stay `NOT_IDENTIFIABLE`; no zero-closure, KL, full-distribution, unseen-region or
nonlinear guarantee follows. The saved signed grid-minus-continuous interval
encloses time bias. The total bound is calculated directly, not by mechanically
adding incompatible components. Model error remains unknown. Sampling is not
applicable to this deterministic approximation; that is not perfect accuracy.

Reference widths, retained-functional error, time bias, direct total error,
scaled transition growth and reference operations are checked against frozen
thresholds. Positive and negative analyses preserve their policy hash and all
component enclosures.
The total-error threshold may not exceed the complete request's accuracy target;
a loose operator threshold cannot widen a stricter requested tolerance.
An unresolved certificate cannot pass. The result retains
the original `APPROXIMATION_ONLY` status and `scientific_qualification=false`.
`PASSED` means those actual numeric checks, not permission to run formal work.

`affine-mixture-qualification` is an explicit pilot-only, restart-only shared
adapter. Its source job retains the existing mixture arm. Kernel work and
fixed worst-case reference-operation allowance (two 200,000-operation components
plus one subtraction) together must fit the million-work quota. Lowering the
qualification's measured-operation threshold never lowers that resource proxy;
the physical state is still four, with existing candidate/live pools plus a
fixed 64 MiB rational/certificate allowance. The frozen qualification job cap
must fit both the mixture cap and the shared 1800 s pilot ceiling. Wrong stage
or job caps are refused before reservation/input access. Direct owner admission
refuses test/final-eval before grant/package lookup, even if a caller supplies
a generic pass report. No budget, grants or store are created by this adapter.

The actual supervisor charges native worker occupancy and keeps failed numeric
checks as completed artifacts. The analysis alone still says
`OWNER_SETTLEMENT_REQUIRED`: successful settlement, exact source/policy linkage,
target-output checking and an independent saved-proof reader are required before
formal target admission. Those gates are not replaced by this producer;
the existing fixture/nonlinear formal mixture refusal is retained. The separate
[settled target adapter](managed-mixture-admission.md) consumes the actual owner
source chain; its independent reader and interrupted-target verification remain
required. This is not a study,
manuscript result, method-wide approval or real-model qualification.
