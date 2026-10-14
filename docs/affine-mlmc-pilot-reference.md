# Bounded affine reference inside an independent MLMC pilot

The explicit `affine-mlmc-qualification-chunk` adapter adds continuous-law and
finest-Euler-grid rational enclosures to a real independent MLMC pilot. It runs
through the existing shared owner, original MLMC arm and pilot budget, not a new
ledger/arm or uncharged owner preflight. It never starts production sampling.

## Frozen inputs and actual work

The cell binds its full frozen affine model/request, original method family,
`MLMCPilotPolicy`, and complete `AffineMLMCReferencePolicy`. The latter freezes
the full level allocation, request/model/code/pilot-policy hashes, reference
width, scaled-transition norm with four SI state scales, operation limit and
job cap (at most1800s). Bias tolerance comes from the already frozen pilot
policy, not a new threshold inferred from output. Three to nine levels, finest
steps at most8192 and the existing cumulative sample/work caps remain enforced.

Only explicit pilot role/mode and training/validation input are accepted.
Registered production seed differs from pilot seed; actual pilot phase2 differs
from production phase1. Missing/substituted policies, formal mode, held-out
input, another method family or wrong stage/cap are refused before worker/data
access. The numerical reference policy's job cap applies before reservation.

The worker measures actual coupled level means, variances and compute-only
costs. Finest steps are `base_steps * 2**(level_count-1)`. It computes an outward
continuous functional enclosure and an explicit Euler finest-grid enclosure,
then bounds signed grid-minus-continuous bias. This is not a last-level mean
extrapolation or comparison against another float64 reference. Frozen width,
absolute bias, scaled growth and operation caps are checked. The resource plan
adds fixed64MiB Fraction/serialization pools; physical state dimension stays4.

## Readiness is not qualification

`bounded_mlmc_pilot_analysis.status` is `NUMERICAL_READY` only when those
necessary numerical checks and original empirical variance/cost/allocation
checks pass. Failures remain `NUMERICAL_FAILED`; computational success may still
publish them. In both cases `qualification: UNQUALIFIED` and
`scientific_qualification:false` remain explicit. The result stays a pilot
fixture. This is not an MLMC complexity theorem, approved production allocation,
family-wide scientific qualification, fitted-model approval or a study result.

The original empirical pilot analysis and original functional error components
are retained unchanged. New saved certificates/bias bounds are separate evidence;
the old float-reference unknown is not silently overwritten. A current scalar's
absolute error upper versus the continuous declared law includes random sampling
and implementation effects; it is explicitly **not** a sampler roundoff bound.
Sampling SE/normal intervals remain empirical, and zero observed rare-event
corrections stay unresolved. Sampler roundoff and real model error remain unknown.
Formal owner admission of this stochastic evidence and independent paper proof
validation still require their own implementation; analytic proof cannot be borrowed.

Compute-only level timing is not the budget charge. Only the owner's measured
whole-worker-tree settlement is authoritative; pilot evidence says
`OWNER_SETTLEMENT_REQUIRED`. Same-arm reuse retains charge. Full numerical work,
including reference generation and repeated reference work after a chunk resume,
is inside the supervised worker and existing cumulative arm/deadline.

Chunk recovery saves completed sampling statistics, not a partial rational
certificate. Reference generation is currently restart-only within that worker;
the hard watchdog remains active. No claim of partial-certificate continuation
or free reference replay is made. No automatic resume, retry or budget extension.

Engineering controls use disposable synthetic stores and explicit test grants.
Actual research still needs operator-supplied shared runtime/store, preregistration
and permissions. Other methods, full axes/study/manuscript/verification/review/
acceptance and merged delivery remain required.
