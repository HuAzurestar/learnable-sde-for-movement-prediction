# Bounded cubature mixture: implementation boundary

This explicit four-state SI kernel is an approximation, not an admitted
scientific comparison or a replacement for path Monte Carlo. The explicit
`affine-mixture-chunk` / `synthetic-mixture-chunk` adapters reuse the existing
shared owner, resource plans, budget ledger and worker-control protocol. They
create no ledger, grants or scientific qualification. Before reservations the
owner checks the frozen job cap. Formal execution is refused before protected
reads, including a generic operator qualification report; fixture/pilot modes
cannot consume test/final-eval blocks either. A dedicated method qualifier
remains required.

`MixturePolicy` freezes the full request, model and current source hashes,
component cap (1–32), scaled merge distance, absolute prune threshold, four
positive state scales, maximum discarded mass, work quota and job cap. There
are no default scientific thresholds. Requests exceeding the frozen work
quota are refused rather than reduced to a smaller grid or component count.
Settings and policies reject malformed, nested or oversized primitive fields
before dataclass copying/serialization; manifest scale arity and primitive
values are checked before tuple construction. The compiler performs the
settings check before serializing arm/request identities.

At each Euler grid step, each live Gaussian produces eight equally weighted
symmetric-root cubature points. Each point is pushed through the frozen drift
and becomes a Gaussian with the additive process covariance for that step.
There are at most `8 * component_cap` candidates and at most `component_cap`
retained components. Candidates are processed by descending absolute weight;
generation order resolves ties. A candidate joins the nearest current mean in
the explicitly scaled Euclidean distance if within the threshold. Once the
cap is full, joining the nearest cluster is forced. Otherwise a new cluster
is created. Counters distinguish threshold merges from forced merges.

Pairwise moment merging preserves total weight, mean and second moment:
with `t = w_b / (w_a + w_b)`, the covariance is
`(1-t) C_a + t C_b + t(1-t)(m_b-m_a)(m_b-m_a)^T`.
This identity is also described in
[Runnalls, Gaussian Mixture Reduction (2007), section II.B](https://www.cs.kent.ac.uk/pubs/2007/2797/content.pdf).
Our selection rule is independently specified above, not that paper's KL-bound
selection. Its positive-definite assumptions do not certify this kernel's
potentially singular components. Preserved moments do not imply preserved
density or tail probabilities.

Intermediate moment accumulators are not sampled, accepted as live components
or saved. Final reduced component covariances and the terminal aggregate
covariance must be finite, symmetric and pass the strict PSD root check. A
negative eigenvalue causes `NUMERICAL_FAILURE`; no clipping, jitter,
symmetrization repair or covariance projection is performed. This includes
negative eigenvalues arising from floating-point roundoff.

Pruned weights remain absolute, and cumulative discarded mass stays visible.
Retained weights are not renormalized between grid steps. The final functional
is normalized by retained mass and explicitly describes the retained
approximation, not the original law. Exceeding the discarded-mass cap or
pruning all mass fails. No pruning error bound is asserted for an unbounded
endpoint functional. Reference, time-grid, closure and model errors remain
`NOT_IDENTIFIABLE`. Sampling error is zero only because the approximation is
deterministic; it has no sampled confidence interval.

Completed-grid checkpoints contain bounded live weights, means, covariances,
compact parent/merge lineage commitments, reduction counters and cumulative
discarded mass. A rolling hash commits the sequence of live reductions, not
a recoverable exponentially expanding ancestry tree or authenticity proof.
The state is limited to 64 KiB and binds the full policy/request/model and
NumPy environment. There is no sampled RNG, and the environment marker says
so. Progress uses a frozen conservative operation proxy, not measured FLOPs
or a wall-time guarantee. A kernel callback replay is not a supervisor
save/ACK/charged linked-resume proof. Only these dedicated adapters accept the
64 KiB checkpoint limit; existing methods retain their 16 KiB worker limit.
The watchdog, 80% signal, ACK grace and original-arm settlement are unchanged.

An engineering integration test runs the actual bounded affine-mixture worker,
receives the owner's real 80% signal, completes a partial-grid save/ACK, verifies
native worker-tree stop before settlement, then reopens the physical store and
resumes only through the verified checkpoint/recovery registry. It compares
full functional/lineage/output content with an uninterrupted unit control and
checks a fresh owner receipt and all-attempt charges on the same original arm.
Resume permission is distinct from evaluate permission. At most two 60 s linked
continuations are declared before fixture creation; only new verified partial
saves may continue, never timeout/fused/numerical/admission failures. No forced
save, fake clock, live budget extension or sleep-only worker is used. This is
an engineering test, not formal mixture qualification or a research experiment.
An intermediate Python3.12 CI control completed normally instead of producing
the required partial save. That failure is retained; it is not numerical-method
failure evidence or a portable recovery pass. The revised initial job sizing
targets the signal at measured startup plus half the numerical work,
`job_seconds = min(20, (startup + 0.5 * compute) / 0.8)`, before reservation.
There is no 3 s minimum that can outlast the whole kernel on a fast host; the same
workload/work quota and frozen 60 s job cap remain. The signal, hard deadline and
ACK grace themselves are not modified.

Engineering regressions cover moment-preserving reduction, the affine Euler
moment limit, distinct nonlinear component centres, pruning accounting and
cap failure, strict PSD refusal, complete policy bindings and exact bounded
kernel replay. They do not establish long-horizon accuracy, tail calibration,
cost superiority, real-model validity or formal admission.
