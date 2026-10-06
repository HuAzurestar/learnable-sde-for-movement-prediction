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

Engineering regressions cover moment-preserving reduction, the affine Euler
moment limit, distinct nonlinear component centres, pruning accounting and
cap failure, strict PSD refusal, complete policy bindings and exact bounded
kernel replay. They do not establish long-horizon accuracy, tail calibration,
cost superiority, real-model validity or formal admission.
