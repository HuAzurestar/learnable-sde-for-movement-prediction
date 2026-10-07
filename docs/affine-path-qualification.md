# Affine path and importance-sampling pilot evidence

`affine-path-qualification` is an explicit restart-only evidence producer for
Euler MC, constant-additive Heun MC, reversible Heun MC and fixed velocity-drift
importance sampling. It uses the existing shared runner, original method-family
arm, authorization, supervisor and ledger. It creates no stores or grants.
Only a registered `pilot` on train/validation inputs with the `qualification`
execution role is eligible. Formal/test/final-eval consumption is refused before
protected input reads. Generic operator reports do not promote this producer.

The immutable `PathQualificationPolicy` binds the complete request, affine
package, current code, method, two-dimensional proposal, four positive SI state
scales, every numerical threshold and pilot job cap (at most1800seconds). It
must be registered before sampling. The total observed scalar error cap cannot
exceed request tolerance. No negative result changes the frozen grid, sample
count, proposal, tolerance, arm or job budget.

One actual sampler invocation captures detached final sufficient statistics,
never raw paths, in at most16KiB. The analysis binds that final state, RNG scheme,
seed, coupling identity, phase, complete functional output and policy by hashes.
The unmodified sampler output is checked against these saved statistics; there
is no second sampler run to obtain an agreeing value. The shared owner must
still settle actual native-worker cost; `OWNER_SETTLEMENT_REQUIRED` is not an
admission receipt or permission.

Continuous and method-specific finite-grid rational certificates enclose the
declared affine endpoint expectation. IS targets the original Euler law; its
proposal is in the declared noise support and the likelihood estimator remains
unnormalized. Reversible Heun has four physical states and four numerical
auxiliaries initialized as copies, not an eight-dimensional physical model.
The scaled norm of its extended transition repeats the physical scales on the
auxiliary axes. Its finite-grid law does not establish A-stability or nonlinear
solver eligibility.

Evidence separates reference interval width, signed grid-minus-continuous time
bias, realized scalar distance to the grid expectation, realized scalar distance
to the continuous expectation, and estimated sampling uncertainty. The two
observed distances include inseparable sampling/implementation effects; they
are not sampler-roundoff certificates, distribution bounds or mechanically
summed confidence guarantees. The original kernel error budget is retained;
the dedicated analysis explicitly leaves implementation roundoff, separate
propagation approximation and model error unidentified.

Ordinary MC uncertainty uses its saved moment standard error and a normal95
approximation. Zero-hit Bernoulli MC retains a missing standard error and a
positive one-sided exact-binomial-formula upper limit. All-hit Bernoulli MC also
gets positive one-sided uncertainty instead of interpreting zero observed
variance as zero error. These formulas refer to an ideal iid Bernoulli law;
float64 evaluation and PRNG coverage are not certified. A zero-variance
non-Bernoulli estimate is unresolved, not a zero-error proof.

IS uncertainty uses its own weighted log-moment variance and ESS. It never
borrows the independent-binomial MC formula, self-normalizes, clips weights or
claims guaranteed normal-interval coverage. No weighted hits, low ESS, a frozen
ESS-threshold failure or zero estimated weighted variance remains negative
evidence. Numerical failure does not remove the charged cell from the matrix
denominator.

Allocation retains the existing `samples*steps` kernel work proxy and adds a
fixed400001 reference-work allowance (two200000-operation hard caps plus one
signed subtraction), with total at most1000000. A stricter observed reference
operation threshold does not reduce this proxy. Two bounded rational and
serialization pools reserve64MiB; physical resource dimension stays four.
The kernel proxy is not a claim that all solvers use identical FLOPs or drift
evaluations. Native elapsed cost remains independently charged.

`PASSED` means only that this one realized, registered engineering pilot met its
own frozen checks. `scientific_qualification` remains false. Settled source
verification, dedicated formal-target admission, qualification of the target's
own independent output, an independent saved-proof reader, actual supervised
formal interruption/restore and the complete study remain separate required
work. This producer does not qualify all requests, a trained nonlinear model,
rare-event accuracy in general, or PIRC-27 delivery.
