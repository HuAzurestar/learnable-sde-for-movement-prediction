# Frozen affine propagation references

The first propagation research slice provides four-state constant-affine Itô
references for `dX = (A X + b) dt + L dW`, in `[x,y,vx,vy]` order with metre,
metre-per-second and second units. Position drift equals velocity and noise acts
only on velocity. There is no model fitting in these recipes.

`domain.frozen_dynamics.FrozenDynamicsPackage` retains canonical JSON bytes and
validates expected content/parameter hashes, units, kinematics, noise convention,
synthetic provenance and fixture-only qualification. Returned manifests are
detached copies. `application.propagation_inputs.validate_oracle_input` also
checks the expected package identity and current numerical implementation hash.
A changed implementation requires a newly frozen package; naming a branch is
not a content identity. Research workers must pass this composition check before
using the numerical functions and still pass shared runtime admission.

`inference.affine_oracle.exact_transition` computes `F`, drift offset and process
covariance. A short Van Loan block exponential integrates covariance, then the
semigroup identities `F(2t)=F(t)^2`, `c(2t)=c(t)+F(t)c(t)` and
`Q(2t)=Q(t)+F(t)Q(t)F(t)^T` extend to long horizons. This avoids the growing inverse
block in a single long stable-horizon exponential and supports singular drift
matrices without inversion. The reference is numerical float64, not an exact
arithmetic result. The declared limit is 60 doubling operations; overflow or a
materially indefinite covariance fails explicitly. Symmetrization removes
floating-point skew; covariance eigenvalues are never clipped or jittered.

`exact_moments` includes Gaussian initial covariance. The endpoint half-space
functional supports open/closed boundaries, singular distributions and stable
Gaussian upper tails using `erfc`. It computes an endpoint probability, not a
continuous first-passage event or PDE solution. General region integrals and
nonlinear dynamics are not provided by this slice.

`experiments.pirc27.oracles.oracle_suite` generates stable, near-critical,
anisotropic/correlated-noise and uncertain-initial-state fixtures with frozen
horizons. `matrix_cardinality` checks a maximum of 10,000 cells before a caller
materializes the Cartesian product. No API here starts research jobs, grants
access to trajectories or publishes scientific conclusions.

Validation:

```console
python -m pytest tests/test_propagation_oracles.py -q
```

The tests compare against integrated Brownian closed forms, stationary oscillator
moments, independent Gauss-Legendre covariance quadrature and semigroup identities.
They also exercise package/code tampering, input covariance refusals, overflow,
degenerate boundaries, rare Gaussian tails and global RNG preservation.

Numerical reference: C. F. Van Loan, *Computing integrals involving the matrix
exponential*, IEEE Transactions on Automatic Control 23(3), 395–404 (1978),
[doi:10.1109/TAC.1978.1101743](https://doi.org/10.1109/TAC.1978.1101743).
