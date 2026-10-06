# Versioned synthetic nonlinear propagation

The `frozen-tanh-dynamics-v1` package is a fixed four-state Itô stress generator,
not a trained model or research grant. It preserves position/velocity kinematics
and additive velocity noise. Drift is `A z + b` with an additional
`amplitude[i] * tanh(position[i] / length_scale[i])` in each velocity coordinate.
Length scales are metres; amplitudes are accelerations. The recipe, parameters,
current implementation, units, context, synthetic provenance and public-generator
scope are content-bound. Scientific qualification remains false.

Euler and additive-noise stochastic Heun, coupled Euler MLMC and unnormalized
drift importance sampling consume the same frozen request and streaming endpoint
machinery as affine methods. Fine/coarse paths share actual adjacent increments.
Completed-chunk statistics and sample-root continuation remain available. Exact
affine transitions and exact discrete Gaussian moments refuse this package.

`cubature_moments` uses eight equal-weight points in four dimensions, mapped
through each Euler step, then adds `dt * L L.T` and re-Gaussianizes. The point
rule follows the spherical third-degree construction in
[Särkkä and Solin, chapter 9](https://users.aalto.fi/~asolin/sde-book/sde-book.pdf).
This discrete assumed-density construction is not an exact nonlinear transition,
continuous Gaussian moment ODE, mixture model, mode switching or PDE. Singular
PSD roots are allowed; negative eigenvalues are refused without projection/jitter.
No more than eight points or fixed 4-by-4 matrices are materialized per step.

Nonlinear reference and time-bias bounds remain NOT_IDENTIFIABLE. Cubature also
retains unknown closure error and APPROXIMATION_ONLY status. No sampling noise
in a deterministic closure does not make its approximation error zero. A tiny
two-well regression counterexample shows that matching mean/covariance can still
misrepresent central probability mass; it is not a preregistered scientific
applicability or gain result. Euler/Heun zero-noise convergence is separately
checked against an independent scalar RK4 calculation.

`propagation_plugin(synthetic=True)` explicitly registers a separate synthetic
adapter; its chunk variant uses `propagation_recovery_plugin(synthetic=True)`.
The affine adapter never silently changes models. The synthetic adapter permits
Euler/Heun MC, MLMC, IS and restart-only cubature, and reserves at least eight
workspace rows even with one path per chunk. Its measured functional estimate is
not labelled an error against a nonexistent nonlinear oracle.

Raw kernel tests and managed fixture grants do not authorize scientific execution
or a new ledger. Actual
research still requires the existing explicit shared root/store, admission,
qualification/preregistration, exposure permission and original arm budget.
