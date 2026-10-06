# Frozen synthetic propagation design preparation

`experiments.pirc27.design.freeze_design` compiles an explicit
`PropagationStudyDesign` into immutable `PropagationStudyManifest` bytes.
This is preparation, not preregistration, authorization, numerical evaluation or
scientific qualification. It creates no ledger, grants, reservations or workers.

The design fixes models and their Gaussian initial distributions, methods and
allocations/proposals/grid/chunk/recovery settings, endpoint functional geometry,
positive horizons, seeds, origin/history cutoff, protocol/input/context/selection
hashes, primary metrics and a fixed stopping rule. Four-state SI/local Cartesian
synthetic packages are required; real, missing or differently dimensioned models
are not silently replaced. Current package implementation identities are checked.
Initial covariance requiring projection is rejected, not repaired.

Axes are immutable, unique and bounded. The candidate product must be at most
10,000 before any Cartesian expansion. A separate conservative 32 MiB manifest
quota is checked before expansion, and the final encoded size is checked again.
Each runnable method configuration is validated against its shared typed resource
contract before matrix construction, including the finest MLMC grid and total
fine/coarse work. No automatic sample allocation or experiment expansion occurs.

Comparators use the same package, initial distribution, endpoint definition,
origin/cutoff, horizon and seed. A common coupling root derives from those inputs,
not method, chunk size or allocation. This declares the pairing identity; it does
not assert identical paths across distinct solvers or independent MLMC levels.
One registered cell has one horizon. This is not first-passage inference.

Arms must be supplied explicitly for exactly the model-family/method/objective
triples. Horizon, seed, region threshold, study ID, package version and numerical
configuration never create extra budget identities. Multiple regions within one
endpoint-probability objective share the same arm. The 86,400 slot-second cap is
unchanged. This compiler neither discovers prior identities nor replaces the
shared store's cross-study identity/conflict and closed-arm checks. An operator
must reuse the existing bindings; renamed identities are not permission to reset
budgets.

The complete expected matrix retains `NOT_IMPLEMENTED` mixture, reversible-Heun
and PDE rows and `INELIGIBLE` affine-only methods on nonlinear packages. These
have reasons and request hashes, not executable bindings. `PLANNED` means only
that the engineering adapter supports the declared input/configuration. It does
not mean scientific eligibility or a passing reference/pilot qualification.

`manifest()` returns detached data. `study_spec(expected_hash=...)` also verifies
the current runtime source and refuses any matrix containing non-runnable rows,
so unsupported comparisons cannot disappear from the denominator. A future
shared disposition workflow is needed to register/run such partial matrices.
For a wholly supported matrix, the returned draft contains all cells and shared
resource bindings but intentionally has no `runtime_binding` or `admission`.
Operators must supply an existing absolute Git-external root/store and obtain
normal shared grants, preregistration, upstream/qualification/pilot decisions and
budget admission before execution. Supplying digest-shaped metadata alone is
not evidence of those checks, and this draft is not an approved study.

`tests/test_propagation_study_design.py` covers immutability, deterministic content
hashing, exact owner preflight compatibility, paired inputs, stable budget arms,
cap-before-expansion, invalid metadata/configuration, retained missing methods,
source freshness and tampered resource bindings without executing research.
