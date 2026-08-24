# Known issues

These limitations are intentionally visible so experimental behavior is not
mistaken for a supported production capability.

## Correctness and numerical stability

1. Energy-score implementations use both the full `2E-E` convention and the
   half `E-1/2E` convention in legacy experiment code. Cross-experiment
   comparisons require a single explicit convention.
2. `models/segment_constant.py` computes a matrix logarithm through eigendecomposition
   and discards the imaginary part. Negative or nearly defective eigenvalues need
   an explicit guard or a more stable implementation.
3. `discrete_to_continuous` uses small additive constants around ill-conditioned
   divisions. It should report or explicitly fall back instead of masking every
   near-singular case.
4. `experiments/backend.py::_load_eval_by_ids` can omit a requested small subset
   because it resamples before filtering. The full evaluation path does not use
   that small-subset pattern.

## Reproducibility and concurrency

5. Some legacy experiment functions still mutate global random state or call
   random sampling without an owned generator.
6. Broad exception handlers remain in a numerical fallback and in the legacy
   experiment backend. Those paths need explicit diagnostics.
7. The experiment backend owns mutable module-level caches and is not thread-safe.
8. On some Windows scientific Python installations, importing pandas after
   selected Torch linear-algebra modules can trigger a BLAS loader conflict. The
   test bootstrap imports NumPy and pandas first as a compatibility guard.
9. Some Windows consoles cannot encode the superscript character printed by the
   verification command. Setting `PYTHONIOENCODING=utf-8` is a temporary
   workaround.

## Incomplete components

10. The neural SDE and Fokker-Planck variants remain research skeletons and are
    deliberately absent from the production registry.
11. Part of the condition-feature alignment remains in the legacy experiment
    backend instead of the declarative data-source adapter.

Issues should be removed from this document only in the same pull request that
adds a regression test for the fix.
