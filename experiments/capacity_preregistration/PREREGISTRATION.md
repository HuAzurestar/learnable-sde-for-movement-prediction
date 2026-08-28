# NEX-381-v7: exact contrast-template preregistration

Status: **draft; training gate closed pending contrast-template differential review**. v7 is a validator-only successor of frozen v6 commit `646e62ec27d586a5793a0a4759b595d4ce758d3b`. Beyond version and supersession metadata, it does not alter the v6 matrix or the model, event, metric, and inference semantics mathematically accepted in v5. Earlier frozen versions are not rewritten.

## Inherited execution and mathematical contract

All v6 structured fields and exact matrix validation remain unchanged: I1 fits and exact-rolls raw `[X,V]` before per-grid affine conversion to common normalized state; bridge/event coordinates obey the frozen chain rule; solar alignment is determined by two hashed one-to-one manifests; seed and eval segment are crossed factors; Energy is the M=1000 U-statistic; HDR uses the accepted 2D KDE and “at least 90.1% forecast-sample mass” wording; event/A/D/bins and delta read normalized X only; inference has two fixed metrics, type-7 intervals, fixed 3/3/1 multiplicity families, complete-cube failure propagation, and exactly 14 contrast rows.

The v6 validator continues to compare split/leakage, RNG, I1/neural training, solar alignment, rollout/event grid, and delta failure objects for full equality. Its canonical parsed-matrix SHA-256 remains a backstop for every matrix field. The 14 reviewer matrix counterexamples and the unlisted-field mutation remain negative tests.

## v7 contrast-template lock

The contrast CSV is now checked at two levels:

- each ordered row must match the registered `contrast_id`, left arm, right arm, metric, `right_minus_left` effect, five paired seeds, metric-specific Holm family, 3/3/1 family size, alpha 0.05, and bootstrap B=2000;
- a SHA-256 of the canonical parsed CSV (JSON encoding of the complete header-and-row cell matrix) locks every template cell, including fields not covered by a named semantic check.

The canonical form is independent of CSV newline style and quoting while remaining sensitive to every parsed cell and row order. Any fixed-cell drift makes `count_parameters.py --templates-root <root>` report a nonempty `matrix_errors` list and exit 1.

CLI-level negative tests independently mutate the left arm, right arm, effect direction, paired-seed count, rejection alpha, and bootstrap count. Every case must emit its field-specific error plus the canonical contrast-template digest error. An additional mutation outside those named fields proves the full-template digest backstop.

## Delivery gate

v7 changes validation only, so the accepted v5 mathematical review and passed v6 matrix review are inherited. Contrast-template differential review must close S1, and `multica issue pull-requests NEX-381` must show the real PR before CRO. Until both gates pass, NEX-381 stays `in_progress` and training remains forbidden.
