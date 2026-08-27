# NEX-381-v6: exact-validator preregistration

Status: **draft; training gate closed pending five-dimension differential review**. v6 is a validator-only successor of frozen, mathematically accepted v5 commit `1081a4ebb0a1f807b0b62799503bb98e1642763e`. It does not alter v5's model, event, metric, or inference semantics and does not rewrite v1–v5.

## Inherited mathematical contract

All v5 mathematical conclusions remain unchanged: I1 fits and exact-rolls raw `[X,V]` before per-grid affine conversion to common normalized state; bridge/event coordinates obey the frozen chain rule; solar alignment is determined by two hashed one-to-one manifests; seed and eval segment are crossed factors; Energy is the M=1000 U-statistic; HDR uses the accepted 2D KDE and “at least 90.1% forecast-sample mass” wording; event/A/D/bins and delta read normalized X only; inference has two fixed metrics, type-7 intervals, fixed 3/3/1 multiplicity families, complete-cube failure propagation, and exactly 14 contrast rows.

## v6 structured execution fields

Fields that previously mixed structure with prose are now typed JSON objects:

- splits state the single source file, keys, allowed roles, `file_id` leakage unit, and mandatory segment-overlap check;
- RNG states paired scopes, `SeedSequence.spawn(3)`, stream order, deterministic dtype/algorithm, and that diagnostic reruns cannot replace a registered run;
- I1 training states estimator/objective, maximum 50 iterations, the exact relative-NLL stopping comparison and tolerance, checkpoint rule, no validation search, and 1000 forecast samples;
- neural training states the exact NLL components, complete Adam parameters, batching, no gradient clipping, maximum epochs, validation stopping/checkpoint rules, and 1000 forecast samples;
- solar alignment states the timestamp column/conversion, hashed start source, duration expression, inclusive bounds, all-and-only row selection, and order invariance;
- rollout/event state a one-second grid object and event reference/hit-rule object;
- delta failure states every failure outcome, literal `NA`, and two explicit prohibitions against delta replacement/reselection.

## Exact validation

`count_parameters.py` compares each object above for full equality, rather than checking selected tokens. It also retains all v5 field-specific checks and verifies a hard-coded SHA-256 of the canonical parsed `experiment_matrix.json` (`sort_keys=True`, UTF-8, compact separators). The digest is a backstop: any matrix field not covered by a named check still fails.

Negative tests independently mutate all 14 reviewer counterexamples:

1. leakage unit;
2. overlap check;
3–5. I1 maximum iterations, stopping, forecast samples;
6–9. neural objective, stopping, checkpoint, forecast samples;
10. solar row selection;
11–12. RNG stream reuse and diagnostic replacement;
13. event grid reference;
14. delta post-failure reselection.

Every mutation must make `validate_matrix()` nonempty with a category-specific error in addition to the full-matrix digest mismatch. A separate unlisted mutation proves the digest backstop.

## Delivery gate

v6 changes validation only, so the accepted v5 mathematical review is inherited. Five-dimension differential review must close S1, and `multica issue pull-requests NEX-381` must show the real PR before CRO. Until both gates pass, NEX-381 stays `in_progress` and training remains forbidden.
