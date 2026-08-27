## Scope and freeze rule

This directory freezes the versioned replacement `NEX-381-v3`. It supersedes, but does not rewrite, frozen v2 commit `5346377f4451c0c0e3774ba71a2b541a1c555f54`. The sole v3 correction is to make the shared state semantics explicit and consistent with `data/loader.py` and `SegmentConstantSDE`: `[normalized_position_X, normalized_velocity_V]`. It defines experiments and audits parameter counts; it does not authorize private-data retraining. Execution starts only after v3 mathematical review, five-dimension review, a locked data manifest, and a neural runtime implementation whose count matches the declaration.

The immutable run identity is `(preregistration_id, execution_git_sha, dataset_id, global_splits_sha256, arm_id, seed)`. A result without every element is invalid.

## Parameter definition

Capacity means stored trainable tensor elements (`requires_grad=True`), not effective statistical degrees of freedom, FLOPs, optimizer state, fixed hyperparameters, or train-split normalization buffers.

| Family | Capacity | Stored trainable elements (primary) | Effective DoF (secondary) | Included |
|---|---:|---:|---:|---|
| I-1 | K=1 | 5 | 4 | Gamma/a/c/g/prior_logits, each length K |
| I-1 | K=3 | 15 | 14 | same |
| I-1 | K=10 | 50 | 49 | same |
| I-1 | K=30 | 150 | 149 | same |
| neural | hidden=(8,8) | 124 | not defined | 3->8->8->2 biased drift MLP + 2 diffusion log-scales |
| neural | hidden=(32,32) | 1252 | not defined | same contract |
| neural | hidden=(64,64) | 4548 | not defined | same contract |
| neural | hidden=(128,128) | 17284 | not defined | same contract |

For I-1, softmax prior logits have `K-1` effective degrees of freedom. Every table/result reports secondary `5K-1` beside primary stored count `5K`; it never substitutes the effective count. `kappa` and `dt_ref` are fixed.

For neural, state is exactly `[state[0],state[1]]=[normalized_position_X, normalized_velocity_V]`; the network input is `[state(2), environment(1)]`, and time is not a feature. The two trainable diffusion values are frozen in coordinate order as `ell=(ell_X,ell_V) in R^2`, `B=diag(exp(ell_X),exp(ell_V))`, and `a=B B^T=diag(exp(2ell_X),exp(2ell_V))`. Thus noise is added to the position and velocity state coordinates, not to two position coordinates. `ell` is state- and time-independent, with no clamp, floor, or softplus. This full-rank `[X,V]` neural model differs from I-1's rank-one underdamped `B=(0,g)^T`; it is only a fitted-pipeline comparator.

Every neural bridge uses `a grad(log h)` in the same `[X,V]` order: correction component 0 acts on `X`, and component 1 acts on `V`. The bridge hit event is nevertheless one-dimensional and reads only `state[0]=X`. No event decision reads velocity. A two-dimensional position state `[X,Y]` is a different model/data/event contract and requires a new preregistration; it cannot reuse this 124-versus-50 comparison.

An underdamped neural alternative would be a new version (H=8 count 114) and require rebuilt arms. Adding time changes the generic (8,8) count to 132 and is also a new preregistration. Feature mean/std are train-only buffers. The current neural class is a zero-parameter skeleton, so neural arms fail readiness rather than execute or fall back.

Run `python -m experiments.capacity_preregistration.count_parameters` to reproduce the table and validate every matrix row.

## Scientific questions and claim boundary

1. Within I-1, how do predictive score, calibration, convergence, and bridge/dual diagnostics change as K grows?
2. Within the frozen neural contract, how do the same outcomes change as width grows?
3. How does I-1 K=10 (50 stored parameters, 49 effective DoF) compare with the frozen `[X,V]` neural (8,8) model (124 stored parameters)?

Question 3 is not an equal-parameter causal architecture comparison: primary ratio `124/50=2.48`, secondary ratio `124/49≈2.5306`, and estimators differ (EM versus Adam/NLL). The only frozen wording is **“2.48×参数量、非参数匹配的 fitted-pipeline 比较”**. Never call it equal-parameter, capacity-matched, or a pure architecture effect.

## Data and leakage controls

Before fitting, create `capacity_data_manifest.json` with every required field in `experiment_matrix.json`. Hash the actual main parquet and `global_splits.json`; refuse stale or absent hashes. Split by `file_id`. Fit normalization only on train, use validation only for neural checkpoint selection, and keep evaluation locked until every planned arm is frozen or registered failed. Use identical ordered segment IDs and seed `20260814` in every arm.

Data amount is fixed. New terrain/OSM/trajectory data creates a new `dataset_id` and complete new matrix; never mix manifests within a capacity curve. Report training segments, transitions, unique files, duration distribution, parameter-to-transition ratio, and parameter-to-file ratio.

## Metrics and inference

- Primary: canonical half Energy Score `E||X-y|| - 0.5 E||X-X'||`; the legacy doubled convention is forbidden.
- HDR90: endpoint membership in the 90% highest-density region; report coverage, absolute calibration error, and paired interval.
- Convergence: success flag, iterations/epochs, objective trace summary, selected checkpoint, wall time, memory, and failures.
- Dual/bridge algebra: state order is `[X,V]`, and `a grad(log h)` acts on X then V. Use the same sampled paths and a discrete per-path boolean hit event `E` that reads only `state[0]=X`; exclusion is exactly `not E`. Require `|pi_exist+surv_excl-1|<=1e-12`. Reconstruct from raw full-support bin counts including underflow/overflow and require `sum_j|p_recon,j-p_prior,j|<=1e-12`. Hit and non-hit strata must both be non-empty; otherwise record `undefined_stratum` and never pass. Report analytic-prior versus Monte-Carlo `fp_mc_l1` separately because sampling/discretization error does not use the algebraic tolerance.
- Robustness delta (one-dimensional position only): coordinate index is frozen to 0. On train after frozen position normalization define `DeltaX_i=state_i[T,0]-state_i[0,0]=X_T_i-X_0_i`; velocity `state[1]=V` is never read for this statistic or hit event. Once for the whole matrix compute `sigma_hist=1.4826*median_i|DeltaX_i-median_j DeltaX_j|` and `delta=0.3*sigma_hist`. Normalize the event to mutually disjoint closed intervals `A=union_j[l_j,r_j]`. Define `A^{+delta}=union_j[l_j-delta,r_j+delta]`, merge overlaps, intersect domain `D`; define `A^{-delta}=union_j[l_j+delta,r_j-delta]`, deleting empty components. Membership uses closed boundaries. Non-finite/zero `sigma_hist` or empty erosion is probe failure/NA, never a reason to replace delta. A `[X,Y]` state or two-dimensional event requires a new preregistration.

Use paired segment-level block bootstrap with B=2000, resampling identical `segment_id` blocks for every contrast. The two within-family scans are separate multiple-comparison families; Holm-adjust their three adjacent contrasts and show simultaneous 95% intervals. Report the cross-family contrast separately. Any seed beyond `20260814` is non-confirmatory and cannot replace it.

## Budgets, stopping, and failure registration

I-1 uses EM, max 50 iterations, relative train-NLL tolerance `1e-5`. Neural uses Adam, learning rate `1e-3`, batch size 256, max 300 epochs, validation-NLL patience 30 and `min_delta=1e-5`; ties choose the earliest epoch. Both use 1000 forecast samples per evaluation segment and B=2000.

Budgets are fixed within a family, not compute-matched across families. Report wall time and peak memory. A crash, NaN, unavailable component, count/hash mismatch, or nonconvergence stays in the intention-to-run denominator. Never silently retry with new settings, drop the seed, impute a metric, or substitute another model.

## Execution order and minimum smoke

1. Obtain v3 mathematical approval, then five-dimension review; v2 approval cannot substitute for v3 review.
2. Lock code SHA/private-data manifest; validate no file/segment leakage.
3. Run matrix and instantiated I-1 count checks.
4. Smoke one synthetic seed for I-1 K=1/K=30 and neural H=8/H=128: one train batch, one validation batch, two evaluation segments, 32 forecast samples, B=20. Smoke validates wiring only.
5. Freeze the smoke failure ledger; then run I-1 arms, neural arms, and the structure contrast.
6. Populate `result_template.csv` without changing columns; review before unlocking evaluation summaries.

## Confounds and controls

| Risk | Required control |
|---|---|
| Data amount changes with count | identical manifest/splits; new data means a new full matrix |
| Larger models see more updates | fixed family-specific budget/stopping; report actual updates |
| Family is confounded with estimator | use the sole fitted-pipeline wording; no pure-structure claim |
| Prior-logit redundancy is hidden | report stored `5K` and effective `5K-1` side by side |
| Normalization leaks or adds hidden parameters | train-only statistics are non-trainable buffers |
| Neural skeleton silently maps to I-1 | mandatory readiness failure until runtime count/contract tests pass |
| K=30 has degenerate modes | report occupancy/collapse/numerics; no post-eval pruning |
| MC noise masks effects | common seed/segment blocks, 1000 samples, paired bootstrap |
| Evaluation is repeatedly inspected | lock eval until arms/failures freeze; no tuning after unlock |

## Separate handoffs

Mathematical review now verifies the v3 state correction: common layout is `[normalized_position_X, normalized_velocity_V]`; diffusion order is `(ell_X,ell_V)`; bridge correction acts X then V; and both the hit event and `DeltaX` read only state index 0. The four v2 items already accepted remain unchanged. Rejection requires another versioned replacement before training, and v3 review must pass before five-dimension review.

Only after approval: implement EnvDriftNet with this exact input/diffusion contract; add runtime-count, gradient, state-dict and RNG tests; create/hash the private manifest; add a runner resolving shared protocol references; implement paired bootstrap/HDR90/dual outputs; run smoke and freeze failures. Implementation may not revise scientific definitions in the same change.
