## Scope and freeze rule

This directory preregisters the parameter-capacity dimension requested by NEX-381. It defines experiments and audits parameter counts; it does not authorize private-data retraining. Execution starts only after mathematical review, five-dimension review, a locked data manifest, and a neural runtime implementation whose count matches the declaration.

The immutable run identity is the tuple `(preregistration_id, execution_git_sha, dataset_id, global_splits_sha256, arm_id, seed)`. A result without every element is invalid.

## Parameter definition

Capacity means stored trainable tensor elements (`requires_grad=True`), not effective statistical degrees of freedom, FLOPs, optimizer state, fixed hyperparameters, or train-split normalization buffers.

| Family | Capacity | Stored trainable elements | Included |
|---|---:|---:|---|
| I-1 | K=1 | 5 | Gamma/a/c/g/prior_logits, each length K |
| I-1 | K=3 | 15 | same |
| I-1 | K=10 | 50 | same |
| I-1 | K=30 | 150 | same |
| neural | hidden=(8,8) | 124 | 3->8->8->2 biased drift MLP + 2 diagonal log-diffusion |
| neural | hidden=(32,32) | 1252 | same contract |
| neural | hidden=(64,64) | 4548 | same contract |
| neural | hidden=(128,128) | 17284 | same contract |

For I-1, the prior logits have only `K-1` effective degrees of freedom, so effective DoF is `5K-1`; the registered storage/training count remains `5K`. `kappa` and `dt_ref` are fixed. For neural, the input is exactly `[state(2), environment(1)]`; time is not a feature. Adding time changes the (8,8) count to 132 and is a new preregistration. Feature mean/std are train-only buffers. The current neural class is a skeleton with zero trainable elements, so all neural arms must fail readiness rather than execute or fall back.

Run `python -m experiments.capacity_preregistration.count_parameters` to reproduce the table and validate every matrix row.

## Scientific questions and claim boundary

1. Within I-1, how do predictive score, calibration, convergence, and bridge/dual diagnostics change as K grows?
2. Within the frozen neural contract, how do the same outcomes change as width grows?
3. How does I-1 K=10 (50 parameters) compare with neural (8,8) (124 parameters) at the same order of magnitude?

Question 3 is not an equal-parameter causal architecture comparison: the ratio is 2.48 and the estimators differ (EM versus Adam/NLL). Report it as a “similar-order fitted-pipeline comparison,” never “same parameter count” or “architecture alone caused the difference.”

## Data and leakage controls

Before fitting, create `capacity_data_manifest.json` beside the private data with the required fields in `experiment_matrix.json`. Hash the actual main parquet and `global_splits.json`; refuse stale or absent hashes. Split by `file_id`. Fit normalization only on train, use validation only for neural checkpoint selection, and keep evaluation locked until every planned arm is either frozen or registered as failed. Use the same ordered segment IDs and the frozen seed `20260814` in every arm.

Data amount is fixed inside this experiment. If new terrain/OSM/trajectory data arrives, it creates a new `dataset_id` and a complete new matrix; never mix old and new manifests within a capacity curve. Alongside every result report training segments, transitions, unique files, duration distribution, parameter-to-transition ratio, and parameter-to-file ratio.

## Metrics and inference

- Primary: canonical half Energy Score `E||X-y|| - 0.5 E||X-X'||`; the legacy doubled convention is forbidden.
- HDR90: endpoint membership in the 90% highest-density region; report coverage, absolute calibration error, and paired interval.
- Convergence: success flag, iterations/epochs, objective trace summary, selected checkpoint, wall time, memory, and all failures.
- Dual/bridge: on full horizon, check `pi_exist + surv_excl = 1` and `pi*p_exist + (1-pi)*p_excl = p`; report absolute scalar error and reconstruction L1. Use common random paths for paired models when supported.
- Robustness delta: compute `sigma_hist = 1.4826*MAD` from train-split endpoint displacement after the frozen normalization. Use `delta=0.3*sigma_hist` only to dilate and erode the predeclared bridge event region. It is not a fit tolerance, HDR bandwidth, stopping threshold, or post-hoc success margin. If `sigma_hist=0` or non-finite, mark the probe failed.

Use paired segment-level block bootstrap with B=2000, resampling identical `segment_id` blocks for every contrast. The two within-family scans are separate multiple-comparison families; apply Holm adjustment to their three adjacent contrasts and show simultaneous 95% intervals. The single cross-family contrast is reported separately. Use only seed `20260814`; any additional diagnostic rerun is labeled non-confirmatory and cannot replace it.

## Budgets, stopping, and failure registration

I-1 uses EM with max 50 iterations and relative train-NLL tolerance `1e-5`. Neural uses Adam, learning rate `1e-3`, batch size 256, max 300 epochs, validation-NLL patience 30 and `min_delta=1e-5`; ties choose the earliest epoch. Both use 1000 forecast samples per evaluation segment and B=2000.

These budgets are fixed within a family, not compute-matched across families. Report wall time and peak memory. A crash, NaN, unavailable component, count mismatch, hash mismatch, or nonconvergence stays in the intention-to-run denominator. Do not silently retry with new settings, drop the seed, impute a metric, or substitute another model.

## Execution order and minimum smoke

1. Freeze mathematical review decisions and matrix version.
2. Lock code SHA and private-data manifest; validate no file/segment leakage.
3. Run the count validator and instantiated I-1 count check.
4. Smoke one synthetic seed for I-1 K=1/K=30 and neural H=8/H=128: one train batch, one validation batch, two evaluation segments, 32 forecast samples, B=20. Smoke validates wiring only and produces no scientific result.
5. Freeze the smoke failure ledger. Then run I-1 capacity arms, neural capacity arms, and finally the predeclared structure contrast.
6. Populate `result_template.csv` without changing columns; review before unlocking evaluation summaries.

## Confounds and controls

| Risk | Required control |
|---|---|
| Data amount changes with parameter count | identical manifest and ordered splits for all arms; new data means a new full matrix |
| Larger models see more optimizer updates | fixed family-specific update budget and stopping; report actual updates |
| Model family is confounded with estimator | call the cross-family result a fitted-pipeline comparison; do not claim pure structure causality |
| Parameter count hides prior-logit redundancy | report stored count and I-1 effective DoF side by side |
| Normalization leaks evaluation data or adds hidden parameters | train-only statistics registered as non-trainable buffers |
| Neural skeleton silently maps to I-1 | readiness failure is mandatory until runtime count and contract tests pass |
| K=30 has empty/degenerate modes | report occupancy, collapsed modes and numerical diagnostics; no pruning after seeing eval |
| MC noise masks capacity effects | common seed/segment blocks, 1000 samples, paired bootstrap |
| Evaluation is repeatedly inspected | lock eval until arms/failures freeze; no tuning after unlock |

## Separate handoffs

Mathematical review must decide: whether `5K` or `5K-1` is the primary comparability axis; whether the frozen neural diffusion parameterization is legitimate; whether the 2.48 ratio supports the phrase “similar order”; exact tolerances for the two dual identities; and whether the delta region perturbation matches the theory. If any item is rejected, record the rejection and issue a versioned replacement contrast before training.

Implementation work, only after approval: implement and register EnvDriftNet with the frozen input/diffusion contract; add runtime-count, gradient, state-dict and RNG tests; create and hash the private data manifest; add a runner that resolves shared protocol references; implement paired segment-block bootstrap/HDR90/dual outputs; then run smoke and freeze the failure ledger. This implementation work must not revise the scientific definitions in the same change.
