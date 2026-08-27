# NEX-381-v5: capacity and structure preregistration

Status: **draft; training gate closed pending v5 mathematical and five-dimension review**. v5 is a versioned child of frozen v4 commit `1c8bc1288c0cd5845dde8aad5b96ca4ffc0bed84`; it does not rewrite v1–v4. `experiment_matrix.json` is the executable source of truth.

## Arms and claim boundary

The eight arms remain I1 `K={1,3,10,30}` with stored counts `{5,15,50,150}` (effective DoF `{4,14,49,149}`), and neural `H={8,32,64,128}` with counts `{124,1252,4548,17284}`. The structure contrast remains I1-K10 versus NN-H008: **2.48×参数量、非参数匹配的 fitted-pipeline 比较**, never equal-parameter, capacity-matched, or a pure architecture effect.

## Raw I1 dynamics and common evaluation coordinates

`to_phase_space_1d(segment, coord=0)` constructs raw `r=[X_raw,V_raw]`; `y` is never read. Train statistics are `m=(m_X,m_V)` and positive diagonal `S=diag(s_X,s_V)`, and the common evaluation state is `z=S^-1(r-m)=[X_n,V_n]`.

I1 is fit and propagated only in raw `r`, where `dX_raw=V_raw dt`; normalized `z` is never passed to `SegmentConstantSDE`. After every registered evaluation-grid transition, each raw I1 path point is transformed to `z` before Energy, HDR, event, histogram, or delta calculations. Neural arms remain fit and propagated in normalized `z`. This preserves the I1 exact kernel and the common normalized evaluation scale without pretending independent z-scores preserve `dX=Vdt`. The affine statistics are fixed buffers, so I1 still has `5K` trainable elements.

For I1 bridge conditioning, if `h` is expressed on normalized event coordinates, `grad_r log h=S^-T grad_z log h`. The raw correction is `a_r S^-T grad_z log h`; after transformation it is `S^-1 a_r S^-T grad_z log h`. Event `A`, domain `D`, and bins are evaluated after transforming to normalized X. The equivalent raw intervals are `m_X+s_X A` and `m_X+s_X D`; `s_X>0` preserves closed boundaries. Neural bridge correction remains `a_z grad_z log h` in X-then-V order.

## Deterministic solar alignment

The data manifest hashes two UTF-8 JSONL files:

- `segment_start_map`: exactly `(segment_id,file_id,absolute_start_epoch)`, unique by segment ID, complete for every used segment, with matching trajectory file ID and finite integer Unix seconds.
- `condition_file_manifest`: exactly `(file_id,relative_path,sha256)`, unique by file ID and complete for every used file. Paths resolve under `cond_root`; hashes must match. Runtime glob selection is forbidden, and zero or duplicate candidates fail.

The inclusive window is `[absolute_start_epoch, absolute_start_epoch + segment duration]` after condition timestamps are converted to integer Unix seconds. It must contain at least one row, and **every** aligned `solar_elev` value must be finite; filtering nonfinite rows is forbidden. The feature is the arithmetic mean over all aligned rows. `day_fraction` and other columns are excluded. Its train-only population z-score is reused unchanged.

## Model, rollout, metrics, and event

Neural input remains `[X_n,V_n,solar_n]`, with two Tanh layers, Xavier-uniform tanh-gain weights, zero biases, `dt_scale=60s`, elementwise drift clipping at 10, frozen Adam options, and non-adaptive left-endpoint one-second Euler–Maruyama. I1 uses raw exact Gaussian transitions on the same grid.

Energy remains the accepted complete normalized `[X,V]`, M=1000 U-statistic. HDR90 remains the accepted 2D Gaussian KDE with Scott covariance and closed density threshold. With `q=ceil(0.10*1000)=100`, `>=` contains 901/1000 forecast samples when densities are unique and at least that many with ties: the set has **at least 90.1% forecast-sample mass**, not exactly 90%.

The accepted event remains `hit_X_origin_interval_v1`: normalized `A=[-0.5,0.5]`, `D=[-8,8]`, closed membership, start/one-second/end grid, and full-support terminal-X edges `[-inf,-8,-4,-2,-1,-0.5,0,0.5,1,2,4,8,+inf]`. Hit, histogram, and `DeltaX=X_T-X_0` read normalized coordinate zero only. Base/dilation/erosion remain distinct arm rows. The ambiguous `delta_probe_value` column is removed; direction-specific event/dual outputs are the probe results.

## Crossed bootstrap and closed inference records

Seeds and eval segments are crossed factors. Each of B=2000 replicates draws one length-5 seed index vector and one length-`N_eval` segment index vector with replacement, then evaluates their Cartesian product with multiplicity. The same two vectors are shared across all eight arms, both registered metrics, and all seven contrasts. All contrasts in a family therefore share replicate `b` before the max-C calculation.

Only two inferential contrast metrics are registered: `energy_half` and `hdr90_abs_calibration_error`. The latter recomputes `abs(mean(HDR membership)-0.90)` inside each bootstrap arm. Effects are right minus left. Pointwise CIs are percentile 0.025/0.975 intervals with Hyndman–Fan type-7 linear interpolation. Raw p is the plus-one centered-bootstrap two-sided Monte-Carlo approximation, not an exact permutation/randomization p-value.

Multiplicity families are separate by `(metric_id, contrast family)`: I1-capacity has exactly 3 contrasts, neural-capacity 3, and structure 1. Holm uses lexical `contrast_id` to break p-value ties. Simultaneous intervals use the type-7 0.95 quantile of `max_c |theta_b,c-theta_hat_c|` within those same fixed families.

For either metric, the complete `8 arms × 5 seeds × N_eval segments` cube is mandatory. If any cell is missing, failed, or nonfinite, all seven contrast rows for that metric become `unavailable`, with estimate/CI/p/Holm fields literal `NA`; complete-case deletion and imputation are forbidden.

The arm primary key is `(preregistration_id,execution_git_sha,dataset_id,global_splits_sha256,arm_id,seed,delta_probe_direction)`. The contrast primary key adds the same execution/data/split identity to `(contrast_id,metric_id)`. Exactly `7×2=14` contrast rows—the Cartesian product of registered contrasts and metrics—are required; no free metrics or extra/missing rows are allowed.

## Gate

The validator and negative tests lock every v5 clause above plus v4's accepted contracts. This document does not authorize training. Mathematical review must accept v5, the five-dimension reviewer must then accept S1/S2, the real PR link must be confirmed, and only then may CRO review.
