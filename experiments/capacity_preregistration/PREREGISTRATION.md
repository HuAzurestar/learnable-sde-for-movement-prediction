# NEX-381-v4: capacity and structure preregistration

Status: **draft; training gate closed pending v4 mathematical review and five-dimension review**. This is a versioned replacement of frozen v3 at `7765dc002732349018dbf0d494f78046a88ddf30`; it does not rewrite v1, v2, or v3. The executable source of truth is `experiment_matrix.json`; prose below explains that contract but may not relax it.

## Questions and arms

Eight arms are frozen: I1 `K={1,3,10,30}` with stored counts `{5,15,50,150}` (effective DoF `{4,14,49,149}`), and neural `H={8,32,64,128}` with counts `{124,1252,4548,17284}`. The structure contrast is I1-K10 versus NN-H008. It is a **2.48× parameter-count, non-parameter-matched fitted-pipeline comparison**, not an equal-parameter, capacity-matched, or pure architecture effect.

Every arm resolves the same manifest, split, state adapter, five paired seeds, RNG, evaluation, and failure contracts. Neural arms additionally resolve the same environment feature, architecture, Adam, and Euler-Maruyama contracts. The matrix validator fails if an arm drops or changes a reference.

## Data and model coordinates

The only adapter is `data.loader.to_phase_space_1d(segment, coord=0)`: it reads parquet `x` through `Segment.x[:,0]`, derives `V_i=(X_{i+1}-X_i)/(t_{i+1}-t_i)` with last-value fill, and never reads `y`. State is always `[normalized_position_X, normalized_velocity_V]`. Coordinatewise mean and population standard deviation are fit on all adapted train timepoints, must be finite with `std>=1e-8`, and are then immutable.

The neural environment is exactly one scalar: the arithmetic mean of finite `solar_elev` values in the inclusive condition window matching the segment. `day_fraction` and all other fields are excluded. Its mean and population standard deviation are fit on train segments only and reused everywhere; missing, empty, nonfinite, or near-constant inputs fail before fit. Thus raw `[x,y]` transitions and the two-column `extract_features(...,"solar_elev")` helper are not admissible adapters for this study.

Neural input order is `[X,V,solar_elev]`. Each network is `Linear(3,H)-Tanh-Linear(H,H)-Tanh-Linear(H,2)`, with Xavier-uniform tanh-gain weights and zero biases. Time is not an input. Drift is elementwise clipped to `[-10,10]`; `dt_scale=60 seconds`. Adam is fully frozen in the matrix, including betas, epsilon, weight decay, flags, batching, shuffle, and no gradient clipping.

Diffusion adds noise to both state coordinates. Its parameter is `ell=(ell_X,ell_V)` in that order, initialized to `(log(0.1),log(0.1))`, with `B=diag(exp(ell_X),exp(ell_V))` and `a=BB^T`. A bridge correction `a grad(log h)` therefore acts first on X and then on V. Neural rollout is left-endpoint Euler-Maruyama with at most one-second steps and no adaptivity; I1 uses its exact Gaussian transition kernel on the same event grid.

## Seeds, fitting, and stopping

All arms use paired seeds `20260814` through `20260818`. Each seed deterministically spawns separate train, rollout, and bootstrap streams. This replaces v3's single-seed, seed-conditional claim: capacity and structure inference now covers both training initialization and segment sampling via a hierarchical paired bootstrap.

I1 runs EM for at most 50 iterations and stops on relative train-NLL improvement below `1e-5`. Neural arms optimize the registered one-step normalized-state Gaussian NLL for at most 300 epochs; validation patience is 30 completed epochs with absolute `min_delta=1e-5`, and an exact validation tie selects the earliest epoch. Evaluation remains locked until all registered successes and failures are frozen.

## Metrics and event

Energy is computed on the complete normalized endpoint state `[X,V]` using the finite-sample U-statistic

`M^-1 sum_m ||z_m-y|| - [2M(M-1)]^-1 sum_{m!=n} ||z_m-z_n||`, with `M=1000`.

HDR90 is a two-dimensional Gaussian KDE on the same 1000 endpoint states. Sample covariance uses denominator `M-1`, Scott factor `M^(-1/6)`, and kernel covariance `H=f^2 S+1e-9 I`. The threshold is the `ceil(0.10M)`-th ascending forecast-sample density. Truth is inside when its density is greater than or equal to the threshold, so ties and the boundary are included. Report coverage and `abs(coverage-0.90)`.

The event is `hit_X_origin_interval_v1`. Its coordinate is only `state[0]=X`; `A=[-0.5,0.5]`, `D=[-8,8]`, and the time grid contains zero, every one-second point, and the exact endpoint. A path hits if X lies in closed A at any grid point, including start or endpoint. Terminal-X histogram edges are exactly `[-inf,-8,-4,-2,-1,-0.5,0,0.5,1,2,4,8,+inf]`; underflow and overflow are retained.

Existence and exclusion use the same paths, with exclusion exactly `not E`. Require `|pi_exist+surv_excl-1|<=1e-12` and full-support reconstruction L1 at most `1e-12`; both strata must be nonempty. Analytic-prior versus Monte-Carlo error is a separate diagnostic.

For robustness, only coordinate zero defines `DeltaX=X_T-X_0`. On normalized train trajectories compute once `sigma_hist=1.4826 median(|DeltaX-median(DeltaX)|)` and `delta=0.3 sigma_hist`. The three result directions are distinct records: `base`, dilation (expand, merge, clip to D), and erosion (shrink, delete empty intervals). Nonfinite/zero scale or empty erosion is failure/NA, never an invitation to change delta.

## Inference and result records

Each of 2,000 hierarchical paired bootstrap replicates resamples the five seeds, then resamples segment IDs inside each selected seed; identical draws are used for both arms. Effects are right minus left. Holm adjustment is performed within contrast family and metric. Family-wise intervals use the registered bootstrap max-absolute-deviation rule. The exact p-value and interval algorithms are in the matrix.

`result_template.csv` has primary key `(preregistration_id, execution_git_sha, arm_id, seed, delta_probe_direction)` and carries separate base/dilation/erosion rows. `contrast_result_template.csv` has primary key `(preregistration_id, contrast_id, metric)` and carries left/right IDs, effect, pointwise and simultaneous intervals, raw p, Holm rank/family/adjusted p, and decision. Validators check both headers and the three direction records.

## Gate and acceptance

`count_parameters.py` validates manifest fields, paired seeds, adapter, environment, neural architecture/optimizer/rollout, metric/event/bin definitions, every arm reference, registered contrasts, and both result schemas. Negative tests mutate each contract and must fail fast. Training cannot begin from this document: mathematical review must first accept the affected model/event/statistical definitions, then the five-dimension reviewer must accept S1/S2, then CRO may review.
