# NEX326 process v2

This directory is the executable PIRC-19 contract for the frozen NEX326 matrix.
`experiment.json` is authoritative for numbering, lineage, controls, subconfigs and
mechanism gates. It contains exactly 22 numbered arms in the frozen 8/4/5/5 groups;
C-4 and C-8 are explicitly unnumbered, and `17-terrain` is an arm 17 subconfig.

Validate the specification without touching data:

```console
python -m experiments.nex326_process --validate-only --output ignored
```

Audit the critical estimator, transfer, Reptile, and bridge routes separately from
scientific validity:

```console
python -m experiments.nex326.fidelity \
  --output experiments/nex326/implementation_fidelity_report.json
```

The report distinguishes a real implemented route from paper-equivalent fidelity.
In particular, the Arm 7 scale search, Arm 9 joint drift/covariance grid, bounded
linear-Gaussian Reptile loop, and finite-particle Schrödinger bridge retain explicit
scope limits and cannot support a formal scientific verdict by themselves.

Run every arm/subconfig on a versioned cohort:

```console
python -m experiments.nex326_process \
  --implementation-fixture \
  --output .local/nex326-process-v2
```

The full PIRC-20 release can instead be consumed without generating another
trajectory JSON. The adapter materializes the hash-bound sources once for all arms:

```console
python -m experiments.nex326_process \
  --cohort /private/pirc20/releases/<cohort-id>/cohort.json \
  --pirc20-trajectory /private/trajectory/unified_full_leg.parquet \
  --pirc20-condition-root /private/cond_slices \
  --final-eval-unlock <cohort-id> \
  --scope-policy experiments/nex326/pirc19_scope_policy.json \
  --output /private/pirc20/runs/<run-id>
```

Omit `--final-eval-unlock` while preparing or validating the training side; the
adapter will not read final-evaluation trajectory or condition rows. The multi-seed
command accepts the same PIRC-20 options and reuses one materialized cohort across
seeds. Aggregate results from the sealed full-source loading audit are recorded in
`pirc20_runtime_audit.json`; it contains no row-level identities or locations.

The PIRC-20 runtime also prevents a resampled two-point segment from initializing a
rollout at its final target. This dataset-specific execution override is hash-bound in
the RunRecord implementation identity; the receipt-bound legacy `runner.py` and
`cohort.py` remain unchanged. Pass the same `--scope-policy` to the multi-seed command
to apply the approved exclusions (Arms 13, 17, and 22) to every replicate and record
the policy hash in the batch manifest.

### PIRC-21 versioned terrain features

`experiments.nex326.pirc21_adapter.FeatureSnapshotAdapter` consumes an immutable
PIRC-21 FeatureRow snapshot independently of the legacy PIRC-20 condition view.
`FeatureSelection` orders variant IDs, composition IDs, causal interaction IDs, and
optional segment aggregation IDs; changing any selection changes the cache identity
without changing adapter code. An empty selection is the registered no-terrain
configuration.

Interactions are evaluated by `pirc21_interactions.py` after exact FeatureRow to
trajectory alignment. They combine registered spatial fields with velocity derived
only from the current point and its visible predecessor. Each interaction records its
formula, dependency columns, units, validity rule, and stable registry hash. Examples
include `surface.velocity` (`dot(v, gradient)`),
`surface.directional_curvature` (a causal finite-difference approximation of
`v^T H_h v`), and `road.motion` (approach speed, tangential speed, distance rate,
time-to-contact, and speed-normalized distance). Interactions are independently
selectable and are never enabled merely because their source factor is present.

```python
selection = FeatureSelection(
    variant_ids=("surface.orientation", "road.distance_log1p", "road.direction"),
    interaction_ids=("surface.velocity", "road.motion"),
)
```

Call `fit()` before transforming a selection containing `train_only` variants. The
adapter fits those statistics from the snapshot's `train` split only and exposes an
`identity_record` suitable for RunRecord configuration. It includes the snapshot,
content inventory, feature spec/version, ordered selection, and fitted-state hashes.
Final evaluation remains sealed unless the exact dataset ID is supplied. Missing
features remain NaN plus an explicit validity mask; `model_matrix()` may replace the
numeric slot with zero only while appending that mask, and refuses to hide missingness
when validity indicators are disabled.

The old `dem_elev/dem_slope/landcover/has_map` reader remains available through the
PIRC-20 path for reproduction only. The PIRC-21 adapter never reads `has_map` and does
not use it to decide whether a configured experiment is valid.

When a validated seed already exists as a standalone run, combine it with later
replicate roots without rerunning or copying prediction artifacts:

```console
python -m experiments.nex326_multi_seed_compose \
  --cohort /private/pirc20/releases/<cohort-id>/cohort.json \
  --run-roots /private/pirc20/runs/seed-1 \
              /private/pirc20/runs/batch/seed-2 \
              /private/pirc20/runs/batch/seed-3 \
  --scope-policy experiments/nex326/pirc19_scope_policy.json \
  --output /private/pirc20/runs/combined-multi-seed-manifest.json
```

Every referenced run root must be below the output manifest's parent directory. The
composer revalidates the exact 28-slot matrix, every RunRecord and artifact hash, the
scope hash, cohort fingerprint, implementation identity, and distinct replicate seeds.
The resulting cross-replicate summary can be sealed with
`experiments.nex326.multi_seed_receipt`; for a scoped PIRC-20 batch the receipt also
requires 28 successful executions per seed and binds the exact exclusions 13, 17, and
22. `pirc20_r1t_multi_seed_receipt.json` is the checked-in, privacy-safe receipt for
the three-seed R1T evaluation.

Discovery-to-confirmation promotion is deliberately one-way. The checked-in
`pirc20_confirmation_policy.json` binds the discovery hashes and
`pirc20_candidate_selection.json` records the three configurations whose Energy
Score and CEP50 deltas were negative on every discovery seed. HDR90 is screened by
absolute error from the 90% target, not by raw coverage direction; all three
candidates have a known calibration-risk flag. Their frozen external-domain
confirmation result is recorded in `pirc20_geolife_confirmation_receipt.json`.

Regenerate the privacy-safe candidate selection with:

```console
python -m experiments.nex326.candidate_selection \
  --policy experiments/nex326/pirc20_confirmation_policy.json \
  --receipt experiments/nex326/pirc20_r1t_multi_seed_receipt.json \
  --replicate-summary /private/pirc20/aggregates/<batch>/nex326_replicate_summary.json \
  --output experiments/nex326/pirc20_candidate_selection.json
```

Before opening any future confirmation target, prove cohort independence and derive
the execution seeds deterministically:

```console
python -m experiments.nex326.confirmation_readiness \
  --policy experiments/nex326/pirc20_confirmation_policy.json \
  --candidate-selection experiments/nex326/pirc20_candidate_selection.json \
  --discovery-cohort /private/pirc20/releases/<discovery>/cohort.json \
  --confirmation-cohort /private/pirc20/releases/<confirmation>/cohort.json \
  --output /private/pirc20/confirmation/<confirmation>/readiness.json
```

The readiness check fully validates both manifests and requires zero overlap in
sample, file, segment, and independent-block identities. It does not read trajectory
targets. Execution remains blocked until the HDR90 absolute-error noninferiority
gate (frozen at zero degradation) and a zero-overlap confirmation cohort are both
satisfied. A user-partitioned GeoLife release can satisfy the identity-independence
gate, but it is predominantly urban Beijing data and therefore supports only an
external-domain confirmation, not an in-distribution replication.

GeoLife execution requires the cleaned trajectory plus a hash-bound solar-condition
receipt reconstructed from the original GMT timestamps. The final-evaluation split
stays sealed unless the exact cohort ID is supplied as the acknowledgement:

```console
python -m experiments.nex326_multi_seed \
  --cohort /private/pirc20/releases/<geolife-confirmation>/cohort.json \
  --pirc20-trajectory /data/trajectory/geolife_leg.parquet \
  --pirc20-geolife-conditions /private/pirc20/conditions/<solar>/receipt.json \
  --scope-policy experiments/nex326/pirc19_scope_policy.json \
  --final-eval-unlock <geolife-confirmation> \
  --seeds <seed-1> <seed-2> <seed-3> \
  --samples 64 \
  --output /private/pirc20/runs/<confirmation-run>
```

The optional exploratory remediation fits one dispersion multiplier per configuration
and seed using validation targets only. It fixes that multiplier before opening the
already-authorized final-evaluation predictions and emits aggregate results only:

```console
python -m experiments.nex326.dispersion_calibration \
  --cohort /private/pirc20/releases/<discovery>/cohort.json \
  --trajectory /data/trajectory/unified_full_leg.parquet \
  --condition-root /data/cond_slices \
  --candidate-selection experiments/nex326/pirc20_candidate_selection.json \
  --run-roots /private/pirc20/runs/seed-1 /private/pirc20/runs/seed-2 /private/pirc20/runs/seed-3 \
  --output /private/pirc20/aggregates/calibrated/pirc20_dispersion_calibration.json
```

This remediation remains discovery analysis. If retained, the same frozen
validation-only operation must be rerun on the future confirmation cohort before its
final-evaluation split is opened.

After a readiness manifest reports `ready_to_execute` and all three registered seed
runs finish, issue the policy-bound verdict with:

```console
python -m experiments.nex326.confirmation_verdict \
  --policy experiments/nex326/pirc20_confirmation_policy.json \
  --readiness /private/pirc20/confirmation/<confirmation>/readiness.json \
  --cohort /private/pirc20/releases/<confirmation>/cohort.json \
  --run-roots /private/pirc20/confirmation/<confirmation>/seed-* \
  --output /private/pirc20/confirmation/<confirmation>/verdict.json
```

The evaluator rechecks the scoped run and prediction hashes, pairs every candidate
with its frozen reference, reduces rows to independent-block sufficient statistics,
performs 2,000 block-bootstrap draws while preserving seed pairing, and applies the
frozen Holm and three-metric gates. Only aggregate results are written.

The completed GeoLife confirmation ran all 28 approved slots on each of the three
registered seeds (84/84 successful executions), excluding Arms 13, 17, and 22 as
authorized. All three frozen candidates were inconclusive and none was retained:
Arm 6/dt30 failed the primary Energy Score and zero-margin HDR90 gates, while both
Arm 9 candidates failed the primary Energy Score and CEP50 gates. Mechanism checks
passed throughout. This is an independent external-domain assessment, not an
in-distribution replication; GeoLife is predominantly urban Beijing data. The
checked-in receipt contains only aggregate counts and hashes. Public redistribution
of the derived GeoLife release still requires separate legal review.

Every execution follows the same load → samples → features → initialization → train →
checkpoint → inference → metrics → mechanism → RunRecord path. The checked-in cohort
is a deterministic, CC0 implementation fixture. Its records deliberately retain
`verdict=not_assessed`; they are not scientific NEX326 results. A scientific run must
provide a separately versioned cohort with disjoint splits and external endpoint-prior
provenance. Missing external data may be reported as `data_unavailable`, but a fixture
must still prove the implementation path.

Each new RunRecord stores SHA-256 references for the execution-critical source bundle,
the environment lock, and the observed Python/NumPy/Pandas/PyArrow versions. Its
`result_id` binds this complete execution identity together with the specification, run
identity, and cohort fingerprint. TSDE rejects an aggregate when source or execution
identities disagree.

The CLI never chooses the fixture implicitly. Scientific execution must pass an
explicit `--cohort`; `--implementation-fixture` keeps the non-scientific intent visible
in both the command and every resulting RunRecord.

## Supplemental four-dimensional benchmark

The frozen 36-execution matrix remains the direct two-dimensional position-transition
benchmark. `phase_space_benchmark.json` registers a separate, unnumbered comparison
whose dynamic state is `[x, y, vx, vy]`. Velocities use causal backward differences on
the cohort's irregular timestamps, position obeys `dX=Vdt` by construction, and both
drift uncertainty and diffusion are confined to the coupled velocity equation.

The first registered phase-space version is deliberately unconditioned. Its condition
contract requires a spatial lookup `C(x,y,t_optional)` during off-observation rollout;
time-aligned values from the held-out trajectory are not silently reused as a spatial
field. Run one seed with:

```console
python -m experiments.nex326_phase_space \
  --cohort .local/nex326-dsde-20pct-pilot/cohort.json \
  --output /tmp/nex326-phase-space-report.json \
  --samples 64 \
  --seed 20260814
```

For descriptive sampling-seed replication and a compact hash-bound receipt:

```console
python -m experiments.nex326_phase_space_multi_seed \
  --cohort .local/nex326-dsde-20pct-pilot/cohort.json \
  --output .local/nex326-phase-space-dsde20-replicates \
  --seeds 20260814 20260815 20260816 \
  --samples 64 \
  --receipt experiments/nex326/phase_space_dsde_20pct_receipt.json
```

This remains a supplemental exploratory benchmark, not a 23rd frozen arm and not a
scientific verdict. Endpoint velocity is a causal finite-difference diagnostic and is
expected to be noisier than endpoint position.

### Static terrain-conditioned comparison

`phase_space_terrain_benchmark.json` registers the first concrete `C(x,y)` field:
SRTM elevation and slope sampled at every simulated position. The DSDE condition
slice recovers only the file-level origin used by the NEX-313 local projection. The
adapter does not build a nearest-neighbour index over trajectory points, which would
expose the geometry of the held-out future route. Source slice and raster hashes are
recorded in every report.

Run the three matched seeds with explicit external roots:

```console
python -m experiments.nex326_phase_space_multi_seed \
  --cohort .local/nex326-dsde-20pct-pilot/cohort.json \
  --spec experiments/nex326/phase_space_terrain_benchmark.json \
  --condition-root /path/to/cond_slices \
  --srtm-root /path/to/map_data/srtm_zj_hgt \
  --output .local/nex326-phase-space-terrain-replicates \
  --seeds 20260814 20260815 20260816 \
  --samples 64 \
  --receipt /tmp/nex326-phase-space-terrain-receipt.json
```

Then compare it against the matched unconditioned manifest:

```console
python -m experiments.nex326_phase_space_contrast \
  --baseline /path/to/unconditioned/phase_space_multi_seed_manifest.json \
  --candidate /path/to/terrain/phase_space_multi_seed_manifest.json \
  --output /tmp/nex326-phase-space-terrain-contrast.json
```

The committed DSDE 20% contrast is explicitly exploratory. Its primary Energy Score
became worse on all three seeds, so it records `no_observed_primary_metric_gain`; it
does not imply that terrain is generally irrelevant or that a nonlinear velocity model
would behave the same way.

### Directional terrain v2

`phase_space_directional_terrain_benchmark.json` preregisters the follow-up before
examining its three-seed result. It replaces scalar slope magnitude with signed east
and north elevation gradients. If `u` is the normalized uphill direction, the velocity
basis contains both `dot(v,u)` (positive uphill, negative downhill) and
`abs(dot(v,u))`. The latter is the Euclidean distance from `v` to the contour tangent
line; it is unchanged when the arbitrary tangent representative `t` is replaced by
`-t`.

The registered primary comparison is directional v2 versus scalar-terrain v1 on the
same DSDE 20% cohort, seeds, cutoffs, and 64 samples. The exploratory gate requires a
negative mean paired Energy Score delta and improvement on at least two of three
seeds. The recorded pilot passed that gate on all three seeds, with mean Energy Score
delta `-4.4971`. This is a pilot signal, not an inferential verdict; calibration and
velocity diagnostics remain mixed.

The preregistered follow-up matrix in `phase_space_directional_ablation.json`
separates the directional basis into gradient-only, signed-uphill-only interaction,
contour-distance-only interaction, and the combined v2 model. On the same three-seed
pilot, mean Energy Scores were `125.7495`, `125.8819`, `122.1404`, and `122.2647`,
respectively. The paired contrasts therefore locate the observed gain primarily in
`abs(dot(v,u))`, the distance to the unoriented contour line: it improved on
gradient-only for all three seeds by a mean `-3.6091`. Adding signed uphill speed alone
did not pass the gate (`+0.1324`), and adding it to contour distance also did not pass
(`+0.1244`). These are feature-attribution results for this affine pilot only; they do
not establish a general behavioral preference for contour-following movement.

### Terrain-aligned drift v4

`phase_space_terrain_aligned_benchmark.json` preregisters a structural follow-up to
the contour-distance result. Instead of giving an unconstrained affine model another
scalar feature, it decomposes velocity with the local terrain-gradient projection:
`v_normal=P_normal v` and `v_tangent=(I-P_normal)v`. The velocity drift learns one
shared response coefficient for each component plus a signed gradient force. This
allows normal and contour-following motion to decay or persist differently, while the
tangent projection is invariant to choosing `t` or `-t`. At a flat raster cell the
registered convention is `v_normal=0` and `v_tangent=v`.

The primary comparator is contour-distance v3 on the same DSDE 20% cohort, three
seeds, cutoffs, and 64 samples. The exploratory gate is frozen before execution: mean
paired Energy Score must improve and at least two of three seeds must improve. As with
the earlier phase-space runs, secondary calibration, position, and velocity metrics
must still be reported even if the primary gate passes.

The recorded run did not pass that gate. Relative to contour-distance v3, v4's mean
paired Energy Score delta was `+3.3755`, with zero of three seeds improving. Across
the 84 seed-segment pairs, 27 improved and 57 worsened; the 10% trimmed mean delta
remained positive at `+2.5073`, so the result is not explained only by one extreme
trajectory. The secondary velocity endpoint RMSE did improve on all three seeds by a
mean `-0.00981`, and validation velocity-increment RMSE moved from `0.56857` to
`0.56815`. The fitted normal and tangent responses were close (`-0.02557` versus
`-0.02424`). Together these diagnostics suggest that the restricted structural model
slightly regularizes velocity prediction but is too constrained for the endpoint
position distribution; they do not support replacing contour-distance v3.

`phase_space_terrain_aligned_residual_benchmark.json` preregisters one bounded v5
follow-up. It preserves the complete contour-distance v3 affine drift and adds only
`lambda_normal*P_normal*v`; setting `lambda_normal=0` recovers the v3 drift family.
This separates the value of a terrain-aligned vector response from the broad isotropic
restriction imposed by v4. The comparator, cohort, seeds, primary metric, and gain
rule remain unchanged. A stopping rule is also registered: if this nested residual
does not pass the exploratory gate, no further terrain feature variants are added in
this reconstruction cycle.

The recorded v5 run passed the exploratory gate, but only as a small pilot effect.
Relative to contour-distance v3, mean paired Energy Score improved by `-0.16278`
with all three sampling seeds in the same direction; CEP50 improved by `-0.12626`
and HDR90 coverage was unchanged. Averaging each trajectory over its three sampling
seeds, 16 of 28 evaluation segments improved and 12 worsened; the median delta was
`-0.11406`. Across all 84 seed-segment pairs, the median (`-0.08609`) and 10% trimmed
mean (`-0.12781`) were also negative, so the mean is not carried by a single extreme
case. Validation velocity-increment RMSE improved slightly from `0.56857` to
`0.56840`, while endpoint velocity RMSE worsened by `+0.00245`. The fitted structural
coefficient was small (`lambda_normal=-0.001384`). This supports retaining v5 as the
best bounded terrain variant, not claiming a scientific verdict: the three runs vary
prediction sampling only and reuse the same fitted cohort and evaluation segments.

`phase_space_uncertainty_protocol.json` freezes the uncertainty follow-up before its
interval is read. It resamples the 28 paired evaluation segments 10,000 times. Inside
each resample it computes the metric delta separately for the three matched prediction
sampling seeds and then averages those deltas, so the seeds are not treated as 84
independent observations. Run it with:

```console
python -m experiments.nex326_phase_space_uncertainty \
  --baseline /path/to/contour-distance/phase_space_multi_seed_manifest.json \
  --candidate /path/to/terrain-residual/phase_space_multi_seed_manifest.json \
  --contrast experiments/nex326/phase_space_terrain_aligned_residual_vs_contour_contrast.json \
  --output /tmp/nex326-phase-space-v5-segment-bootstrap.json
```

The equal-tailed percentile intervals quantify held-out segment sampling only. They do
not cover training-data, fitted-cohort, or dataset-version uncertainty. HDR90 is not
bootstrapped because the legacy compact per-segment report omitted its inclusion flag;
the analysis records that limitation instead of reconstructing or inventing it.

The recorded Energy Score delta was `-0.16278`, with a 95% paired-segment interval
of `[-0.34787, +0.01303]`; 96.58% of bootstrap draws were below zero. The interval
slightly crosses zero, so the descriptive improvement is not promoted to a stable
effect. CEP50 (`[-1.06742, +0.40763]`) and endpoint velocity RMSE
(`[-0.00194, +0.00812]`) also crossed zero. Consequently v5 remains the lowest
observed Energy Score in this bounded pilot, while the scientific assessment remains
`not_assessed` and requires genuinely independent data to strengthen.

For a bounded run on the registered DSDE Zhejiang holdout, first materialize the
deterministic 20%-by-segment pilot cohort without copying the source parquet into this
repository:

```console
python -m experiments.nex326.dsde_pilot \
  --trajectory /path/to/zhejiang_holdout.parquet \
  --splits ../DSDE-SDE/trajectory/zhejiang_splits.json \
  --output /tmp/nex326-dsde-20pct-cohort.json \
  --fraction 0.2
python -m experiments.nex326_process \
  --cohort /tmp/nex326-dsde-20pct-cohort.json \
  --condition-root /path/to/cond_slices \
  --srtm-root /path/to/map_data/srtm_zj_hgt \
  --output /tmp/nex326-dsde-20pct-runs
```

The two terrain arguments are optional as a pair. When present, the frozen Arm 17
terrain execution samples SRTM at observed positions during fitting and at each
simulated position during inference. It never indexes terrain by a future route point.
The frozen finite-propagation (`fp`) setting is retained as a local Gaussian
mean-closure approximation and is declared in the RunRecord. Without the pair, Arm 17
terrain remains `data_unavailable` rather than receiving fabricated columns.

For training/prediction-replicate evidence, keep the frozen protocol seed in the spec
and provide distinct replicate seeds explicitly:

```console
python -m experiments.nex326_multi_seed \
  --cohort .local/nex326-dsde-20pct-pilot/cohort.json \
  --condition-root /path/to/cond_slices \
  --srtm-root /path/to/map_data/srtm_zj_hgt \
  --output .local/nex326-dsde-20pct-replicates \
  --seeds 20260814 20260815 20260816 \
  --strict-environment
```

`--strict-environment` refuses execution unless the runtime exactly matches
`environment.lock.json`. Without the flag, the observed versions and conformance state
are still recorded and remain part of the execution identity.

Each seed gets a complete 36-execution directory and the batch manifest binds every
per-seed manifest by SHA-256. Multi-seed runs remain `not_combined_verdict` until TSDE
performs an explicitly registered cross-replicate analysis.

The tracked 20% DSDE receipt uses seeds `20260814..20260816`: all three replicas contain
31 succeeded and 5 `data_unavailable` executions (93 succeeded of 108 total). Arm 17
terrain is the additional succeeded execution. The remaining unavailable set is Arm 13
animal transfer, Arm 17 weather, and the three expert-prior Arm 22 variants.

Every succeeded RunRecord hashes the ordered segment IDs for each selected split. TSDE
therefore compares Arm 17 terrain with the Arm 16 Full reference only when their
evaluation-selection hashes match exactly. In the tracked pilot they do match. Across
three seeds, terrain's Energy Score delta is consistently positive (mean `+5.91289`,
lower is better), while CEP50 error is consistently lower (mean `-21.36536`) and HDR90
coverage increases by `+0.11905` on average. The per-seed paired-segment Energy and
CEP50 intervals all cross zero, so this mixed result remains exploratory and
`not_assessed`.

Regenerate the dimensioned PIRC-19 completion audit from the tracked spec, fidelity
report, implementation gaps, and both DSDE receipts with:

```console
python -m experiments.nex326.completion \
  --output experiments/nex326/pirc19_completion_report.json
```

The audit deliberately does not publish one blended completion percentage. It reports
the frozen-contract implementation, current-data execution coverage, replicated
execution coverage, and scientific-assessment coverage separately, and fails if source
hashes, receipt identities, unavailable scopes, or the Arm 22 extension boundary drift.

The accepted PIRC-19 empirical scope is recorded in
`pirc19_scope_policy.json`. Arms 13, 17, and 22 are approved exclusions: Arm 13 lacks a
licensed animal cohort, Arm 17 condition variants are not required in the current
scope, and Arm 22 remains an expert-assisted extension. This excludes 8 of the 36
frozen execution slots. All 28 required slots succeeded on the registered DSDE 20%
pilot; three Arm 17 variants also succeeded as supplemental evidence. PIRC-19 is
therefore complete within the approved scope. This completion status does not promote
the exploratory pilot results to paper-equivalent scientific claims.

After aggregating each seed independently, produce a descriptive cross-replicate table:

```console
python scripts/aggregate_nex326_replicates.py \
  --batch-manifest /path/to/multi_seed_manifest.json \
  --summaries /path/to/aggregate-seed-1/nex326_summary.json \
              /path/to/aggregate-seed-2/nex326_summary.json \
              /path/to/aggregate-seed-3/nex326_summary.json \
  --output /path/to/aggregate-replicates
```

The cross-replicate output reports mean, sample standard deviation, range, and sign
consistency for each registered delta. Three seeds are not treated as sufficient for an
inferential confidence interval; the output remains `exploratory_only/not_assessed`.

After cross-replicate aggregation, create the compact tracked receipt:

```console
python -m experiments.nex326.multi_seed_receipt \
  --batch-manifest .local/nex326-dsde-20pct-replicates/multi_seed_manifest.json \
  --replicate-summary .local/nex326-dsde-20pct-replicates/aggregate-replicates/nex326_replicate_summary.json \
  --output experiments/nex326/dsde_20pct_multi_seed_receipt.json
```

The receipt re-verifies all three per-seed manifests and summaries, their comparison
CSVs, and the cross-replicate table before recording the complete hash chain. Scoped
receipts additionally require the batch, each seed manifest, and the aggregate to
agree on the approved 28-slot policy.

The adapter preserves the DSDE validation/evaluation file partitions, divides the
finetune files into disjoint train/adapt partitions, and records source hashes. Solar
elevation is a declared Zhejiang-centroid approximation derived from timestamps. It
uses the DSDE city label as the meta-learning task unit, with region as fallback. It
does not fabricate missing animal, weather, or endpoint-prior data. Terrain is available
only when both registered condition slices and SRTM tiles are supplied; its file and
tile hashes are bound into the Arm 17 RunRecord. Other affected executions emit
`data_unavailable`. The pilot purpose remains `not_final_scientific_evidence`.

When an independent endpoint-prior feed becomes available, attach it without modifying
the original DSDE pilot cohort:

```console
python -m experiments.nex326.endpoint_prior \
  --cohort .local/nex326-dsde-20pct-pilot/cohort.json \
  --prior /path/to/endpoint-prior-v1.json \
  --output .local/nex326-dsde-20pct-pilot/cohort-with-endpoint-prior.json
```

Before requesting that feed, export a provider package containing only each evaluation
segment's observed inference prefix and forecast horizon. It deliberately excludes all
states after the inference cutoff and the evaluation endpoint:

```console
python -m experiments.nex326.endpoint_prior \
  --cohort .local/nex326-dsde-20pct-pilot/cohort.json \
  --emit-request \
  --output /tmp/nex326-endpoint-prior-request-v1.json
```

The feed must satisfy `endpoint_prior.schema.json`, cover every evaluation segment
exactly once, contain finite positive-definite 2D covariances, and include a responsible
party's explicit attestation that it was not derived from evaluation truth. The output
records hashes for both the base cohort and prior feed; the adapter refuses to overwrite
an existing enriched cohort.

Arm 22's `sb` subconfig freezes the finite-particle dynamic solver at epsilon scale
0.5, eight Brownian-reference time steps, at most 500 log-domain Sinkhorn/IPF
iterations, and marginal tolerance 1e-8. Prediction artifacts retain each sampled
path and convergence diagnostics. Failure to meet the registered tolerance aborts that
execution; it never falls back to the soft-endpoint transform.

Arm 22 is retained in the reconstructed 22-arm contract, but its DSDE endpoint prior is
an expert-assisted extension rather than part of the current PIRC-19 core evidence.
Until an independently attested expert feed exists, all three Arm 22 executions remain
`data_unavailable`; no mock prior is used in the core receipts.

After TSDE aggregation, bind the cohort, all RunRecords, and the aggregate summary
into a portable receipt (only hashes and compact metadata are committed):

```console
python -m experiments.nex326.pilot_receipt \
  --cohort .local/nex326-dsde-20pct-pilot/cohort.json \
  --records .local/nex326-dsde-20pct-pilot/runs \
  --aggregate-summary .local/nex326-dsde-20pct-pilot/aggregate/nex326_summary.json \
  --output experiments/nex326/dsde_20pct_pilot_receipt.json
```

TSDE consumes only the emitted RunRecords:

```console
python scripts/aggregate_nex326.py \
  --records /path/to/.local/nex326-process-v2 \
  --output /path/to/aggregate
```

The aggregator requires the exact 36-execution arm/subconfig matrix, rejects incomplete
arm sets (including the historical same-name eight-arm JSON), `implementation_missing`,
inconsistent Full anchors, and mechanism claims that provide only a boolean without a
statistic and threshold. For directory inputs it also verifies that every artifact stays
inside the records root and matches its declared size and SHA-256. Aggregate summaries
separate mechanism `passed`, `failed`, and `not_run` counts. It also emits
`nex326_pilot_comparisons.csv`: registered Full/internal-reference point differences are
explicitly marked `exploratory_only` and `not_assessed`. Arm 17 terrain is not compared
without a coverage-matched reference, and arm 22 is not assigned an invented no-bridge
reference. When directory prediction artifacts are available, TSDE adds deterministic
paired-evaluation-segment bootstrap intervals (1,000 resamples); these quantify held-out
segment sampling only, not training-seed or dataset-version uncertainty. Closed-form d=2
energy remains explicitly unresampled. The summary hash-binds every generated CSV/SVG
artifact.
