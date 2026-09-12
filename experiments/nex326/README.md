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
  --output /tmp/nex326-dsde-20pct-runs
```

For training/prediction-replicate evidence, keep the frozen protocol seed in the spec
and provide distinct replicate seeds explicitly:

```console
python -m experiments.nex326_multi_seed \
  --cohort .local/nex326-dsde-20pct-pilot/cohort.json \
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
CSVs, and the cross-replicate table before recording the complete hash chain.

The adapter preserves the DSDE validation/evaluation file partitions, divides the
finetune files into disjoint train/adapt partitions, and records source hashes. Solar
elevation is a declared Zhejiang-centroid approximation derived from timestamps. It
uses the DSDE city label as the meta-learning task unit, with region as fallback. It
does not fabricate missing animal, weather, terrain, or endpoint-prior data; affected
executions emit `data_unavailable`. The pilot purpose remains
`not_final_scientific_evidence`.

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
