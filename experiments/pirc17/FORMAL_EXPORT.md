# PIRC-17 public aggregate replay

`formal_export.py` is the final aggregate consumer in the fixed formal matrix,
not an authorization or experiment-launch command. The formal worker calls
its guarded entry only after the saved analysis and independent audit have
actually completed. Publication remains unaccepted until scientific review
and human ACCEPT-02. A successful synthetic test is not a formal result.

## Contents and privacy boundary

The complete bundle has a fixed 128 MiB file bound. Its 11,368 possible score
rows can exceed the ordinary 32 MiB per-record limit; only the export reader
uses this larger bound. IPC messages and other scientific record limits do
not change. Full serialization and validation remain within the original
export phase allocation, not an additional computation allowance.

The content-addressed JSON exports the complete method ledger, work/score
dispositions, ordinal block/origin labels, per-seed scalar scores, descriptive
metric tables, all registered paired families, numeric mechanism summaries
and diagnostics, runtime/replay evidence, limitations and source hashes.

No trajectory points, original sample/block/origin IDs, region centers, raw
fitted parameters, map values, absolute DLL paths or private exception text
are included. The ordinal labels preserve the original sorted block order so
fixed bootstrap draws are unchanged. Hashes link private source evidence;
they are not a claim that public arithmetic repeats private-source validation.

The metric table has 40 subjects per mode: 28 required method slots, ten
terrain configurations, one diagnostic reference and one deterministic
inertial baseline. The 120 rows are descriptive, not 120 hypothesis tests.
All four original scoring times and coverage levels 0.5/0.8/0.9/0.95 remain.
Missing/failed required forecasts suppress the corresponding mean; they are
not silently omitted. The eight human-exempted slots remain in the separate
36-slot method ledger, not fabricated as measured table rows.

Numerical diagnostics retain four-time errors and scoring-only estimates.
Variance evidence retains MC-FP and CRN-FP absolute variances, five-seed
paired differences and the fixed descriptive block bootstrap. A zero
denominator remains undefined. No new forecast, fit, seed or refinement is
performed during export or replay.

The cost snapshot precedes export completion. It is explicitly not the final
closed execution cost; consult the final controller ledger at run closure.
The original 48-hour active-compute cap is not a project ETA.

## Offline reproduction

From the PSDE repository, in the registered research Python environment with
the sibling DSDE repository on `PYTHONPATH`, run:

```text
python -m experiments.pirc17.formal_export --input PUBLIC_AGGREGATE.json --sha256 PINNED_CONTENT_SHA256 --output-directory REPLAY_OUTPUT
```

Pin the input content SHA from the reviewed result handoff, not merely from
the same untrusted file. The command requires only the public JSON and the
declared code/environment; it takes no trajectory, model, map, approval or
native-session path. It recomputes numeric predicates, diagnostic arithmetic,
paired inference, factor verdicts and descriptive metrics, compares them with
the exported results and publishes a new immutable replay receipt. Existing
receipts are never overwritten. A mismatch or wrong hash fails before a
successful receipt is written.

The code's fixed decision policy still references the registered public
development evidence in this repository. Keep those versioned files with the
source revision. Replaying self-consistent numbers proves arithmetic
reproducibility, not their provenance, new numerical qualification or human
acceptance. Full evidence cards and the complete TSDE manuscript/PDF remain
separate required deliverables.
