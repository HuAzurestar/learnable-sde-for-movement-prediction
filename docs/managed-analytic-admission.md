# Managed analytic qualification before formal input exposure

The affine restart-only `exact` and `gaussian` methods now have a positive
owner-controlled admission path from an **actual settled qualification pilot**
to a preregistered formal execution. This is request-level qualification of a
declared synthetic affine endpoint law, not approval of a fitted real model,
path/MLMC/IS method, nonlinear closure, whole method family or complete study.
The four general propagation adapters are version1.2.0; the dedicated analytic
qualification adapter remains1.0.0. No store/grant/arm is initialized here.

## Frozen source pointer and preregistration

The execution package's payload must contain `managed_analytic_qualification`
with exactly these fields:

```
schema_version: managed-affine-analytic-qualification-v1
policy: complete AffineQualificationPolicy manifest
source_attempt_id: actual successful qualification attempt
source_artifact_id: exact result artifact from that attempt
source_authorization_id: explicit existing source-study grant
source_authorization_version: explicit version or null for an unversioned grant
```

The formal preregistration contains the same policy exactly once in
`propagation_qualification_policies`. Policies cannot be duplicated or silently
replaced after a pilot result. Its primary metric is the fixed
`absolute_error_upper_vs_declared_affine_law`, in meters or unit1 according to
the registered functional. Generic operator-imported contract qualification and
blind preregistration remain mandatory but are **not sufficient**.

Only an affine restart-only analytic target can consume this evidence. Other
propagation adapters/methods cannot borrow it; their currently unsupported
formal qualification is refused before protected input reads.

## What the owner verifies before the held-out read

Before `EvaluationExposureLedger.verify` exposes test/final-eval content, the
owner checks the source against the current authoritative store, not a caller
copy or an unverified external report:

- exact source attempt/artifact with terminal `SUCCEEDED` state;
- current source hash and the same root/store, complete frozen dynamics,
  request, method, policy and sole model/method/objective arm identity;
- explicit current unexpired source grant permitting evaluation by the consumer;
- canonical result bytes matching the attempt's completed artifact metadata;
- actual qualification adapter, pilot admission and train/validation input;
- `verified_reuse` checks for actual source run/admission/worker/reservation/
  settlement/completion provenance without executing another worker;
- source native worker-tree stop followed by successful settled measured charge,
  with a positive cost no greater than its reservation or frozen job cap;
- component certificate hashes/schema/source/request/model/grid/profile,
  actual saved interval arithmetic values, all passed numerical checks and the
  frozen error/bias/scaled-growth/operation thresholds.

These checks do not recompute exponentials or Gaussian series in uncharged
preflight. The expensive mathematical enclosures were generated inside the
supervised qualification worker. Bounded Fraction arithmetic over their saved
intervals verifies threshold/result consistency. A saved content hash alone is
not a mathematical proof; the trusted numerical computation is established by
the actual current-source owner/worker/artifact chain. Owner honesty and host
integrity are not cryptographically inferred from arbitrary imported packets.

The recorded evidence is bounded to2MiB and the source result to512KiB. It retains
the source admission, canonical result, metadata, policy, grant, worker/stop/
reservation/settlement events and target spec/cell/preregistration bindings.
The source grant is rechecked after subsequent admission I/O. Missing, failed,
unsettled, stale, resealed-false or out-of-policy evidence is not repaired or
promoted. Qualification and formal work share the **same existing cumulative
arm**, even across registered studies. Qualification cost and prediction cost
remain separate measured records, both counted against the arm.

## Worker and result boundary

The owner appends a private `--admission-hash` argument only after successful
admission. The worker resolves the canonical receipt and confirms its owner
event, current running attempt, spec/cell/run and actual result output path.
Direct formal `execute_propagation` without its owner receipt is refused; a
caller cannot request a `qualified` flag to bypass admission.

The current deterministic scalar output is checked against the saved target
interval again. If its implementation error now exceeds the frozen tolerance,
execution fails rather than borrowing the pilot's floating value. The main
metric bounds the error of this actual output versus the continuous declared
law, not merely its difference from another float reference. Qualified output
separately records continuous reference interval width, current implementation
error, absolute/signed grid bias, model error unknown, and source/policy hashes.
The former uncertified float-reference diagnostic is not the formal metric.
The owner independently checks this metric, separated components and provenance
before publishing/reusing a formal result. Intervals are not mechanically summed.

The source qualification analysis remains `scientific_qualification:false` and
its pilot result remains `fixture`. Only the formal target owner receipt/result
has `qualified`, and only for the explicitly declared request-level numerical
scope. Frozen synthetic dynamics metadata is unchanged; no real-data model
qualification is fabricated. These paths do not make unit controls into actual
research permission or completed research.

## Disclosure and independent consumers

The source artifact's actual visibility is included both in the unexecuted
package lineage and in the admitted result lineage. Synthetic labels on the
consumer cannot declassify a restricted qualification source. Export separately
checks the source grant's export purpose, consumer scope, visibility, expiry and
canonical source bytes; execution/evaluation permission is not export permission.
Final export grant checks include the source authorization too.

The independent paper reader still needs an equivalent managed-analytic proof
validator and counterexamples. Existing generic admission validation is not
claimed to verify this new numerical chain. Reproduction claims must retain the
source attempt/cost/policy/certificate bindings and clear declared-law limits.
No official research, manuscript, review/acceptance or merged delivery is claimed
by adding this runtime capability.

## Engineering validation

Tests use real qualification and formal worker processes in disposable stores.
Training and held-out controls have different content **and** source-block
identities; renaming/reseeding an already-read block would not regain blindness.
They verify the pre-read source settlement order, actual cumulative charge,
qualified scalar/component output, no-charge reuse, source export and refusal of
missing/substituted source pointers, changed thresholds, absent preregistration,
wrong metric/grant, resealed false numerical values, direct flag-only execution
and false current-output/provenance. No actual worker or cost event is fabricated.
