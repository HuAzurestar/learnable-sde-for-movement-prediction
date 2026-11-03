B1 implements SOL-014@v2. It does not register production permissions or run
research automatically. The original walking policy and BudgetLedger remain
the budget authority, including every historical arm charge and closure.

Use a new immutable config with an `execution_id`, a new study, the exact new
code identity, and an independent qualification for that code. Keep the old
source inventory and geometry references. Source grant versions must preserve
the original source study and scope while binding `pilot_study_id` to the new
execution study. Internal grants bind the new study. Keep all old objects.
`rebind_grant` produces the exact new-version template without registering it.
Activation compares every field with the old grant: original owner evidence,
expiry, purposes, consumers and source scope are preserved. Only version and
the exact execution study binding change.

Publish the config, qualification, study and a content-addressed B1 decision
receipt before calling `application.walking_execution.activate_execution`.
The decision receipt contains `user: azurestarhu`,
`decision_record: PIRC-26/DECISION-03@v1`, `solution: SOL-014@v2/B1`, and
`implementation: true`; it records the existing decision, never grants one.

The activation argument is an immutable `walking-execution-v1` object with:

- `execution_id`, `budget_pilot_id`, `study_id`, `code_hash`, `arms`, `expires_at`;
- `config_ref`, `qualification_ref`, `decision_ref` as exact
  `{manifest_id, sha256}` references;
- `source_inventory_hash`, `source_geometry_hash` from the original config;
- `previous_execution_ref` from `legacy_ref(store, original_policy)` for the
  first activation, or the exact preceding activation's execution reference.

Activation appends one `WALKING_EXECUTION_ACTIVATED` event. Pending costs,
nonterminal attempts, or missing native stop evidence prevent replacement.
Exact current replay is idempotent. Retired execution identities cannot be
reactivated. `resolve_execution` selects only the active exact version; old v1
operation remains available when there has been no activation. A new name
never creates a new budget. Original-arm continuation accepts only the exact
active study pointing back to its originally approved study.

The normal controller accepts the new config through its existing `--config`
option. A B1 stage reads protected input in the foreground parent after funding
and RUNNING admission. The input read is part of the same charged stage clock.
Only bounded bytes, numeric settings, lineage hashes and exact attempt identity
enter `walking-stage-input-v2`. The child receives no grants or store handle and
uses `application.walking_compute`. Legacy worker admission rejects B1 configs.
Input copies are removed after confirmed native stop and settlement; retained
results remain governed by the existing restricted artifact rules.

Each stage is capped by 120 seconds, actual global remainder, permission expiry
and the controller's absolute deadline. Training also shares the original
cumulative arm allowance. Windows Job containment, 2 GiB whole-tree memory and
independent watchdogs remain in force. Actual overruns are charged without
clipping and parent reads are not billed again as owner work.

During computation, the parent probes head identity/content, event-directory
identity/change times, hold presence and the nonblocking OS writer lock. It
writes no authoritative HEARTBEAT and performs no full-chain poll. The target
cadence is 25 ms; observations stalled beyond 250 ms conservatively stop the
tree on resumption. These are observation targets, not realtime guarantees.
The native absolute deadline remains independent of journal I/O.

Any change or read failure stops the whole native tree before authoritative
review. A normal exit also requires native stop and fresh full physical chain,
current permission/budget, original artifact and package/result binding checks
before publication. In-place historical content changes without a head or
directory change may be found only at that final boundary. No result is allowed
through that boundary on failed verification. Validated fatal diagnosis is
retained separately from timeout/budget status and actual settlement cost.

Shared changes are limited to the optional B1 supervisor adapter and the exact
walking continuation mapping in `research_budget.reserve`; this is not a
second local implementation of the parent shared budget contract. Independent
qualification and artificial 12k production-scale measurements still belong
to the subsequent assigned validation tasks. This document is an implementation
handoff, not evidence of research readiness or a final completion decision.
