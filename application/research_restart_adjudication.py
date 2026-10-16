"""Append-only PIRC-28 restart adjudication across runtime source epochs.

The old interruption, charge and arm closure remain authoritative.  A reviewed
source bridge may register exactly one child attempt for the same frozen
scientific cell.  It does not generally reopen the arm or start a worker.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import os
import re

from infrastructure.research_store import ResearchError, digest


SCHEMA = "pirc28-restart-adjudication-v1"
PRELAUNCH_SCHEMA = "pirc28-restart-adjudication-v2"
SUPERSESSION_SCHEMA = "pirc28-restart-adjudication-v3"
BRIDGE_SCHEMA = "pirc28-source-epoch-bridge-v1"
CONTINUATION_SCHEMA = "pirc28-resource-arm-continuation-v1"
RESOURCE_RETRY_SCHEMA = "pirc28-resource-retry-adjudication-v1"
PREFLIGHT_RECOVERY_SCHEMA = "pirc28-preflight-recovery-adjudication-v1"
REASON = "PIRC-28 approved source-epoch infrastructure restart"
BRIDGE_FIELDS = {"schema_version", "execution_policy", "historical_definition_hash",
    "current_definition_hash", "historical_code_hash", "current_code_hash",
    "changed_fields", "study_pairs"}
BRIDGE_METADATA_FIELDS = {"preregistration_hash", "renewal_scope_hash", "physical_recipe_count"}
RECEIPT_FIELDS = {"schema_version", "source_bridge", "execution_policy", "current_code_hash",
    "parent_attempt_id", "historical_run_id", "historical_run_hash", "historical_event_hashes",
    "retained_cost_ms", "scientific_cell_hash", "current_run_id", "current_cell_hash",
    "retry_attempt_id", "retry_ordinal", "maximum_infrastructure_retries", "recovery_level",
    "native_identity"}
RESOURCE_RETRY_FIELDS = {"schema_version", "execution_policy", "current_code_hash",
    "parent_attempt_id", "run_id", "cell_hash", "parent_event_hashes", "parent_cost_ms",
    "retry_attempt_id", "retry_ordinal", "maximum_infrastructure_retries", "native_identity"}


def _require(condition, message, code="CONTRACT_MISMATCH"):
    if not condition:
        raise ResearchError(code, message)


def _reference(value, label):
    _require(type(value) is dict and set(value) == {"manifest_id", "sha256"}
             and type(value["manifest_id"]) is str
             and re.fullmatch(r"[0-9a-f]{64}", str(value["sha256"])),
             "exact " + label + " reference required")


def scientific_cell_identity(cell):
    """Identity fields frozen by the PIRC-28 plan, independent of adapters."""
    fields = ("arm_id", "fit_K", "replicate_id", "block_id", "seed")
    _require(type(cell) is dict and all(name in cell for name in fields),
             "complete scientific cell identity required")
    return {name: deepcopy(cell[name]) for name in fields}


def _published(store, reference, label):
    _reference(reference, label)
    value = store._manifest(reference["manifest_id"])
    _require(digest(value) == reference["sha256"], label + " hash differs",
             "IDENTITY_CONFLICT")
    return value


def _grant(store, study):
    settings = study.get("admission", {})
    return store.authorization(settings.get("authorization_id"),
                               version=settings.get("authorization_version"))


def prime_source_definitions(store, bridge_reference):
    """Validate bounded immutable definitions outside any owned writer scope.

    This is not authority: resolvers still reread and hash their actual bridge,
    definitions, study/permission lineage, and journal under the later lock.
    """
    from experiments.pirc28.study import SyntheticStudyDefinition

    _require(store._read_snapshot() is None,
             "cold source definition validation must run outside the writer scope")
    _reference(bridge_reference, "source epoch bridge")
    bridge = store.manifest(bridge_reference["manifest_id"])
    _require(digest(bridge) == bridge_reference["sha256"], "source bridge hash differs")
    for field in ("historical_definition_hash", "current_definition_hash"):
        reference = bridge.get(field)
        _require(type(reference) is str and re.fullmatch("[0-9a-f]{64}", reference),
                 "source definition hash required")
        value = store.manifest("pirc28-study-definition-" + reference)
        _require(digest(value) == reference, "source definition hash differs")
        SyntheticStudyDefinition.from_dict(value)


def _prime_restart(store, reference):
    if store._read_snapshot() is not None:
        return
    _reference(reference, "retry adjudication")
    receipt = store.manifest(reference["manifest_id"])
    _require(digest(receipt) == reference["sha256"], "retry adjudication hash differs")
    if receipt.get("schema_version") in {SCHEMA, PRELAUNCH_SCHEMA, SUPERSESSION_SCHEMA}:
        prime_source_definitions(store, receipt["source_bridge"])
    elif receipt.get("schema_version") == RESOURCE_RETRY_SCHEMA:
        prime_policy_definitions(store, receipt["execution_policy"], receipt["current_code_hash"])
    elif receipt.get("schema_version") == PREFLIGHT_RECOVERY_SCHEMA:
        from .research_preflight_recovery import prime_recovery
        prime_recovery(store, receipt)


def prime_policy_definitions(store, execution_policy, source_code_hash):
    """Find only this policy's source bridge, then prime its pure definitions."""
    if store._read_snapshot() is not None:
        return
    references = []
    preflight_recoveries = []
    with store._read_transaction():
        for event in store._manifest_events("pirc28-source-epoch-bridge-"):
            payload = event["payload"]
            bridge = store._manifest(payload["object_id"])
            if (bridge.get("execution_policy") == execution_policy
                    and bridge.get("current_code_hash") == source_code_hash):
                references.append({"manifest_id": payload["object_id"], "sha256": payload["sha256"]})
        for event in store._events():
            if event['event_kind'] == 'PREFLIGHT_RECOVERY_ADJUDICATED':
                receipt = store._manifest(event['payload']['manifest_id'])
                if receipt.get('execution_policy') == execution_policy and receipt.get('current_code_hash') == source_code_hash:
                    preflight_recoveries.append(event['payload'])
    _require(len(references) <= 1, "source policy has ambiguous epoch bridges", "IDENTITY_CONFLICT")
    for reference in references:
        prime_source_definitions(store, reference)
    # Fresh workers also need the original source definitions underlying the
    # exception. Never cold-validate those only after taking the budget writer.
    _require(len(preflight_recoveries) <= 1, 'one-time policy has multiple recoveries', 'IDENTITY_CONFLICT')
    for reference in preflight_recoveries:
        _prime_restart(store, reference)


def resolve_source_bridge(store, reference, *, execution_policy, source_code_hash):
    """Verify the exact old/new definitions and scope-preserving grant lineage."""
    from application.research_execution_policy import resolve_policy
    from experiments.pirc28.study import SyntheticStudyDefinition

    bridge = _published(store, reference, "source epoch bridge")
    _require(set(bridge) in (BRIDGE_FIELDS, BRIDGE_FIELDS | BRIDGE_METADATA_FIELDS)
             and bridge.get("schema_version") == BRIDGE_SCHEMA
             and bridge.get("execution_policy") == execution_policy
             and bridge.get("current_code_hash") == source_code_hash
             and bridge.get("changed_fields") == ["runtime_code_hash"],
             "source epoch bridge policy/code differs", "UNAUTHORIZED_DATA")
    old_hash = bridge.get("historical_definition_hash")
    new_hash = bridge.get("current_definition_hash")
    old = SyntheticStudyDefinition.from_dict(store._manifest(
        "pirc28-study-definition-" + str(old_hash)))
    new = SyntheticStudyDefinition.from_dict(store._manifest(
        "pirc28-study-definition-" + str(new_hash)))
    old_value, new_value = old.to_dict(), new.to_dict()
    changed = [key for key in sorted(old_value) if old_value.get(key) != new_value.get(key)]
    _require(old.definition_hash == old_hash and new.definition_hash == new_hash
             and set(old_value) == set(new_value) and changed == ["runtime_code_hash"]
             and old_value["runtime_code_hash"] == bridge.get("historical_code_hash")
             and new_value["runtime_code_hash"] == source_code_hash,
             "source bridge changes frozen scientific design", "IDENTITY_CONFLICT")
    resolve_policy(store, execution_policy, arm_id="pirc28-k1",
                   source_code_hash=source_code_hash)

    pairs = bridge.get("study_pairs")
    _require(type(pairs) is list and len(pairs) == len(old.study_ids) == len(new.study_ids),
             "complete source study lineage required")
    by_old, seen_new = {}, set()
    for expected_old, expected_new, pair in zip(old.study_ids, new.study_ids, pairs):
        _require(type(pair) is dict and pair.get("historical_study_id") == expected_old
                 and pair.get("current_study_id") == expected_new
                 and expected_old not in by_old and expected_new not in seen_new,
                 "ordered one-to-one source study lineage differs")
        historical = store._manifest("study-" + expected_old)["spec"]
        current = store._manifest("study-" + expected_new)["spec"]
        _require(digest(historical) == pair.get("historical_spec_hash")
                 and digest(current) == pair.get("current_spec_hash")
                 and historical.get("pirc28_study_definition_hash") == old_hash
                 and current.get("pirc28_study_definition_hash") == new_hash
                 and historical.get("code_hash") == bridge.get("historical_code_hash")
                 and current.get("code_hash") == source_code_hash,
                 "registered source study lineage differs", "IDENTITY_CONFLICT")
        old_grant, new_grant = _grant(store, historical), _grant(store, current)
        _require(digest(old_grant) == pair.get("historical_grant_hash")
                 and digest(new_grant) == pair.get("current_grant_hash"),
                 "source grant hash lineage differs", "UNAUTHORIZED_DATA")
        stable_grant = ("purposes", "visibilities", "block_ids", "data_root", "test_authorization")
        _require(all(old_grant.get(field) == new_grant.get(field) for field in stable_grant)
                 and new_grant.get("study_id") == expected_new
                 and new_grant.get("protocol_hash") == current.get("protocol_hash")
                 and datetime.fromisoformat(new_grant["expires_at"]) > datetime.now(timezone.utc),
                 "current synthetic grant expands or expires outside lineage", "UNAUTHORIZED_DATA")
        cell_pairs = pair.get("cell_pairs")
        _require(type(cell_pairs) is list and len(cell_pairs) == len(historical["cells"]) == len(current["cells"]),
                 "complete source cell lineage required")
        old_cells = {digest(cell): cell for cell in historical["cells"]}
        new_cells = {digest(cell): cell for cell in current["cells"]}
        for cell_pair in cell_pairs:
            old_cell = old_cells.get(cell_pair.get("historical_cell_hash"))
            new_cell = new_cells.get(cell_pair.get("current_cell_hash"))
            _require(old_cell is not None and new_cell is not None,
                     "source bridge cell is absent", "IDENTITY_CONFLICT")
            identity = scientific_cell_identity(old_cell)
            _require(scientific_cell_identity(new_cell) == identity
                     and cell_pair.get("scientific_cell_hash") == digest(identity),
                     "scientific cell lineage differs", "IDENTITY_CONFLICT")
        by_old[expected_old] = {"pair": pair, "historical": historical, "current": current}
        seen_new.add(expected_new)
    return {"bridge": deepcopy(bridge), "historical_definition": old,
            "current_definition": new, "by_historical_study": by_old}


def _pid_reuse_observation(pid, recorded_started_at):
    """Prove that a live Windows PID was born after the recorded worker."""
    if os.name != "nt" or type(pid) is not int or pid <= 1:
        return None
    import ctypes
    from ctypes import wintypes
    try:
        recorded = datetime.fromisoformat(recorded_started_at)
        if recorded.tzinfo is None:
            return None
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *[ctypes.POINTER(wintypes.FILETIME)] * 4]
        kernel.GetProcessTimes.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return None
        try:
            creation, exited, kernel_time, user_time = [wintypes.FILETIME() for _ in range(4)]
            if not kernel.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exited),
                                          ctypes.byref(kernel_time), ctypes.byref(user_time)):
                return None
            ticks = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
            born = datetime.fromtimestamp((ticks - 116444736000000000) / 10000000, timezone.utc)
            if (born - recorded).total_seconds() <= 2:
                return None
            return {"pid": pid, "recorded_started_at": recorded_started_at,
                    "current_process_created_at": born.isoformat(),
                    "creation_filetime": ticks, "source": "native-GetProcessTimes-query-only"}
        finally:
            kernel.CloseHandle(handle)
    except (ValueError, TypeError, OSError, OverflowError):
        return None


def _attempt_payload(receipt):
    return {"attempt_id": receipt["retry_attempt_id"], "run_id": receipt["current_run_id"],
            "parent_attempt_id": receipt["parent_attempt_id"], "reason": REASON,
            "state": "REGISTERED", "started_at": None, "ended_at": None,
            "error_code": None, "artifact_manifest_hash": None}


def _prelaunch_predecessors(store, references, *, parent_attempt_id, current_run,
                            current_definition, before_sequence, unfunded_supersession=None):
    """Carry every older prelaunch sibling forward across source renewals."""
    from .research_queue_recovery import resolve_prelaunch_cancellation

    _require(type(references) is list and len(references) <= 1,
             "source renewal infrastructure retries exhausted", "RETRY_EXHAUSTED")
    events = store._events()
    expected = {event["payload"]["attempt_id"] for event in events
        if event["event_kind"] == "ATTEMPT" and event["sequence"] < before_sequence
        and event["payload"].get("state") == "REGISTERED"
        and event["payload"].get("parent_attempt_id") == parent_attempt_id
        and event["payload"].get("run_id") != current_run["run_id"]}
    seen = set()
    participants = [(resolve_prelaunch_cancellation(store, reference), True) for reference in references]
    if unfunded_supersession is not None:
        from .research_unfunded_source import resolve_unfunded_supersession
        participants.append((resolve_unfunded_supersession(store, unfunded_supersession,
            source_code_hash=current_definition['runtime_code_hash']), False))
    for cancellation, funded in participants:
        attempt = store._attempts()[cancellation["attempt_id"]]
        run = store._manifest("run-" + attempt["run_id"])
        study = store._manifest("study-" + run["study_id"])["spec"]
        definition_hash = study.get("pirc28_study_definition_hash")
        previous = store._manifest("pirc28-study-definition-" + str(definition_hash))
        _require(digest(previous) == definition_hash
                 and {k: v for k, v in previous.items() if k != "runtime_code_hash"}
                     == {k: v for k, v in current_definition.items() if k != "runtime_code_hash"}
                 and previous.get("runtime_code_hash") != current_definition.get("runtime_code_hash")
                 and digest(scientific_cell_identity(run["cell"]))
                     == digest(scientific_cell_identity(current_run["cell"]))
                 and attempt.get("parent_attempt_id") == parent_attempt_id
                 and attempt.get("reason") == REASON and attempt["attempt_id"] not in seen,
                 "prelaunch predecessor changes frozen design or retry lineage", "IDENTITY_CONFLICT")
        if funded:
            reserve = next(event for event in events
                           if event["hash"] == cancellation["event_hashes"]["reserve"])
            prior_reference = reserve["payload"].get("restart_adjudication")
            terminal_hash = cancellation["event_hashes"]["terminal"]
        else:
            prior_reference = cancellation['restart_adjudication']
            terminal_hash = cancellation['terminal_event_hash']
        prior = _published(store, prior_reference, "prior restart")
        _require(prior.get("parent_attempt_id") == parent_attempt_id
                 and prior.get("retry_attempt_id") == attempt["attempt_id"]
                 and prior.get("current_run_id") == run["run_id"]
                 and prior.get("current_code_hash") == previous["runtime_code_hash"]
                 and any(event["hash"] == terminal_hash
                         and event["sequence"] < before_sequence for event in events),
                 "prelaunch predecessor adjudication or terminal order differs")
        seen.add(attempt["attempt_id"])
    _require(seen == expected, "all prior source retry registrations must be retained",
             "RETRY_EXHAUSTED")
    # Only the previously FUNDED queue cancellation consumes its retained
    # retry ordinal. The additional, explicitly recorded source registration
    # had no reservation, queue, admission or possible worker execution.
    return len(references)


def _resolve_receipt(store, reference, *, allow_missing_attempt=False):
    receipt = _published(store, reference, "restart adjudication")
    expected_fields = (RECEIPT_FIELDS | {"prelaunch_cancellations"}
                       if receipt.get("schema_version") == PRELAUNCH_SCHEMA else RECEIPT_FIELDS)
    if receipt.get("schema_version") == SUPERSESSION_SCHEMA:
        expected_fields = RECEIPT_FIELDS | {"prelaunch_cancellations", "unfunded_supersession"}
    _require(set(receipt) == expected_fields
             and receipt.get("schema_version") in {SCHEMA, PRELAUNCH_SCHEMA, SUPERSESSION_SCHEMA},
             "restart adjudication schema differs")
    bridge = resolve_source_bridge(store, receipt["source_bridge"],
        execution_policy=receipt["execution_policy"], source_code_hash=receipt["current_code_hash"])
    events = store._events()
    matches = [event for event in events if event["event_kind"] == "RESTART_ADJUDICATED"
               and event["payload"] == reference]
    _require(len(matches) == 1,
             "restart adjudication publication/event differs")
    parent = store._attempts().get(receipt.get("parent_attempt_id"))
    old_run = store._manifest("run-" + receipt.get("historical_run_id", ""))
    _require(parent is not None and parent.get("run_id") == old_run.get("run_id")
             and parent.get("state") == "INTERRUPTED" and parent.get("error_code") == "SUPERVISOR_ERROR"
             and parent.get("artifact_id") is None and digest(old_run) == receipt.get("historical_run_hash"),
             "historical interrupted parent differs", "RECOVERY_REQUIRED")
    lineage = bridge["by_historical_study"].get(old_run["study_id"])
    pair = lineage and lineage["pair"]
    cell_pair = next((value for value in (pair or {}).get("cell_pairs", [])
                      if value.get("historical_cell_hash") == old_run["cell_hash"]), None)
    _require(cell_pair is not None and cell_pair.get("current_cell_hash") == receipt.get("current_cell_hash")
             and cell_pair.get("scientific_cell_hash") == receipt.get("scientific_cell_hash"),
             "adjudicated scientific cell lineage differs")
    current_run = store._manifest("run-" + receipt.get("current_run_id", ""))
    _require(current_run.get("study_id") == pair["current_study_id"]
             and current_run.get("cell_hash") == cell_pair["current_cell_hash"],
             "adjudicated current run differs", "IDENTITY_CONFLICT")
    predecessor_count = _prelaunch_predecessors(store, receipt.get("prelaunch_cancellations", []),
        parent_attempt_id=receipt["parent_attempt_id"], current_run=current_run,
        current_definition=bridge["current_definition"].to_dict(), before_sequence=matches[0]["sequence"],
        unfunded_supersession=receipt.get('unfunded_supersession'))
    hashes = receipt.get("historical_event_hashes", {})
    _require(type(hashes) is dict and len(hashes) == 6
             and all(sum(event["hash"] == value for event in events) == 1 for value in hashes.values()),
             "historical interruption event evidence moved", "RECOVERY_REQUIRED")
    evidence = {name: next(event for event in events if event["hash"] == value)
                for name, value in hashes.items()}
    reserve, launch, stop = evidence["reserve"], evidence["launch"], evidence["stop"]
    settlement, closure, terminal = (evidence["settlement"], evidence["closure"], evidence["terminal"])
    reservation_id = reserve["payload"].get("reservation_id")
    charged = settlement["payload"]
    _require(reserve["event_kind"] == "RESERVE"
             and reserve["payload"].get("attempt_id") == parent["attempt_id"]
             and reserve["payload"].get("run_id") == old_run["run_id"]
             and launch["event_kind"] == "WORKER_STARTED"
             and launch["payload"].get("reservation_id") == reservation_id
             and launch["payload"].get("attempt_id") == parent["attempt_id"]
             and stop["event_kind"] == "WORKER_TREE_STOPPED"
             and stop["payload"].get("reservation_id") == reservation_id
             and stop["payload"].get("attempt_id") == parent["attempt_id"]
             and stop["payload"].get("confirmation") == "native-job-or-process-group-no-running-descendants"
             and settlement["event_kind"] == "SETTLE" and charged.get("settled") is True
             and charged.get("outcome") == "INTERRUPTED"
             and type(charged.get("monotonic_elapsed_ms")) is int
             and charged["monotonic_elapsed_ms"] == charged.get("charged_ms")
             == stop["payload"].get("observed_elapsed_ms") == receipt.get("retained_cost_ms")
             and closure["event_kind"] == "ARM_CLOSED"
             and closure["payload"] == {"arm_id": old_run["arm_id"], "reason": "INTERRUPTED"}
             and terminal["event_kind"] == "ATTEMPT" and terminal["payload"] == parent
             and reserve["sequence"] < launch["sequence"] < stop["sequence"]
             < settlement["sequence"] < closure["sequence"] < terminal["sequence"],
             "historical stop/cost/closure evidence differs", "RECOVERY_REQUIRED")
    attempts = store._attempts()
    child = attempts.get(receipt["retry_attempt_id"])
    if child is None:
        _require(allow_missing_attempt, "adjudicated retry attempt missing", "RECOVERY_REQUIRED")
    else:
        initial = _attempt_payload(receipt)
        registrations = [event for event in events if event["event_kind"] == "ATTEMPT"
                         and event["payload"] == initial]
        _require(len(registrations) == 1
                 and all(child.get(field) == initial[field] for field in
                         ("attempt_id", "run_id", "parent_attempt_id", "reason")),
                 "adjudicated retry attempt differs", "IDENTITY_CONFLICT")
    same_cell = [value for value in attempts.values()
                 if value["run_id"] in {old_run["run_id"], current_run["run_id"]}]
    if child is None:
        expected_ordinal = len(same_cell)
    else:
        expected_ordinal, cursor, seen = 0, child, set()
        while cursor.get("parent_attempt_id") is not None:
            _require(cursor["attempt_id"] not in seen,
                     "cross-epoch retry ancestry cycles", "IDENTITY_CONFLICT")
            seen.add(cursor["attempt_id"])
            cursor = attempts.get(cursor["parent_attempt_id"])
            _require(cursor is not None and cursor["run_id"] in
                     {old_run["run_id"], current_run["run_id"]},
                     "cross-epoch retry ancestry leaves its cell", "IDENTITY_CONFLICT")
            expected_ordinal += 1
    old_attempts = [value for value in same_cell if value["run_id"] == old_run["run_id"]]
    current_attempts = [value for value in same_cell if value["run_id"] == current_run["run_id"]]
    _require(len(old_attempts) == 1 and len(current_attempts) <= 3
             and receipt.get("retry_ordinal") == expected_ordinal + predecessor_count <= 2,
             "cross-epoch infrastructure retry allowance differs", "RETRY_EXHAUSTED")
    return deepcopy(receipt)


def resolve_restart_adjudication(store, reference, *, execution_policy, source_code_hash):
    """Resolve one registered child without converting the old closure."""
    _prime_restart(store, reference)
    with store._read_transaction():
        receipt = _resolve_receipt(store, reference)
        _require(receipt["execution_policy"] == execution_policy
                 and receipt["current_code_hash"] == source_code_hash,
                 "restart adjudication funding binding differs", "UNAUTHORIZED_DATA")
        store._read_completion(lambda: _resolve_receipt(store, reference), lambda: None)
    return receipt


def adjudicate_restart(store, parent_attempt_id, *, source_bridge, execution_policy,
                       source_code_hash, prelaunch_cancellations=None, unfunded_supersession=None):
    """Register one deterministic restart child; never launch or reserve it."""
    from application.research_restart_review import inspect_restart_review
    from infrastructure.process_tree import process_may_be_alive

    if store._read_snapshot() is None:
        prime_source_definitions(store, source_bridge)
    event_id = "pirc28-restart-adjudication-" + digest(
        [store.store_id, parent_attempt_id, source_bridge, execution_policy, source_code_hash])
    with store._read_transaction():
        existing = [event for event in store._events() if event["event_id"] == event_id]
        if existing:
            _require(len(existing) == 1 and existing[0]["event_kind"] == "RESTART_ADJUDICATED",
                     "restart adjudication replay differs", "IDENTITY_CONFLICT")
            reference = existing[0]["payload"]
            receipt = _resolve_receipt(store, reference, allow_missing_attempt=True)
            _require(receipt.get("prelaunch_cancellations", []) == (prelaunch_cancellations or []),
                     "restart replay changed prelaunch history", "IDENTITY_CONFLICT")
            _require(receipt.get('unfunded_supersession') == unfunded_supersession,
                     'restart replay changed unfunded source history', 'IDENTITY_CONFLICT')
            if store._attempts().get(receipt["retry_attempt_id"]) is None:
                store._append("ATTEMPT", _attempt_payload(receipt),
                              "attempt-" + receipt["retry_attempt_id"])
            return reference

        candidate = inspect_restart_review(store, parent_attempt_id, _final_completion=False)
        bridge = resolve_source_bridge(store, source_bridge, execution_policy=execution_policy,
                                       source_code_hash=source_code_hash)
        lineage = bridge["by_historical_study"].get(candidate["study_id"])
        _require(lineage is not None, "historical source study absent from current bridge")
        pair = lineage["pair"]
        cell_pair = next((value for value in pair["cell_pairs"]
                          if value["historical_cell_hash"] == candidate["cell_hash"]), None)
        _require(cell_pair is not None, "historical scientific cell absent from current bridge")
        current_cell = next(cell for cell in lineage["current"]["cells"]
                            if digest(cell) == cell_pair["current_cell_hash"])
        current_run_id = store.register_run(pair["current_study_id"], current_cell)
        predecessor_count = _prelaunch_predecessors(store,
            [] if prelaunch_cancellations is None else prelaunch_cancellations,
            parent_attempt_id=parent_attempt_id,
            current_run=store._manifest("run-" + current_run_id),
            current_definition=bridge["current_definition"].to_dict(),
            before_sequence=store._events()[-1]["sequence"] + 1,
            unfunded_supersession=unfunded_supersession)
        attempts = store._attempts()
        lineage_attempts = [value for value in attempts.values()
                            if value["run_id"] in {candidate["run_id"], current_run_id}]
        _require(len(lineage_attempts) + predecessor_count < 3 and all(value["state"] in store.TERMINAL
                 for value in lineage_attempts), "cross-epoch retry unavailable", "RETRY_EXHAUSTED")

        worker = next(event for event in store._events()
                      if event["hash"] == candidate["event_hashes"]["launch"])
        pid = worker["payload"]["pid"]
        native = {"status": "recorded-worker-absent", "pid": pid,
                  "worker_event_hash": worker["hash"]}
        if process_may_be_alive(pid):
            observation = _pid_reuse_observation(pid, worker["created_at"])
            _require(observation is not None, "recorded worker identity may still be alive", "WORKER_ACTIVE")
            native = {"status": "pid-reused-by-later-process", "worker_event_hash": worker["hash"],
                      **observation}

        retry_id = digest([event_id, current_run_id, cell_pair["scientific_cell_hash"]])[:32]
        receipt = {"schema_version": SCHEMA, "source_bridge": deepcopy(source_bridge),
            "execution_policy": deepcopy(execution_policy), "current_code_hash": source_code_hash,
            "parent_attempt_id": parent_attempt_id, "historical_run_id": candidate["run_id"],
            "historical_run_hash": candidate["source_run_hash"],
            "historical_event_hashes": candidate["event_hashes"],
            "retained_cost_ms": candidate["retained_cost_ms"],
            "scientific_cell_hash": cell_pair["scientific_cell_hash"],
            "current_run_id": current_run_id, "current_cell_hash": cell_pair["current_cell_hash"],
            "retry_attempt_id": retry_id, "retry_ordinal": len(lineage_attempts) + predecessor_count,
            "maximum_infrastructure_retries": 2, "recovery_level": "restart-only-no-checkpoint",
            "native_identity": native}
        if prelaunch_cancellations:
            receipt.update(schema_version=PRELAUNCH_SCHEMA,
                           prelaunch_cancellations=deepcopy(prelaunch_cancellations))
        if unfunded_supersession is not None:
            receipt.update(schema_version=SUPERSESSION_SCHEMA,
                           prelaunch_cancellations=deepcopy(prelaunch_cancellations or []),
                           unfunded_supersession=deepcopy(unfunded_supersession))
        object_id = "pirc28-restart-adjudication-" + digest(receipt)
        reference = {"manifest_id": object_id, "sha256": digest(receipt)}
        store._publish(object_id, receipt)
        store._append("MANIFEST", {"object_id": object_id, "sha256": reference["sha256"]},
                      "manifest-" + digest([object_id, reference["sha256"]]))
        event = store._append("RESTART_ADJUDICATED", reference, event_id)
        # Event hash is authoritative in the journal and does not need a
        # self-referential manifest field.
        store._append("ATTEMPT", _attempt_payload(receipt), "attempt-" + retry_id)
        _require(event["payload"] == reference, "adjudication event publication failed")
        return reference


def _resource_retry_evidence(store, parent_attempt_id, *, execution_policy, source_code_hash,
                             allow_child=False):
    """Verify one current-epoch, known-cost, stopped infrastructure failure."""
    from application.research_execution_policy import resolve_policy
    from .research_preflight_recovery import forbid_extra_attempt
    forbid_extra_attempt(store, parent_attempt_id=parent_attempt_id)

    attempts = store._attempts()
    parent = attempts.get(parent_attempt_id)
    _require(parent is not None and parent["state"] == "INTERRUPTED"
             and parent.get("error_code") in {"SUPERVISOR_ERROR", "RESOURCE_LIMIT", "HEARTBEAT_LOST"}
             and parent.get("artifact_id") is None and parent.get("artifact_manifest_hash") is None,
             "resource retry parent is not an artifact-free infrastructure interruption",
             "RECOVERY_REQUIRED")
    run = store._manifest("run-" + parent["run_id"])
    study = store._manifest("study-" + run["study_id"])["spec"]
    _require(study.get("code_hash") == source_code_hash
             and run["cell"].get("visibility") == "synthetic"
             and run["arm_id"] in {"pirc28-k1", "pirc28-k2"},
             "resource retry source/cell/arm differs", "UNAUTHORIZED_DATA")
    resolve_policy(store, execution_policy, arm_id=run["arm_id"],
                   source_code_hash=source_code_hash)
    related = [value for value in attempts.values() if value["run_id"] == run["run_id"]]
    live = [value for value in related if value["state"] not in store.TERMINAL]
    if allow_child:
        _require(len(live) <= 1, "multiple current retry children are live", "IDENTITY_CONFLICT")
    else:
        _require(not live, "current source cell already has a live retry", "IDENTITY_CONFLICT")
    _require(len(related) <= 3 and (len(related) < 3 or allow_child),
             "current source cell retries exhausted", "RETRY_EXHAUSTED")
    events = store._events()
    parent_terminals = [event for event in events if event["event_kind"] == "ATTEMPT"
                        and event["payload"] == parent]
    _require(len(parent_terminals) == 1, "one resource retry parent terminal required")
    later_same_run = [event for event in events if event["event_kind"] == "ATTEMPT"
                      and event["payload"].get("run_id") == run["run_id"]
                      and event["sequence"] > parent_terminals[0]["sequence"]]
    if not allow_child:
        _require(not later_same_run, "resource retry parent is not the latest attempt")
    reservations = [event for event in events if event["event_kind"] == "RESERVE"
                    and event["payload"].get("attempt_id") == parent_attempt_id]
    _require(len(reservations) == 1, "one resource retry reservation required")
    reserve = reservations[0]
    reservation = reserve["payload"]
    _require(reservation.get("reserved_ms") is None
             and reservation.get("budget_mode") == "resource-guarded-no-time-cap"
             and reservation.get("execution_policy") == execution_policy
             and reservation.get("source_code_hash") == source_code_hash
             and reservation.get("run_id") == run["run_id"]
             and reservation.get("arm_id") == run["arm_id"],
             "resource retry funding binding differs", "UNAUTHORIZED_DATA")
    reservation_id = reservation["reservation_id"]
    launches = [event for event in events if event["event_kind"] == "WORKER_STARTED"
                and event["payload"].get("reservation_id") == reservation_id]
    stops = [event for event in events if event["event_kind"] == "WORKER_TREE_STOPPED"
             and event["payload"].get("reservation_id") == reservation_id]
    settlements = [event for event in events if event["event_kind"] == "SETTLE"
                   and event["payload"].get("reservation_id") == reservation_id]
    _require(len(launches) == len(stops) == len(settlements) == 1,
             "complete resource launch/stop/cost/closure evidence required", "RECOVERY_REQUIRED")
    launch, stop, settlement = launches[0], stops[0], settlements[0]
    all_parent_closures = [event for event in events if event["event_kind"] == "ARM_CLOSED"
                and event["payload"].get("arm_id") == run["arm_id"]
                and reserve["sequence"] < event["sequence"]
                < parent_terminals[0]["sequence"]]
    closures = [event for event in all_parent_closures
                if settlement["sequence"] < event["sequence"]]
    _require(len(all_parent_closures) == len(closures) == 1,
             "one exact resource closure required", "RECOVERY_REQUIRED")
    closure = closures[0]
    cost = settlement["payload"]
    _require(launch["payload"].get("attempt_id") == parent_attempt_id
             and launch["payload"].get("run_id") == run["run_id"]
             and stop["payload"].get("confirmation") == "native-job-or-process-group-no-running-descendants"
             and stop["payload"].get("attempt_id") == parent_attempt_id
             and all(cost.get(key) == value for key, value in {
                 "attempt_id": parent_attempt_id, "run_id": run["run_id"],
                 "study_id": run["study_id"], "arm_id": run["arm_id"]}.items())
             and type(cost.get("monotonic_elapsed_ms")) is int and cost["monotonic_elapsed_ms"] >= 0
             and cost.get("charged_ms") == cost["monotonic_elapsed_ms"]
             == stop["payload"].get("observed_elapsed_ms")
             and cost.get("settled") is True and cost.get("outcome") == "INTERRUPTED"
             and closure["payload"].get("reason") == "INTERRUPTED"
             and reserve["sequence"] < launch["sequence"] < stop["sequence"]
             < settlement["sequence"] < closure["sequence"] < parent_terminals[0]["sequence"]
             and not any(event["event_kind"] == "CHECKPOINT"
                         and event["payload"].get("attempt_id") == parent_attempt_id for event in events),
             "resource retry stop/cost/order differs", "RECOVERY_REQUIRED")
    # Before this reservation the arm must either have had no closure or have
    # been released by a verified continuation. A new unrelated closure cannot
    # be borrowed as retry authority.
    prior_closures = [event for event in events[:reserve["sequence"] - 1]
                      if event["event_kind"] == "ARM_CLOSED"
                      and event["payload"].get("arm_id") == run["arm_id"]]
    prior_continuations = [event for event in events[:reserve["sequence"] - 1]
                           if event["event_kind"] == "RESOURCE_ARM_CONTINUATION"
                           and store._manifest(event["payload"]["manifest_id"]).get("arm_id") == run["arm_id"]
                           and store._manifest(event["payload"]["manifest_id"]).get("execution_policy") == execution_policy
                           and store._manifest(event["payload"]["manifest_id"]).get("source_code_hash") == source_code_hash]
    authorized_after_closure = bool(prior_continuations
        and prior_continuations[-1]["sequence"] > prior_closures[-1]["sequence"])
    if prior_closures and authorized_after_closure:
        _resolve_arm_continuation(store, prior_continuations[-1]["payload"],
                                  allow_later_closure=True)
    restart_reference = reservation.get("restart_adjudication")
    if prior_closures and restart_reference is not None:
        previous = resolve_retry_adjudication(store, restart_reference,
            execution_policy=execution_policy, source_code_hash=source_code_hash)
        previous_closure = (previous.get("parent_event_hashes", {}).get("closure")
            or previous.get("historical_event_hashes", {}).get("closure"))
        authorized_after_closure = (previous.get("retry_attempt_id") == parent_attempt_id
            and previous_closure == prior_closures[-1]["hash"])
    _require(not prior_closures or authorized_after_closure,
             "resource retry parent ran behind an unrelated closed arm", "RECOVERY_REQUIRED")
    return {"parent": parent, "run": run, "related": related, "reserve": reserve,
        "launch": launch, "stop": stop, "settlement": settlement, "closure": closure,
        "terminal": parent_terminals[0], "cost_ms": cost["charged_ms"]}


def _resource_retry_attempt_payload(receipt):
    return {"attempt_id": receipt["retry_attempt_id"], "run_id": receipt["run_id"],
            "parent_attempt_id": receipt["parent_attempt_id"],
            "reason": "PIRC-28 explicit resource infrastructure retry",
            "state": "REGISTERED", "started_at": None, "ended_at": None,
            "error_code": None, "artifact_manifest_hash": None}


def _resolve_resource_retry(store, reference, *, allow_missing_attempt=False):
    receipt = _published(store, reference, "resource retry adjudication")
    _require(set(receipt) == RESOURCE_RETRY_FIELDS
             and receipt.get("schema_version") == RESOURCE_RETRY_SCHEMA,
             "resource retry adjudication schema differs")
    evidence = _resource_retry_evidence(store, receipt["parent_attempt_id"],
        execution_policy=receipt["execution_policy"], source_code_hash=receipt["current_code_hash"],
        allow_child=True)
    events = store._events()
    publications = [event for event in events if event["event_kind"] == "RESOURCE_RETRY_ADJUDICATED"
                    and event["payload"] == reference]
    expected_event_id = "pirc28-resource-retry-" + digest([store.store_id,
        receipt["parent_attempt_id"], receipt["execution_policy"], receipt["current_code_hash"]])
    expected_retry_id = digest([expected_event_id, evidence["run"]["run_id"],
                                evidence["run"]["cell_hash"]])[:32]
    expected_hashes = {name: evidence[name]["hash"] for name in
        ("reserve", "launch", "stop", "settlement", "closure", "terminal")}
    _require(len(publications) == 1 and publications[0]["event_id"] == expected_event_id
             and publications[0]["sequence"] > evidence["terminal"]["sequence"]
             and receipt["retry_attempt_id"] == expected_retry_id
             and receipt["run_id"] == evidence["run"]["run_id"]
             and receipt["cell_hash"] == evidence["run"]["cell_hash"]
             and receipt["parent_event_hashes"] == expected_hashes
             and receipt["parent_cost_ms"] == evidence["cost_ms"]
             and receipt["maximum_infrastructure_retries"] == 2,
             "resource retry evidence binding differs", "RECOVERY_REQUIRED")
    child = store._attempts().get(receipt["retry_attempt_id"])
    if child is None:
        _require(allow_missing_attempt, "resource retry child missing", "RECOVERY_REQUIRED")
        expected_ordinal = len(evidence["related"])
    else:
        initial = _resource_retry_attempt_payload(receipt)
        registrations = [event for event in events if event["event_kind"] == "ATTEMPT"
                         and event["payload"] == initial]
        _require(len(registrations) == 1 and all(child.get(field) == initial[field] for field in
            ("attempt_id", "run_id", "parent_attempt_id", "reason")),
            "resource retry child differs", "IDENTITY_CONFLICT")
        _require(registrations[0]["sequence"] > evidence["terminal"]["sequence"],
                 "resource retry child predates its parent terminal", "IDENTITY_CONFLICT")
        _require(publications[0]["sequence"] < registrations[0]["sequence"],
                 "resource retry child predates adjudication", "IDENTITY_CONFLICT")
        expected_ordinal, cursor, seen = 0, child, set()
        while cursor.get("parent_attempt_id") is not None:
            _require(cursor["attempt_id"] not in seen,
                     "resource retry ancestry cycles", "IDENTITY_CONFLICT")
            seen.add(cursor["attempt_id"])
            parent = store._attempts().get(cursor["parent_attempt_id"])
            _require(parent is not None, "resource retry ancestor missing", "IDENTITY_CONFLICT")
            if parent["run_id"] != receipt["run_id"]:
                break
            cursor = parent
            expected_ordinal += 1
    _require(receipt["retry_ordinal"] == expected_ordinal <= 2,
             "resource retry ordinal differs", "RETRY_EXHAUSTED")
    return deepcopy(receipt)


def resolve_retry_adjudication(store, reference, *, execution_policy, source_code_hash):
    """Resolve either the historical cross-epoch child or a current retry."""
    _prime_restart(store, reference)
    with store._read_transaction():
        _reference(reference, "retry adjudication")
        value = store._manifest(reference["manifest_id"])
        if value.get("schema_version") in {SCHEMA, PRELAUNCH_SCHEMA, SUPERSESSION_SCHEMA}:
            receipt = _resolve_receipt(store, reference)
        elif value.get("schema_version") == RESOURCE_RETRY_SCHEMA:
            receipt = _resolve_resource_retry(store, reference)
        elif value.get("schema_version") == PREFLIGHT_RECOVERY_SCHEMA:
            from .research_preflight_recovery import resolve_preflight_recovery
            receipt = resolve_preflight_recovery(store, reference)
        else:
            raise ResearchError("CONTRACT_MISMATCH", "unknown retry adjudication schema")
        _require(receipt["execution_policy"] == execution_policy
                 and receipt["current_code_hash"] == source_code_hash,
                 "retry adjudication funding binding differs", "UNAUTHORIZED_DATA")
        return receipt


def adjudicate_resource_retry(store, parent_attempt_id, *, execution_policy, source_code_hash):
    """Register one same-source retry after exact measured-stop adjudication."""
    from infrastructure.process_tree import process_may_be_alive

    prime_policy_definitions(store, execution_policy, source_code_hash)
    event_id = "pirc28-resource-retry-" + digest(
        [store.store_id, parent_attempt_id, execution_policy, source_code_hash])
    with store._read_transaction():
        from .research_preflight_recovery import forbid_extra_attempt
        forbid_extra_attempt(store, parent_attempt_id=parent_attempt_id)
        existing = [event for event in store._events() if event["event_id"] == event_id]
        if existing:
            _require(len(existing) == 1 and existing[0]["event_kind"] == "RESOURCE_RETRY_ADJUDICATED",
                     "resource retry replay differs", "IDENTITY_CONFLICT")
            reference = existing[0]["payload"]
            receipt = _resolve_resource_retry(store, reference, allow_missing_attempt=True)
            if store._attempts().get(receipt["retry_attempt_id"]) is None:
                store._append("ATTEMPT", _resource_retry_attempt_payload(receipt),
                              "attempt-" + receipt["retry_attempt_id"])
            return reference
        evidence = _resource_retry_evidence(store, parent_attempt_id,
            execution_policy=execution_policy, source_code_hash=source_code_hash)
        launch = evidence["launch"]
        pid = launch["payload"].get("pid")
        native = {"status": "recorded-worker-absent", "pid": pid,
                  "worker_event_hash": launch["hash"]}
        if process_may_be_alive(pid):
            observation = _pid_reuse_observation(pid, launch["created_at"])
            _require(observation is not None, "resource retry worker may still be alive", "WORKER_ACTIVE")
            native = {"status": "pid-reused-by-later-process", "worker_event_hash": launch["hash"],
                      **observation}
        retry_id = digest([event_id, evidence["run"]["run_id"], evidence["run"]["cell_hash"]])[:32]
        receipt = {"schema_version": RESOURCE_RETRY_SCHEMA,
            "execution_policy": deepcopy(execution_policy), "current_code_hash": source_code_hash,
            "parent_attempt_id": parent_attempt_id, "run_id": evidence["run"]["run_id"],
            "cell_hash": evidence["run"]["cell_hash"],
            "parent_event_hashes": {name: evidence[name]["hash"] for name in
                ("reserve", "launch", "stop", "settlement", "closure", "terminal")},
            "parent_cost_ms": evidence["cost_ms"], "retry_attempt_id": retry_id,
            "retry_ordinal": len(evidence["related"]), "maximum_infrastructure_retries": 2,
            "native_identity": native}
        object_id = "pirc28-resource-retry-adjudication-" + digest(receipt)
        reference = {"manifest_id": object_id, "sha256": digest(receipt)}
        store._publish(object_id, receipt)
        store._append("MANIFEST", {"object_id": object_id, "sha256": reference["sha256"]},
                      "manifest-" + digest([object_id, reference["sha256"]]))
        store._append("RESOURCE_RETRY_ADJUDICATED", reference, event_id)
        store._append("ATTEMPT", _resource_retry_attempt_payload(receipt), "attempt-" + retry_id)
        return reference


def adjudication_for_attempt(store, attempt_id, *, execution_policy, source_code_hash):
    """Find exactly one adjudication for an exact child attempt."""
    with store._read_transaction():
        references = []
        for event in store._events():
            if event["event_kind"] not in {"RESTART_ADJUDICATED", "RESOURCE_RETRY_ADJUDICATED",
                                           "PREFLIGHT_RECOVERY_ADJUDICATED"}:
                continue
            receipt = store._manifest(event["payload"]["manifest_id"])
            if receipt.get("retry_attempt_id") == attempt_id:
                references.append(event["payload"])
        _require(len(references) == 1, "exact restart adjudication missing", "BUDGET_EXHAUSTED")
        receipt = resolve_retry_adjudication(store, references[0],
            execution_policy=execution_policy, source_code_hash=source_code_hash)
        _require(receipt["execution_policy"] == execution_policy
                 and receipt["current_code_hash"] == source_code_hash,
                 "restart adjudication policy/source differs", "UNAUTHORIZED_DATA")
        if receipt.get("schema_version") == PREFLIGHT_RECOVERY_SCHEMA:
            _require(store._attempts()[attempt_id]['state'] in {'REGISTERED', 'RUNNING'},
                     'one-time terminal attempt is not new execution authority', 'BUDGET_EXHAUSTED')
        return {"reference": deepcopy(references[0]), "receipt": receipt}


def _resolve_arm_continuation(store, reference, *, allow_later_closure=False):
    value = _published(store, reference, "resource arm continuation")
    _require(set(value) == {"schema_version", "restart_adjudication", "execution_policy",
                "source_code_hash", "arm_id", "retry_attempt_id", "adjudicated_parent_cost_ms"}
             and value.get("schema_version") == CONTINUATION_SCHEMA,
             "resource arm continuation schema differs")
    adjudication = resolve_retry_adjudication(store, value.get("restart_adjudication"),
        execution_policy=value.get("execution_policy"), source_code_hash=value.get("source_code_hash"))
    parent_cost = adjudication.get("retained_cost_ms", adjudication.get("parent_cost_ms"))
    run_id = adjudication.get("current_run_id", adjudication.get("run_id"))
    run = store._manifest("run-" + run_id)
    _require(value.get("execution_policy") == adjudication["execution_policy"]
             and value.get("source_code_hash") == adjudication["current_code_hash"]
             and value.get("arm_id") == run["arm_id"]
             and value.get("retry_attempt_id") == adjudication["retry_attempt_id"]
             and value.get("adjudicated_parent_cost_ms") == parent_cost,
             "resource continuation adjudication binding differs", "UNAUTHORIZED_DATA")
    attempt = store._attempts().get(adjudication["retry_attempt_id"])
    events = store._events()
    publications = [event for event in events if event["event_kind"] == "RESOURCE_ARM_CONTINUATION"
                    and event["payload"] == reference]
    verified = [event for event in events if event["event_kind"] in {"REUSE_VERIFIED", "FAILURE_VERIFIED"}
             and event["payload"].get("attempt_id") == adjudication["retry_attempt_id"]]
    settlements = [event for event in events if event["event_kind"] == "SETTLE"
                   and event["payload"].get("attempt_id") == adjudication["retry_attempt_id"]]
    scientific_terminal = (attempt is not None and attempt["state"] in {"SUCCEEDED", "FAILED"}
        and attempt.get("artifact_id") and attempt.get("error_code") not in
        {"SUPERVISOR_ERROR", "RESOURCE_LIMIT", "HEARTBEAT_LOST"})
    if adjudication.get('schema_version') == PREFLIGHT_RECOVERY_SCHEMA:
        _require(attempt is not None and attempt['state'] == 'SUCCEEDED',
                 'one-time recovery failure cannot promote its arm', 'UNQUALIFIED')
    expected_verification = "REUSE_VERIFIED" if attempt and attempt["state"] == "SUCCEEDED" else "FAILURE_VERIFIED"
    _require(scientific_terminal and len(verified) >= 1 and len(settlements) == 1
             and settlements[0]["payload"].get("outcome") == attempt["state"]
             and type(settlements[0]["payload"].get("charged_ms")) is int
             and len(publications) == 1
             and any(event["event_kind"] == expected_verification
                     and event["sequence"] < publications[0]["sequence"] for event in verified),
             "verified scientific restart required before arm continuation", "UNQUALIFIED")
    later_closures = [event for event in events[publications[0]["sequence"]:]
                      if event["event_kind"] == "ARM_CLOSED"
                      and event["payload"].get("arm_id") == value["arm_id"]]
    _require(allow_later_closure or not later_closures,
             "resource arm closed again after continuation", "BUDGET_EXHAUSTED")
    return deepcopy(value)


def resource_arm_was_promoted(store, arm_id, *, execution_policy, source_code_hash):
    """Verify that this source/arm had at least one valid continuation."""
    matched = []
    for event in store._events():
        if event["event_kind"] != "RESOURCE_ARM_CONTINUATION":
            continue
        value = store._manifest(event["payload"]["manifest_id"])
        if (value.get("arm_id") == arm_id and value.get("execution_policy") == execution_policy
                and value.get("source_code_hash") == source_code_hash):
            matched.append(event["payload"])
    if not matched:
        return False
    _resolve_arm_continuation(store, matched[0], allow_later_closure=True)
    return True


def resource_arm_continuation(store, arm_id, *, execution_policy, source_code_hash):
    """Return the latest active policy-bound continuation, or ``None``."""
    events = store._events()
    publications = [event for event in events
                    if event["event_kind"] == "RESOURCE_ARM_CONTINUATION"]
    matched = []
    for event in publications:
        reference = event["payload"]
        value = store._manifest(reference["manifest_id"])
        if (value.get("arm_id") == arm_id and value.get("execution_policy") == execution_policy
                and value.get("source_code_hash") == source_code_hash):
            matched.append(event)
    if not matched:
        return None
    latest = matched[-1]
    later_closure = any(event["event_kind"] == "ARM_CLOSED"
        and event["payload"].get("arm_id") == arm_id
        and event["sequence"] > latest["sequence"] for event in events)
    if later_closure:
        return None
    reference = latest["payload"]
    return {"reference": deepcopy(reference), "receipt": _resolve_arm_continuation(store, reference)}


def promote_resource_arm(store, restart_adjudication):
    """After a verified scientific terminal, permit remaining arm cells."""
    with store._read_transaction():
        _reference(restart_adjudication, "retry adjudication")
        raw = store._manifest(restart_adjudication["manifest_id"])
        policy = raw.get("execution_policy")
        source = raw.get("current_code_hash")
        adjudication = resolve_retry_adjudication(store, restart_adjudication,
            execution_policy=policy, source_code_hash=source)
        if adjudication.get('schema_version') == PREFLIGHT_RECOVERY_SCHEMA:
            _require(store._attempts()[adjudication['retry_attempt_id']]['state'] == 'SUCCEEDED',
                     'one-time recovery failure cannot promote its arm', 'UNQUALIFIED')
        run_id = adjudication.get("current_run_id", adjudication.get("run_id"))
        run = store._manifest("run-" + run_id)
        parent_cost = adjudication.get("retained_cost_ms", adjudication.get("parent_cost_ms"))
        event_id = "pirc28-resource-arm-continuation-" + digest(restart_adjudication)
        existing = [event for event in store._events() if event["event_id"] == event_id]
        if existing:
            _require(len(existing) == 1 and existing[0]["event_kind"] == "RESOURCE_ARM_CONTINUATION",
                     "resource continuation replay differs", "IDENTITY_CONFLICT")
            _resolve_arm_continuation(store, existing[0]["payload"])
            return existing[0]["payload"]
        value = {"schema_version": CONTINUATION_SCHEMA,
            "restart_adjudication": deepcopy(restart_adjudication),
            "execution_policy": deepcopy(adjudication["execution_policy"]),
            "source_code_hash": adjudication["current_code_hash"], "arm_id": run["arm_id"],
            "retry_attempt_id": adjudication["retry_attempt_id"],
            "adjudicated_parent_cost_ms": parent_cost}
        object_id = "pirc28-resource-arm-continuation-" + digest(value)
        reference = {"manifest_id": object_id, "sha256": digest(value)}
        store._publish(object_id, value)
        store._append("MANIFEST", {"object_id": object_id, "sha256": reference["sha256"]},
                      "manifest-" + digest([object_id, reference["sha256"]]))
        store._append("RESOURCE_ARM_CONTINUATION", reference, event_id)
        _resolve_arm_continuation(store, reference)
        return reference
