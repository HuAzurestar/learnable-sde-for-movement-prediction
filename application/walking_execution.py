"""Append-only walking execution identity; the original policy owns all money.

Registration is deliberately explicit. Publishing a config or a study alone
never activates it. All resolution happens within the caller's verified scope.
"""
from datetime import datetime

from infrastructure.research_store import ResearchError, digest, identifier


def reference(store, ref):
    if not isinstance(ref, dict) or set(ref) != {"manifest_id", "sha256"}:
        raise ResearchError("UNAUTHORIZED_DATA", "exact immutable reference required")
    value = store._manifest(ref["manifest_id"])
    if digest(value) != ref["sha256"]:
        raise ResearchError("UNAUTHORIZED_DATA", "execution reference hash differs")
    return value


def activations(store, pilot_id):
    return [e["payload"] for e in store._events()
            if e["event_kind"] == "WALKING_EXECUTION_ACTIVATED"
            and e["payload"]["budget_pilot_id"] == pilot_id]


def _validate(store, binding, policy, previous):
    from .walking_pilot import cells, code_identity
    if (binding.get("schema") not in {"walking-execution-v1","walking-execution-grid-v1"}
            or binding.get("budget_pilot_id") != policy["pilot_id"]
            or binding.get("arms") != policy["arms"]
            or binding.get("expires_at") != policy["expires_at"]
            or binding.get("previous_execution_ref") != previous):
        raise ResearchError("UNAUTHORIZED_DATA", "execution scope or predecessor differs")
    identifier(binding["execution_id"])
    if binding["study_id"] == policy["study_id"]:
        raise ResearchError("IDENTITY_CONFLICT", "execution requires a new immutable study")
    config = reference(store, binding["config_ref"])
    from .walking_pilot import validate_config
    from .walking_pilot_budget import GRID_FIELDS,GRID_BASIS,LIMIT_BASIS
    validate_config(config)
    grid=binding["schema"]=="walking-execution-grid-v1"
    if grid!=(config.get("schema_version")=="pirc26-walking-config-grid-v1"):
        raise ResearchError("CONTRACT_MISMATCH","execution/config billing basis differs")
    if grid:
        if binding.get("billing_basis")!=GRID_BASIS or binding.get("limit_basis")!=LIMIT_BASIS:
            raise ResearchError("CONTRACT_MISMATCH","unknown execution billing basis")
        receipt=reference(store,binding["billing_decision_ref"])
        if not receipt or any(config.get(k)!=binding.get(k) for k in GRID_FIELDS):
            raise ResearchError("UNAUTHORIZED_DATA","billing decision binding differs")
    elif any(k in binding or k in config for k in GRID_FIELDS):
        raise ResearchError("CONTRACT_MISMATCH","legacy execution cannot carry grid billing")
    old = store._manifest("walking-config-" + policy["pilot_id"])
    study = store._manifest("study-" + binding["study_id"])
    qualification = reference(store, binding["qualification_ref"])
    if grid and any(qualification.get(k)!=binding.get(k) for k in GRID_FIELDS):
        raise ResearchError("UNAUTHORIZED_DATA","qualification billing binding differs")
    decision = reference(store, binding["decision_ref"])
    if (decision.get("user") != "azurestarhu"
            or decision.get("decision_record") != "PIRC-26/DECISION-03@v1"
            or decision.get("solution") != "SOL-014@v2/B1"
            or decision.get("implementation") is not True):
        raise ResearchError("UNAUTHORIZED_DATA", "accurate B1 decision receipt required")
    if (config.get("execution_id") != binding["execution_id"]
            or config.get("pilot_id") != policy["pilot_id"]
            or config.get("study_id") != binding["study_id"]
            or config.get("arms") != policy["arms"]
            or config.get("qualification") != binding["qualification_ref"]
            or binding.get("code_hash") != code_identity()
            or study["spec"].get("code_hash") != binding["code_hash"]
            or qualification.get("code_hash") != binding["code_hash"]
            or qualification.get("scope") != "independent-walking-pilot"
            or qualification.get("status") != "passed"
            or study["spec"]["cells"] != cells(config)
            or study["spec"]["data_hash"] != digest(config["sources"])):
        raise ResearchError("UNAUTHORIZED_DATA", "execution code/config/qualification differs")
    # Registration cannot alter the original numerical or source contract.
    for key in ("motion", "final_test", "speed_limit", "speed_basis", "speed_frozen_at",
                "train_steps", "preparation_arm", "source_mode", "source_inventory", "source_geometry"):
        if config.get(key) != old.get(key):
            raise ResearchError("UNAUTHORIZED_DATA", "execution changed original scope")
    for key in ("source_inventory", "source_geometry"):
        if binding.get(key + "_hash") != old[key]["sha256"]:
            raise ResearchError("UNAUTHORIZED_DATA", "original source identity differs")
    strip = lambda rows: [{k: v for k, v in row.items()
                           if k not in {"authorization_id", "authorization_version"}} for row in rows]
    if strip(config["sources"]) != strip(old["sources"]):
        raise ResearchError("UNAUTHORIZED_DATA", "execution changed source contents or roles")
    # A code/study rebind is never a new data authority. Require exact new
    # versions of the already registered grants, preserving their owner
    # evidence, expiry, source study, consumer restrictions and every scope
    # field. Only internal outputs move to the new execution study.
    pairs=[(a,b,False) for a,b in zip(old["sources"],config["sources"])]
    pairs.append((old["artifact_grant"],config["artifact_grant"],True))
    for prior,current,internal in pairs:
        before=store.authorization(prior["authorization_id"],version=prior.get("version") if internal else prior.get("authorization_version"))
        after=store.authorization(current["authorization_id"],version=current.get("version") if internal else current.get("authorization_version"))
        expected=rebind_grant(before,binding["study_id"],after.get("version"),internal=internal)
        if after!=expected:
            raise ResearchError("UNAUTHORIZED_DATA", "execution grant expanded or replaced original authority")

    if datetime.fromisoformat(binding["expires_at"]) > datetime.fromisoformat(policy["expires_at"]):
        raise ResearchError("UNAUTHORIZED_DATA", "execution extended permission")
    return config


def legacy_ref(store, policy):
    return {"policy_hash": digest(policy), "config_hash": policy["config_hash"],
            "study_hash": digest(store._manifest("study-" + policy["study_id"]))}


def resolve_execution(store, policy, execution_id=None):
    """Resolve only the exact active execution, or retain the old v1 behavior."""
    with store._read_transaction():
        active = activations(store, policy["pilot_id"])
        if not active:
            if execution_id is not None:
                raise ResearchError("UNAUTHORIZED_DATA", "execution is not activated")
            return policy, store._manifest("walking-config-" + policy["pilot_id"])
        previous = legacy_ref(store, policy)
        binding = config = None
        for event in active:
            binding = reference(store, event["execution_ref"])
            # Old execution code need not match this process's code. Its
            # immutable predecessor/hash and scope are verified by activation;
            # the current binding is checked in full below.
            if binding["previous_execution_ref"] != previous:
                raise ResearchError("UNAUTHORIZED_DATA", "execution activation chain differs")
            previous = event["execution_ref"]
        if execution_id is not None and binding["execution_id"] != execution_id:
            raise ResearchError("UNAUTHORIZED_DATA", "execution is no longer active")
        before = legacy_ref(store, policy) if len(active) == 1 else active[-2]["execution_ref"]
        config = _validate(store, binding, policy, before)
        return {**policy, "study_id": binding["study_id"], "config_hash": digest(config),
                "execution_id": binding["execution_id"], "original_study_id": policy["study_id"],
                **{k:binding[k] for k in ("billing_basis","limit_basis","billing_decision_ref") if k in binding}}, config


def activate_execution(store, binding):
    """Explicit operator action, idempotent only for the current exact binding."""
    from .research_budget import BudgetLedger
    from .walking_pilot_budget import _policy_in_scope, usage, owner_opens
    with store._read_transaction():
        ledger = BudgetLedger(store)
        p = _policy_in_scope(ledger, binding["budget_pilot_id"])
        if owner_opens(store,p["pilot_id"]):
            raise ResearchError("RECOVERY_REQUIRED","controller costs are unresolved")
        active = activations(store, p["pilot_id"])
        ref = {"manifest_id": "walking-execution-" + identifier(binding["execution_id"]), "sha256": digest(binding)}
        if active and active[-1]["execution_ref"] == ref:
            resolve_execution(store, p, binding["execution_id"])
            return ref
        if any(e["execution_ref"]["manifest_id"] == ref["manifest_id"] for e in active):
            raise ResearchError("IDENTITY_CONFLICT", "retired execution cannot reactivate")
        if usage(ledger, p["pilot_id"])["pending"] or any(
                a["state"] not in store.TERMINAL and
                store._manifest("run-" + a["run_id"])["cell"].get("pilot_id") == p["pilot_id"]
                for a in store._attempts().values()):
            raise ResearchError("RECOVERY_REQUIRED", "prior execution has unresolved attempts")
        stopped = {e["payload"].get("reservation_id") for e in store._events()
                   if e["event_kind"] in {"WORKER_TREE_STOPPED", "WORKER_STOP_CONFIRMED"}}
        for event in store._events():
            if event["event_kind"] != "WORKER_STARTED":
                continue
            worker = event["payload"]
            attempt = store._attempts().get(worker["attempt_id"])
            if (attempt and store._manifest("run-" + attempt["run_id"])["cell"].get("pilot_id") == p["pilot_id"]
                    and worker.get("reservation_id") not in stopped):
                raise ResearchError("WORKER_ACTIVE", "prior execution lacks native whole-tree stop proof")
        previous = active[-1]["execution_ref"] if active else legacy_ref(store, p)
        _validate(store, binding, p, previous)
        store.publish(ref["manifest_id"], binding)
        store._append("WALKING_EXECUTION_ACTIVATED", {"budget_pilot_id": p["pilot_id"], "execution_ref": ref},
                      "walking-activation-" + binding["execution_id"])
        return ref


def execution_config(store, pilot_id, execution_id=None):
    from .walking_pilot_budget import _policy_in_scope
    from .research_budget import BudgetLedger
    p = _policy_in_scope(BudgetLedger(store), pilot_id)
    return resolve_execution(store, p, execution_id)[1]


def rebind_grant(original, study_id, version, *, internal=False):
    """Produce a scope-preserving registration template, with no side effects."""
    identifier(study_id); identifier(version)
    if (version==original.get("version") or original.get("test_authorization") is not False
            or original.get("public_export") is not False or original.get("movement_scope")!="walking"):
        raise ResearchError("UNAUTHORIZED_DATA", "exact new version of original walking grant required")
    return {**original,"version":version,"pilot_study_id":study_id,
            **({"study_id":study_id} if internal else {})}
