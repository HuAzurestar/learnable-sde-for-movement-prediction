"""Explicit, append-only concurrent compute permission; never a budget bypass.

Compute tickets are independent from the single authoritative metadata writer.
An activation does not modify the immutable legacy runtime-resources manifest,
existing reservations, data grants, job limits or attempt/claim uniqueness.
"""
from infrastructure.research_store import ResearchError, digest

SCHEMA = "pirc25-concurrent-dispatch-policy-v1"
EVENT = "DISPATCH_POLICY_ACTIVATED"
FIELDS = {"schema_version", "store_id", "decision", "human_issuer", "quoted_authorization",
          "resources", "arm_prefixes", "write_policy", "budget_policy_changed", "data_scope_changed"}


def validate_policy(store, value):
    def require(condition, detail):
        if not condition:
            raise ResearchError("CONTRACT_MISMATCH", detail)
    require(type(value) is dict and set(value) == FIELDS, "closed dispatch policy required")
    require(value["schema_version"] == SCHEMA and value["store_id"] == store.store_id
            and value["decision"] == "approved", "dispatch policy identity/approval differs")
    require(all(type(value[key]) is str and 0 < len(value[key].strip()) <= 8192
                for key in ("human_issuer", "quoted_authorization")), "explicit human dispatch authorization required")
    resources, prefixes = value["resources"], value["arm_prefixes"]
    require(type(resources) is list and bool(resources)
            and all(type(r) is str and r in ("cpu", "gpu") for r in resources)
            and resources == sorted(set(resources)), "explicit unique compute resources required")
    require(type(prefixes) is list and 1 <= len(prefixes) <= 64
            and all(type(p) is str and 0 < len(p) <= 128 for p in prefixes)
            and prefixes == sorted(set(prefixes)), "explicit nonempty arm scope required")
    require(value["write_policy"] == "atomic-single-writer"
            and value["budget_policy_changed"] is False and value["data_scope_changed"] is False,
            "compute permission must preserve write exclusion, budget and data scope")
    return value


def _resolve(store, reference):
    if (type(reference) is not dict or set(reference) != {"manifest_id", "sha256"}
            or type(reference["manifest_id"]) is not str or type(reference["sha256"]) is not str):
        raise ResearchError("CONTRACT_MISMATCH", "exact published dispatch reference required")
    value = store._manifest(reference["manifest_id"])
    if digest(value) != reference["sha256"]:
        raise ResearchError("CORRUPT_ARTIFACT", "dispatch policy reference hash differs")
    return validate_policy(store, value)


def activate_policy(store, reference):
    """Explicit owner operation; a worker never grants itself concurrency."""
    with store._read_transaction():
        if (store.path / "recovery-hold.json").exists():
            raise ResearchError("RECOVERY_REQUIRED", "recovery hold prohibits dispatch activation")
        value = _resolve(store, reference)
        store._append(EVENT, {"reference": reference, "store_id": value["store_id"]},
                      "activate-dispatch-" + digest(reference))
        store._read_completion(lambda: _resolve(store, reference), lambda: None)
    return reference


def concurrent_permission(store, arm_id, resource):
    """Fresh exact activation, scoped by arm and resource; no implicit latest grant.

Separate CPU/GPU activations compose without replacing prior immutable facts.
Every activation is revalidated against physical manifests in the caller's
transaction; invalid authority never silently falls back or opens a slot.
"""
    allowed = None
    for event in store._events():
        if event["event_kind"] != EVENT:
            continue
        payload = event["payload"]
        if type(payload) is not dict or set(payload) != {"reference", "store_id"} or payload["store_id"] != store.store_id:
            raise ResearchError("CONTRACT_MISMATCH", "dispatch activation identity differs")
        value = _resolve(store, payload["reference"])
        publication = store._manifest_event(payload["reference"]["manifest_id"], last=True)
        if publication is None or publication["sequence"] >= event["sequence"]:
            raise ResearchError("CONTRACT_MISMATCH", "dispatch policy was not published before activation")
        if resource in value["resources"] and any(arm_id.startswith(p) for p in value["arm_prefixes"]):
            allowed = payload["reference"]
    return allowed
