"""Hash-bound opt-in execution policies; inspection never authorizes a run.

V2 is scoped to the human-requested PIRC-28 experiment. Legacy budgets and
other participants are untouched. A local decision record is not an external
signature, a data grant, a historical closure adjudication or a worker lease.
"""

from copy import deepcopy
import re

from infrastructure.research_store import ResearchError, digest


SCHEMA = "pirc25-execution-policy-v2"
RESOURCE_FIELDS = frozenset({"cpu_slots", "gpu_slots", "minimum_system_free_bytes",
    "minimum_runtime_free_bytes", "minimum_available_memory_bytes", "maximum_worker_rss_bytes"})
POLICY_FIELDS = frozenset({"schema_version", "store_id", "feature_id", "arm_ids",
    "decision", "budget_mode", "job_seconds", "arm_seconds", "max_infrastructure_retries",
    "resources", "source_code_hash", "decision_basis", "real_data_authorized",
    "test_authorization", "public_export"})


def _require(condition, message, code="CONTRACT_MISMATCH"):
    if not condition:
        raise ResearchError(code, message)


def _hash(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_policy(value, *, store_id, arm_id, source_code_hash):
    """Strict metadata validation, with no default or implicit latest policy."""
    _require(type(value) is dict and set(value) == POLICY_FIELDS,
             "execution policy fields differ")
    _require(value["schema_version"] == SCHEMA and value["decision"] == "approved"
             and value["feature_id"] == "PIRC-28" and value["store_id"] == store_id
             and value["arm_ids"] == ["pirc28-k1", "pirc28-k2"]
             and arm_id in value["arm_ids"], "execution policy participant/store/arm differs",
             "UNAUTHORIZED_DATA")
    _require(value["budget_mode"] == "resource-guarded-no-time-cap"
             and value["job_seconds"] is None and value["arm_seconds"] is None
             and type(value["max_infrastructure_retries"]) is int
             and value["max_infrastructure_retries"] == 2,
             "explicit nullable time caps and finite retries required")
    _require(_hash(source_code_hash) and value["source_code_hash"] == source_code_hash,
             "execution policy source epoch differs", "IDENTITY_CONFLICT")
    _require(all(value[field] is False for field in
                 ("real_data_authorized", "test_authorization", "public_export")),
             "time policy cannot broaden data/test/export permission", "UNAUTHORIZED_DATA")
    basis = value["decision_basis"]
    _require(type(basis) is dict and set(basis) == {"authority", "task_id", "session_id", "instructions"}
             and basis["authority"] == "local-human-decision-record"
             and all(type(basis[field]) is str and 0 < len(basis[field]) <= 128
                     for field in ("task_id", "session_id"))
             and type(basis["instructions"]) is list and 1 <= len(basis["instructions"]) <= 8
             and all(type(text) is str and 0 < len(text) <= 4096 for text in basis["instructions"]),
             "bounded exact human decision provenance required")
    resources = value["resources"]
    _require(type(resources) is dict and set(resources) == RESOURCE_FIELDS
             and type(resources["cpu_slots"]) is int and resources["cpu_slots"] == 1
             and type(resources["gpu_slots"]) is int and resources["gpu_slots"] == 0
             and all(type(resources[field]) is int and 0 < resources[field] < 2**63
                     for field in RESOURCE_FIELDS - {"cpu_slots", "gpu_slots"}),
             "single-worker finite storage/memory guards required")
    return deepcopy(value)


def resolve_policy(store, reference, *, arm_id, source_code_hash):
    """Recheck immutable policy and the final physical prefix on every call."""
    _require(type(reference) is dict and set(reference) == {"manifest_id", "sha256"}
             and type(reference["manifest_id"]) is str and _hash(reference["sha256"]),
             "exact execution policy reference required")
    result = {}
    def check():
        if (store.path / "recovery-hold.json").exists():
            raise ResearchError("RECOVERY_REQUIRED", "execution policy blocked by recovery hold")
        value = store._manifest(reference["manifest_id"])
        _require(digest(value) == reference["sha256"], "execution policy hash differs",
                 "IDENTITY_CONFLICT")
        _require(any(event["event_kind"] == "MANIFEST"
                     and event["payload"] == {"object_id": reference["manifest_id"],
                                               "sha256": reference["sha256"]}
                     for event in store._events()), "execution policy lacks authoritative publication")
        result.clear()
        result.update(validate_policy(value, store_id=store.store_id, arm_id=arm_id,
                                      source_code_hash=source_code_hash))
    with store._read_transaction():
        check()
        store._read_completion(check, lambda: None)
    return result


def check_resource_observation(policy, observation):
    """Fresh trusted-owner samples only; no observation is reusable admission.

    This pure checker neither samples the OS nor terminates processes. The
    supervisor integration must supply current observations independently of
    ledger I/O; unknown measurements fail closed rather than assuming space.
    """
    fields = {"system_free_bytes", "runtime_free_bytes", "available_memory_bytes", "worker_rss_bytes"}
    _require(type(observation) is dict and set(observation) == fields
             and all(type(value) is int and 0 <= value < 2**63 for value in observation.values()),
             "unknown/malformed resource observation", "RESOURCE_PLAN_REJECTED")
    rules = policy["resources"]
    for actual, minimum in (("system_free_bytes", "minimum_system_free_bytes"),
                            ("runtime_free_bytes", "minimum_runtime_free_bytes"),
                            ("available_memory_bytes", "minimum_available_memory_bytes")):
        _require(observation[actual] >= rules[minimum], "resource floor reached: " + actual,
                 "RESOURCE_LIMIT")
    _require(observation["worker_rss_bytes"] <= rules["maximum_worker_rss_bytes"],
             "worker memory envelope exceeded", "RESOURCE_LIMIT")
    return {"state": "RESOURCE_SAMPLE_WITHIN_LIMITS", "execution_permitted": False}
