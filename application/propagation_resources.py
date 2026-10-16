"""Per-attempt propagation observations; never budget or qualification tokens.

Reuse the delivered native self-process observer. Duration covers the worker
validation/method/reference/qualification pipeline, not just solver arithmetic,
imports, parent admission, output encoding or ledger settlement. Memory is a
lifetime peak including interpreter/dependencies, not a phase delta/tree peak.
"""
import re
import time
import os
from pathlib import Path

from infrastructure.pirc26_process_resources import process_resources, SCHEMA as OBSERVER_SCHEMA
from infrastructure.research_store import ResearchError, digest, encode


SCHEMA = "pirc27-propagation-resource-observation-v1"
DURATION_SCOPE = "worker-validation-method-reference-qualification-pipeline"
MEMORY_SCOPE = "current-process-lifetime-through-observation"
POLICY = {"schema_version": "pirc27-propagation-resource-policy-v1", "required": True,
          "duration_scope": DURATION_SCOPE, "memory_scope": MEMORY_SCOPE}
POINTS = {"SUCCEEDED": "job-payload-before-encoding", "FAILED": "observer-call",
          "CHECKPOINTED": "checkpoint-save-before-ack"}
PLUGINS = frozenset({"affine-propagation", "affine-propagation-chunk", "synthetic-propagation",
    "synthetic-propagation-chunk", "affine-propagation-qualification", "affine-mlmc-qualification-chunk",
    "affine-mlmc-production-chunk", "affine-mixture-qualification", "affine-mixture-production-chunk",
    "affine-mixture-chunk", "synthetic-mixture-chunk", "affine-path-qualification",
    "affine-path-production-chunk", "affine-cubature", "affine-cubature-qualification",
    "affine-halfspace-calibration"})
MEMORY_KEYS = frozenset({"observer_schema", "status", "point", "platform", "scope", "units",
    "includes_children", "includes_previous_attempts", "is_phase_isolated",
    "resident_counter", "peak_resident_bytes", "reason"})
PACKET_KEYS = frozenset({"schema_version", "study_id", "spec_hash", "cell_hash", "code_hash",
    "attempt_id", "policy_hash", "execution_state", "resumed", "includes_previous_attempts", "duration", "memory"})


def require(condition, message):
    if not condition:
        raise ResearchError("CONTRACT_MISMATCH", "propagation resource evidence: " + message)


def resource_policy():
    return dict(POLICY)


def observation_enabled(cell):
    if "resource_observation_policy" not in cell:
        return False  # Legacy evidence remains unmeasured, never zero.
    require(type(cell["resource_observation_policy"]) is dict
            and encode(cell["resource_observation_policy"]) == encode(POLICY), "unknown or changed frozen policy")
    require(cell.get("resource_class") == "cpu" and cell.get("plugin_id") in PLUGINS
            and type(cell.get("propagation_request")) is dict,
            "only the explicit CPU propagation pipeline is supported")
    return True


def attempt_from_output(spec, output):
    """Lexical binding only; admission/native owner checks remain authoritative."""
    binding = spec.get("runtime_binding")
    require(type(binding) is dict and type(binding.get("root")) is str, "existing runtime root required")
    root = Path(binding["root"])
    output = Path(os.path.abspath(output))
    require(root.is_absolute() and output.name == "result.json"
            and output.parent.parent == root/"pirc25"/"artifacts"
            and re.fullmatch(r"\.attempt-[0-9a-f]{32}", output.parent.name) is not None,
            "output must belong to this existing owner's unique attempt")
    return output.parent.name[len(".attempt-"):]


def validate_packet(value, spec, cell, attempt_id, *, execution_state=None):
    require(observation_enabled(cell), "measurement policy absent")
    require(type(value) is dict and set(value) == PACKET_KEYS, "packet fields differ")
    require(type(attempt_id) is str and re.fullmatch(r"[0-9a-f]{32}", attempt_id) is not None,
            "actual attempt identity required")
    expected = {"schema_version": SCHEMA, "study_id": spec["study_id"], "spec_hash": digest(spec),
        "cell_hash": digest(cell), "code_hash": spec["code_hash"], "attempt_id": attempt_id,
        "policy_hash": digest(cell["resource_observation_policy"]), "includes_previous_attempts": False}
    require(encode({k: value[k] for k in expected}) == encode(expected), "source or attempt binding differs")
    state = value["execution_state"]
    require(type(state) is str and state in POINTS and (execution_state is None or state == execution_state),
            "execution disposition differs")
    require(type(value["resumed"]) is bool, "resume marker must be a boolean")
    duration = value["duration"]
    require(type(duration) is dict and set(duration) == {"wall_ns", "cpu_ns", "units", "scope", "cpu_scope"},
            "duration fields differ")
    require(type(duration["wall_ns"]) is int and 0 < duration["wall_ns"] < 1 << 63
            and type(duration["cpu_ns"]) is int and 0 <= duration["cpu_ns"] < 1 << 63,
            "finite nonnegative integer durations required")
    require(duration["units"] == "ns" and duration["scope"] == DURATION_SCOPE
            and duration["cpu_scope"] == "current-process-all-threads", "duration scope/units differ")
    # CPU sums all process threads and can exceed wall time; do not clamp it.
    memory = value["memory"]
    require(type(memory) is dict and set(memory) == MEMORY_KEYS, "memory fields differ")
    fixed = {"observer_schema": OBSERVER_SCHEMA, "point": POINTS[state], "scope": MEMORY_SCOPE,
        "units": "bytes", "includes_children": False, "includes_previous_attempts": False, "is_phase_isolated": False}
    require(encode({k: memory[k] for k in fixed}) == encode(fixed), "memory scope/units differ")
    platform = memory["platform"]
    require(type(platform) is str and re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", platform) is not None,
            "native platform identifier differs")
    native = ("windows.PROCESS_MEMORY_COUNTERS_EX.PeakWorkingSetSize" if platform == "win32" else
              "linux.getrusage.RUSAGE_SELF.ru_maxrss" if platform.startswith("linux") else None)
    if memory["status"] == "MEASURED":
        require(native is not None and memory["resident_counter"] == native
                and type(memory["peak_resident_bytes"]) is int and 0 < memory["peak_resident_bytes"] < 1 << 63
                and memory["reason"] is None, "native measured RSS differs")
    else:
        require(memory["status"] == "UNAVAILABLE" and memory["peak_resident_bytes"] is None
                and memory["resident_counter"] is None
                and memory["reason"] == ("NATIVE_COUNTER_UNAVAILABLE" if native else "UNSUPPORTED_PLATFORM"),
                "unavailable observation must retain null and a reason")
    require(len(encode(value)) <= 16384, "resource packet byte bound exceeded")


class PropagationResourceProbe:
    def __init__(self, spec, cell, attempt_id, *, resumed=False):
        require(observation_enabled(cell), "explicit policy required before measurement")
        require(type(resumed) is bool and type(attempt_id) is str
                and re.fullmatch(r"[0-9a-f]{32}", attempt_id) is not None, "invalid attempt/resume identity")
        require(sum(digest(c) == digest(cell) for c in spec["cells"]) == 1, "cell is not uniquely registered")
        self.spec, self.cell, self.attempt_id, self.resumed = spec, cell, attempt_id, resumed
        self.binding = (digest(spec), digest(cell), spec["code_hash"], digest(cell["resource_observation_policy"]))
        self.wall_start, self.cpu_start = time.perf_counter_ns(), time.process_time_ns()

    def snapshot(self, execution_state):
        require(type(execution_state) is str and execution_state in POINTS, "unsupported disposition")
        require((digest(self.spec), digest(self.cell), self.spec["code_hash"],
                 digest(self.cell["resource_observation_policy"])) == self.binding,
                "frozen measurement context changed during execution")
        wall, cpu = time.perf_counter_ns()-self.wall_start, time.process_time_ns()-self.cpu_start
        observed = process_resources(POINTS[execution_state])
        # Drop process IDs, absolute clocks, private commit and unrelated fields
        # from public synthetic evidence. Keep the original counter semantics.
        memory = {k: observed[k] for k in MEMORY_KEYS - {"observer_schema"}}
        memory["observer_schema"] = observed["schema_version"]
        value = {"schema_version": SCHEMA, "study_id": self.spec["study_id"],
            "spec_hash": digest(self.spec), "cell_hash": digest(self.cell), "code_hash": self.spec["code_hash"],
            "attempt_id": self.attempt_id, "policy_hash": digest(self.cell["resource_observation_policy"]),
            "execution_state": execution_state, "resumed": self.resumed, "includes_previous_attempts": False,
            "duration": {"wall_ns": wall, "cpu_ns": cpu, "units": "ns", "scope": DURATION_SCOPE,
                         "cpu_scope": "current-process-all-threads"}, "memory": memory}
        validate_packet(value, self.spec, self.cell, self.attempt_id, execution_state=execution_state)
        return value
