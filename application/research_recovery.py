"""Declared method recovery; checkpoint state never includes budget balances."""

from __future__ import annotations

from dataclasses import dataclass
import json

from .research_budget import BudgetLedger, BudgetSpec
from .research_admission import AdmissionGate
from infrastructure.research_store import ResearchError, ResearchStore, digest


@dataclass(frozen=True)
class RecoveryPlugin:
    plugin_id: str
    resume_level: str
    command_builder: object


class RecoveryRegistry:
    def __init__(self):
        self.plugins = {}

    def register(self, plugin: RecoveryPlugin):
        if (plugin.plugin_id in self.plugins or plugin.resume_level not in {"exact", "chunk"}
                or not callable(plugin.command_builder)):
            raise ResearchError("CONTRACT_MISMATCH", "invalid recovery plugin")
        self.plugins[plugin.plugin_id] = plugin

    def resolve(self, plugin_id, level):
        plugin = self.plugins.get(plugin_id)
        if level == "restart-only" or plugin is None or plugin.resume_level != level:
            raise ResearchError("CONTRACT_MISMATCH", "method does not implement requested recovery level")
        return plugin


def bindings(spec, cell):
    arms = [arm for arm in spec["arms"] if arm["arm_id"] == cell["arm_id"]]
    return {"schema_version": spec["schema_version"], "code_hash": spec["code_hash"],
            "data_hash": spec["data_hash"], "protocol_hash": spec["protocol_hash"],
            "feature_hash": spec["feature_hash"], "selection_hash": spec["selection_hash"],
            "model_hash": digest(arms[0]["model_family_id"]),
            "objective_hash": digest(arms[0]["objective_id"]), "cell_hash": digest(cell)}


class SharedRecovery:
    def __init__(self, store: ResearchStore, execution_registry, recovery_registry):
        self.store = store
        self.execution = execution_registry
        self.recovery = recovery_registry

    def _context(self, attempt_id):
        attempts = self.store.attempts()
        if attempt_id not in attempts:
            raise ResearchError("MISSING_INPUT", "recovery parent absent")
        attempt = attempts[attempt_id]
        run = self.store.manifest("run-" + attempt["run_id"])
        spec = self.store.manifest("study-" + run["study_id"])["spec"]
        cell = run["cell"]
        plugin = self.execution.resolve(cell["plugin_id"], cell["capability"])
        return attempt, run, spec, cell, plugin

    def checkpoint(self, attempt_id, state: dict):
        attempt, run, spec, cell, plugin = self._context(attempt_id)
        self.recovery.resolve(plugin.plugin_id, plugin.resume_level)
        required = {"step", "data_position", "method_state", "rng_state"}
        if not required <= state.keys() or not isinstance(state["step"], int) or state["step"] < 0:
            raise ResearchError("CONTRACT_MISMATCH", "checkpoint needs method, position and actual RNG state")
        if plugin.resume_level == "chunk" and state.get("chunk_complete") is not True:
            raise ResearchError("CONTRACT_MISMATCH", "chunk checkpoint must be at a completed boundary")
        if "budget" in state or "remaining_seconds" in state:
            raise ResearchError("CONTRACT_MISMATCH", "checkpoint cannot restore budget")
        value = {"schema_version": "pirc25-checkpoint-v1", "parent_attempt_id": attempt_id,
                 "run_id": run["run_id"], "plugin_id": plugin.plugin_id,
                 "resume_level": plugin.resume_level, "bindings": bindings(spec, cell),
                 "payload_hash": digest(state), "state": state}
        from infrastructure.research_store import encode
        from infrastructure.research_visibility import study_visibility
        artifact = self.store.artifact(encode(value), role="checkpoint", visibility=study_visibility(self.store.manifest, spec),
                                      block_ids=[cell["block_id"]], study_id=run["study_id"])
        self.store.append("CHECKPOINT", {"attempt_id": attempt_id, "artifact_id": artifact["artifact_id"],
                                        "resume_level": plugin.resume_level, "bindings_hash": digest(value["bindings"])})
        return artifact["artifact_id"]

    def prepare(self, attempt_id, checkpoint_id, *, authorization):
        attempt, run, spec, cell, plugin = self._context(attempt_id)
        adapter = self.recovery.resolve(plugin.plugin_id, plugin.resume_level)
        if attempt["state"] not in {"FAILED", "INTERRUPTED"}:
            raise ResearchError("CONTRACT_MISMATCH", "only a stopped failed attempt may resume")
        if BudgetLedger(self.store).balance(run["arm_id"])["closed"]:
            raise ResearchError("BUDGET_EXHAUSTED", "checkpoint cannot reopen a fused arm")
        content = self.store.read_artifact(checkpoint_id, purpose="resume", authorization=authorization)
        value = json.loads(content)
        if (value.get("schema_version") != "pirc25-checkpoint-v1"
                or value.get("parent_attempt_id") != attempt_id or value.get("run_id") != run["run_id"]
                or value.get("plugin_id") != plugin.plugin_id or value.get("resume_level") != plugin.resume_level
                or value.get("bindings") != bindings(spec, cell) or digest(value.get("state")) != value.get("payload_hash")):
            raise ResearchError("CONTRACT_MISMATCH", "checkpoint schema, identity or code/data/method hashes differ")
        return {"run": run, "spec": spec, "cell": cell, "plugin": plugin,
                "adapter": adapter, "state": value["state"], "checkpoint_id": checkpoint_id}

    def resume(self, attempt_id, checkpoint_id, *, authorization, budget=BudgetSpec()):
        prepared = self.prepare(attempt_id, checkpoint_id, authorization=authorization)
        retry = self.store.new_attempt(prepared["run"]["run_id"], parent_attempt_id=attempt_id,
                                       reason="resume from verified " + checkpoint_id)
        self.store.append("RESUME", {"attempt_id": retry, "parent_attempt_id": attempt_id,
                                    "checkpoint_id": checkpoint_id, "resume_level": prepared["plugin"].resume_level})
        return AdmissionGate(self.store).run(retry, prepared["spec"], prepared["cell"], prepared["plugin"],
            lambda output: prepared["adapter"].command_builder(output, prepared["spec"], prepared["cell"], prepared["state"]),
            budget, recovery_builder=prepared["adapter"].command_builder)
