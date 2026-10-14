"""Declared method recovery; checkpoint state never includes budget balances."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time

from .research_budget import BudgetLedger, BudgetSpec
from .research_admission import AdmissionGate
from .research_contracts import invoke_validator
from .research_execution import resolve_execution, execution_plan
from .research_registry import implementation_hash
from infrastructure.research_store import ResearchError, ResearchStore, digest


@dataclass(frozen=True)
class RecoveryPlugin:
    plugin_id: str
    resume_level: str
    command_builder: object
    version: str


class RecoveryRegistry:
    def __init__(self):
        self.plugins = {}

    def register(self, plugin: RecoveryPlugin):
        from infrastructure.research_store import identifier
        identifier(plugin.plugin_id)
        identifier(plugin.version)
        if (plugin.resume_level not in {"exact", "numerical-tolerance", "chunk"}
                or not callable(plugin.command_builder)):
            raise ResearchError("CONTRACT_MISMATCH", "invalid recovery plugin")
        key = (plugin.plugin_id, plugin.version)
        content = implementation_hash(plugin.command_builder)
        if key in self.plugins and self.plugins[key][1:] != (plugin.resume_level, content):
            raise ResearchError("IDENTITY_CONFLICT", "same recovery version has different immutable content")
        self.plugins[key] = (plugin, plugin.resume_level, content)

    def resolve(self, plugin_id, level, version):
        registered = self.plugins.get((plugin_id, version))
        if level == "restart-only" or registered is None or registered[1] != level:
            raise ResearchError("CONTRACT_MISMATCH", "method does not implement requested recovery level")
        if implementation_hash(registered[0].command_builder) != registered[2]:
            raise ResearchError("IDENTITY_CONFLICT", "registered recovery implementation changed")
        return registered[0]


def bindings(spec, cell):
    arms = [arm for arm in spec["arms"] if arm["arm_id"] == cell["arm_id"]]
    return {"schema_version": spec["schema_version"], "code_hash": spec["code_hash"],
            "data_hash": spec["data_hash"], "protocol_hash": spec["protocol_hash"],
            "feature_hash": spec["feature_hash"], "selection_hash": spec["selection_hash"],
            "model_hash": digest(arms[0]["model_family_id"]),
            "objective_hash": digest(arms[0]["objective_id"]), "cell_hash": digest(cell),
            "execution_binding_hash": digest(cell["execution"])}


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
        plugin = resolve_execution(self.execution, cell)
        execution_plan(spec, cell, plugin)
        return attempt, run, spec, cell, plugin

    def checkpoint_handler(self, attempt_id):
        def save(state, progress, receipt, deadline):
            # Only this short owner save phase holds the writer lock; the
            # waiting worker and its independent hard fuse remain outside it.
            with self.store._checkpoint_publication():
                # Validate the receipt against the same fresh context used to
                # publish, not a second full version/resource context. This is
                # one owned save phase, never reusable admission or authority.
                reference = self._checkpoint(attempt_id, state, admission_hash=receipt["admission_hash"],
                    progress=progress, deadline=deadline, receipt=receipt)
            # The final uncached physical verification is owner I/O too.
            if deadline is not None and time.monotonic() >= deadline:
                raise ResearchError("TIMEOUT", "checkpoint publication reached the hard deadline")
            return reference
        return save

    def checkpoint(self, attempt_id, state: dict, *, admission_hash=None, progress=None, deadline=None):
        with self.store._checkpoint_publication():
            artifact = self._checkpoint(attempt_id, state, admission_hash=admission_hash,
                                        progress=progress, deadline=deadline)
        if deadline is not None and time.monotonic() >= deadline:
            raise ResearchError("TIMEOUT", "checkpoint publication reached the hard deadline")
        return artifact

    def _checkpoint(self, attempt_id, state, *, admission_hash, progress, deadline, receipt=None):
        attempt, run, spec, cell, plugin = self._context(attempt_id)
        if receipt is not None:
            if (attempt["state"] != "RUNNING" or receipt["attempt_id"] != attempt_id
                    or receipt["cell_hash"] != digest(cell) or type(state) is not dict
                    or state.get("step") != progress["completed_steps"]
                    or plugin.checkpoint_validator is None and progress["total_steps"] > receipt["resource_plan"]["counts"]["steps"]):
                raise ResearchError("CONTRACT_MISMATCH", "checkpoint differs from the admitted running job")
            if plugin.checkpoint_validator is not None:
                invoke_validator(plugin, "checkpoint_validator", self.store, receipt, state, progress)
        adapter = self.recovery.resolve(plugin.plugin_id, plugin.resume_level, plugin.registry_entry.version)
        required = {"step", "data_position", "method_state", "rng_state"}
        from .research_registry import _bounded_json
        _bounded_json(state, nodes=65536, depth=32)
        if type(state) is not dict or not required <= state.keys() or type(state["step"]) is not int or state["step"] < 0:
            raise ResearchError("CONTRACT_MISMATCH", "checkpoint needs method, position and actual RNG state")
        if plugin.resume_level == "chunk" and state.get("chunk_complete") is not True:
            raise ResearchError("CONTRACT_MISMATCH", "chunk checkpoint must be at a completed boundary")
        if "budget" in state or "remaining_seconds" in state:
            raise ResearchError("CONTRACT_MISMATCH", "checkpoint cannot restore budget")
        value = {"schema_version": "pirc25-checkpoint-v1", "parent_attempt_id": attempt_id,
                 "run_id": run["run_id"], "plugin_id": plugin.plugin_id,
                 "plugin_version": plugin.registry_entry.version, "recovery_command_hash": implementation_hash(adapter.command_builder),
                 "resume_level": plugin.resume_level, "bindings": bindings(spec, cell),
                 "payload_hash": digest(state), "state": state}
        if admission_hash is not None:
            value.update(admission_hash=admission_hash, progress=progress)
        from infrastructure.research_store import encode
        from infrastructure.research_visibility import study_visibility
        if deadline is not None and time.monotonic() >= deadline:
            raise ResearchError("TIMEOUT", "checkpoint publication reached the hard deadline")
        artifact = self.store.artifact(encode(value), role="checkpoint", visibility=study_visibility(self.store.manifest, spec),
                                      block_ids=[cell["block_id"]], study_id=run["study_id"])
        if deadline is not None and time.monotonic() >= deadline:
            raise ResearchError("TIMEOUT", "checkpoint publication reached the hard deadline")
        self.store.append("CHECKPOINT", {"attempt_id": attempt_id, "artifact_id": artifact["artifact_id"],
                                        "resume_level": plugin.resume_level, "bindings_hash": digest(value["bindings"])})
        if receipt is not None:
            return {"artifact_id": artifact["artifact_id"], "resume_level": plugin.resume_level}
        return artifact["artifact_id"]

    def prepare(self, attempt_id, checkpoint_id, *, authorization):
        context = self._context(attempt_id)
        if not context[-1].validator_store_context:
            return self._prepare(attempt_id, checkpoint_id, authorization=authorization, context=context)
        # Opt-in source validators register guards on this SAME owned scope.
        # The state cannot leave after a later checkpoint I/O outlives a raw
        # source/model grant. Legacy callbacks retain their original scope.
        with self.store._read_transaction():
            return self._prepare(attempt_id, checkpoint_id, authorization=authorization,
                context=self._context(attempt_id))

    def _prepare(self, attempt_id, checkpoint_id, *, authorization, context):
        attempt, run, spec, cell, plugin = context
        adapter = self.recovery.resolve(plugin.plugin_id, plugin.resume_level, plugin.registry_entry.version)
        if attempt["state"] not in {"FAILED", "INTERRUPTED"}:
            raise ResearchError("CONTRACT_MISMATCH", "only a stopped failed attempt may resume")
        if BudgetLedger(self.store).balance(run["arm_id"])["closed"]:
            raise ResearchError("BUDGET_EXHAUSTED", "checkpoint cannot reopen a fused arm")
        content = self.store.read_artifact(checkpoint_id, purpose="resume", authorization=authorization)
        try:
            value = json.loads(content)
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise ResearchError("CONTRACT_MISMATCH", "checkpoint is invalid JSON") from exc
        if (type(value) is not dict or value.get("schema_version") != "pirc25-checkpoint-v1"
                or value.get("parent_attempt_id") != attempt_id or value.get("run_id") != run["run_id"]
                or value.get("plugin_id") != plugin.plugin_id or value.get("resume_level") != plugin.resume_level
                or value.get("plugin_version") != plugin.registry_entry.version
                or value.get("recovery_command_hash") != implementation_hash(adapter.command_builder)
                or value.get("bindings") != bindings(spec, cell) or digest(value.get("state")) != value.get("payload_hash")):
            raise ResearchError("CONTRACT_MISMATCH", "checkpoint schema, identity or code/data/method hashes differ")
        metadata = self.store.manifest("artifact-" + checkpoint_id)
        saved = {"attempt_id": attempt_id, "artifact_id": checkpoint_id,
                 "resume_level": plugin.resume_level, "bindings_hash": digest(value["bindings"])}
        if (metadata.get("role") != "checkpoint" or metadata.get("study_id") != run["study_id"]
                or metadata.get("block_ids") != [cell["block_id"]]
                or not any(event["event_kind"] == "CHECKPOINT" and event["payload"] == saved
                           for event in self.store.events())):
            raise ResearchError("CONTRACT_MISMATCH", "checkpoint lacks authoritative source save and scope")
        if plugin.checkpoint_validator is not None:
            reference = value.get("admission_hash")
            if not isinstance(reference, str) or not isinstance(value.get("progress"), dict):
                raise ResearchError("CONTRACT_MISMATCH", "owned phase recovery requires its original admission and progress")
            receipt = self.store.manifest("admission-" + reference)
            if (receipt.get("admission_hash") != reference
                    or digest({k:v for k,v in receipt.items() if k != "admission_hash"}) != reference
                    or receipt.get("attempt_id") != attempt_id or receipt.get("spec_hash") != digest(spec)
                    or receipt.get("cell_hash") != digest(cell)):
                raise ResearchError("CONTRACT_MISMATCH", "checkpoint original admission differs")
            # Before retry creation/admission/provider reads, not only after a
            # worker has consumed the data. Validation grants no new authority.
            invoke_validator(plugin, "checkpoint_validator", self.store, receipt, value["state"], value["progress"])
        self.store.verify_artifact_read(checkpoint_id, purpose="resume", authorization=authorization)
        return {"run": run, "spec": spec, "cell": cell, "plugin": plugin,
                "adapter": adapter, "state": value["state"], "checkpoint_id": checkpoint_id}

    def resume(self, attempt_id, checkpoint_id, *, authorization, budget=BudgetSpec()):
        prepared = self.prepare(attempt_id, checkpoint_id, authorization=authorization)
        retry = self.store.new_attempt(prepared["run"]["run_id"], parent_attempt_id=attempt_id,
                                       reason="resume from verified " + checkpoint_id)
        self.store.append("RESUME", {"attempt_id": retry, "parent_attempt_id": attempt_id,
                                    "checkpoint_id": checkpoint_id, "resume_level": prepared["plugin"].resume_level})

        def command(output):
            # prepare's grant cannot authorize handing state to a plugin after
            # retry/resume/admission I/O. The distinct input grant is not a
            # substitute for current permission to read this checkpoint state.
            # A denial here follows the supervisor's normal failed-preflight
            # settlement, before the builder writes state or launches a worker.
            self.store.verify_artifact_read(checkpoint_id, purpose="resume", authorization=authorization)
            return prepared["adapter"].command_builder(output, prepared["spec"], prepared["cell"], prepared["state"])

        return AdmissionGate(self.store).run(retry, prepared["spec"], prepared["cell"], prepared["plugin"],
            command,
            budget, recovery_builder=prepared["adapter"].command_builder,
            checkpoint_handler=self.checkpoint_handler(retry))
