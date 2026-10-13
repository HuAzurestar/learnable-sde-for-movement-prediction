"""Capability-selected command adapters always execute through the supervisor."""

from __future__ import annotations

from application.research_budget import BudgetSpec
from application.research_contracts import CapabilityRegistry
from application.research_admission import AdmissionGate
from application.research_execution import resolve_execution
from infrastructure.research_store import ResearchStore, ResearchError, digest


def affine_registry(store: ResearchStore):
    from .plugins import affine_plugin
    registry = CapabilityRegistry()
    for dimensions in (1, 4):
        registry.register(affine_plugin(dimensions))
    return registry


class SharedRunner:
    def __init__(self, store: ResearchStore, registry=None, recovery_registry=None):
        self.store = store
        self.builtin_fixture = registry is None
        self.registry = registry if registry is not None else affine_registry(store)
        self.recovery_registry = recovery_registry

    def run_cell(self, study_id, cell_hash, *, budget=BudgetSpec(60, category="smoke"), parent_attempt_id=None,
                 reason=None, restart_adjudication=None):
        spec = self.store.manifest("study-" + study_id)["spec"]
        cells = [cell for cell in spec["cells"] if digest(cell) == cell_hash]
        if len(cells) != 1:
            raise ResearchError("CONTRACT_MISMATCH", "cell not registered")
        cell = cells[0]
        plugin = resolve_execution(self.registry, cell)
        checkpoint_handler = None
        if self.recovery_registry is not None and plugin.resume_level != "restart-only":
            from application.research_recovery import SharedRecovery
            self.recovery_registry.resolve(plugin.plugin_id, plugin.resume_level, plugin.registry_entry.version)
            recovery = SharedRecovery(self.store, self.registry, self.recovery_registry)
        run_id = self.store.register_run(study_id, cell)
        for attempt in self.store.attempts().values():
            if attempt["run_id"] == run_id and attempt["state"] == "SUCCEEDED":
                from application.research_reuse import verified_reuse
                return verified_reuse(self.store, attempt, spec, cell, plugin)
        if restart_adjudication is None:
            attempt = self.store.new_attempt(run_id, parent_attempt_id=parent_attempt_id, reason=reason)
        else:
            from application.research_budget import ResourceBudgetSpec
            from application.research_restart_adjudication import resolve_retry_adjudication
            if type(budget) is not ResourceBudgetSpec or parent_attempt_id is not None or reason is not None:
                raise ResearchError("CONTRACT_MISMATCH", "adjudicated restart owns its parent and funding")
            receipt = resolve_retry_adjudication(self.store, restart_adjudication,
                execution_policy=budget.execution_policy, source_code_hash=budget.source_code_hash)
            attempt = receipt["retry_attempt_id"]
            registered = self.store.attempts().get(attempt)
            receipt_run = receipt.get("current_run_id", receipt.get("run_id"))
            receipt_cell = receipt.get("current_cell_hash", receipt.get("cell_hash"))
            if (receipt_run != run_id or receipt_cell != cell_hash
                    or registered is None or registered["run_id"] != run_id
                    or registered["state"] != "REGISTERED"):
                raise ResearchError("IDENTITY_CONFLICT", "adjudicated restart target differs")
        if self.recovery_registry is not None and plugin.resume_level != "restart-only":
            checkpoint_handler = recovery.checkpoint_handler(attempt)
        context = {"root": str(self.store.path.parent), "store_id": self.store.store_id}
        return AdmissionGate(self.store).run(attempt, spec, cell, plugin,
            lambda output: plugin.command_builder(output, spec, cell, context) if self.builtin_fixture
                else plugin.command_builder(output, spec, cell), budget, builtin_fixture=self.builtin_fixture,
            checkpoint_handler=checkpoint_handler)
