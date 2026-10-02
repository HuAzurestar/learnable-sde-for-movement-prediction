"""Capability-selected command adapters always execute through the supervisor."""

from __future__ import annotations

import sys

from application.research_budget import BudgetSpec
from application.research_contracts import CapabilityRegistry, ExecutionPlugin
from application.research_admission import AdmissionGate
from application.research_execution import resolve_execution, execution_plan
from infrastructure.research_store import ResearchStore, ResearchError, digest


def affine_registry(store: ResearchStore):
    from .plugins import affine_plugin
    registry = CapabilityRegistry()
    for dimensions in (1, 4):
        registry.register(affine_plugin(dimensions))
    return registry


class SharedRunner:
    def __init__(self, store: ResearchStore, registry=None):
        self.store = store
        self.builtin_fixture = registry is None
        self.registry = registry if registry is not None else affine_registry(store)

    def run_cell(self, study_id, cell_hash, *, budget=BudgetSpec(60, category="smoke"), parent_attempt_id=None, reason=None):
        spec = self.store.manifest("study-" + study_id)["spec"]
        cells = [cell for cell in spec["cells"] if digest(cell) == cell_hash]
        if len(cells) != 1:
            raise ResearchError("CONTRACT_MISMATCH", "cell not registered")
        cell = cells[0]
        plugin = resolve_execution(self.registry, cell)
        run_id = self.store.register_run(study_id, cell)
        for attempt in self.store.attempts().values():
            if attempt["run_id"] == run_id and attempt["state"] == "SUCCEEDED":
                from application.research_reuse import verified_reuse
                return verified_reuse(self.store, attempt, spec, cell, plugin)
        attempt = self.store.new_attempt(run_id, parent_attempt_id=parent_attempt_id, reason=reason)
        context = {"root": str(self.store.path.parent), "store_id": self.store.store_id}
        return AdmissionGate(self.store).run(attempt, spec, cell, plugin,
            lambda output: plugin.command_builder(output, spec, cell, context) if self.builtin_fixture
                else plugin.command_builder(output, spec, cell), budget, builtin_fixture=self.builtin_fixture)
