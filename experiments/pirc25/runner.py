"""Capability-selected command adapters always execute through the supervisor."""

from __future__ import annotations

import sys

from application.research_budget import BudgetSpec
from application.research_contracts import CapabilityRegistry, ExecutionPlugin
from application.research_admission import AdmissionGate
from infrastructure.research_store import ResearchStore, ResearchError, digest


def affine_registry(store: ResearchStore):
    registry = CapabilityRegistry()
    for dimensions in (1, 4):
        plugin_id = "affine-" + str(dimensions)

        def command(output, spec, cell):
            return [sys.executable, "-m", "experiments.pirc25.worker", str(store.path.parent),
                    store.store_id, spec["study_id"], digest(cell), str(output)]

        registry.register(ExecutionPlugin(plugin_id,
            frozenset({"exact-transition" if dimensions == 1 else "generic-rollout"}),
            ("x", "vx") if dimensions == 1 else ("x", "y", "vx", "vy"),
            ("m", "m/s") if dimensions == 1 else ("m", "m", "m/s", "m/s"), "restart-only", command))
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
        plugin = self.registry.resolve(cell["plugin_id"], cell["capability"])
        run_id = self.store.register_run(study_id, cell)
        for attempt in self.store.attempts().values():
            if attempt["run_id"] == run_id and attempt["state"] == "SUCCEEDED":
                from application.research_reuse import verified_reuse
                return verified_reuse(self.store, attempt, spec, cell, plugin)
        attempt = self.store.new_attempt(run_id, parent_attempt_id=parent_attempt_id, reason=reason)
        return AdmissionGate(self.store).run(attempt, spec, cell, plugin,
            lambda output: plugin.command_builder(output, spec, cell), budget, builtin_fixture=self.builtin_fixture)
