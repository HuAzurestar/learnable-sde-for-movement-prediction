"""Instance-owned component registry used only at the composition root."""

from __future__ import annotations

from typing import Callable, Dict, Generic, TypeVar

from domain import ConfigurationError
from .research_registry import VersionedRegistry, plan_resources
from .research_execution import BINDING_FIELDS
from infrastructure.research_store import ResearchError, encode
import json

ConfigT = TypeVar("ConfigT")
ComponentT = TypeVar("ComponentT")


class ComponentRegistry(Generic[ConfigT, ComponentT]):
    def __init__(self) -> None:
        self._builders: Dict[str, Callable[[ConfigT], ComponentT]] = {}
        self._versions = VersionedRegistry()

    def register_version(self, entry, builder) -> str:
        """Register an explicit immutable implementation without constructing it.

        Versioned factories take bounded JSON config/inputs and explicit runtime
        context. Legacy factories remain separate; resolution never falls back.
        """
        return self._versions.register(entry, builder)

    def plan_bound(self, binding, *, matrix_cells, **compatibility):
        if (type(binding) is not dict or set(binding) != BINDING_FIELDS or
                binding.get("schema_version") != "pirc25-execution-binding-v1"):
            raise ResearchError("CONTRACT_MISMATCH", "explicit sealed component binding required")
        registration = self._versions.resolve(binding["component_id"], binding["component_version"],
            entry_hash=binding["registry_entry_hash"], **compatibility)
        entry = registration.entry
        if binding["resource_class"] != entry.resource_class:
            raise ResearchError("CONTRACT_MISMATCH", "component resource class differs")
        plan = plan_resources(entry, binding["config"], binding["inputs"], matrix_cells=matrix_cells)
        if binding["resource_plan_hash"] != plan["resource_plan_hash"]:
            raise ResearchError("CONTRACT_MISMATCH", "component plan differs from frozen inputs")
        return plan, registration

    def create_bound(self, binding, *, matrix_cells, context=None, **compatibility):
        """Construct only after exact resolution and allocation preflight.

        This constructs the existing component, not its numerical result; the
        component's typed model/fit/forecast interfaces validate those results.
        It must be invoked inside a managed worker for budgeted research jobs.
        """
        # First establish bounded JSON; then detach and revalidate exactly the
        # content used by the constructor. A caller mutation between planning
        # and snapshotting cannot smuggle unplanned configuration into it.
        self.plan_bound(binding, matrix_cells=matrix_cells, **compatibility)
        sealed = json.loads(encode(binding))
        _, registration = self.plan_bound(sealed, matrix_cells=matrix_cells, **compatibility)
        return registration.builder(sealed["config"], sealed["inputs"], context)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders))

    def register(self, name: str, builder: Callable[[ConfigT], ComponentT]) -> None:
        if not name or name in self._builders:
            raise ConfigurationError(f"组件名为空或重复: {name!r}")
        self._builders[name] = builder

    def create(self, name: str, config: ConfigT) -> ComponentT:
        try:
            return self._builders[name](config)
        except KeyError as exc:
            raise ConfigurationError(
                f"未知组件 {name!r}；可用组件: {self.names}"
            ) from exc
