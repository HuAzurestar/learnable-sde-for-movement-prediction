"""Instance-owned component registry used only at the composition root."""

from __future__ import annotations

from typing import Callable, Dict, Generic, TypeVar

from domain import ConfigurationError

ConfigT = TypeVar("ConfigT")
ComponentT = TypeVar("ComponentT")


class ComponentRegistry(Generic[ConfigT, ComponentT]):
    def __init__(self) -> None:
        self._builders: Dict[str, Callable[[ConfigT], ComponentT]] = {}

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
