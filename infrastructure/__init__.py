"""Filesystem and serialization adapters."""

from importlib import import_module

__all__ = [
    "ArtifactWriter",
    "AtomicRunStore",
    "JsonArtifactStore",
    "TorchModelStore",
]

_EXPORTS = {"ArtifactWriter": "runs", "AtomicRunStore": "runs",
            "JsonArtifactStore": "artifacts", "TorchModelStore": "checkpoint"}


def __getattr__(name):
    # Store/journal metadata and statistical workers are independent of torch.
    # A requested legacy adapter still imports its real defining module.
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("." + _EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
