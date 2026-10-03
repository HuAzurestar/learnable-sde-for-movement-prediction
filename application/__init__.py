"""Application orchestration and the composition root."""

from importlib import import_module

__all__ = [
    "EvaluationPipeline",
    "EvidenceConditioner",
    "RandomStreams",
    "RunContext",
]

_EXPORTS = {"EvaluationPipeline": "pipelines", "EvidenceConditioner": "pipelines",
            "RandomStreams": "runtime", "RunContext": "runtime"}


def __getattr__(name):
    # Importing a stdlib research submodule must not initialize the unrelated
    # numerical execution chain. Public classes remain the original objects.
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("." + _EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
