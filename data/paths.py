"""Resolve local-only paths without embedding workstation details.

Environment variables named ``LEARNABLE_SDE_<NAME>`` take precedence. Safe
defaults live below the repository's ignored ``.local`` directory.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_PREFIX = "LEARNABLE_SDE"
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_ROOT = _PROJECT_ROOT / ".local"

_DEFAULTS = {
    "data_root": _LOCAL_ROOT / "data",
    "cond_root": _LOCAL_ROOT / "conditions",
    "checkpoint": _LOCAL_ROOT / "checkpoints",
    "output_root": _LOCAL_ROOT / "outputs",
}


def resolve(name: str) -> Path:
    """Resolve a named path from the environment or a safe local default."""
    env = os.environ.get(f"{_ENV_PREFIX}_{name.upper()}")
    if env:
        return Path(env)
    if name not in _DEFAULTS:
        raise KeyError(f"未知路径名 {name!r}（可用: {sorted(_DEFAULTS)}）")
    return _DEFAULTS[name]
