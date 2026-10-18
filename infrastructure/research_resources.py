"""Fresh stdlib-only system and contained-worker resource observations."""

from __future__ import annotations

import os
from pathlib import Path
import shutil

from infrastructure.research_store import ResearchError


def available_memory_bytes():
    try:
        if os.name == "nt":
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            value = MemoryStatus()
            value.dwLength = ctypes.sizeof(value)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
                raise OSError(ctypes.get_last_error(), "cannot sample physical memory")
            return int(value.ullAvailPhys)
        if hasattr(os, "sysconf"):
            return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, TypeError) as exc:
        raise ResearchError("RESOURCE_PLAN_REJECTED", "available memory observation failed") from exc
    raise ResearchError("RESOURCE_PLAN_REJECTED", "available memory observation is unsupported")


def observe_resources(*, system_path, runtime_path, tree=None):
    """Sample independently on every call; errors are never treated as capacity."""
    try:
        system = Path(system_path).resolve(strict=True)
        runtime = Path(runtime_path).resolve(strict=True)
        system_free = int(shutil.disk_usage(system).free)
        runtime_free = int(shutil.disk_usage(runtime).free)
        worker_rss = 0 if tree is None else tree.peak_memory_bytes()
    except (OSError, ValueError, TypeError) as exc:
        raise ResearchError("RESOURCE_PLAN_REJECTED", "disk or worker resource observation failed") from exc
    values = {"system_free_bytes": system_free, "runtime_free_bytes": runtime_free,
              "available_memory_bytes": available_memory_bytes(), "worker_rss_bytes": worker_rss}
    if any(type(value) is not int or value < 0 for value in values.values()):
        raise ResearchError("RESOURCE_PLAN_REJECTED", "resource observation is invalid")
    return values
