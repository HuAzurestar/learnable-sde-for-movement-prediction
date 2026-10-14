"""Fresh Windows available-physical-memory observation, without broad counters.

This module is deliberately separate from live/registered forecast entry points.
It changes no RAM threshold or polling cadence and does not cache measurements.

Microsoft documents both MEMORYSTATUSEX.ullAvailPhys (bytes) and
PERFORMANCE_INFORMATION.PhysicalAvailable (pages) as immediately reusable
physical memory. psutil 5.9.5 reads the latter with GetPerformanceInfo; this
reader requests the former without also retrieving process/commit counters.
Sequential readings are volatile, not an atomic equivalence oracle.

Sources:
https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-globalmemorystatusex
https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/ns-sysinfoapi-memorystatusex
https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-performance_information
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import hashlib
from pathlib import Path

VERSION = "pirc17-fresh-physical-memory-observer-v1"


class _MemoryStatusEx(ctypes.Structure):
    # Explicit-width integers also make software ABI tests portable. c_ulong
    # is not a portable spelling of Windows DWORD on non-Windows hosts.
    _fields_ = [
        ("dwLength", ctypes.c_uint32),
        ("dwMemoryLoad", ctypes.c_uint32),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


@lru_cache(maxsize=1)
def _windows_query():
    """Cache only the immutable DLL function binding, never its returned data."""
    loader = getattr(ctypes, "WinDLL", None)
    if loader is None:
        raise OSError("Windows physical-memory observer is unavailable on this platform")
    api = loader("kernel32.dll", use_last_error=True).GlobalMemoryStatusEx
    api.argtypes = [ctypes.POINTER(_MemoryStatusEx)]
    api.restype = ctypes.c_int32
    return api


def available_physical_bytes(*, query=None):
    """One fresh OS read per invocation; API/ABI failures never imply free RAM.

    The optional callable is an explicit software-test seam. Production callers
    omit it. The caller owns the resource threshold and stop policy.
    """
    api = _windows_query() if query is None else query
    state = _MemoryStatusEx()
    state.dwLength = ctypes.sizeof(state)
    if not api(ctypes.byref(state)):
        error = getattr(ctypes, "get_last_error", lambda: 0)()
        raise OSError(error, "GlobalMemoryStatusEx failed; available RAM is unknown")
    if (state.dwLength != ctypes.sizeof(state) or state.dwMemoryLoad > 100
            or state.ullTotalPhys == 0 or state.ullAvailPhys > state.ullTotalPhys):
        raise OSError("GlobalMemoryStatusEx returned invalid physical-memory fields")
    # Zero is a valid observation and must reach the caller's low-memory guard.
    return int(state.ullAvailPhys)


def identity():
    """Metadata for a future explicitly versioned consumer, not adoption itself."""
    return {"version": VERSION, "platform": "Windows", "api": "kernel32.GlobalMemoryStatusEx",
            "field": "MEMORYSTATUSEX.ullAvailPhys", "unit": "bytes",
            "semantics": "currently immediately reusable physical memory across NUMA nodes",
            "measurement_cache": False, "one_native_query_per_call": True,
            "failure_policy": "raise; never synthesize availability or reuse a prior reading",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

