"""Read-only native memory observations for the current numerical process.

No tensor engine, data/store access, limit changes or process enumeration.
These are lifetime peaks through observation, not isolated phase deltas,
whole-tree peaks, previous-attempt peaks or an OS memory sandbox.
"""

import math
import os
import sys
import time


SCHEMA = "pirc26-process-resource-observation-v1"
POINTS = frozenset({"observer-call", "fit-return-before-encoding", "basis-return-before-encoding",
    "job-payload-before-encoding", "checkpoint-save-before-ack", "worker-exit-after-output-or-ack"})


def _windows_counters():
    # Native ABI: SIZE_T is pointer-sized; DWORD remains 32 bits on Win64.
    # https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters_ex
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
            "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
            "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.argtypes, kernel.GetCurrentProcess.restype = [], wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    value = Counters()
    value.cb = ctypes.sizeof(value)
    # Pseudo-handle for self: no OpenProcess, inherited handle or CloseHandle.
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    commit_available = value.PeakPagefileUsage > 0 and value.PrivateUsage > 0
    return {"resident_counter": "windows.PROCESS_MEMORY_COUNTERS_EX.PeakWorkingSetSize",
            "peak_resident_bytes": int(value.PeakWorkingSetSize), "current_resident_bytes": int(value.WorkingSetSize),
            "commit_counter": "windows.PROCESS_MEMORY_COUNTERS_EX.PeakPagefileUsage",
            "peak_private_commit_bytes": int(value.PeakPagefileUsage) if commit_available else None,
            "current_private_commit_bytes": int(value.PrivateUsage) if commit_available else None,
            "commit_reason": None if commit_available else "NATIVE_COMMIT_COUNTER_UNAVAILABLE"}


def _linux_counters():
    # Linux ru_maxrss is KiB for RUSAGE_SELF, not a process-tree measurement.
    # https://man7.org/linux/man-pages/man2/getrusage.2.html
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if type(value) is not int or not 0 < value < (1 << 63) // 1024:
        raise ValueError("invalid native RSS counter")
    return {"resident_counter": "linux.getrusage.RUSAGE_SELF.ru_maxrss",
            "peak_resident_bytes": value * 1024, "current_resident_bytes": None,
            "commit_counter": None, "peak_private_commit_bytes": None, "current_private_commit_bytes": None,
            "commit_reason": "not provided by getrusage.RUSAGE_SELF"}


def process_resources(point="observer-call"):
    """Fresh native facts, or explicit unavailable values; never a zero fallback.

    Supported counter semantics are Windows and Linux. Unknown platforms or
    failed queries do not fabricate comparable measurements/qualification.
    The CPU-only component never initializes CUDA to collect telemetry.
    """
    if type(point) is not str or point not in POINTS:
        raise ValueError("unsupported resource observation point")
    started = time.monotonic()
    result = {"schema_version": SCHEMA, "status": "UNAVAILABLE", "point": point,
        "process_id": os.getpid(), "parent_process_id": os.getppid(), "platform": sys.platform,
        "scope": "current-process-lifetime-through-observation", "units": "bytes",
        "includes_children": False, "includes_previous_attempts": False, "is_phase_isolated": False,
        "observation_started_monotonic_seconds": started, "observed_monotonic_seconds": started, "resident_counter": None,
        "peak_resident_bytes": None, "current_resident_bytes": None, "commit_counter": None,
        "peak_private_commit_bytes": None, "current_private_commit_bytes": None, "commit_reason": None,
        "reason": "UNSUPPORTED_PLATFORM", "gpu_status": "NOT_APPLICABLE",
        "peak_gpu_memory_bytes": None, "gpu_reason": "CPU-only registered numerical component"}
    if not math.isfinite(result["observed_monotonic_seconds"]):
        raise ValueError("invalid resource observation clock")
    if sys.platform != "win32" and not sys.platform.startswith("linux"):
        return result
    try:
        counters = _windows_counters() if sys.platform == "win32" else _linux_counters()
        peak = counters["peak_resident_bytes"]
        if type(peak) is not int or not 0 < peak < (1 << 63):
            raise ValueError("invalid native resident peak")
        for name, peak_name in (("current_resident_bytes", "peak_resident_bytes"),
                                ("current_private_commit_bytes", "peak_private_commit_bytes")):
            current, maximum = counters[name], counters[peak_name]
            if maximum is not None and (type(maximum) is not int or not 0 <= maximum < (1 << 63)):
                raise ValueError("invalid native memory peak")
            if current is not None and (type(current) is not int or type(maximum) is not int
                    or not 0 <= current <= maximum < (1 << 63)):
                raise ValueError("inconsistent native memory counters")
        result.update(counters, status="MEASURED", reason=None)
    except (OSError, ValueError, TypeError, OverflowError, AttributeError, ImportError, KeyError):
        # Do not leak native messages or let unavailable telemetry grant/deny
        # data authority, reset budgets or masquerade as a zero-memory result.
        result["reason"] = "NATIVE_COUNTER_UNAVAILABLE"
    result["observed_monotonic_seconds"] = time.monotonic()
    if (not math.isfinite(result["observed_monotonic_seconds"])
            or result["observed_monotonic_seconds"] < started):
        raise ValueError("invalid resource observation interval")
    return result
