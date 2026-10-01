"""Contain a worker tree; Windows uses kill-on-close job objects.

API reference: https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects
"""

from __future__ import annotations

import os
import signal


def process_may_be_alive(pid):
    """Read-only conservative probe; errors never count as proof of exit."""
    if type(pid) is not int or pid <= 1:
        return True
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() != 87  # nonexistent PID, not access denied
        try:
            code = wintypes.DWORD()
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        pass
    except (PermissionError, OSError):
        return True
    # The POSIX wrapper leads a process group; its descendants may outlive it.
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True


class ProcessTree:
    def __init__(self, process):
        self.process = process
        self.job = None
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class Basic(ctypes.Structure):
                _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                            ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                            ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                            ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                            ("SchedulingClass", wintypes.DWORD)]

            class IO(ctypes.Structure):
                _fields_ = [(name, ctypes.c_uint64) for name in (
                    "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                    "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

            class Extended(ctypes.Structure):
                _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", IO),
                            ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                            ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

            self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
            self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
            self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
            self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            self.kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
            self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            self.job = self.kernel.CreateJobObjectW(None, None)
            limits = Extended()
            limits.BasicLimitInformation.LimitFlags = 0x2000
            if (not self.job or not self.kernel.SetInformationJobObject(self.job, 9, ctypes.byref(limits), ctypes.sizeof(limits))
                    or not self.kernel.AssignProcessToJobObject(self.job, wintypes.HANDLE(int(process._handle)))):
                error = ctypes.get_last_error()
                self.close()
                process.kill()
                process.wait()
                raise OSError(error, "cannot contain worker process tree")

    def terminate(self):
        if os.name == "nt":
            if self.job:
                self.kernel.TerminateJobObject(self.job, 124)
        else:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def close(self):
        if self.job:
            self.kernel.CloseHandle(self.job)
            self.job = None
