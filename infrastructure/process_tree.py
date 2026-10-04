"""Contain a worker tree; Windows uses kill-on-close job objects.

API reference: https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects
"""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path


def _linux_proc_visible():
    """A filtered procfs cannot establish that every group member has exited.

    https://docs.kernel.org/filesystems/proc.html#mount-options
    Restricted/unknown configurations conservatively retain reservations.
    """
    try:
        with Path("/proc/self/mounts").open(encoding="utf-8") as stream:
            mounts = stream.read(1_048_577)
        if len(mounts) > 1_048_576:
            raise OSError("process mount observation quota exceeded")
        records = [line.split() for line in mounts.splitlines()]
        records = [fields for fields in records if len(fields) >= 4 and fields[1] == "/proc"]
        if len(records) != 1 or records[0][2] != "proc":
            raise OSError("cannot establish process observation mount")
        for option in records[0][3].split(","):
            if option.startswith("hidepid=") and option not in {"hidepid=0", "hidepid=off"}:
                raise OSError("filtered process observation mount")
            if option.startswith("pidns="):
                raise OSError("alternate process observation namespace")
    except UnicodeError as exc:
        raise OSError("cannot parse process observation mount") from exc


def _linux_group_active(pid, *, include_pid=False):
    """Observe executing group members; unreadable records are not exit proof.

    Recovery also considers the recorded PID itself, conservatively retaining
    a hold if that identity is now executing outside the expected group.
    """
    _linux_proc_visible()
    observed_group, observed_pid = False, False
    for count, path in enumerate(Path("/proc").iterdir()):
        if count > 100_000:
            raise OSError("process observation quota exceeded")
        if not path.name.isdecimal():
            continue
        try:
            # stat: pid (comm) state ppid pgrp ...; comm may contain ')'.
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            group = int(fields[2])
            if len(fields[0]) != 1 or group < 0:
                raise ValueError("invalid process state/group")
            matches = group == pid or (include_pid and int(path.name) == pid)
            observed_group = observed_group or group == pid
            observed_pid = observed_pid or int(path.name) == pid
            if matches and fields[0] not in {"Z", "X"}:
                return True
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (ValueError, IndexError) as exc:
            raise OSError("cannot parse process group observation") from exc
    _linux_proc_visible()  # Fresh restriction check before using exit evidence.
    for observed, probe in ((observed_group, "killpg"),
                            (observed_pid or not include_pid, "kill")):
        if not observed:
            try:
                getattr(os, probe)(pid, 0)
            except ProcessLookupError:
                continue
            raise OSError("kernel process exists without an observable record")
    return False


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
    if sys.platform.startswith("linux") and Path("/proc").is_dir():
        try:
            return _linux_group_active(pid, include_pid=True)
        except OSError:
            return True  # Unknown group observations keep recovery reserved.
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
    @classmethod
    def contain_current_process(cls):
        """Windows wrapper owns a nested job before spawning descendants.

        Its native deadline stop does not launch another external process.
        The kill-on-close handle dies with the wrapper, including on exit 0.
        """
        if os.name != "nt":
            raise OSError("current-process job containment requires Windows")
        return cls(None)

    def __init__(self, process):
        if process is None and os.name != "nt":
            raise OSError("current-process job containment requires Windows")
        self.process = process
        self.job = None
        self._termination_sent = False
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
            self.kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                                            wintypes.DWORD, ctypes.c_void_p]
            self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            self.job = self.kernel.CreateJobObjectW(None, None)
            limits = Extended()
            limits.BasicLimitInformation.LimitFlags = 0x2000
            if (not self.job or not self.kernel.SetInformationJobObject(self.job, 9, ctypes.byref(limits), ctypes.sizeof(limits))
                    or not self.kernel.AssignProcessToJobObject(self.job,
                        wintypes.HANDLE(int(process._handle) if process is not None else -1))):
                error = ctypes.get_last_error()
                self.close()
                if process is not None:
                    process.kill()
                    process.wait()
                raise OSError(error, "cannot contain worker process tree")

    def terminate(self):
        if self._termination_sent:
            return
        # Claim the one-shot operation before a native call can release the GIL.
        # A numeric POSIX group may be reused after its last member is reaped.
        self._termination_sent = True
        try:
            if os.name == "nt":
                if self.job and not self.kernel.TerminateJobObject(self.job, 124):
                    import ctypes
                    raise OSError(ctypes.get_last_error(), "cannot terminate contained process tree")
            else:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        except OSError:
            if os.name == "nt":
                self._termination_sent = False  # This immutable job handle can be retried safely.
            raise

    def active(self):
        """Confirm the entire group/job, not just its already-dead leader.

        Linux zombies are terminated, not executing descendants. Unknown OS
        observations are errors: callers must retain the reservation.
        """
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            class Accounting(ctypes.Structure):
                _fields_ = [(name, ctypes.c_int64) for name in (
                    "TotalUserTime", "TotalKernelTime", "ThisPeriodTotalUserTime", "ThisPeriodTotalKernelTime")]
                _fields_ += [(name, wintypes.DWORD) for name in (
                    "TotalPageFaultCount", "TotalProcesses", "ActiveProcesses", "TotalTerminatedProcesses")]
            data = Accounting()
            if not self.job or not self.kernel.QueryInformationJobObject(
                    self.job, 1, ctypes.byref(data), ctypes.sizeof(data), None):
                raise OSError(ctypes.get_last_error(), "cannot confirm whole-job stop")
            return data.ActiveProcesses != 0
        if sys.platform.startswith("linux") and Path("/proc").is_dir():
            return _linux_group_active(self.process.pid)
        try:
            os.killpg(self.process.pid, 0)
            return True
        except ProcessLookupError:
            return False

    def wait_stopped(self, timeout=1):
        """Bounded observation after kill, never an extra computation grace."""
        until = time.monotonic() + timeout
        while True:
            if self.process is not None:
                self.process.poll()
            if not self.active():
                return True
            if time.monotonic() >= until:
                return False
            time.sleep(0.005)

    def close(self):
        if self.job:
            self.kernel.CloseHandle(self.job)
            self.job = None
