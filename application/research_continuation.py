"""Explicit owner-approved addenda; no latest fallback or erased history.

Approval is an operator-controlled human-decision record, not cryptographic
proof of an off-platform issuer. Every reference is content/hash bound.
"""
from copy import deepcopy
from datetime import datetime, timezone
import os

from infrastructure.research_store import ResearchError, digest


def require(condition, message, code="CONTRACT_MISMATCH"):
    if not condition:
        raise ResearchError(code, message)


def approval(store, reference, *, live=False):
    require(type(reference) is dict and set(reference) == {"manifest_id", "sha256"},
            "exact approval reference required")
    value = store._manifest(reference["manifest_id"])
    require(digest(value) == reference["sha256"]
            and value.get("schema_version") == "pirc25-continuation-approval-v1"
            and value.get("decision") == "approved"
            and value.get("arm_seconds") == 86400
            and value.get("budget_mode") == "cumulative-only-hard"
            and type(value.get("arm_ids")) is list and value["arm_ids"]
            and value.get("test_authorization") is False
            and value.get("public_export") is False,
            "registered continuation approval differs")
    try:
        deadline_policy = value.get("work_deadline_policy", "hard-stop")
        require(deadline_policy in {"hard-stop", "report-only"}, "unknown work deadline policy")
        start = datetime.fromisoformat(value["work_started_at"])
        deadline = datetime.fromisoformat(value["work_deadline"])
        expiry = datetime.fromisoformat(value["authorization_expires_at"])
        require(start.tzinfo is not None and deadline.tzinfo is not None
                and expiry.tzinfo is not None and 0 < (deadline - start).total_seconds() <= 86400
                and expiry >= deadline, "bounded work/authorization window required")
        if live:
            execution_expiry = expiry if deadline_policy == "report-only" else deadline
            require(start <= datetime.now(timezone.utc) < execution_expiry,
                    "continuation execution authorization expired", "UNAUTHORIZED_DATA")
    except ResearchError:
        raise
    except (ValueError, KeyError, TypeError) as exc:
        raise ResearchError("CONTRACT_MISMATCH", "malformed continuation dates") from exc
    return deepcopy(value)


def renew_grant(store, authorization_id, *, old_version, new_version, approval_ref):
    """Append one exact version and a scope-preserving renewal receipt."""
    with store._read_transaction():
        decision = approval(store, approval_ref, live=True)
        old = store.authorization(authorization_id, version=old_version)
        require(authorization_id in decision.get("authorization_ids", [])
                and old_version != new_version and old.get("test_authorization") is False,
                "grant renewal outside approved scope", "UNAUTHORIZED_DATA")
        new = {**deepcopy(old), "version": new_version,
               "expires_at": decision["authorization_expires_at"],
               "evidence_hash": approval_ref["sha256"]}
        require(datetime.fromisoformat(new["expires_at"]) > datetime.fromisoformat(old["expires_at"]),
                "renewal must extend the original expiry")
        receipt = {"schema_version": "pirc25-grant-renewal-v1",
                   "approval": deepcopy(approval_ref), "authorization_id": authorization_id,
                   "old_version": old_version, "old_hash": digest(old),
                   "new_version": new_version, "new_hash": digest(new)}
        reference = {"manifest_id": "grant-renewal-" + digest(receipt), "sha256": digest(receipt)}
        store.authorize(new)
        store.publish(reference["manifest_id"], receipt)
        resolve_grant(store, old, reference)
    return reference


def resolve_grant(store, original, renewal_reference):
    """Explicit renewal only; both frozen versions and unchanged scope checked."""
    require(type(renewal_reference) is dict and set(renewal_reference) == {"manifest_id", "sha256"},
            "exact renewal reference required")
    receipt = store._manifest(renewal_reference["manifest_id"])
    require(digest(receipt) == renewal_reference["sha256"]
            and receipt.get("schema_version") == "pirc25-grant-renewal-v1",
            "renewal receipt differs")
    decision = approval(store, receipt["approval"])
    current_old = store.authorization(receipt["authorization_id"], version=receipt["old_version"])
    new = store.authorization(receipt["authorization_id"], version=receipt["new_version"])
    require(original == current_old and digest(original) == receipt["old_hash"]
            and digest(new) == receipt["new_hash"]
            and receipt["authorization_id"] in decision.get("authorization_ids", [])
            and new == {**original, "version": receipt["new_version"],
                        "expires_at": decision["authorization_expires_at"],
                        "evidence_hash": receipt["approval"]["sha256"]}
            and original.get("test_authorization") is False
            and datetime.fromisoformat(new["expires_at"]) > datetime.now(timezone.utc),
            "renewal scope/version/expiry differs", "UNAUTHORIZED_DATA")
    return deepcopy(new)


def renewed_source_grant(store, original, renewals):
    """No implicit fallback, even when another same-ID version exists."""
    key = digest(original)
    require(type(renewals) is dict and key in renewals,
            "explicit upstream grant renewal absent", "UNAUTHORIZED_DATA")
    return resolve_grant(store, original, renewals[key])


def pid_reuse_observation(pid, recorded_started_at):
    """Read-only native birth identity; unknown observations NEVER clear a veto.

    Caller must already have durable whole-tree-stop evidence. This only proves
    that a currently existing Windows PID is a different, later-born process.
    https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getprocesstimes
    """
    if os.name != "nt" or type(pid) is not int or not 1 < pid < 2**32:
        return None
    import ctypes
    from ctypes import wintypes
    try:
        recorded = datetime.fromisoformat(recorded_started_at)
        if recorded.tzinfo is None:
            return None
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *[ctypes.POINTER(wintypes.FILETIME)] * 4]
        kernel.GetProcessTimes.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            # OpenProcess can deny access to a retired process object that is
            # absent from the executing-process table. Never infer exit from
            # that error: require two complete native snapshots, including our
            # own process, and the caller's pre-existing whole-tree-stop proof.
            error = ctypes.get_last_error()
            absent = pid_absence_observation(pid, recorded_started_at, open_error=error)
            return absent or (cim_pid_reuse_observation(pid, recorded_started_at) if error == 5 else None)
        try:
            creation, exit_time, kernel_time, user_time = [wintypes.FILETIME() for _ in range(4)]
            if not kernel.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time),
                                          ctypes.byref(kernel_time), ctypes.byref(user_time)):
                return cim_pid_reuse_observation(pid, recorded_started_at) if ctypes.get_last_error() == 5 else None
            ticks = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
            birth = datetime.fromtimestamp((ticks - 116444736000000000) / 10000000, timezone.utc)
            # Whole-tree launch always precedes WORKER_STARTED journal time.
            # Two seconds is a conservative timestamp/observation margin.
            if (birth - recorded).total_seconds() <= 2:
                return None
            return {"pid": pid, "recorded_started_at": recorded_started_at,
                    "current_process_created_at": birth.isoformat(), "creation_filetime": ticks,
                    "source": "native-GetProcessTimes-query-only"}
        finally:
            kernel.CloseHandle(handle)
    except (ValueError, TypeError, OSError, OverflowError):
        return None


_CIM_BIRTH_QUERY = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
try {
  $queryPid = [uint32]$env:PSDE_QUERY_PROCESS_ID
  $rows = @(Get-CimInstance -ClassName Win32_Process -Filter ('ProcessId = ' + $queryPid) -ErrorAction Stop)
  if ($rows.Count -ne 1 -or $rows[0].ProcessId -ne $queryPid -or $null -eq $rows[0].CreationDate) { exit 3 }
  @{ ProcessId = [uint32]$rows[0].ProcessId; Name = [string]$rows[0].Name;
     CreationDateUtc = $rows[0].CreationDate.ToUniversalTime().ToString('o') } | ConvertTo-Json -Compress
} catch { exit 4 }
"""


def _cim_process_birth(pid):
    """Fixed read-only query, numeric input outside script, 5s/2048-byte caps.

    stdout is not consumed into an unbounded communicate buffer. If output fills
    the pipe, the wait times out; only our owned query helper is stopped/reaped.
    No target process method (Create/Terminate/etc.) is called.
    """
    if os.name != "nt" or type(pid) is not int or not 1 < pid < 2**32:
        return None
    import json
    import subprocess
    from pathlib import Path
    executable=Path(os.environ.get("SystemRoot",r"C:\Windows"))/"System32/WindowsPowerShell/v1.0/powershell.exe"
    environment=dict(os.environ,PSDE_QUERY_PROCESS_ID=str(pid))
    process=None
    try:
        process=subprocess.Popen([str(executable),"-NoProfile","-NonInteractive","-Command",_CIM_BIRTH_QUERY],
            stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,env=environment)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(timeout=2)
            return None
        content=process.stdout.read(2049)
        if process.returncode or len(content)>2048:
            return None
        value=json.loads(content.decode("utf-8-sig"))
        if (type(value) is not dict or set(value)!={"ProcessId","Name","CreationDateUtc"}
                or type(value["ProcessId"]) is not int or value["ProcessId"]!=pid
                or type(value["Name"]) is not str or not 1<=len(value["Name"])<=260
                or type(value["CreationDateUtc"]) is not str or len(value["CreationDateUtc"])>64):
            return None
        # PowerShell round-trip dates have seven fractional digits. Python
        # 3.10 accepts at most six; discard only sub-microsecond precision,
        # far below the conservative two-second reuse margin.
        import re
        stamp=re.fullmatch(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,7}))?(?:Z|\+00:00)",value["CreationDateUtc"])
        if stamp is None:
            return None
        birth=datetime.fromisoformat(stamp[1]+"."+(stamp[2] or "0")[:6].ljust(6,"0")+"+00:00")
        if birth.tzinfo is None or birth.utcoffset().total_seconds()!=0 or birth>datetime.now(timezone.utc):
            return None
        return {"pid":pid,"name":value["Name"],"created_at_utc":birth.isoformat()}
    except (OSError,ValueError,TypeError,OverflowError,subprocess.TimeoutExpired):
        return None
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill(); process.wait(timeout=2)
            process.stdout.close()


def cim_pid_reuse_observation(pid,recorded_started_at):
    """Two fresh matching UTC births can prove identity reuse, never tree stop.

    Win32_Process.CreationDate is read-only process start time:
    https://learn.microsoft.com/en-us/windows/win32/cimwin32prov/win32-process
    continue_arm's prior settled/terminal/whole-tree-stop checks still apply.
    """
    if os.name != "nt" or type(pid) is not int or not 1<pid<2**32:
        return None
    try:
        recorded=datetime.fromisoformat(recorded_started_at)
        if recorded.tzinfo is None:
            return None
        first,second=_cim_process_birth(pid),_cim_process_birth(pid)
        if not first or first!=second or first.get("pid")!=pid:
            return None
        birth=datetime.fromisoformat(first["created_at_utc"])
        if (birth.tzinfo is None or birth.utcoffset().total_seconds()!=0
                or (birth-recorded).total_seconds()<=2 or birth>datetime.now(timezone.utc)):
            return None
        return {"pid":pid,"recorded_started_at":recorded_started_at,"current_process_created_at":birth.isoformat(),
                "cim_observations":[first,second],"source":"CIM-Win32_Process-two-fresh-matching-births-query-only"}
    except (ValueError,TypeError,KeyError,OSError,OverflowError):
        return None


def _snapshot_pid_presence(pid):
    """Complete read-only native process table; any enumeration fault unknown.

    https://learn.microsoft.com/en-us/windows/win32/api/tlhelp32/nf-tlhelp32-createtoolhelp32snapshot
    TH32CS_SNAPPROCESS includes all system processes, not their restricted
    modules/heaps. A query denial is not confused with absence.
    """
    if os.name != "nt" or type(pid) is not int or not 1 < pid < 2**32:
        return None
    import ctypes
    from ctypes import wintypes
    class Entry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    for name in ("Process32FirstW", "Process32NextW"):
        function = getattr(kernel, name)
        function.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
        function.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateToolhelp32Snapshot(2, 0)
    if not handle or handle == ctypes.c_void_p(-1).value:
        return None
    try:
        entry = Entry()
        entry.dwSize = ctypes.sizeof(Entry)
        if not kernel.Process32FirstW(handle, ctypes.byref(entry)):
            return None
        seen = set()
        while True:
            if entry.th32ProcessID in seen or len(seen) >= 100000:
                return None
            seen.add(entry.th32ProcessID)
            if not kernel.Process32NextW(handle, ctypes.byref(entry)):
                if ctypes.get_last_error() != 18 or os.getpid() not in seen:
                    return None
                return {"present": pid in seen, "process_count": len(seen), "self_present": True}
    finally:
        kernel.CloseHandle(handle)


def pid_absence_observation(pid, recorded_started_at, *, open_error):
    """Only complete fresh absence clears a historical identity veto.

    This is not permission to stop/recover a live tree. continue_arm already
    requires durable old whole-tree stop, terminal attempts and settled costs.
    Present/protected/unknown identities remain vetoed; no process is killed.
    """
    if type(pid) is not int or not 1 < pid < 2**32 or open_error not in (5, 87):
        return None
    try:
        if datetime.fromisoformat(recorded_started_at).tzinfo is None:
            return None
        samples = [_snapshot_pid_presence(pid), _snapshot_pid_presence(pid)]
        if not all(s and s.get("present") is False and s.get("self_present") is True
                   and type(s.get("process_count")) is int and s["process_count"] > 0 for s in samples):
            return None
        return {"pid": pid, "recorded_started_at": recorded_started_at,
            "observation_kind": "absent-from-executing-process-table", "open_process_error": open_error,
            "native_snapshots": samples, "source": "native-TH32CS_SNAPPROCESS-query-only"}
    except (ValueError, TypeError, OSError, OverflowError):
        return None
