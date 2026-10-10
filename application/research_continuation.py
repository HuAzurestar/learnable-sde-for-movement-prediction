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
        start = datetime.fromisoformat(value["work_started_at"])
        deadline = datetime.fromisoformat(value["work_deadline"])
        expiry = datetime.fromisoformat(value["authorization_expires_at"])
        require(start.tzinfo is not None and deadline.tzinfo is not None
                and expiry.tzinfo is not None and 0 < (deadline - start).total_seconds() <= 86400
                and expiry >= deadline, "bounded work/authorization window required")
        if live:
            require(start <= datetime.now(timezone.utc) < deadline,
                    "continuation work window expired", "UNAUTHORIZED_DATA")
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
    if os.name != "nt" or type(pid) is not int or pid <= 1:
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
            return None
        try:
            creation, exit_time, kernel_time, user_time = [wintypes.FILETIME() for _ in range(4)]
            if not kernel.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time),
                                          ctypes.byref(kernel_time), ctypes.byref(user_time)):
                return None
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
