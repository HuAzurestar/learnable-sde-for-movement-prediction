"""Bounded file frames between the budget owner and its managed worker.

Stdlib-only so a numerical worker can poll without importing model packages.
This is a trusted-worker protocol, not an OS sandbox or a data authorization.
"""

from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re
import secrets
import time
import uuid


ENVIRONMENT = "PIRC25_WORKER_CONTROL"
SCHEMA = "pirc25-worker-control-v1"


class ControlError(ValueError):
    pass


def canonical(value, limit):
    if type(limit) is not int or not 0 < limit <= 64 * 1024 * 1024:
        raise ControlError("checkpoint frame byte quota is invalid")
    remaining = 65536
    remaining_bytes = limit
    def charge(size):
        nonlocal remaining_bytes
        remaining_bytes -= size
        if remaining_bytes < 0:
            raise ControlError("checkpoint frame exceeds byte quota")
    def charge_string(item):
        # Count the exact ensure_ascii=False JSON UTF-8 representation without
        # allocating it. In particular, many individually small strings must
        # not reach json.dumps when their aggregate is already over quota.
        charge(2)
        for character in item:
            code = ord(character)
            if 0xD800 <= code <= 0xDFFF:
                raise ControlError("checkpoint frame must be finite UTF-8 JSON")
            if character in ('"', '\\', '\b', '\f', '\n', '\r', '\t'):
                charge(2)
            elif code < 0x20:
                charge(6)
            else:
                charge(1 if code < 0x80 else 2 if code < 0x800 else 3 if code < 0x10000 else 4)
    stack = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        remaining -= 1
        if remaining < 0 or depth > 32:
            raise ControlError("checkpoint frame exceeds structural quota")
        if type(item) is dict:
            if len(item) > remaining or any(type(key) is not str or len(key) > 128 for key in item):
                raise ControlError("checkpoint frame keys exceed quota")
            charge(2 + len(item) + max(0, len(item) - 1))
            for key in item:
                charge_string(key)
            stack.extend((child, depth + 1) for child in item.values())
        elif type(item) in {list, tuple}:
            if len(item) > remaining:
                raise ControlError("checkpoint frame array exceeds quota")
            charge(2 + max(0, len(item) - 1))
            stack.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            if len(item) > min(limit, 65536):
                raise ControlError("checkpoint frame string exceeds quota")
            charge_string(item)
        elif type(item) is int:
            if item.bit_length() > 256:
                raise ControlError("checkpoint frame integer exceeds quota")
            charge(len(str(item)))
        elif type(item) is float:
            if not math.isfinite(item):
                raise ControlError("checkpoint frame must be finite")
            charge(len(repr(item)))
        elif item is None or type(item) is bool:
            charge(4 if item is None or item is True else 5)
        elif item is not None and type(item) is not bool:
            raise ControlError("checkpoint frame must be JSON")
    try:
        content = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ControlError("checkpoint frame must be finite UTF-8 JSON") from exc
    if len(content) > limit:
        raise ControlError("checkpoint frame exceeds byte quota")
    return content


@lru_cache(maxsize=1)
def _windows_frame_api(load_library):
    """Reuse only native ABI definitions, not a handle or control-frame fact.

    The loader is part of the definition identity: replacing the native
    binding (including forwarded fault observations) must not borrow an older
    binding. Ordinary owner/worker polls use the same ctypes loader.
    """
    import ctypes
    from ctypes import wintypes

    class AttributeTag(ctypes.Structure):
        _fields_ = [("FileAttributes", wintypes.DWORD), ("ReparseTag", wintypes.DWORD)]

    kernel = load_library("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
        ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.GetFileInformationByHandleEx.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                    ctypes.c_void_p, wintypes.DWORD]
    kernel.GetFileInformationByHandleEx.restype = wintypes.BOOL
    kernel.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                                ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    kernel.ReadFile.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    return ctypes, wintypes, AttributeTag, kernel


def _windows_frame_content(path, limit):
    """Preserve native IPC errors that the CRT collapses into errno13.

    Only an actual sharing/byte-lock conflict defers this poll. No deadline
    or retry loop is added here; the existing owner/worker fuse stays in charge.
    https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew
    https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-readfile
    """
    import ctypes
    ctypes, wintypes, AttributeTag, kernel = _windows_frame_api(ctypes.WinDLL)
    # Read/write/delete sharing accommodates actual atomic control publication;
    # OPEN_EXISTING never creates a missing frame, OPEN_REPARSE_POINT never
    # follows a final link swapped after the caller's lexical link check.
    handle = kernel.CreateFileW(str(path), 0x80000000, 7, None, 3, 0x00200000, None)
    if handle in (None, ctypes.c_void_p(-1).value):
        error = ctypes.get_last_error()
        if error in {32, 33}:
            return None
        raise ctypes.WinError(error)
    try:
        attributes = AttributeTag()
        if not kernel.GetFileInformationByHandleEx(handle, 9, ctypes.byref(attributes),
                                                  ctypes.sizeof(attributes)):
            raise ctypes.WinError(ctypes.get_last_error())
        if attributes.FileAttributes & (0x400 | 0x10):
            raise ControlError("checkpoint frame must be a regular non-reparse file")
        # Never allocate a buffer sized from an untrusted file. Read at most
        # the original limit+1 bytes, even on a short read or a growing frame.
        remaining, content = limit + 1, bytearray()
        buffer = ctypes.create_string_buffer(min(65536, remaining))
        count = wintypes.DWORD()
        while remaining:
            if not kernel.ReadFile(handle, buffer, min(len(buffer), remaining), ctypes.byref(count), None):
                error = ctypes.get_last_error()
                if error in {32, 33}:
                    return None  # Discard partial bytes; next poll reads afresh.
                raise ctypes.WinError(error)
            if not count.value:
                break
            content.extend(buffer.raw[:count.value])
            remaining -= count.value
        return bytes(content)
    finally:
        if not kernel.CloseHandle(handle):
            raise ctypes.WinError(ctypes.get_last_error())


def read_frame(path, limit):
    if type(limit) is not int or not 0 < limit <= 64 * 1024 * 1024:
        raise ControlError("checkpoint frame byte quota is invalid")
    if path.is_symlink():
        raise ControlError("checkpoint frame cannot be a symlink")
    try:
        if os.name == "nt":
            content = _windows_frame_content(path, limit)
            if content is None:
                return None
        else:
            with path.open("rb") as stream:
                content = stream.read(limit + 1)
    except FileNotFoundError:
        return None
    if len(content) > limit:
        raise ControlError("checkpoint frame exceeds byte quota")
    try:
        value = json.loads(content)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ControlError("checkpoint frame is invalid JSON") from exc
    if content != canonical(value, limit):
        raise ControlError("checkpoint frame must be canonical JSON")
    return value


def require_live(deadline):
    if deadline is not None and (type(deadline) not in {int, float}
            or (type(deadline) is int and deadline.bit_length() > 256)
            or not math.isfinite(deadline) or time.monotonic() >= deadline):
        raise ControlError("checkpoint control reached the hard deadline")


def write_frame(path, value, limit, *, deadline=None):
    content = canonical(value, limit)
    require_live(deadline)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        require_live(deadline)
        os.replace(temporary, path)
        require_live(deadline)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class CheckpointExchange:
    def __init__(self, directory, attempt_id, deadline, byte_limit):
        self.directory, self.attempt_id = Path(directory), attempt_id
        self.deadline, self.byte_limit = deadline, byte_limit
        self.token = secrets.token_hex(32)
        self.request_id = None
        self.accepted = False

    def environment(self):
        descriptor = {"schema_version": SCHEMA, "directory": str(self.directory), "attempt_id": self.attempt_id,
            "deadline": self.deadline, "byte_limit": self.byte_limit, "token": self.token}
        return {ENVIRONMENT: canonical(descriptor, 16384).decode()}

    def request(self):
        self.request_id = uuid.uuid4().hex
        value = {"schema_version": SCHEMA, "attempt_id": self.attempt_id, "token": self.token,
            "request_id": self.request_id, "deadline": self.deadline}
        write_frame(self.directory / "checkpoint-request.json", value, 16384, deadline=self.deadline)
        return self.request_id

    def response(self):
        if self.accepted:
            return None
        value = read_frame(self.directory / "checkpoint-response.json", self.byte_limit)
        if value is None:
            return None
        if (type(value) is not dict or set(value) != {"schema_version", "attempt_id", "token", "request_id", "state", "progress"}
                or value["schema_version"] != SCHEMA or value["attempt_id"] != self.attempt_id
                or value["token"] != self.token or self.request_id is None or value["request_id"] != self.request_id):
            raise ControlError("checkpoint response identity differs")
        return value

    def acknowledge(self, artifact_id):
        if (self.request_id is None or type(artifact_id) is not str
                or re.fullmatch(r"[0-9a-f]{64}", artifact_id) is None):
            raise ControlError("checkpoint acknowledgement needs issued request and valid artifact")
        write_frame(self.directory / "checkpoint-ack.json", {"schema_version": SCHEMA,
            "request_id": self.request_id, "artifact_id": artifact_id}, 16384, deadline=self.deadline)
        self.accepted = True


class WorkerControl:
    def __init__(self, descriptor):
        if (type(descriptor) is not dict or set(descriptor) != {"schema_version", "directory", "attempt_id", "deadline", "byte_limit", "token"}
                or descriptor["schema_version"] != SCHEMA or type(descriptor["directory"]) is not str
                or "\x00" in descriptor["directory"] or not Path(descriptor["directory"]).is_absolute()
                or type(descriptor["attempt_id"]) is not str
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", descriptor["attempt_id"]) is None
                or type(descriptor["token"]) is not str or re.fullmatch(r"[0-9a-f]{64}", descriptor["token"]) is None
                or type(descriptor["deadline"]) not in {float, int}
                or (type(descriptor["deadline"]) is int and descriptor["deadline"].bit_length() > 256)
                or not math.isfinite(descriptor["deadline"])
                or type(descriptor["byte_limit"]) is not int or not 0 < descriptor["byte_limit"] <= 64 * 1024 * 1024):
            raise ControlError("checkpoint control descriptor is invalid")
        self.descriptor = json.loads(canonical(descriptor, 16384))
        self.directory = Path(self.descriptor["directory"])
        self.request = None

    @classmethod
    def from_environment(cls):
        content = os.environ.get(ENVIRONMENT)
        if content is None:
            return None
        if len(content) > 16384:
            raise ControlError("checkpoint descriptor exceeds quota")
        try:
            if len(content.encode("utf-8")) > 16384:
                raise ControlError("checkpoint descriptor exceeds quota")
            descriptor = json.loads(content)
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise ControlError("checkpoint descriptor is invalid UTF-8 JSON") from exc
        return cls(descriptor)

    def poll(self):
        value = read_frame(self.directory / "checkpoint-request.json", 16384)
        if value is None:
            return None
        if (type(value) is not dict or set(value) != {"schema_version", "attempt_id", "token", "request_id", "deadline"}
                or any(value[key] != self.descriptor[key] for key in ("schema_version", "attempt_id", "token", "deadline"))
                or type(value["request_id"]) is not str or re.fullmatch(r"[0-9a-f]{32}", value["request_id"]) is None):
            raise ControlError("checkpoint request identity differs")
        self.request = value
        return value

    def save(self, state, progress):
        if self.request is None or time.monotonic() >= self.descriptor["deadline"]:
            raise ControlError("checkpoint save has no live request")
        frame = {key: self.request[key] for key in ("schema_version", "attempt_id", "token", "request_id")}
        write_frame(self.directory / "checkpoint-response.json", {**frame, "state": state, "progress": progress},
            self.descriptor["byte_limit"], deadline=self.descriptor["deadline"])
        while time.monotonic() < self.descriptor["deadline"]:
            ack = read_frame(self.directory / "checkpoint-ack.json", 16384)
            if ack is not None:
                require_live(self.descriptor["deadline"])
                if (type(ack) is not dict or set(ack) != {"schema_version", "request_id", "artifact_id"}
                        or ack["schema_version"] != SCHEMA or ack["request_id"] != self.request["request_id"]
                        or type(ack["artifact_id"]) is not str or re.fullmatch(r"[0-9a-f]{64}", ack["artifact_id"]) is None):
                    raise ControlError("checkpoint acknowledgement identity differs")
                return ack["artifact_id"]
            time.sleep(0.01)
        raise ControlError("checkpoint save was not acknowledged before deadline")
