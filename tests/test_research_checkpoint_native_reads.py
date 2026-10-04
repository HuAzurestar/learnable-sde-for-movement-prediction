"""Native transient IPC reads never bypass identity, quota or original fuse.

These are trusted-worker control frames, not data-read authorization. Native
handles below belong only to each disposable fixture and are always closed.
"""

from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from infrastructure import research_control as control


WINDOWS = pytest.mark.skipif(os.name != "nt", reason="actual Windows native IPC sharing/byte lock")


class NativeFrameLock:
    def __init__(self, path, kind):
        from ctypes import wintypes

        class Overlapped(ctypes.Structure):
            _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t),
                        ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD),
                        ("hEvent", wintypes.HANDLE)]

        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
            ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        self.kernel.CreateFileW.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.kernel.LockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
            wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(Overlapped)]
        self.kernel.LockFileEx.restype = wintypes.BOOL
        self.kernel.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                                        ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        self.kernel.ReadFile.restype = wintypes.BOOL
        self.overlapped = Overlapped()
        self.handle = self.kernel.CreateFileW(str(path), 0x80000000, 0 if kind == "share" else 7,
                                               None, 3, 0x80, None)
        if self.handle in (None, ctypes.c_void_p(-1).value):
            raise ctypes.WinError(ctypes.get_last_error())
        if kind == "byte" and not self.kernel.LockFileEx(self.handle, 3, 0, 16384, 0,
                                                         ctypes.byref(self.overlapped)):
            error = ctypes.get_last_error()
            self.close()
            raise ctypes.WinError(error)
        self.expected_error = 32 if kind == "share" else 33

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        handle, self.handle = self.handle, None
        if handle is not None and not self.kernel.CloseHandle(handle):
            raise ctypes.WinError(ctypes.get_last_error())

    def prove_native_denial(self, path):
        from ctypes import wintypes
        probe = self.kernel.CreateFileW(str(path), 0x80000000, 7, None, 3, 0x80, None)
        if probe in (None, ctypes.c_void_p(-1).value):
            error = ctypes.get_last_error()
        else:
            try:
                buffer, count = ctypes.create_string_buffer(1), wintypes.DWORD()
                assert not self.kernel.ReadFile(probe, buffer, 1, ctypes.byref(count), None)
                error = ctypes.get_last_error()
            finally:
                assert self.kernel.CloseHandle(probe)
        assert error == self.expected_error
        # The real CRT reader collapses both native conditions into errno13;
        # it need not retain a winerror. This was the first harness error.
        with pytest.raises(PermissionError) as observed:
            with path.open("rb") as stream:
                stream.read(1)
        assert observed.value.errno == 13


def exchange_worker(tmp_path, seconds=3):
    exchange = control.CheckpointExchange(tmp_path, "native-control", time.monotonic() + seconds, 4096)
    exchange.request()
    worker = control.WorkerControl(json.loads(exchange.environment()[control.ENVIRONMENT]))
    return exchange, worker


@WINDOWS
@pytest.mark.parametrize("kind", ["share", "byte"])
@pytest.mark.parametrize("role", ["request", "response", "ack"])
def test_native_busy_frame_defers_without_consuming_then_reads_fresh(tmp_path, kind, role):
    path = tmp_path / ("checkpoint-" + role + ".json")
    value = {"fixture": "only actual canonical bytes"}
    control.write_frame(path, value, 4096)
    with NativeFrameLock(path, kind) as held:
        held.prove_native_denial(path)
        assert control.read_frame(path, 4096) is None
    assert control.read_frame(path, 4096) == value


@WINDOWS
@pytest.mark.parametrize("kind", ["share", "byte"])
def test_actual_worker_poll_does_not_consume_busy_request(tmp_path, kind):
    exchange, worker = exchange_worker(tmp_path)
    path = tmp_path / "checkpoint-request.json"
    with NativeFrameLock(path, kind) as held:
        held.prove_native_denial(path)
        assert worker.poll() is None and worker.request is None
    assert worker.poll()["request_id"] == exchange.request_id


@WINDOWS
@pytest.mark.parametrize("kind", ["share", "byte"])
def test_actual_worker_save_waits_for_released_ack_with_same_deadline(tmp_path, monkeypatch, kind):
    exchange, worker = exchange_worker(tmp_path)
    worker.poll()
    path = tmp_path / "checkpoint-ack.json"
    control.write_frame(path, {"schema_version": control.SCHEMA, "request_id": exchange.request_id,
                              "artifact_id": "a" * 64}, 16384)
    attempted = threading.Event()
    original_read = control.read_frame

    def read(frame, limit):
        try:
            result = original_read(frame, limit)
        except BaseException:
            if frame == path:
                attempted.set()
            raise
        if frame == path:
            attempted.set()  # Native read has finished while the handle is held.
        return result

    monkeypatch.setattr(control, "read_frame", read)
    with NativeFrameLock(path, kind) as held:
        held.prove_native_denial(path)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(worker.save, {}, {})
            try:
                assert attempted.wait(timeout=1), "actual ACK read was not reached"
            finally:
                held.close()
            assert future.result(timeout=3) == "a" * 64
    assert worker.descriptor["deadline"] == exchange.deadline
    assert control.read_frame(tmp_path / "checkpoint-response.json", 4096)["state"] == {}


@WINDOWS
@pytest.mark.parametrize("kind", ["share", "byte"])
def test_busy_ack_still_reaches_original_hard_fuse_without_acceptance(tmp_path, kind):
    exchange, worker = exchange_worker(tmp_path, seconds=1)
    worker.poll()
    path = tmp_path / "checkpoint-ack.json"
    control.write_frame(path, {"schema_version": control.SCHEMA, "request_id": exchange.request_id,
                              "artifact_id": "a" * 64}, 16384)
    deadline = worker.descriptor["deadline"]
    with NativeFrameLock(path, kind) as held:
        held.prove_native_denial(path)
        with pytest.raises(control.ControlError, match="not acknowledged before deadline"):
            worker.save({}, {})
    assert time.monotonic() >= deadline and exchange.accepted is False
    assert worker.descriptor["deadline"] == deadline
    assert (tmp_path / "checkpoint-response.json").is_file()
    with pytest.raises(control.ControlError, match="no live request"):
        worker.save({}, {})


@WINDOWS
@pytest.mark.parametrize("kind", ["share", "byte"])
def test_released_busy_frame_does_not_borrow_identity(tmp_path, kind):
    _, worker = exchange_worker(tmp_path)
    path = tmp_path / "checkpoint-request.json"
    with NativeFrameLock(path, kind) as held:
        held.prove_native_denial(path)
        assert worker.poll() is None
    frame = control.read_frame(path, 16384)
    frame["token"] = "0" * 64
    control.write_frame(path, frame, 16384)
    with pytest.raises(control.ControlError, match="identity differs"):
        worker.poll()
    assert worker.request is None


@WINDOWS
@pytest.mark.parametrize("kind", ["share", "byte"])
def test_released_busy_frame_still_enforces_byte_quota(tmp_path, kind):
    path = tmp_path / "checkpoint-response.json"
    control.write_frame(path, {"payload": "x" * 256}, 4096)
    with NativeFrameLock(path, kind) as held:
        held.prove_native_denial(path)
        assert control.read_frame(path, 128) is None
    with pytest.raises(control.ControlError, match="byte quota"):
        control.read_frame(path, 128)


@WINDOWS
def test_actual_native_access_denied_is_not_a_missing_frame(tmp_path):
    path = tmp_path / "checkpoint-request.json"
    path.mkdir()  # File-only native open of a directory gives permanent error5.
    with pytest.raises(PermissionError) as native:
        NativeFrameLock(path, "share")
    assert native.value.winerror == 5
    with pytest.raises(PermissionError) as observed:
        control.read_frame(path, 4096)
    assert observed.value.errno == 13


@pytest.mark.parametrize("kind", ["permission", "io", "posix-with-winerror"])
def test_other_read_failures_are_not_deferred(tmp_path, monkeypatch, kind):
    path = tmp_path / "checkpoint-request.json"
    control.write_frame(path, {}, 4096)
    if kind == "io":
        failure = OSError("actual read unavailable")
    else:
        failure = PermissionError("actual read denied")
        if kind == "posix-with-winerror":
            failure.winerror = 32
            monkeypatch.setattr(control, "os", SimpleNamespace(name="posix"))

    def unavailable(*args, **kwargs):
        raise failure

    monkeypatch.setattr(Path, "open", unavailable)
    if os.name == "nt":
        monkeypatch.setattr(ctypes, "WinDLL", unavailable)
    with pytest.raises(type(failure)) as observed:
        control.read_frame(path, 4096)
    assert observed.value is failure
