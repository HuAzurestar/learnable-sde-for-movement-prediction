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


@pytest.mark.parametrize("value", [{"payload": ["x" * 1024] * 65},
                                  {"payload": ["中😀"] * 10000}, {}],
                         ids=["multiple-native-chunks", "split-utf8", "empty-object"])
def test_actual_frame_chunks_preserve_canonical_bytes_and_exact_quota(tmp_path, value):
    path = tmp_path / "checkpoint-response.json"
    content = control.canonical(value, 256 * 1024)
    control.write_frame(path, value, len(content))
    assert control.read_frame(path, len(content)) == value
    with pytest.raises(control.ControlError):
        control.read_frame(path, len(content) - 1)


@pytest.mark.parametrize("limit", [True, 0, -1, 64 * 1024 * 1024 + 1])
def test_invalid_read_quota_never_reaches_native_or_python_io(tmp_path, monkeypatch, limit):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid frame quota reached file I/O")

    monkeypatch.setattr(Path, "open", unexpected)
    if os.name == "nt":
        monkeypatch.setattr(ctypes, "WinDLL", unexpected)
    with pytest.raises(control.ControlError, match="byte quota is invalid"):
        control.read_frame(tmp_path / "missing-frame.json", limit)


@WINDOWS
def test_actual_native_short_reads_are_accumulated_without_changing_bytes(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint-response.json"
    value = {"payload": ["中😀", "ascii", "escaped\n\t"] * 16}
    control.write_frame(path, value, 4096)
    kernel, calls = ctypes.WinDLL("kernel32", use_last_error=True), []
    actual_read = kernel.ReadFile

    class ShortRead:
        def __call__(self, handle, buffer, count, output, overlapped):
            actual_read.argtypes = self.argtypes
            actual_read.restype = self.restype
            calls.append(count)
            return actual_read(handle, buffer, min(count, 7), output, overlapped)

    proxy = SimpleNamespace(CreateFileW=kernel.CreateFileW,
        GetFileInformationByHandleEx=kernel.GetFileInformationByHandleEx,
        ReadFile=ShortRead(), CloseHandle=kernel.CloseHandle)
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: proxy)
    assert control.read_frame(path, 4096) == value
    assert len(calls) > 1, "actual native short reads were not exercised"


@WINDOWS
def test_native_byte_lock_deferred_read_closes_its_actual_frame_handle(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint-ack.json"
    control.write_frame(path, {}, 4096)
    with NativeFrameLock(path, "byte") as held:
        held.prove_native_denial(path)
        kernel, opened, closed = ctypes.WinDLL("kernel32", use_last_error=True), [], []

        class Forward:
            def __init__(self, function, records):
                self.function, self.records = function, records

            def __call__(self, *args):
                self.function.argtypes, self.function.restype = self.argtypes, self.restype
                result = self.function(*args)
                self.records.append(result if self.function is kernel.CreateFileW else args[0])
                return result

        proxy = SimpleNamespace(CreateFileW=Forward(kernel.CreateFileW, opened),
            GetFileInformationByHandleEx=kernel.GetFileInformationByHandleEx,
            ReadFile=kernel.ReadFile, CloseHandle=Forward(kernel.CloseHandle, closed))
        monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: proxy)
        assert control.read_frame(path, 4096) is None
        assert len(opened) == 1 and closed == opened


@WINDOWS
@pytest.mark.parametrize("extended", [False, True])
def test_native_control_reader_keeps_legal_long_path_spellings(tmp_path, extended):
    long_directory = tmp_path / ("a" * 80) / ("b" * 80)
    created = Path("\\\\?\\" + str(long_directory))
    created.mkdir(parents=True)
    value = {"actual": "long native control path"}
    control.write_frame(created / "checkpoint-response.json", value, 4096)
    path = (created if extended else long_directory) / "checkpoint-response.json"
    assert len(str(path)) > 260
    assert control.read_frame(path, 4096) == value


def test_frame_symlink_is_never_consumed(tmp_path):
    source, link = tmp_path / "actual-frame.json", tmp_path / "checkpoint-request.json"
    control.write_frame(source, {}, 4096)
    try:
        link.symlink_to(source)
    except OSError as exc:
        if os.name == "nt" and getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows symlink creation privilege unavailable")
        raise
    with pytest.raises(control.ControlError, match="symlink"):
        control.read_frame(link, 4096)


@WINDOWS
def test_native_open_never_follows_link_after_lexical_check(tmp_path, monkeypatch):
    source, link = tmp_path / "actual-frame.json", tmp_path / "checkpoint-response.json"
    control.write_frame(source, {}, 4096)
    try:
        link.symlink_to(source)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows symlink creation privilege unavailable")
        raise
    monkeypatch.setattr(Path, "is_symlink", lambda path: False)
    with pytest.raises(control.ControlError, match="non-reparse"):
        control.read_frame(link, 4096)
