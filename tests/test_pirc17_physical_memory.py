"""ABI, fresh-reading and fail-closed tests; no forecast or research data."""
import ctypes
from hashlib import sha256
from pathlib import Path

import pytest

from experiments.pirc17 import physical_memory as memory


def api_returning(available, total=16*1024**3, load=25, success=1, length=None):
    def query(pointer):
        state = ctypes.cast(pointer, ctypes.POINTER(memory._MemoryStatusEx)).contents
        assert state.dwLength == 64
        state.ullTotalPhys = total
        state.ullAvailPhys = available
        state.dwMemoryLoad = load
        if length is not None:
            state.dwLength = length
        return success
    return query


def test_native_structure_width_and_offsets_match_windows_abi():
    assert ctypes.sizeof(memory._MemoryStatusEx) == 64
    fields = [name for name, _ in memory._MemoryStatusEx._fields_]
    assert [getattr(memory._MemoryStatusEx, name).offset for name in fields] == [0,4,8,16,24,32,40,48,56]


@pytest.mark.parametrize("available", [0, 1, 2*1024**3-1, 2*1024**3, 2*1024**3+1, 8*1024**3, 16*1024**3])
def test_exact_unsigned_bytes_and_zero_are_preserved(available):
    value = memory.available_physical_bytes(query=api_returning(available))
    assert type(value) is int and value == available


def test_every_call_observes_fresh_values_without_changing_threshold(monkeypatch):
    readings = iter([8*1024**3, 2*1024**3-1, 0, 9*1024**3])
    calls = []
    def api(pointer):
        calls.append(1)
        return api_returning(next(readings))(pointer)
    monkeypatch.setattr(memory, "_windows_query", lambda:api)
    values = [memory.available_physical_bytes() for _ in range(4)]
    assert len(calls) == 4
    assert [v < 2*1024**3 for v in values] == [False,True,True,False]


@pytest.mark.parametrize("arguments", [
    {"available":10,"total":0},
    {"available":11,"total":10},
    {"available":-1},
    {"available":1,"load":101},
    {"available":1,"length":32},
])
def test_invalid_native_fields_fail_closed(arguments):
    with pytest.raises(OSError, match="invalid physical-memory"):
        memory.available_physical_bytes(query=api_returning(**arguments))


def test_native_failure_does_not_return_even_a_plausible_stale_value(monkeypatch):
    monkeypatch.setattr(ctypes, "get_last_error", lambda:5, raising=False)
    with pytest.raises(OSError, match="available RAM is unknown") as failed:
        memory.available_physical_bytes(query=api_returning(8*1024**3, success=0))
    assert failed.value.errno == 5


def test_failure_after_success_is_not_masked_by_cache(monkeypatch):
    readings = iter([1,0])
    def api(pointer):
        return api_returning(8*1024**3, success=next(readings))(pointer)
    monkeypatch.setattr(memory, "_windows_query", lambda:api)
    assert memory.available_physical_bytes() == 8*1024**3
    with pytest.raises(OSError): memory.available_physical_bytes()


def test_native_exception_propagates_without_fallback():
    def failed(pointer): raise RuntimeError("injected native binding failure")
    with pytest.raises(RuntimeError, match="binding failure"):
        memory.available_physical_bytes(query=failed)


def test_function_binding_uses_windows_calling_convention_and_error_capture(monkeypatch):
    loaded = []
    class Library:
        GlobalMemoryStatusEx = staticmethod(api_returning(1024))
    def loader(name, **kwargs):
        loaded.append((name,kwargs))
        return Library()
    memory._windows_query.cache_clear()
    monkeypatch.setattr(ctypes, "WinDLL", loader, raising=False)
    try:
        query = memory._windows_query()
        assert memory._windows_query() is query
        assert loaded == [("kernel32.dll",{"use_last_error":True})]
        assert query.argtypes == [ctypes.POINTER(memory._MemoryStatusEx)]
        assert query.restype is ctypes.c_int32
        assert memory.available_physical_bytes() == 1024
    finally:
        memory._windows_query.cache_clear()


def test_unsupported_platform_is_explicit_not_a_silent_new_policy(monkeypatch):
    memory._windows_query.cache_clear()
    monkeypatch.delattr(ctypes, "WinDLL", raising=False)
    try:
        with pytest.raises(OSError, match="unavailable on this platform"):
            memory.available_physical_bytes()
    finally:
        memory._windows_query.cache_clear()


def test_identity_declares_field_freshness_failure_policy_and_exact_source():
    identity = memory.identity()
    assert identity["version"] == memory.VERSION and identity["unit"] == "bytes"
    assert identity["measurement_cache"] is False and identity["one_native_query_per_call"] is True
    assert identity["source_sha256"] == sha256(Path(memory.__file__).read_bytes()).hexdigest()


@pytest.mark.skipif(not hasattr(ctypes, "WinDLL"), reason="actual Windows API only")
def test_real_windows_observation_is_available_without_forecasting():
    assert type(memory.available_physical_bytes()) is int
    assert memory.available_physical_bytes() >= 0

