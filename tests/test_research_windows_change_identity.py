"""Native Windows metadata calls: real handles, work and fail-closed boundaries."""
import os

import pytest

from infrastructure.research_files import opened_regular_file
from infrastructure.research_store import ResearchError
from tests.research_file_observation import observe_file


pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows native handle API contract')


def observe_native(monkeypatch, *, fail_query=None, missing_path=None, unknown_change=False):
    from infrastructure import research_windows_files as native
    api = native._api()
    ctypes, BasicInfo, real_query, real_create, real_close = api
    calls, data_fds = [], []
    real_open = os.open

    def create(path, access, share, security, disposition, flags, template):
        handle = real_create(str(missing_path) if missing_path else path,
                             access, share, security, disposition, flags, template)
        calls.append(('create', handle, access, share, disposition, flags))
        return handle

    def query(handle, kind, output, size):
        number = sum(call[0] == 'query' for call in calls) + 1
        # Failure injection uses the actual OS rejection of an invalid handle,
        # not a fabricated error/result/identity. Retain the held real handle.
        target = ctypes.c_void_p(-1).value if number == fail_query else handle
        result = real_query(target, kind, output, size)
        error = ctypes.get_last_error()
        value = ctypes.cast(output, ctypes.POINTER(BasicInfo)).contents
        calls.append(('query', handle, kind, size, bool(result), value.ChangeTime))
        if result and unknown_change:
            # Explicit API-compatibility fault, NOT a native filesystem observation.
            value.ChangeTime = 0
        ctypes.set_last_error(error)
        return result

    def close(handle):
        result = real_close(handle)
        calls.append(('close', handle, bool(result)))
        return result

    def data_open(*args, **kwargs):
        fd = real_open(*args, **kwargs)
        data_fds.append(fd)
        return fd

    monkeypatch.setattr(native, '_api', lambda: (ctypes, BasicInfo, query, create, close))
    monkeypatch.setattr(os, 'open', data_open)
    return native, calls, data_fds, api


def assert_closed(data_fds):
    for fd in data_fds:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_native_guard_has_one_attributes_open_and_two_fresh_held_queries(tmp_path, monkeypatch):
    import msvcrt
    path = tmp_path / 'source with spaces.py'
    path.write_bytes(b'# unchanged\n')
    reads, handles = observe_file(monkeypatch, path)
    _, calls, data_fds, api = observe_native(monkeypatch)
    with opened_regular_file(tmp_path, path) as (stream, size, _):
        assert stream.read(size + 1) == b'# unchanged\n'
        held = msvcrt.get_osfhandle(stream.fileno())
    creates = [call for call in calls if call[0] == 'create']
    queries = [call for call in calls if call[0] == 'query']
    assert len(creates) == 1 and creates[0][2:] == (0x80, 0x7, 3, 0x00200000)
    assert len(queries) == 3 and queries[0][1] == creates[0][1]
    assert [call[1] for call in queries[1:]] == [held, held]
    assert all(call[2:5] == (0, api[0].sizeof(api[1]), True) for call in queries)
    assert all(call[5] > 0 for call in queries)
    assert [call[0] for call in calls] == ['create', 'query', 'close', 'query', 'query']
    assert calls[2][2] is True and reads == [{'size': size + 1, 'bytes': size}] and handles == [True]
    assert len(data_fds) == 1
    assert_closed(data_fds)


@pytest.mark.parametrize('fail_query', [1, 2, 3, 4])
def test_actual_native_query_failure_never_returns_success_and_closes_handles(tmp_path, monkeypatch, fail_query):
    path = tmp_path / 'source.py'
    path.write_bytes(b'# unchanged\n')
    reads, handles = observe_file(monkeypatch, path)
    _, calls, data_fds, _ = observe_native(monkeypatch, fail_query=fail_query)
    with pytest.raises(OSError) as error:
        with opened_regular_file(tmp_path, path) as (stream, size, verify):
            stream.read(size + 1)
            verify()  # Query 3; context completion is query 4.
    assert error.value.winerror == 6  # ERROR_INVALID_HANDLE, from actual kernel call.
    assert len([call for call in calls if call[0] == 'query']) == fail_query
    assert [call for call in calls if call[0] == 'close'][0][2] is True
    assert handles == ([] if fail_query <= 2 else [True])
    assert len(data_fds) == (0 if fail_query == 1 else 1)
    if fail_query <= 2:
        assert reads == []
    else:
        assert reads == [{'size': size + 1, 'bytes': size}]
    assert_closed(data_fds)


def test_failed_native_attributes_open_never_opens_data_or_closes_invalid_handle(tmp_path, monkeypatch):
    path = tmp_path / 'source.py'
    path.write_bytes(b'# unchanged\n')
    reads, handles = observe_file(monkeypatch, path)
    _, calls, data_fds, _ = observe_native(monkeypatch, missing_path=tmp_path / 'missing.py')
    with pytest.raises(OSError) as error:
        with opened_regular_file(tmp_path, path):
            pytest.fail('failed metadata open yielded a data stream')
    assert error.value.winerror == 2
    assert [call[0] for call in calls] == ['create']
    assert reads == handles == data_fds == []


def test_explicit_unknown_native_change_identity_is_not_a_fallback_stamp(tmp_path, monkeypatch):
    path = tmp_path / 'source.py'
    path.write_bytes(b'# unchanged\n')
    reads, handles = observe_file(monkeypatch, path)
    _, calls, data_fds, _ = observe_native(monkeypatch, unknown_change=True)
    with pytest.raises(ResearchError) as error:
        with opened_regular_file(tmp_path, path):
            pytest.fail('unknown change identity yielded a data stream')
    assert error.value.code == 'CORRUPT_ARTIFACT'
    assert [call[0] for call in calls] == ['create', 'query', 'close']
    assert calls[-1][2] is True and reads == handles == data_fds == []


def test_descriptor_queries_are_fresh_and_do_not_close_the_callers_handle(tmp_path):
    from infrastructure.research_windows_files import descriptor_change_time, path_change_time
    path = tmp_path / 'source.py'
    path.write_bytes(b'# original\n')
    before = path.stat()
    with path.open('rb') as stream:
        initial = descriptor_change_time(stream.fileno())
        assert initial == path_change_time(path)
        path.write_bytes(b'# modified\n')
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        changed = descriptor_change_time(stream.fileno())
        assert changed != initial and changed == path_change_time(path)
        assert os.fstat(stream.fileno()).st_ino == before.st_ino
        assert stream.read() == b'# modified\n'
