"""Forward native frame I/O; share definitions only, never a frame result."""

from collections import Counter
import ctypes
import os

import pytest

from infrastructure import research_control as control
from infrastructure.research_store import encode


WINDOWS = pytest.mark.skipif(os.name != 'nt', reason='actual Windows frame API bindings')


def observe_bindings(monkeypatch):
    original_loader = ctypes.WinDLL
    bindings, calls, handles = [], Counter(), {'opened': [], 'closed': []}

    class Function:
        def __init__(self, name, actual):
            object.__setattr__(self, 'name', name)
            object.__setattr__(self, 'actual', actual)

        def __getattr__(self, name):
            return getattr(self.actual, name)

        def __setattr__(self, name, value):
            setattr(self.actual, name, value)

        def __call__(self, *args):
            result = self.actual(*args)
            calls[self.name] += 1
            if self.name == 'CreateFileW' and result not in (None, ctypes.c_void_p(-1).value):
                handles['opened'].append(result)
            if self.name == 'CloseHandle' and result:
                handles['closed'].append(args[0])
            return result

    class Library:
        def __init__(self, actual):
            self.actual, self.functions = actual, {}

        def __getattr__(self, name):
            if name not in self.functions:
                self.functions[name] = Function(name, getattr(self.actual, name))
            return self.functions[name]

    def load(name, *args, **kwargs):
        actual = original_loader(name, *args, **kwargs)
        assert name == 'kernel32' and kwargs == {'use_last_error': True}
        bindings.append(name)
        return Library(actual)

    monkeypatch.setattr(ctypes, 'WinDLL', load)
    return bindings, calls, handles


@WINDOWS
@pytest.mark.parametrize('count', [8, 32])
def test_repeated_actual_frame_reads_bind_definitions_once_but_read_every_new_value(
        tmp_path, monkeypatch, count):
    bindings, calls, handles = observe_bindings(monkeypatch)
    path = tmp_path / 'checkpoint-response.json'
    values = []
    for position in range(count):
        value = {'position': position, 'payload': '中😀' + str(position)}
        control.write_frame(path, value, 4096)
        actual = control.read_frame(path, 4096)
        assert actual == value
        values.append(actual)
    observed = {'bindings': len(bindings), 'calls': dict(calls), 'handles': handles,
                'values': values, 'count': count}
    (tmp_path / 'frame-api-work-observed.json').write_bytes(encode(observed))
    # These are actual OS calls and bytes, not cached identity or parser facts.
    assert calls['CreateFileW'] == calls['GetFileInformationByHandleEx'] == count, observed
    assert calls['ReadFile'] >= count and calls['CloseHandle'] == count, observed
    assert handles['closed'] == handles['opened'] and len(handles['opened']) == count, observed
    assert len(bindings) == 1, observed


@WINDOWS
def test_missing_frame_polls_share_definitions_but_reopen_the_actual_name(tmp_path, monkeypatch):
    bindings, calls, handles = observe_bindings(monkeypatch)
    path = tmp_path / 'checkpoint-request.json'
    for _ in range(16):
        assert control.read_frame(path, 4096) is None
    observed = {'bindings': len(bindings), 'calls': dict(calls), 'handles': handles}
    (tmp_path / 'missing-frame-api-work-observed.json').write_bytes(encode(observed))
    assert calls['CreateFileW'] == 16, observed
    assert not calls['ReadFile'] and not calls['CloseHandle'] and not handles['opened'], observed
    assert len(bindings) == 1, observed


@WINDOWS
def test_warmed_api_definitions_cannot_hide_new_quota_or_permanent_native_denial(tmp_path):
    path = tmp_path / 'checkpoint-response.json'
    control.write_frame(path, {}, 4096)
    assert control.read_frame(path, 128) == {}
    control.write_frame(path, {'payload': 'x' * 256}, 4096)
    with pytest.raises(control.ControlError, match='byte quota'):
        control.read_frame(path, 128)
    # Both paths are disposable children of this test's fixture; no deletion.
    path.rename(tmp_path / 'parked-response.json')
    path.mkdir()
    with pytest.raises(PermissionError) as denied:
        control.read_frame(path, 4096)
    assert denied.value.winerror == 5
