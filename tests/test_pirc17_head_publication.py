"""Bounded metadata publication regressions; no research or real authority."""
import ctypes
import errno
import os
import threading
import time

import pytest

from experiments.pirc17 import formal_budget as budget
from experiments.pirc17.protocol_core import digest, file_hash, read_json, unpack


def _denial(code=5):
    error = PermissionError(errno.EACCES, 'metadata fixture sharing denial')
    error.winerror = code
    return error


def _held_reader(path):
    api = ctypes.WinDLL('kernel32.dll', use_last_error=True)
    api.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                               ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    api.CreateFileW.restype = ctypes.c_void_p
    api.CloseHandle.argtypes, api.CloseHandle.restype = [ctypes.c_void_p], ctypes.c_int
    handle = api.CreateFileW(str(path), 0x80000000, 3, None, 3, 0x80, None)
    if handle in (None, ctypes.c_void_p(-1).value):
        raise ctypes.WinError(ctypes.get_last_error())
    def close():
        if not api.CloseHandle(handle):
            raise ctypes.WinError(ctypes.get_last_error())
    return close


@pytest.mark.skipif(os.name != 'nt', reason='actual Windows sharing regression')
def test_actual_reader_release_allows_same_atomic_head_without_extra_event(tmp_path):
    head = tmp_path/'head.json'
    budget._publish(head, dict(sequence=0))
    close = _held_reader(head)
    errors = []
    def release():
        time.sleep(0.08)
        try:
            close()
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=release)
    thread.start()
    begin = time.monotonic()
    try:
        record = budget._publish(head, dict(sequence=1), replace=True)
    finally:
        thread.join(timeout=2)
    assert not thread.is_alive() and not errors
    assert 0.04 < time.monotonic()-begin < 2
    assert read_json(head) == record
    assert unpack(record) == dict(sequence=1)
    assert list(tmp_path.iterdir()) == [head]


@pytest.mark.skipif(os.name != 'nt', reason='actual Windows sharing regression')
def test_persistent_actual_reader_is_bounded_and_old_head_is_preserved(tmp_path):
    head = tmp_path/'head.json'
    budget._publish(head, dict(sequence=0))
    before = file_hash(head)
    close = _held_reader(head)
    begin = time.monotonic()
    try:
        with pytest.raises(PermissionError) as caught:
            budget._publish(head, dict(sequence=1), replace=True)
        assert caught.value.winerror == 5
        assert budget.HEAD_REPLACE_RETRY_SECONDS <= time.monotonic()-begin < 2
        assert file_hash(head) == before
        assert list(tmp_path.iterdir()) == [head]
    finally:
        close()


@pytest.mark.skipif(os.name != 'nt', reason='Windows transient-error policy')
def test_transient_replace_retries_same_flushed_bytes_only(tmp_path, monkeypatch):
    head = tmp_path/'head.json'
    budget._publish(head, dict(sequence=0))
    real_replace = os.replace
    attempts = []
    def replace(source, target):
        attempts.append((source, target, file_hash(source)))
        if len(attempts) < 3:
            raise _denial(32)
        return real_replace(source, target)
    monkeypatch.setattr(budget.os, 'replace', replace)
    expected = budget._publish(head, dict(sequence=1), replace=True)
    assert len(attempts) == 3 and len(set(attempts)) == 1
    assert read_json(head) == expected
    assert list(tmp_path.iterdir()) == [head]


@pytest.mark.parametrize('error', [OSError(errno.ENOSPC, 'fixture full disk'), _denial(2)])
def test_other_failures_propagate_without_retry_or_overwriting_head(tmp_path, monkeypatch, error):
    head = tmp_path/'head.json'
    budget._publish(head, dict(sequence=0))
    before = file_hash(head)
    attempts = []
    def replace(*args):
        attempts.append(args)
        raise error
    monkeypatch.setattr(budget.os, 'replace', replace)
    with pytest.raises(type(error)) as caught:
        budget._publish(head, dict(sequence=1), replace=True)
    assert caught.value is error and len(attempts) == 1
    assert file_hash(head) == before
    assert list(tmp_path.iterdir()) == [head]


def test_immutable_publications_are_not_replaced_or_retried(tmp_path, monkeypatch):
    path = tmp_path/'event.json'
    budget._publish(path, dict(sequence=0))
    def forbidden(*args):
        raise AssertionError('immutable events must not use replacement')
    monkeypatch.setattr(budget.os, 'replace', forbidden)
    with pytest.raises(FileExistsError):
        budget._publish(path, dict(sequence=1))
    assert unpack(read_json(path)) == dict(sequence=0)
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.skipif(os.name != 'nt', reason='Windows transient-error policy')
def test_head_retry_never_duplicates_work_event_attempt_or_generation_debit(tmp_path, monkeypatch):
    directory = tmp_path/'ledger'
    contract = dict(schema_version=budget.VERSION,
        **{name:digest('SOFTWARE-NOT-AUTHORITY-'+name) for name in
           ('protocol_sha256','execution_sha256','matrix_sha256','runtime_manifest_sha256','approval_sha256')},
        ledger_directory=str(directory.resolve()), phase_caps_ns={'forecast':1_000_000}, total_cap_ns=1_000_000,
        max_generated_forecasts=1, max_attempts_per_item=1,
        workloads=[dict(work_id=digest('fixture-work'), phase='forecast', max_active_ns=500_000, generated_forecasts=1)])
    with budget.Ledger.create(directory, contract) as ledger:
        real_replace = os.replace
        attempts = []
        def replace(source, target):
            attempts.append((source,target))
            if len(attempts) < 3:
                raise _denial()
            return real_replace(source,target)
        monkeypatch.setattr(budget.os, 'replace', replace)
        reservation = ledger.reserve(digest('fixture-work'))
        summary = ledger.summary()
        assert len(attempts) == 3 and summary['event_count'] == 1
        assert summary['attempted_work_items'] == summary['generated_forecasts_reserved'] == 1
        assert summary['charged_ns_by_phase']['forecast'] == 500_000
        assert summary['pending']['reservation_sha256'] == reservation['reservation_sha256']
        assert unpack(read_json(directory/'head.json')) == ledger.tip
        assert len(list((directory/'events').iterdir())) == 1
        assert not ledger._poisoned
