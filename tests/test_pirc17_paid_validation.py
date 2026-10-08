"""Regression for the actual profile -> meter.guard self-deadlock."""
import sys
import threading

import pytest

from experiments.pirc17.formal_controller import _Meter
from experiments.pirc17.formal_paid_validation import paid_call


def test_clock_boundary_under_meter_lock_does_not_reenter_tick():
    now, calls, armed = [0], [], [False]
    def clock():
        if armed[0]:
            now[0] = 1_000_000_000
        return now[0]
    meter = _Meter(0, 100_000_000_000, 30_000_000_000, clock,
                   lambda reason: None, lambda: 8*1024**3, lambda: False)
    def tick():
        calls.append(threading.get_ident())
        meter.guard()
    def operation():
        armed[0] = True
        meter.guard()  # Clock returns while its non-reentrant lock is held.
        return 'completed'
    result, errors = [], []
    def run():
        try:
            result.append(paid_call(operation, _tick=tick, _clock=clock))
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(2)
    assert not thread.is_alive(), 'self-deadlock reproduced'
    assert result == ['completed'] and not errors and len(calls) == 2
    assert not meter._lock.locked()


def test_only_boundary_checks_and_existing_profiler_unchanged():
    events = []
    profile = lambda *args: None
    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        def operation():
            assert sys.getprofile() is profile
            events.append('operation')
            return 42
        assert paid_call(operation, _tick=lambda: events.append('tick')) == 42
        assert events == ['tick', 'operation', 'tick']
        assert sys.getprofile() is profile
    finally:
        sys.setprofile(previous)


def test_domain_failure_not_suppressed():
    events = []
    def reject():
        raise ValueError('domain rejected')
    with pytest.raises(ValueError, match='domain rejected'):
        paid_call(reject, _tick=lambda: events.append('tick'))
    assert events == ['tick']


def test_boundary_failure_and_backwards_clock():
    with pytest.raises(RuntimeError, match='stop'):
        paid_call(lambda: pytest.fail('must not enter'),
                  _tick=lambda: (_ for _ in ()).throw(RuntimeError('stop')))
    times = iter([2, 1])
    with pytest.raises(ValueError, match='backwards'):
        paid_call(lambda: None, _tick=lambda: None, _clock=lambda: next(times))


@pytest.mark.parametrize('field', ['function', '_tick', '_clock'])
def test_invalid_callbacks(field):
    values = dict(function=lambda: None, _tick=lambda: None, _clock=lambda: 0)
    values[field] = None
    with pytest.raises(ValueError, match='callable'):
        paid_call(**values)
