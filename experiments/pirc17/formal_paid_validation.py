"""Legacy boundary checks, without profiling or reentrant credit callbacks.

The old sys.setprofile pulse reentered the controller's non-reentrant meter
lock. The checkpoint runner instead uses an external process deadline and
does not use this legacy controller/credit transport at all.
"""
import time


def paid_call(function, *args, _tick, _clock=time.monotonic_ns, **kwargs):
    """Check only at boundaries; preserve the operation and installed profiler."""
    if not callable(function) or not callable(_tick) or not callable(_clock):
        raise ValueError('callable validator, original credit callback and clock required')
    _tick()
    started = _clock()
    result = function(*args, **kwargs)
    if _clock() < started:
        raise ValueError('paid validation clock moved backwards')
    _tick()
    return result
