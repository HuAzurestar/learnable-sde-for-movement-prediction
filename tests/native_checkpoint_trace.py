"""Closed, bounded test-only observations; never replaces native execution.

Only scalar timing/phase/presence/exit facts are retained in memory. No frame,
path, identity, scientific state, return payload or exception message is copied.
The caller emits a report only after an actual original assertion fails.
"""

from collections import deque
import math
import time


OPERATIONS = frozenset({"request", "response", "ack", "native-stop", "close",
                        "journal-request", "journal-save", "journal-stop"})


def finite(value):
    if type(value) not in (int, float) or abs(value) > 1e12 or not math.isfinite(value):
        return None
    return value


class NativeCheckpointTrace:
    LIMIT = 32

    def __init__(self, *, clock=time.monotonic):
        self.clock = clock
        self.rows = deque(maxlen=self.LIMIT)

    def call(self, operation, callback, *, deadline=None, native_code=None,
             frame=False, omit_empty=False):
        if operation not in OPERATIONS:
            raise ValueError("closed native diagnostic operation required")
        started = self.clock()
        result, raised = None, False
        try:
            result = callback()
            return result
        except BaseException:
            raised = True
            raise  # Same original exception, never a replacement success.
        finally:
            ended = self.clock()
            if raised or not (omit_empty and result is None):
                first, last, limit = finite(started), finite(ended), finite(deadline)
                duration = (last - first) * 1000 if first is not None and last is not None else None
                remaining = (limit - last) * 1000 if limit is not None and last is not None else None
                try:
                    code = native_code() if native_code is not None else None
                except Exception:
                    code = None  # Optional observation failure cannot mask the callback.
                self.rows.append({
                    "operation": operation,
                    "duration_ms": finite(duration),
                    "remaining_ms": finite(remaining),
                    "native_code": code if type(code) is int and abs(code) <= 2**31 else None,
                    "frame_present": (result is not None) if frame and not raised else None,
                    "raised": raised,
                })

    def report(self):
        return {"schema": "native-checkpoint-test-trace-v1",
                "observations": [dict(row) for row in self.rows]}
