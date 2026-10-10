"""Read the shared owner's actual process/reservation facts; never fund work.

Restart-only methods receive no optimizer checkpoint channel. They still need
proof of the actual shared wrapper parent, running attempt and held reservation.
This is a trusted-worker check, not a second runtime, grant or OS sandbox.
"""

import math
import os
from pathlib import Path
import time

from infrastructure.research_control import read_frame
from infrastructure.research_store import ResearchError, ResearchStore


class ManagedAttempt:
    def __init__(self, output, attempt_id, parent_pid, deadline, reserved_ms):
        self.directory = Path(output).parent
        self.attempt_id, self.parent_pid, self.deadline = attempt_id, parent_pid, deadline
        self.soft_threshold = deadline - reserved_ms / 1000 * .2

    def requested(self):
        return os.getppid() != self.parent_pid or time.monotonic() >= self.soft_threshold


def require_owned_worker(output):
    output = Path(output).absolute()
    if (len(output.parents) < 4 or output.name != "result.json" or output.parents[1].name != "artifacts"
            or output.parents[2].name != "pirc25" or not output.parent.name.startswith(".attempt-")):
        raise ResearchError("UNAUTHORIZED_DATA", "worker output is not a shared owner attempt")
    identity = read_frame(output.parents[2] / "store.json", 16384)
    if type(identity) is not dict or type(identity.get("store_id")) is not str:
        raise ResearchError("UNAUTHORIZED_DATA", "existing owner store identity is required")
    store = ResearchStore(output.parents[3], identity["store_id"])
    attempt_id = output.parent.name.removeprefix(".attempt-")
    with store._read_transaction():
        attempt = store._attempts().get(attempt_id)
        events = store._events()
        workers = [e["payload"] for e in events if e["event_kind"] == "WORKER_STARTED"
                   and e["payload"].get("attempt_id") == attempt_id]
        if not attempt or attempt["state"] != "RUNNING" or len(workers) != 1:
            raise ResearchError("UNAUTHORIZED_DATA", "actual shared running worker is required")
        worker = workers[0]
        holds = [e["payload"] for e in events if e["event_kind"] == "RESERVE"
                 and e["payload"].get("reservation_id") == worker["reservation_id"]]
        settled = any(e["event_kind"] == "SETTLE" and e["payload"].get("reservation_id") == worker["reservation_id"] for e in events)
        deadline = worker.get("deadline_monotonic")
        if (worker.get("pid") != os.getppid() or len(holds) != 1 or settled
                or holds[0].get("attempt_id") != attempt_id or holds[0].get("run_id") != attempt["run_id"]
                or type(holds[0].get("reserved_ms")) is not int or holds[0]["reserved_ms"] <= 0
                or type(deadline) not in (int, float) or not math.isfinite(deadline) or time.monotonic() >= deadline):
            raise ResearchError("UNAUTHORIZED_DATA", "actual funded wrapper parent/deadline differs")
    return ManagedAttempt(output, attempt_id, worker["pid"], deadline, holds[0]["reserved_ms"])
