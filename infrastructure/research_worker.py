"""Internal stdlib-only process wrapper; waits for containment before execution."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


def main():
    heartbeat, deadline, *command = sys.argv[1:]
    deadline = float(deadline)
    # Parent assigns the Windows Job Object before releasing this barrier.
    if sys.stdin.readline().strip() != "GO":
        return 125
    job = None
    if os.name == "nt":
        # Imported from this script's own directory, never package initializers.
        # Assign self before Popen: no child can race out of the nested job.
        from process_tree import ProcessTree
        job = ProcessTree.contain_current_process()

    def stop():
        if job is not None:
            job.terminate()
        else:
            os.killpg(os.getpgrp(), signal.SIGKILL)

    # Independent of heartbeat writes, child polling and the owner's threads.
    # All startup stays inside the original absolute deadline.
    watchdog = threading.Timer(max(0, deadline - time.monotonic()), stop)
    watchdog.daemon = True
    watchdog.start()
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL)

    def parent_watch():
        if sys.stdin.read(1) == "":
            stop()

    threading.Thread(target=parent_watch, daemon=True).start()
    while child.poll() is None:
        if time.monotonic() >= deadline:
            stop()
            return 124
        Path(heartbeat).write_text(json.dumps({"monotonic": time.monotonic(), "pid": child.pid}))
        # Child exit must wake the wrapper immediately, including checkpoint
        # exit 85. A heartbeat sleep needlessly consumes its remaining margin.
        # Keep the same heartbeat cadence and absolute, independently fused
        # deadline; a timeout merely means the next heartbeat is due.
        try:
            child.wait(timeout=min(0.1, max(0, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            pass
    watchdog.cancel()
    # Do not explicitly close the self-containing job before returning: that
    # would terminate this wrapper without preserving the child's exit code.
    # Process exit closes its last handle and kills leftover descendants.
    return child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
