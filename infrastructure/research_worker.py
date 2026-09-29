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
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL)

    def parent_watch():
        if sys.stdin.read(1) == "":
            if os.name != "nt":
                os.killpg(os.getpgrp(), signal.SIGKILL)
            else:
                child.kill()  # Job-object handle closure also kills descendants.

    threading.Thread(target=parent_watch, daemon=True).start()
    while child.poll() is None:
        if time.monotonic() >= deadline:
            if os.name != "nt":
                os.killpg(os.getpgrp(), signal.SIGKILL)
            subprocess.run(["taskkill", "/PID", str(os.getpid()), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=subprocess.CREATE_NO_WINDOW)
            child.kill()
            return 124
        Path(heartbeat).write_text(json.dumps({"monotonic": time.monotonic(), "pid": child.pid}))
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    return child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
