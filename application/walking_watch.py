"""Conservative change detector, never proof of journal validity.

Only use between two full physical validations. Directory/head sameness cannot
prove historical contents unchanged. A stall or read failure requires native
stop before the next authoritative read.
"""
import stat
import time

from infrastructure.research_store import ResearchError, file_lock
from .walking_stage import regular


class WalkingWatch:
    interval = .025
    stall_seconds = .250

    def __init__(self, path, monotonic=time.monotonic):
        self.path = path
        self.clock = monotonic
        self.previous = self.clock()
        self.baseline = self.snapshot(check_writer=False)
        self.previous = self.clock()

    def snapshot(self, check_writer=True):
        # Probe the real OS writer mutex; never hold it during computation.
        from contextlib import nullcontext
        with file_lock(self.path / ".writer.lock", timeout=0) if check_writer else nullcontext():
            hold = self.path / "recovery-hold.json"
            try:
                hold.lstat()
            except FileNotFoundError:
                pass
            else:
                raise ResearchError("RECOVERY_REQUIRED", "walking change detector observed hold")
            head = self.path / "head.json"
            content = regular(head, 4096)
            def identity(path, directory=False):
                info = path.lstat()
                if (getattr(info, "st_file_attributes", 0) & 1024 or stat.S_ISLNK(info.st_mode)
                        or directory and not stat.S_ISDIR(info.st_mode)):
                    raise ResearchError("RECOVERY_REQUIRED", "walking observation path changed")
                return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns
            return content, identity(head), identity(self.path / "events", True)

    def check(self):
        before = self.clock()
        if before - self.previous > self.stall_seconds:
            raise ResearchError("RECOVERY_REQUIRED", "walking observation stalled")
        current = self.snapshot()
        now = self.clock()
        if current != self.baseline or now - before > self.stall_seconds:
            raise ResearchError("RECOVERY_REQUIRED", "walking authority changed during computation")
        self.previous = now
