"""Test-only bounded helper protocol and exact Windows descendant identity.

No product imports. Importing this module performs no native operations.
"""
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

HISTORY_SHA256 = 'bb1b9fe99f89f841090d632f884a62ee830378a539a936c2884ca2e0c7c74df2'
HISTORY_BYTES = 7797


def historical_source(path=None):
    path = path or Path(__file__).with_name('fixtures') / 'walking_budget_2f3b15c.py.txt'
    try:
        with Path(path).open('rb') as stream:
            raw = stream.read(HISTORY_BYTES + 1)
    except OSError as exc:
        raise AssertionError('historical negative-control fixture unavailable') from exc
    if len(raw) != HISTORY_BYTES or hashlib.sha256(raw).hexdigest() != HISTORY_SHA256:
        raise AssertionError('historical negative-control fixture identity mismatch')
    return raw.decode('utf-8')


def creation_receipt(raw, nonce):
    """Pure bounded identity decoding; unknown or ambiguous receipts fail."""
    def unique(pairs):
        row={}
        for key,value in pairs:
            if key in row:
                raise AssertionError('duplicate descendant receipt field')
            row[key]=value
        return row
    if len(raw)>512:
        raise AssertionError('creation receipt quota')
    try:
        row=json.loads(raw,object_pairs_hook=unique)
    except (ValueError,UnicodeError) as exc:
        raise AssertionError('malformed creation receipt') from exc
    if (type(row) is not dict or set(row)!={'pid','birth','nonce'} or row['nonce']!=nonce
            or type(row['pid']) is not int or not 1<row['pid']<2**32
            or type(row['birth']) is not int or not 0<row['birth']<2**64):
        raise AssertionError('creation receipt identity unknown')
    return row


class HelperProtocol:
    """One absolute operational deadline, including spawn and all handshakes.

    Readers drain both pipes continuously with bounded retained bytes/messages.
    Failure cleanup has a separate bounded reap allowance, never compute grace.
    Only use with owned helpers that do not delegate pipe handles.
    """
    def __init__(self, command, deadline, **kwargs):
        self.deadline = deadline
        self.lines = queue.Queue(maxsize=16)
        self.errors = queue.Queue(maxsize=1)
        self.stderr = bytearray()
        self.threads = []
        self.remaining()
        self.child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, bufsize=0,
                                      creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                                      **kwargs)
        try:
            for name, stream in [('stdout', self.child.stdout), ('stderr', self.child.stderr)]:
                thread = threading.Thread(target=self._drain, args=(name, stream), name='walking-harness-'+name)
                self.threads.append(thread)
                thread.start()
            self.remaining()
        except BaseException:
            self.close()
            raise

    def remaining(self):
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise AssertionError('helper absolute deadline expired')
        return left

    def _error(self, message):
        try:
            self.errors.put_nowait(message)
        except queue.Full:
            pass

    def _drain(self, name, stream):
        pending = bytearray()
        total = 0
        try:
            while True:
                chunk = os.read(stream.fileno(), 4096)
                if not chunk:
                    if name == 'stdout':
                        if pending:
                            self._error('helper partial line at EOF')
                        self.lines.put_nowait(None)
                    return
                if name == 'stderr':
                    self.stderr.extend(chunk[:max(0, 4096-len(self.stderr))])
                    continue
                total += len(chunk)
                if total > 65536:
                    self._error('helper stdout quota exceeded')
                    continue
                pending.extend(chunk)
                while b'\n' in pending:
                    line, _, rest = pending.partition(b'\n')
                    pending = bytearray(rest)
                    if len(line) > 128:
                        self._error('helper line quota exceeded')
                    else:
                        try:
                            self.lines.put_nowait(bytes(line).rstrip(b'\r'))
                        except queue.Full:
                            self._error('helper message quota exceeded')
                if len(pending) > 128:
                    self._error('helper partial line quota exceeded')
                    pending.clear()
        except BaseException as exc:
            self._error('helper reader failed: '+type(exc).__name__)

    def _check(self):
        self.remaining()
        if not self.errors.empty():
            raise AssertionError(self.errors.get_nowait())

    def expect(self, token):
        while True:
            self._check()
            try:
                line = self.lines.get(timeout=min(.02, self.remaining()))
            except queue.Empty:
                continue
            self._check()
            if line != token.encode('ascii'):
                raise AssertionError('helper unexpected token or EOF: '+repr(line))
            return

    def go(self):
        self._check()
        # Sole writer sends only these three bytes to the initially empty pipe.
        if self.child.stdin.write(b'GO\n') != 3:
            raise AssertionError('helper incomplete GO')
        self.child.stdin.close()
        self._check()

    def finish(self):
        self.expect('DONE')
        self.child.wait(timeout=self.remaining())
        self._join(self.deadline)
        self._check()
        assert self.child.returncode == 0, bytes(self.stderr)

    def _join(self, until):
        for thread in self.threads:
            if thread.ident is not None:
                thread.join(timeout=max(0, until-time.monotonic()))
        if any(t.is_alive() for t in self.threads):
            raise AssertionError('helper reader failed to stop')

    def close(self):
        # Reap even if the deadline or a token failed, before closing read pipes.
        until = time.monotonic()+3
        try:
            if self.child.poll() is None:
                self.child.kill()
            self.child.wait(timeout=max(.001, until-time.monotonic()))
            self._join(until)
        finally:
            for stream in (self.child.stdin, self.child.stdout, self.child.stderr):
                if not stream.closed:
                    stream.close()


class ExactWindowsProcess:
    """Retain the exact kernel handle; PID reuse never establishes absence."""
    def __init__(self, pid, birth=None):
        import ctypes
        from ctypes import wintypes
        if os.name != 'nt' or type(pid) is not int or not 1 < pid < 2**32:
            raise AssertionError('invalid descendant PID or platform')
        self.ctypes = ctypes
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        k = self.kernel
        k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.OpenProcess.restype = wintypes.HANDLE
        k.GetProcessTimes.argtypes = [wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
        k.GetProcessTimes.restype = wintypes.BOOL
        k.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k.WaitForSingleObject.restype = wintypes.DWORD
        k.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k.TerminateProcess.restype = wintypes.BOOL
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        k.CloseHandle.restype = wintypes.BOOL
        self.handle = k.OpenProcess(0x100000 | 0x1000 | 1, False, pid)
        if not self.handle:
            raise AssertionError('descendant identity unavailable/access denied')
        try:
            times = [wintypes.FILETIME() for _ in range(4)]
            if not k.GetProcessTimes(self.handle, *(ctypes.byref(t) for t in times)):
                raise AssertionError('descendant birth unknown')
            self.birth = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
            self.pid = pid
            if not self.birth or (birth is not None and (type(birth) is not int or self.birth != birth)):
                raise AssertionError('descendant birth mismatch/reused PID')
            if self.state() != 258:
                raise AssertionError('descendant not observed alive at creation')
        except BaseException:
            self.close()
            raise

    def state(self):
        value = self.kernel.WaitForSingleObject(self.handle, 0)
        if value not in (0, 258):
            raise AssertionError('descendant state unknown')
        return value

    def close(self):
        if self.handle:
            if not self.kernel.CloseHandle(self.handle):
                raise AssertionError('descendant handle close failed')
            self.handle = None

    def cleanup(self):
        try:
            if self.state() == 258:
                if not self.kernel.TerminateProcess(self.handle, 124):
                    raise AssertionError('owned descendant cleanup failed')
                if self.kernel.WaitForSingleObject(self.handle, 2000) != 0:
                    raise AssertionError('owned descendant cleanup unconfirmed')
        finally:
            self.close()
