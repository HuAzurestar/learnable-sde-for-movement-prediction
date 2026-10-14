"""One-time Windows deployment handoff, NOT a predictor synchronization lock.

The legacy live controller has no drain command. Pause only its verified main
thread while an ALREADY dispatched forecast finishes in its separate child.
After its successful durable receipt, retire only that owned controller/tree.
Recover the one receipt normally; no successful rerun or failed extra attempt.
On any pre-retirement error the legacy main thread is always resumed.
"""
import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import time

import psutil

from .checkpoint_resume import load, status
from .checkpoint_state import NS, Progress, atomic_json, owner_alive, single_writer
from .protocol_core import read_json


class MainThreadPause:
    """Hold one exact OS thread handle; never guess a process by its name."""
    def __init__(self, process):
        if os.name != 'nt':
            raise RuntimeError('this one-time handoff is Windows-only')
        self.api = ctypes.WinDLL('kernel32', use_last_error=True)
        self.api.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.api.OpenThread.restype = wintypes.HANDLE
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.api.CloseHandle.restype = wintypes.BOOL
        self.api.GetThreadTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)]*4)]
        self.api.GetThreadTimes.restype = wintypes.BOOL
        self.api.GetProcessIdOfThread.argtypes = [wintypes.HANDLE]
        self.api.GetProcessIdOfThread.restype = wintypes.DWORD
        for name in ('SuspendThread','ResumeThread'):
            method = getattr(self.api, name)
            method.argtypes = [wintypes.HANDLE]
            method.restype = wintypes.DWORD
        threads = []
        for thread in process.threads():
            handle = self.api.OpenThread(0x0800, False, thread.id)
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                stamp = self._created(handle)
                threads.append((stamp, thread.id))
            finally:
                self.api.CloseHandle(handle)
        threads.sort()
        if not threads or (len(threads)>1 and threads[0][0] == threads[1][0]):
            raise RuntimeError('unique original main thread cannot be identified')
        created, self.thread_id = threads[0]
        if abs(created-process.create_time()) > .002:
            raise RuntimeError('oldest thread is not the original process main thread')
        self.handle = self.api.OpenThread(0x0802, False, self.thread_id)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self.suspended = False
        if self.api.GetProcessIdOfThread(self.handle) != process.pid or self._created(self.handle) != created:
            self.close()
            raise RuntimeError('main thread identity changed')

    def _created(self, handle):
        values = [wintypes.FILETIME() for _ in range(4)]
        if not self.api.GetThreadTimes(handle, *(ctypes.byref(v) for v in values)):
            raise ctypes.WinError(ctypes.get_last_error())
        return ((values[0].dwHighDateTime<<32)|values[0].dwLowDateTime)/10_000_000-11644473600

    def suspend(self):
        count = self.api.SuspendThread(self.handle)
        if count == 0xffffffff:
            raise ctypes.WinError(ctypes.get_last_error())
        self.suspended = True
        if count != 0:
            self.resume()
            raise RuntimeError('main thread was already externally suspended; not taking over')

    def resume(self):
        if self.suspended:
            count = self.api.ResumeThread(self.handle)
            if count == 0xffffffff:
                raise ctypes.WinError(ctypes.get_last_error())
            self.suspended = False

    def close(self):
        if getattr(self, 'handle', None):
            self.api.CloseHandle(self.handle)
            self.handle = None


def inflight(directory, session, owner):
    """Called ONLY after pause: require a published request and absent reply."""
    if read_json(directory/'writer.json') != owner:
        raise RuntimeError('writer identity changed')
    progress = read_json(directory/'progress.json')
    pending = progress['pending']
    if not pending or not pending['work_id'] or progress.get('parallel_pending'):
        raise RuntimeError('legacy runner is not waiting for one dispatched forecast')
    paths = list((session/'requests').glob('??????.json'))  # request filenames only; no old outputs opened.
    if not paths:
        raise RuntimeError('no dispatched request')
    request = read_json(max(paths, key=lambda p:p.name))
    output = directory/'outputs'/pending['work_id']
    if (request['work_id'] != pending['work_id'] or Path(request['output_directory']).resolve() != output.resolve()
            or Path(pending['reply']).resolve() != (output/'reply.json').resolve()
            or Path(pending['reply']).exists()):
        raise RuntimeError('not a safe already-dispatched/unfinished forecast handoff point')
    return progress


def handoff(directory, *, expected_pid, session, timeout_seconds=600, _pause_type=MainThreadPause):
    directory, session = Path(directory).resolve(), Path(session).resolve()
    if session.parent != directory/'sessions' or timeout_seconds <= 0:
        raise ValueError('exact owned session and positive handoff timeout required')
    owner = read_json(directory/'writer.json')
    if owner['pid'] != expected_pid or not owner_alive(owner) or read_json(session/'run.json')['owner'] != owner:
        raise RuntimeError('live legacy owner/session does not match requested target')
    process = psutil.Process(expected_pid)
    command = process.cmdline()
    if ('experiments.pirc17.checkpoint_resume' not in command or 'run' not in command
            or '--directory' not in command
            or Path(command[command.index('--directory')+1]).resolve() != directory):
        raise RuntimeError('target is not this checkpoint legacy controller')
    if (session/'parallel-handoff.json').exists() or (session/'parallel-handoff-request.json').exists():
        raise FileExistsError('this session handoff is already recorded; inspect it, do not repeat')
    descendants = process.children(recursive=True)
    identities = [dict(pid=p.pid, created=p.create_time()) for p in descendants]
    pause = _pause_type(process)
    retired = False
    started = time.monotonic_ns()
    try:
        pause.suspend()
        frozen = inflight(directory, session, owner)
        pending = frozen['pending']
        reply = Path(pending['reply'])
        next_report = started
        while not reply.is_file():
            now = time.monotonic_ns()
            if now-started > timeout_seconds*NS:
                raise TimeoutError('handoff abandoned; original controller will resume')
            if not owner_alive(owner):
                raise RuntimeError('original owner exited before receipt')
            if now >= next_report:
                print(json.dumps(dict(event='handoff_waiting_for_original_forecast',
                    work_id=pending['work_id'], controller_pid=expected_pid,
                    controller_main_thread=pause.thread_id, wait_seconds=(now-started)/NS)), flush=True)
                next_report = now+30*NS
            time.sleep(.1)
        message = read_json(reply)
        if message['work_id'] != pending['work_id'] or message.get('result', {}).get('status') != 'success':
            raise RuntimeError('no successful receipt: original controller will resume')
        if read_json(directory/'progress.json') != frozen or read_json(directory/'writer.json') != owner:
            raise RuntimeError('checkpoint changed while controller was paused')
        request = dict(old_owner=owner, session=str(session), work_id=pending['work_id'],
            original_failure_count=len(frozen['failures']), completed_receipt_preserved=True,
            old_cost_scope='unknown original elapsed retains original reservation; no refund/reset',
            descendants=identities)
        atomic_json(session/'parallel-handoff-request.json', request)
        process.kill()  # ONLY the verified owned supervisor; its existing Job closes its idle children.
        retired = True
        gone, alive = psutil.wait_procs([process,*descendants], timeout=20)
        if alive:
            raise RuntimeError(f'old owned tree did not close: {[p.pid for p in alive]}')
        # No writer runs now. Reclaim the dead marker and settle ONE saved
        # original success normally. Do not mutate a live writer's snapshot.
        with single_writer(directory):
            settings, imported = load(directory)
            progress = Progress(directory, settings=settings, imported_ids=imported)
            progress.recover_pending()
            if (progress.value['failures'] != frozen['failures']
                    or pending['work_id'] not in progress.value['completed']):
                raise RuntimeError('handoff did not preserve original successful work/failures')
            progress.charge('input_qualification_and_binding', time.monotonic_ns()-started)
            receipt = dict(**request, old_process_tree_closed=True,
                observed_closed_pids=[p.pid for p in gone], status=status(progress),
                helper_elapsed_seconds=(time.monotonic_ns()-started)/NS)
            atomic_json(session/'parallel-handoff.json', receipt)
        print(json.dumps(dict(event='parallel_handoff_complete', **receipt)), flush=True)
        return receipt
    finally:
        try:
            if not retired and owner_alive(owner):
                pause.resume()
        finally:
            pause.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--expected-pid', type=int, required=True)
    parser.add_argument('--timeout-seconds', type=float, default=600)
    args = parser.parse_args(argv)
    handoff(args.directory, expected_pid=args.expected_pid, session=args.session, timeout_seconds=args.timeout_seconds)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
