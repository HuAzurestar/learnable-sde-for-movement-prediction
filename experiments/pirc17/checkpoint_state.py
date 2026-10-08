"""One experiment owner, one atomic progress snapshot; no thread/file waits.

Completed outputs are indexed, not copied or scanned at restart. A killed
attempt keeps its reserved cost and is never silently retried or marked done.
The writer marker only rejects simultaneous duplicate runners; dead owners
are reclaimed immediately using PID AND creation time (PID reuse is normal).
"""
from contextlib import contextmanager
import os
from pathlib import Path
import time
import uuid

import psutil

from .protocol_core import canonical, read_json, sha256

NS = 1_000_000_000


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(canonical(value)+b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(21):
            try:
                os.replace(temporary, path)
                break
            except PermissionError as exc:
                # Windows readers/antivirus can briefly deny atomic replace.
                # Bounded IO retry only (<=200ms), never a lock or recovery
                # scan. A genuine ACL/write failure still propagates.
                if getattr(exc, 'winerror', None) not in {5, 32, 33} or attempt == 20:
                    raise
                time.sleep(.01)
    finally:
        if temporary.exists():
            temporary.unlink()  # Only our exact temporary file, never outputs.


def owner_alive(owner):
    try:
        process = psutil.Process(owner['pid'])
        return process.create_time() == owner['created'] and process.is_running()
    except psutil.NoSuchProcess:
        return False


@contextmanager
def single_writer(directory):
    """Fail immediately for a live owner; no locks, watchdog threads or wait."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory/'writer.json'
    owner = dict(pid=os.getpid(), created=psutil.Process().create_time(), token=uuid.uuid4().hex)
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        # An empty marker can only be from a crash during these few bytes.
        # Don't guess whether it is stale; operator can remove that ONE marker.
        previous = marker.read_bytes()
        old = read_json(marker)
        if owner_alive(old):
            raise RuntimeError('checkpoint already has a live runner')
        if marker.read_bytes() != previous:
            raise RuntimeError('checkpoint owner changed during reclamation')
        marker.unlink()
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(canonical(owner))
            stream.flush()
            os.fsync(stream.fileno())
        yield owner
    finally:
        if marker.exists() and read_json(marker) == owner:
            marker.unlink()


class Progress:
    def __init__(self, directory, *, settings, imported_ids):
        self.directory = Path(directory)
        self.path = self.directory/'progress.json'
        self.settings = settings
        self.imported_ids = frozenset(imported_ids)
        self.work = {w['work_id']: w for w in settings['workloads']}
        if self.path.exists():
            self.value = read_json(self.path)
            if self.value['settings_id'] != settings['settings_id']:
                raise ValueError('checkpoint settings changed')
        else:
            self.value = dict(settings_id=settings['settings_id'], completed={}, attempted=[],
                charged_ns_by_phase=dict(settings['charged_ns_by_phase']),
                generated=settings['generated'], pending=None, failures={})
            self.save()

    def save(self):
        atomic_json(self.path, self.value)

    def remaining_ns(self, phase):
        charged = self.value['charged_ns_by_phase']
        return max(0, min(self.settings['phase_caps_ns'][phase]-charged[phase],
                          self.settings['total_cap_ns']-sum(charged.values())))

    def charge(self, phase, elapsed_ns):
        """Controller IO/closure also costs time; record overrun, never hide it."""
        if type(elapsed_ns) is not int or elapsed_ns < 0:
            raise ValueError('nonnegative measured elapsed time required')
        self.value['charged_ns_by_phase'][phase] += elapsed_ns
        self.save()

    def missing(self, *, kinds):
        done = self.imported_ids | self.value['completed'].keys() | set(self.value['attempted'])
        return [w for w in self.work.values() if w['kind'] in kinds and w['work_id'] not in done]

    def reserve(self, *, phase, maximum_ns, work_id=None, reply=None, ignore_time_budgets=False):
        if self.value['pending'] is not None:
            raise RuntimeError('unfinished attempt must be accounted before dispatch')
        self.value['pending'] = self._reservation(phase=phase, maximum_ns=maximum_ns,
            work_id=work_id, reply=reply, ignore_time_budgets=ignore_time_budgets)
        self.save()  # Durable BEFORE worker sees the request.

    def _reservation(self, *, phase, maximum_ns, work_id, reply, ignore_time_budgets):
        if (type(maximum_ns) is not int or maximum_ns <= 0
                or (not ignore_time_budgets and maximum_ns > self.remaining_ns(phase))):
            raise TimeoutError('insufficient remaining cumulative phase/total budget')
        if work_id is not None:
            sha256(work_id)
            work = self.work[work_id]
            if (phase != work['phase'] or maximum_ns > int(work['max_active_seconds']*NS)
                    or work_id in self.imported_ids or work_id in self.value['attempted']):
                raise ValueError('already completed/attempted or changed work budget')
            generation = work['generated_forecasts']
            if self.value['generated']+generation > self.settings['generation_limit']:
                raise TimeoutError('cumulative generation allowance exhausted')
            self.value['attempted'].append(work_id)
            self.value['generated'] += generation
        self.value['charged_ns_by_phase'][phase] += maximum_ns
        return dict(phase=phase, work_id=work_id, reserved_ns=maximum_ns, reply=reply)

    def reserve_lane(self, lane, **kwargs):
        """ONE supervisor writes the snapshot; no locks inside predictors."""
        if lane not in {'0', '1'}:
            raise ValueError('only the two explicitly authorized lanes are supported')
        pending = self.value.setdefault('parallel_pending', {})
        if lane in pending or (self.value['pending'] is not None and not pending):
            raise RuntimeError('unfinished attempt in lane')
        pending[lane] = self._reservation(**kwargs)
        self.value['pending'] = next(iter(pending.values()))
        self.save()

    def settle(self, *, elapsed_ns=None, result=None, failure=None):
        if self.value.get('parallel_pending'):
            raise RuntimeError('settle the specific parallel lane, not its representative')
        pending = self.value['pending']
        if pending is None:
            raise RuntimeError('no pending attempt')
        self._settlement(pending, elapsed_ns=elapsed_ns, result=result, failure=failure)
        self.value['pending'] = None
        self.save()

    def _settlement(self, pending, *, elapsed_ns, result, failure):
        # Only a measured, normally observed closure releases unused reserve.
        # After supervisor death retain the whole reservation conservatively.
        if elapsed_ns is not None:
            if type(elapsed_ns) is not int or elapsed_ns < 0:
                raise ValueError('nonnegative measured elapsed time required')
            self.value['charged_ns_by_phase'][pending['phase']] += elapsed_ns-pending['reserved_ns']
        wid = pending['work_id']
        if wid is not None:
            if elapsed_ns is not None:
                recent = self.value.setdefault('recent_work', [])
                work = self.work[wid]
                recent.append(dict(phase=work['phase'], matrix=work['matrix'], kind=work['kind'],
                    elapsed_ns=elapsed_ns, success=result is not None and result.get('status') == 'success'))
                del recent[:-100]
            if result is not None and result.get('status') == 'success':
                self.value['completed'][wid] = result
            else:
                self.value['failures'][wid] = failure or result or 'interrupted; no silent retry'

    def settle_lane(self, lane, *, elapsed_ns=None, result=None, failure=None):
        pending = self.value.get('parallel_pending', {})
        if lane not in pending:
            raise RuntimeError('no pending attempt in lane')
        self._settlement(pending[lane], elapsed_ns=elapsed_ns, result=result, failure=failure)
        del pending[lane]
        self.value['pending'] = next(iter(pending.values()), None)
        self.save()

    def recover_pending(self):
        """Inspect only in-flight replies (one serial or at most two lanes)."""
        for lane, pending in list(self.value.get('parallel_pending', {}).items()):
            self.settle_lane(lane, result=self._pending_result(pending),
                failure='interrupted; full reserved cost retained')
        pending = self.value['pending']
        if pending is None:
            return
        result = self.pending_result()
        self.settle(result=result, failure='interrupted; full reserved cost retained')

    def pending_result(self):
        return self._pending_result(self.value['pending'])

    @staticmethod
    def _pending_result(pending):
        reply = Path(pending['reply']) if pending and pending['reply'] else None
        if reply is not None and reply.is_file():
            message = read_json(reply)
            if message['work_id'] != pending['work_id']:
                raise ValueError('pending reply belongs to another work item')
            return message.get('result')
        return None
