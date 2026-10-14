"""Bounded diagnostic timings, NEVER admission, authority or charged cost.

The session sidecar is outside bootstrap's scientific artifact inventory.
Loss/corruption of this optional log cannot grant or deny scientific admission;
original metering, independent validators and native closure remain mandatory.
No private sample/work/source identities or scientific values are logged.
"""
import os
import time
from pathlib import Path

from .protocol_core import canonical, under

MAX_EVENTS = 128


class RestorationCosts:
    def __init__(self, directory, actor, *, _clock=time.monotonic_ns):
        if actor not in {'worker', 'controller'}:
            raise ValueError('fixed restoration actor required')
        self.actor, self.clock = actor, _clock
        self.started = self.stage_started = _clock()
        self.stage, self.completed, self.total = None, 0, None
        self.events, self.closed, self.stream = 0, False, None
        try:
            self.stream = under(Path(directory), 'restoration-cost-' + actor + '.jsonl').open('x', encoding='utf-8')
        except OSError:
            pass  # Optional diagnostic sink, not a scientific/control gate.

    def _emit(self, event, **extra):
        if self.stream is None or self.events >= MAX_EVENTS:
            return
        now = self.clock()
        row = dict(schema_version='pirc17-restoration-cost-diagnostic-v1', actor=self.actor,
            pid=os.getpid(), event=event, stage=self.stage, completed=self.completed, total=self.total,
            started_ns=self.started, observed_ns=now,
            elapsed_ns=now-self.started, stage_elapsed_ns=now-self.stage_started,
            diagnostic_only=True, authorizes_execution=False, scientific_admission=False, **extra)
        try:
            self.stream.write(canonical(row).decode('utf-8') + '\n')
            self.stream.flush()
            self.events += 1
        except OSError:
            self._close_stream()
            self.stream = None

    def _close_stream(self):
        try:
            self.stream.close()
        except OSError:
            pass

    def mark(self, stage, *, total=None):
        if (self.closed or not isinstance(stage, str) or not 0 < len(stage) <= 64
                or not stage.isascii() or not stage.replace('-', '').isalpha()):
            raise ValueError('open named diagnostic stage required')
        if total is not None and (type(total) is not int or total < 0):
            raise ValueError('nonnegative stage count required')
        if self.stage is not None:
            self._emit('stage-finished', status='returned')
        self.stage, self.completed, self.total = stage, 0, total
        self.stage_started = self.clock()
        self._emit('stage-started')

    def progress(self, completed):
        if (self.closed or self.stage is None or type(completed) is not int or completed < self.completed
                or self.total is not None and completed > self.total):
            raise ValueError('monotone bounded stage count required')
        self.completed = completed
        if completed % 250 == 0 or completed == self.total:
            self._emit('progress')

    def close(self, status):
        if status not in {'returned', 'failed'}:
            raise ValueError('fixed diagnostic terminal status required')
        if not self.closed:
            self._emit('finished', status=status)
            self.closed = True
            if self.stream is not None:
                self._close_stream()
                self.stream = None
