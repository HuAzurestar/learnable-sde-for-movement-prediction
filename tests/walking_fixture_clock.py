"""Historical wall time for synthetic walking fixtures; never a real grant.

Only named application bindings change. OS time, stdlib datetime and monotonic
billing/deadlines remain real. Child commands enter the same test context in
place, retaining the original process tree and absolute deadline.
"""
import builtins
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import runpy
import subprocess
import sys
import types
from unittest.mock import patch

EPOCH = datetime(2026, 11, 3, 0, 0, tzinfo=timezone.utc)
MODULES = frozenset({
    'infrastructure.research_store', 'application.walking_pilot',
    'application.walking_pilot_sources', 'application.walking_pilot_budget',
    'application.walking_stage', 'application.research_supervisor',
    'application.research_budget', 'application.walking_execution',
})
FIXTURE_MODULES = frozenset({
    'test_walking_pilot_budget', 'test_walking_pilot_sources',
    'test_walking_b1', 'test_walking_b1_native', 'test_walking_owner_terminal',
    'test_walking_pilot_serial_scope', 'test_walking_pilot_bounds',
    'test_walking_pilot_fatal_limits',
})


def child_command(args, instant=EPOCH):
    """Wrap only explicit current-interpreter argv; never shell/native commands."""
    if not isinstance(args, (list, tuple)) or not args:
        return args
    if str(args[0]) != sys.executable:
        return args
    tail = list(args[1:]); flags = []
    while tail and tail[0] in {'-B', '-u'}:
        flags.append(tail.pop(0))
    if not tail or (tail[0].startswith('-') and tail[0] not in {'-c', '-m'}):
        raise ValueError('unsupported synthetic Python invocation')
    runner = str(Path(__file__).resolve())
    if tail[0] == runner:
        return args
    return [args[0], *flags, runner, instant.isoformat(), '--', *tail]


@contextmanager
def synthetic_clock(instant=EPOCH):
    """Scope named datetime imports/bindings and owned Python child launches."""
    if instant.tzinfo is None:
        raise ValueError('synthetic wall clock must be timezone-aware')
    class HistoricalDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = instant.astimezone(tz) if tz else instant.astimezone().replace(tzinfo=None)
            return cls.fromisoformat(value.isoformat())
        @classmethod
        def utcnow(cls):
            return cls.now(timezone.utc).replace(tzinfo=None)
    real_import = builtins.__import__
    real_popen = subprocess.Popen
    proxy = types.ModuleType('datetime')
    proxy.__dict__.update(vars(sys.modules['datetime']))
    proxy.datetime = HistoricalDatetime
    saved = {}
    def scoped_import(name, globals=None, locals=None, fromlist=(), level=0):
        result = real_import(name, globals, locals, fromlist, level)
        namespace = globals or {}
        spec = namespace.get('__spec__')
        owner = getattr(spec, 'name', None) or namespace.get('__name__')
        if level == 0 and name == 'datetime' and owner in MODULES:
            return proxy
        return result
    def scoped_popen(args, *a, **kw):
        if not kw.get('shell') and kw.get('executable') in (None, sys.executable):
            args = child_command(args, instant)
        return real_popen(args, *a, **kw)
    for name in MODULES:
        module = sys.modules.get(name)
        if module is not None and hasattr(module, 'datetime'):
            saved[name] = (module, module.datetime)
            module.datetime = HistoricalDatetime
    try:
        with patch.object(builtins, '__import__', scoped_import), patch.object(subprocess, 'Popen', scoped_popen):
            yield HistoricalDatetime
    finally:
        # Include modules first imported during this scope; they must not leak
        # the historical class into later, unrelated tests.
        for name in MODULES:
            module = sys.modules.get(name)
            if name in saved:
                original_module, original = saved[name]
                original_module.datetime = original
            elif module is not None and getattr(module, 'datetime', None) is HistoricalDatetime:
                module.datetime = datetime


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    instant = datetime.fromisoformat(args.pop(0))
    if args.pop(0) != '--':
        raise ValueError('missing synthetic runner separator')
    mode = args.pop(0)
    old_argv, old_path = sys.argv[:], sys.path[:]
    root = str(Path(__file__).resolve().parents[1])
    try:
        # The tests package is needed by recursively owned child launches.
        sys.path.insert(0, root)
        with synthetic_clock(instant):
            if mode == '-c':
                code = args.pop(0); sys.argv = ['-c', *args]
                exec(compile(code, '<string>', 'exec'), {'__name__': '__main__'})
            elif mode == '-m':
                module = args.pop(0); sys.argv = [module, *args]
                runpy.run_module(module, run_name='__main__', alter_sys=True)
            else:
                sys.argv = [mode, *args]
                sys.path.insert(0, str(Path(mode).resolve().parent))
                runpy.run_path(mode, run_name='__main__')
    finally:
        sys.argv, sys.path[:] = old_argv, old_path


if __name__ == '__main__':
    main()
