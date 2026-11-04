"""DEV-only clock checks: fake modules/endpoints/Popen; no product imports."""
import ast
import builtins
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

from tests.walking_fixture_clock import EPOCH, MODULES, child_command, main, synthetic_clock

ROOT = Path(__file__).resolve().parents[1]
CEILING = datetime.fromisoformat('2026-11-03T23:59:00+08:00')


class FakeError(Exception):
    def __init__(self, code, message):
        self.code = code
        super().__init__(code + ': ' + message)


def predicate_module(name, path, function):
    """Compile only the unchanged pure predicate, never import product code."""
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8-sig'))
    nodes = list(ast.walk(tree))
    target = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == function)
    target.decorator_list = []
    module = types.ModuleType(name)
    module.__dict__.update(ResearchError=FakeError, re=re, identifier=lambda v: v,
                          authorization_key_for_grant=lambda v: 'fake-only')
    exec('from datetime import datetime, timezone', module.__dict__)
    exec(compile(ast.Module(body=[target], type_ignores=[]), str(path), 'exec'), module.__dict__)
    return module


class ClockTests(unittest.TestCase):
    def test_existing_late_imports_and_exception_restore_without_global_clock(self):
        name = 'application.walking_pilot_budget'
        fake = types.ModuleType(name); fake.datetime = datetime
        original_import, original_popen = builtins.__import__, subprocess.Popen
        clocks = (time.monotonic, time.monotonic_ns, time.perf_counter, time.sleep, time.time)
        with patch.dict(sys.modules, {name: fake}):
            with self.assertRaisesRegex(RuntimeError, 'synthetic'):
                with synthetic_clock() as historical:
                    self.assertEqual(fake.datetime.now(timezone.utc), EPOCH)
                    namespace = {'__name__': name}
                    exec('from datetime import datetime, timezone\nvalue=datetime.now(timezone.utc)', namespace)
                    self.assertEqual(namespace['value'], EPOCH)
                    unrelated = {'__name__': 'unrelated'}
                    exec('from datetime import datetime', unrelated)
                    self.assertIs(unrelated['datetime'], datetime)
                    self.assertIs(sys.modules['datetime'].datetime, datetime)
                    self.assertEqual(clocks, (time.monotonic, time.monotonic_ns, time.perf_counter, time.sleep, time.time))
                    late = types.ModuleType('application.walking_stage')
                    exec('from datetime import datetime', late.__dict__)
                    sys.modules[late.__name__] = late
                    self.assertIs(late.datetime, historical)
                    raise RuntimeError('synthetic')
            self.assertIs(fake.datetime, datetime)
            self.assertIs(late.datetime, datetime)
            del sys.modules[late.__name__]
        self.assertIs(builtins.__import__, original_import)
        self.assertIs(subprocess.Popen, original_popen)

    def test_unchanged_expiry_and_authorize_predicates_before_at_after_ceiling(self):
        policy = predicate_module('application.walking_pilot_budget', 'application/walking_pilot_budget.py', '_policy_expiry')
        grant = predicate_module('infrastructure.research_store', 'infrastructure/research_store.py', 'authorize')
        class Endpoint:
            def __init__(self): self.published = []
            def publish(self, *args): self.published.append(args)
        original = {'authorization_id': 'synthetic', 'study_id': 'synthetic', 'purposes': ['resume'],
                    'visibilities': ['synthetic'], 'block_ids': [], 'evidence_hash': 'a'*64,
                    'expires_at': CEILING.isoformat()}
        with patch.dict(sys.modules, {policy.__name__: policy, grant.__name__: grant}):
            for offset in (-1, 0, 1):
                with self.subTest(offset=offset), synthetic_clock(CEILING + timedelta(seconds=offset)):
                    endpoint = Endpoint()
                    if offset < 0:
                        policy._policy_expiry(original)
                        grant.authorize(endpoint, original)
                        self.assertEqual(len(endpoint.published), 1)
                    else:
                        with self.assertRaisesRegex(FakeError, 'expired or extended'):
                            policy._policy_expiry(original)
                        with self.assertRaisesRegex(FakeError, 'expired or missing'):
                            grant.authorize(endpoint, original)
                        self.assertFalse(endpoint.published)
                    with self.assertRaisesRegex(FakeError, 'expired or extended'):
                        policy._policy_expiry({'expires_at': (CEILING+timedelta(seconds=1)).isoformat()})
                    with self.assertRaisesRegex(FakeError, 'CONTRACT_MISMATCH'):
                        grant.authorize(endpoint, {**original, 'purposes': []})
            with synthetic_clock(EPOCH):
                for expiry in ('2026-11-02T00:00:00+00:00', '2026-11-03T00:00:00'):
                    with self.assertRaises(FakeError): policy._policy_expiry({'expires_at': expiry})
                    with self.assertRaises(FakeError): grant.authorize(Endpoint(), {**original, 'expires_at': expiry})
        self.assertIs(policy.datetime, datetime)
        self.assertIs(grant.datetime, datetime)

    def test_fixture_selection_and_negative_override_teardown(self):
        source = ast.parse((ROOT/'tests/conftest.py').read_text(encoding='utf-8-sig'))
        node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'walking_synthetic_wall_clock')
        node.decorator_list = []; namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<fixture>', 'exec'), namespace)
        fake = types.ModuleType('application.walking_pilot_budget'); fake.datetime = datetime
        with patch.dict(sys.modules, {fake.__name__: fake}):
            request = types.SimpleNamespace(module=types.SimpleNamespace(__name__='tests.test_walking_pilot_sources'))
            fixture = namespace[node.name](request); next(fixture)
            historical = fake.datetime
            self.assertEqual(historical.now(timezone.utc), EPOCH)
            # Match pytest ordering: per-test override is undone before the
            # outer autouse scope closes, including an exception path.
            with patch.object(fake, 'datetime', object()): pass
            self.assertIs(fake.datetime, historical)
            fixture.close(); self.assertIs(fake.datetime, datetime)
            request.module.__name__ = 'tests.test_research_store'
            fixture = namespace[node.name](request); next(fixture)
            self.assertIs(fake.datetime, datetime); fixture.close()

    def test_owned_child_argv_env_and_native_delegation_without_spawning(self):
        calls = []
        def fake_popen(*args, **kw): calls.append((args, kw)); return 'fake-process'
        env = {'SYNTHETIC_ONLY': 'yes'}
        with patch.object(subprocess, 'Popen', fake_popen):
            with synthetic_clock():
                for tail in (['-B', '-c', 'pass', 'value'], ['-u', '-m', 'fake.module', 'arg'], ['fake.py', 'arg']):
                    argv = [sys.executable, *tail]
                    result = subprocess.Popen(argv, env=env, cwd='fake-cwd', stdin='fake-input', start_new_session=True)
                    self.assertEqual(result, 'fake-process')
                    wrapped = calls[-1][0][0]
                    self.assertEqual(wrapped, child_command(argv))
                    self.assertEqual(wrapped[wrapped.index('--')+1:], tail[1:] if tail[0] in ('-B','-u') else tail)
                    self.assertIs(calls[-1][1]['env'], env)
                    self.assertEqual(calls[-1][1]['cwd'], 'fake-cwd')
                    self.assertEqual(calls[-1][1]['stdin'], 'fake-input')
                    self.assertIs(calls[-1][1]['start_new_session'], True)
                    self.assertEqual(child_command(wrapped), wrapped)
                native = ['fake-native', 'arg']; subprocess.Popen(native)
                self.assertIs(calls[-1][0][0], native)
                shell = [sys.executable, '-c', 'pass']; subprocess.Popen(shell, shell=True)
                self.assertIs(calls[-1][0][0], shell)
        with self.assertRaises(ValueError): child_command([sys.executable, '-I', '-c', 'pass'])

    def test_runner_command_module_script_and_restoration_no_process(self):
        argv, paths = sys.argv[:], sys.path[:]
        with tempfile.TemporaryDirectory(prefix='synthetic-clock-') as directory:
            script = Path(directory)/'fake_script.py'
            script.write_text("import sys\nassert sys.argv[1:]==['argument']\n", encoding='utf-8')
            main([EPOCH.isoformat(), '--', str(script), 'argument'])
        main([EPOCH.isoformat(), '--', '-c', "import sys; assert sys.argv==['-c','argument']", 'argument'])
        def fake_run_module(module, **kw):
            self.assertEqual(module, 'fake.module')
            self.assertEqual(sys.argv, ['fake.module', 'argument'])
            self.assertEqual(kw, {'run_name':'__main__','alter_sys':True})
            # runpy's __main__ uses its original module spec for scoped imports.
            scope = {'__name__':'__main__', '__spec__':types.SimpleNamespace(name='application.walking_pilot')}
            exec('from datetime import datetime, timezone\nassert datetime.now(timezone.utc).isoformat()=="2026-11-03T00:00:00+00:00"', scope)
            raise RuntimeError('fake-runner')
        with patch('runpy.run_module', fake_run_module):
            with self.assertRaisesRegex(RuntimeError, 'fake-runner'):
                main([EPOCH.isoformat(), '--', '-m', 'fake.module', 'argument'])
        with self.assertRaises(SystemExit) as stopped:
            main([EPOCH.isoformat(), '--', '-c', 'raise SystemExit(23)'])
        self.assertEqual(stopped.exception.code, 23)
        self.assertEqual(sys.argv, argv); self.assertEqual(sys.path, paths)


if __name__ == '__main__':
    unittest.main()
