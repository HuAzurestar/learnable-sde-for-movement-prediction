"""Owned artificial harness selftests; never import the product or original suites.

Run directly: python -B tests/test_walking_harness_selftest.py
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('owned_walking_harness', Path(__file__).with_name('walking_harness.py'))
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


class HelperTests(unittest.TestCase):
    def exercise(self, code, *, failure=None, token='READY', go=False):
        start=time.monotonic()
        helper=harness.HelperProtocol([sys.executable,'-u','-c',code],start+.65)
        try:
            if failure:
                with self.assertRaises((AssertionError,subprocess.TimeoutExpired,BrokenPipeError)):
                    helper.expect(token)
                    if go:
                        helper.go(); helper.expect('TRYING')
                    helper.finish()
            else:
                helper.expect(token)
                if go:
                    helper.go(); helper.expect('TRYING')
                helper.finish()
        finally:
            helper.close()
        self.assertIsNotNone(helper.child.returncode)
        self.assertTrue(all(not t.is_alive() for t in helper.threads))
        self.assertTrue(all(s.closed for s in (helper.child.stdin,helper.child.stdout,helper.child.stderr)))
        self.assertLess(time.monotonic()-start,3.7)
        self.assertLessEqual(len(helper.stderr),4096)
        return helper

    def test_stall_before_ready(self):
        self.exercise('import time; time.sleep(30)',failure=True)

    def test_stall_before_trying(self):
        self.exercise("import sys,time; print('READY',flush=True); sys.stdin.readline(); time.sleep(30)",failure=True,go=True)

    def test_partial_line_stays_open(self):
        self.exercise("import sys,time; sys.stdout.write('REA'); sys.stdout.flush(); time.sleep(30)",failure=True)

    def test_partial_line_eof(self):
        self.exercise("import sys; sys.stdout.write('REA')",failure=True)

    def test_eof(self):
        self.exercise('pass',failure=True)

    def test_wrong_token(self):
        self.exercise("print('OTHER',flush=True)",failure=True)

    def test_oversized_line(self):
        self.exercise("print('x'*10000,flush=True)",failure=True)

    def test_message_flood(self):
        self.exercise("import os; os.write(1,b'x\\n'*10000)",failure=True)

    def test_stderr_flood_is_drained_and_capped(self):
        helper=self.exercise("import os; os.write(2,b'x'*2000000); print('READY',flush=True); print('DONE',flush=True)")
        self.assertEqual(len(helper.stderr),4096)

    def test_stderr_flood_then_stall(self):
        self.exercise("import os,time; os.write(2,b'x'*2000000); time.sleep(30)",failure=True)

    def test_exit_error(self):
        self.exercise("import sys; print('READY',flush=True); print('DONE',flush=True); sys.exit(7)",failure=True)

    def test_normal_handshakes(self):
        self.exercise("import sys; print('READY',flush=True); assert sys.stdin.readline()=='GO\\n'; print('TRYING',flush=True); print('DONE',flush=True)",go=True)

    def test_startup_and_final_wait_share_absolute_deadline(self):
        self.exercise("import time; time.sleep(.4); print('READY',flush=True); time.sleep(.4); print('DONE',flush=True)",failure=True)

    def test_expired_before_spawn(self):
        with patch.object(harness.subprocess,'Popen',side_effect=AssertionError('must not spawn')) as spawn:
            with self.assertRaisesRegex(AssertionError,'deadline'):
                harness.HelperProtocol([],time.monotonic()-1)
        spawn.assert_not_called()

    def test_current_style_exclusion_releases_before_done(self):
        with tempfile.TemporaryDirectory() as directory:
            release=Path(directory)/'release'
            code=("import pathlib,sys,time; print('READY',flush=True); sys.stdin.readline(); print('TRYING',flush=True); "
                  "p=pathlib.Path(sys.argv[1]); exec('while not p.exists(): time.sleep(.005)'); print('DONE',flush=True)")
            helper=harness.HelperProtocol([sys.executable,'-u','-c',code,str(release)],time.monotonic()+2)
            try:
                helper.expect('READY'); helper.go(); helper.expect('TRYING')
                # Simulated protected read ends before waiting for DONE.
                self.assertIsNone(helper.child.poll())
                release.touch(); helper.finish()
            finally:
                helper.close()
            self.assertFalse(any(t.is_alive() for t in helper.threads))


class FixtureTests(unittest.TestCase):
    def test_exact_raw_fixture_validation_without_compilation(self):
        self.assertEqual(len(harness.historical_source().encode('utf-8')),7797)

    def test_missing_and_mutated_fixture_fail_before_spawn(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'fixture'
            for raw in (None,b'',b'x'*7797,b'x'*100000):
                if raw is not None: p.write_bytes(raw)
                with patch.object(harness.subprocess,'Popen') as spawn:
                    with self.assertRaisesRegex(AssertionError,'historical negative-control fixture'):
                        harness.historical_source(p)
                spawn.assert_not_called()


class ReceiptTests(unittest.TestCase):
    def test_strict_identity(self):
        good={'pid':123,'birth':987,'nonce':'owned'}
        self.assertEqual(harness.creation_receipt(json.dumps(good).encode(),'owned'),good)
        for changes in ({'pid':True},{'pid':0},{'pid':2**32},{'birth':None},{'birth':False},
                        {'birth':0},{'birth':2**64},{'nonce':'other'},{'extra':1}):
            with self.assertRaises(AssertionError):
                harness.creation_receipt(json.dumps({**good,**changes}).encode(),'owned')

    def test_missing_partial_duplicate_and_oversized_receipts(self):
        for raw in (b'',b'{',b'null',b'[]',b'x'*513,
                    b'{"pid":123,"pid":124,"birth":987,"nonce":"owned"}'):
            with self.assertRaises(AssertionError): harness.creation_receipt(raw,'owned')


if __name__=='__main__':
    unittest.main(verbosity=2)
