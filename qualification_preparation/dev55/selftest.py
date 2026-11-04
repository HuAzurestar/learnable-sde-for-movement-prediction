"""Own DEV55 stdlib synthetic checks; never calls future_entry or native driver."""
import dataclasses
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

p = Path(__file__).with_name('host.py')
spec = importlib.util.spec_from_file_location('dev55_host', p)
h = importlib.util.module_from_spec(spec); sys.modules[spec.name] = h; spec.loader.exec_module(h)

class SyntheticAuthority:
    def __init__(self, events, deny=None):
        self.events, self.deny = events, deny
    @contextmanager
    def hold(self, request, *, reference, limits):
        self.events.append('authority-enter')
        if self.deny == 'hold':
            raise h.Rejected('authority', 'synthetic denied')
        self.assertions = (reference, limits)
        try:
            yield self
        finally:
            self.events.append('authority-exit')
    def checkpoint(self, phase):
        self.events.append(phase)
        if phase == self.deny:
            raise h.Rejected('authority', 'synthetic revoked')

class Checks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.file = self.root/'source.py'; self.file.write_text('synthetic only\n')
        self.pin = h.identity(self.file)
        self.f = h.Frozen(str(self.root), str(self.root/'future'), (sys.executable, 0, 'synthetic'), (self.pin,), (), self.pin, (), (), 2**31, 2**22, 'synthetic-only-no-grant', self.pin)
        self.events=[]; self.evidence=[]; self.a=SyntheticAuthority(self.events)
    def factory(self, probe=None):
        def observed(f, sid):
            self.events.append('probe'); return {'runtime_margin':'UNKNOWN','synthetic':True}
        return h._admission(self.f, self.a, 'S-1-5-synthetic', self.evidence, probe or observed)
    def test01_exact_lifetime(self):
        with self.factory()(self.f.request()):
            self.events.append('simulated-original-owner-exit')
        self.assertEqual(self.events,['authority-enter','before-preflight','probe','before-native','simulated-original-owner-exit','authority-exit'])
    def test02_replay(self):
        a=self.factory()
        with a(self.f.request()): pass
        with self.assertRaisesRegex(h.Rejected,'single-use'):
            with a(self.f.request()): self.fail('entered')
    def test03_all_request_fields(self):
        for key in self.f.request():
            with self.subTest(key=key):
                request=self.f.request(); request[key]='tampered'
                with self.assertRaisesRegex(h.Rejected,'request'):
                    with self.factory()(request): self.fail('entered')
    def test04_added_field(self):
        request=self.f.request(); request['approved']=True
        with self.assertRaises(h.Rejected):
            with self.factory()(request): self.fail('entered')
    def test05_denied_hold(self):
        self.a.deny='hold'
        with self.assertRaises(h.Rejected):
            with self.factory()(self.f.request()): self.fail('entered')
        self.assertNotIn('probe',self.events)
    def test06_revoked_preflight(self):
        self.a.deny='before-preflight'
        with self.assertRaises(h.Rejected):
            with self.factory()(self.f.request()): self.fail('entered')
        self.assertEqual(self.events[-1],'authority-exit')
    def test07_revoked_native(self):
        self.a.deny='before-native'
        with self.assertRaises(h.Rejected):
            with self.factory()(self.f.request()): self.fail('entered')
    def test08_probe_failure(self):
        def probe(f,s): raise h.Rejected('acl','synthetic denied')
        with self.assertRaises(h.Rejected):
            with self.factory(probe)(self.f.request()): self.fail('entered')
        self.assertEqual(self.events[-1],'authority-exit')
    def test09_mutated_request(self):
        r=self.f.request()
        def probe(f,s): r['science']=True; return {}
        with self.assertRaises(h.Rejected):
            with self.factory(probe)(r): self.fail('entered')
    def test10_body_exception_identity(self):
        error=RuntimeError('synthetic original failure')
        try:
            with self.factory()(self.f.request()): raise error
        except RuntimeError as got: self.assertIs(got,error)
        self.assertEqual(self.events[-1],'authority-exit')
        self.assertEqual(self.evidence[-1]['error'],'RuntimeError')
    def test11_no_root_creation(self):
        with self.factory()(self.f.request()): pass
        self.assertFalse(Path(self.f.runtime_root).exists())
    def test12_identity_bytes(self):
        self.file.write_text('changed')
        self.assertNotEqual(h.identity(self.file),self.pin)
    def test13_relative_identity(self):
        with self.assertRaises(h.Rejected): h.identity('source.py')
    def test14_link(self):
        linked=self.root/'hardlink'; os.link(self.file,linked)
        with self.assertRaises(h.Rejected): h.identity(self.file)
    def test15_existing_root(self):
        with self.assertRaises(h.Rejected): h.parent_paths(self.root)
    def test16_missing_parent(self):
        with self.assertRaises(h.Rejected): h.parent_paths(self.root/'missing'/'future')
    def test17_fresh_parent(self):
        self.assertEqual(h.parent_paths(self.root/'future')[0],str(self.root))
    def test18_git_parent(self):
        (self.root/'.git').mkdir()
        with self.assertRaises(h.Rejected): h.parent_paths(self.root/'future')
    def test19_acl_safe(self):
        h.restrictive_acl((('x','S-1-5-test','O:BAD:(A;;FA;;;SY)(A;;FRFX;;;BU)'),),'S-1-5-test')
    def test20_acl_write(self):
        with self.assertRaises(h.Rejected): h.restrictive_acl((('x','S-1-5-test','D:(A;;FA;;;BU)'),),'S-1-5-test')
    def test21_acl_null(self):
        with self.assertRaises(h.Rejected): h.restrictive_acl((('x','S-1-5-test','D:'),),'S-1-5-test')
    def test22_acl_unknown_owner(self):
        with self.assertRaises(h.Rejected): h.restrictive_acl((('x','unknown','D:(A;;FA;;;SY)'),),'S-1-5-test')
    def test23_acl_callback(self):
        with self.assertRaises(h.Rejected): h.restrictive_acl((('x','SY','D:(XA;;FA;;;SY;condition)'),),'S-1-5-test')
    def test24_nonfinite(self):
        with self.assertRaises(ValueError): h.canonical({'n':float('nan')})
    def test25_clean_process_guard(self):
        sys.modules['application.synthetic_forbidden']=object()
        try:
            with self.assertRaises(h.Rejected): h.loaded_modules()
        finally: del sys.modules['application.synthetic_forbidden']
    def test26_provider_pin_required(self):
        f=dataclasses.replace(self.f,authority_code=(self.pin[0],0,'0'*64))
        with self.assertRaises(h.Rejected): h.make_admission(f,authority=self.a,trusted_sid='S-1-5-test',evidence=[])
    def test27_no_cli_grant(self):
        import subprocess
        r=subprocess.run([sys.executable,'-B',str(p)],capture_output=True,text=True)
        self.assertEqual(r.returncode,2); self.assertIn('no CLI authority',r.stdout)

    def test28_exit_error_preserves_body(self):
        @contextmanager
        def broken_hold(*args,**kwargs):
            try: yield self.a
            finally: raise ValueError('synthetic exit failure')
        self.a.hold=broken_hold
        error=RuntimeError('body')
        try:
            with self.factory()(self.f.request()): raise error
        except RuntimeError as got:
            self.assertIs(got,error); self.assertIsInstance(got.__cause__,ValueError)
    def test29_wrong_thread(self):
        import threading
        admission=self.factory(); errors=[]
        def run():
            try:
                with admission(self.f.request()): errors.append('entered')
            except h.Rejected as exc: errors.append(exc.check)
        thread=threading.Thread(target=run);thread.start();thread.join()
        self.assertEqual(errors,['owner'])

    def test30_prospective_inherit_only_writer(self):
        rows=(('parent','SY','D:(A;;FA;;;SY)(A;OICIIO;FA;;;BU)'),)
        h.restrictive_acl(rows,'S-1-5-test')  # ineffective on existing parent
        with self.assertRaisesRegex(h.Rejected,'acl-write'):
            h.restrictive_acl(rows,'S-1-5-test',descendants=True)

    def test31_propagation_variants(self):
        for flags in ('CIIO','OIIO','OICIIO','CINPIO','OINPIO','OI','CI','OICIID'):
            with self.subTest(flags=flags), self.assertRaisesRegex(h.Rejected,'acl-write'):
                h.restrictive_acl((('p','SY','D:(A;;FA;;;SY)(A;'+flags+';FA;;;BU)'),),
                                  'S-1-5-test',descendants=True)

    def test32_prospective_safe(self):
        for ace in ('A;OICIIO;FA;;;SY','A;OICIIO;FRFX;;;BU','A;IO;FA;;;BU',
                    'D;OICIIO;FA;;;BU'):
            with self.subTest(ace=ace):
                h.restrictive_acl((('p','SY','D:(A;;FA;;;SY)('+ace+')'),),
                                  'S-1-5-test',descendants=True)

    def test33_unknown_flags(self):
        for flags in ('ZZ','C','CIZZ'):
            with self.subTest(flags=flags), self.assertRaisesRegex(h.Rejected,'acl-ace'):
                h.restrictive_acl((('p','SY','D:(A;'+flags+';FA;;;BU)'),), 'S-1-5-test')

    def environment_probe(self, unsafe=None, inherited=False):
        # Only synthetic OS observations. No original driver or native calls.
        from unittest.mock import patch
        from types import SimpleNamespace
        reader=self.root/'reader.exe';reader.write_text('synthetic reader')
        interp=self.root/'interpreter.exe';interp.write_text('synthetic interpreter')
        parent=str(self.root); calls=[]
        f=dataclasses.replace(self.f,interpreter=(str(interp),0,'synthetic'),
            acl_reader=(str(reader),0,'synthetic'),modules=(),
            acl=((parent,'SY','D:(A;;FA;;;SY)'+('(A;OICIIO;FA;;;BU)' if inherited else '')),),
            distributions=(('synthetic','1'),))
        def acl(paths, executable):
            calls.append(tuple(paths))
            if tuple(paths)==(parent,): return f.acl
            return tuple((x,'SY','D:(A;;FA;;;'+('BU' if x==unsafe else 'SY')+')') for x in paths)
        def ident(path):
            path=str(path)
            if path==h.sys.executable: return f.interpreter
            return next(row for row in (self.pin,f.interpreter,f.acl_reader) if row[0]==path)
        with patch.object(h.Path,'cwd',return_value=self.root), \
             patch.object(h,'parent_paths',return_value=(parent,)), \
             patch.object(h,'identity',side_effect=ident), \
             patch.object(h,'native_images',return_value=()), \
             patch.object(h,'loaded_modules',return_value=()), \
             patch.object(h.Path,'rglob',return_value=iter([self.file])), \
             patch.object(h,'read_acl',side_effect=acl), \
             patch.object(h.metadata,'distributions',return_value=[SimpleNamespace(metadata={'Name':'synthetic'},version='1')]), \
             patch.object(h,'resources',return_value=2**32), \
             patch.object(h.shutil,'disk_usage',return_value=SimpleNamespace(free=2**24)):
            result=h.environment(f,'S-1-5-test')
        return result,calls,f

    def test34_environment_all_file_descriptors(self):
        result,calls,f=self.environment_probe()
        self.assertEqual(result['runtime_margin'],'UNKNOWN')
        self.assertIn(tuple(sorted((self.pin[0],f.interpreter[0],f.acl_reader[0]))),calls)

    def test35_environment_explicit_file_writer(self):
        for name in ('source.py','interpreter.exe','reader.exe'):
            with self.subTest(name=name), self.assertRaisesRegex(h.Rejected,'acl-write'):
                self.environment_probe(unsafe=str(self.root/name))

    def test36_environment_prospective_writer(self):
        with self.assertRaisesRegex(h.Rejected,'acl-write'):
            self.environment_probe(inherited=True)

if __name__=='__main__': unittest.main(verbosity=2)
