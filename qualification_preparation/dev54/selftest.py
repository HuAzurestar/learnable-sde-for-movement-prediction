"""DEV selfchecks of new stdlib code only. Never import baseline modules."""
import ast
from contextlib import contextmanager, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('dev54_driver',HERE/'driver.py')
d=importlib.util.module_from_spec(spec); spec.loader.exec_module(d)

class SourceOnlyTests(unittest.TestCase):
    def test_import_has_no_original_modules(self):
        import sys
        self.assertFalse(any(k.split('.')[0] in {'application','infrastructure','models','estimation','evaluation'} for k in sys.modules))

    def test_default_explain_is_inert_and_unqualified(self):
        with patch.object(d,'verify_source',side_effect=AssertionError('unexpected source IO')), patch.object(d,'load_native',side_effect=AssertionError('native')), redirect_stdout(io.StringIO()) as out:
            self.assertEqual(d.main([]),2)
        self.assertEqual(json.loads(out.getvalue())['status'],'SOURCE_ONLY_NATIVE_UNEXECUTED')

    def test_execute_without_host_refused_before_io(self):
        with patch.object(d,'verify_source',side_effect=AssertionError('IO')):
            with self.assertRaisesRegex(d.Refusal,'NOT_ADMITTED'): d.main(['--mode','execute'])

    def test_json_cannot_be_admission(self):
        with self.assertRaisesRegex(d.Refusal,'in-process'): d.execute_after_admission(d.REPOSITORY,Path('x'),{'approved':True},[])

    def test_all704_source_bytes_and_manifest(self):
        self.assertEqual(d.verify_source(d.REPOSITORY),d.REPOSITORY)
        self.assertEqual(len(json.loads((HERE/'baseline.json').read_bytes())),704)

    def test_manifest_mutation_refused(self):
        original=Path.read_bytes
        def changed(path):
            return b'[]' if path==HERE/'baseline.json' else original(path)
        with patch.object(Path,'read_bytes',changed):
            with self.assertRaisesRegex(d.Refusal,'manifest differs'): d.verify_source(d.REPOSITORY)

    def test_source_byte_mutation_refused(self):
        row=json.loads((HERE/'baseline.json').read_bytes())[0]; target=d.REPOSITORY/row['path']; original=Path.read_bytes
        def changed(path): return b'mutation' if path==target else original(path)
        with patch.object(Path,'read_bytes',changed):
            with self.assertRaisesRegex(d.Refusal,'baseline bytes differ'): d.verify_source(d.REPOSITORY)

    def test_preflight_does_not_create_root_or_load_native(self):
        with tempfile.TemporaryDirectory(prefix='dev54-guards-') as tmp:
            root=Path(tmp).resolve()/'future'
            with patch.object(d,'load_native',side_effect=AssertionError('native')),redirect_stdout(io.StringIO()) as out:
                self.assertEqual(d.main(['--mode','preflight','--runtime-root',str(root)]),2)
            self.assertFalse(root.exists())
            self.assertEqual(json.loads(out.getvalue())['future_argv'][-1],str(root))

    def test_existing_root_refused_without_mutation(self):
        with tempfile.TemporaryDirectory(prefix='dev54-guards-') as tmp:
            root=Path(tmp).resolve(); keep=root/'keep';keep.write_bytes(b'original')
            with self.assertRaises(d.Refusal): d.future_root(root)
            self.assertEqual(keep.read_bytes(),b'original')

    def test_git_ancestor_refused(self):
        with tempfile.TemporaryDirectory(prefix='dev54-guards-') as tmp:
            parent=Path(tmp).resolve();(parent/'.git').write_bytes(b'gitdir: elsewhere')
            with self.assertRaisesRegex(d.Refusal,'outside Git'): d.future_root(parent/'future')

    def test_relative_and_missing_parent_refused(self):
        with self.assertRaises(d.Refusal): d.future_root(Path('relative'))
        with tempfile.TemporaryDirectory(prefix='dev54-guards-') as tmp:
            with self.assertRaises(d.Refusal): d.future_root(Path(tmp).resolve()/'absent'/'future')

    def test_reparse_guard_without_platform_link_privilege(self):
        with tempfile.TemporaryDirectory(prefix='dev54-guards-') as tmp:
            parent=Path(tmp).resolve(); actual=Path.lstat
            def reparse(path,*a,**kw):
                s=actual(path,*a,**kw)
                return SimpleNamespace(st_mode=s.st_mode,st_file_attributes=1024) if path==parent else s
            with patch.object(Path,'lstat',reparse):
                with self.assertRaisesRegex(d.Refusal,'reparse'): d.future_root(parent/'future')

    def test_fresh_os_lock_and_no_replay(self):
        with tempfile.TemporaryDirectory(prefix='dev54-guards-') as tmp:
            root=Path(tmp).resolve()/'owned'
            with d.claim_root(root):
                self.assertTrue((root/'.dev54-owner').is_file())
            marker=json.loads((root/'.dev54-owner').read_bytes());self.assertEqual(marker['pid'],os.getpid())
            self.assertTrue((root/'.dev54-owner').is_file())
            with self.assertRaises(d.Refusal):
                with d.claim_root(root): pass

    def test_denied_host_never_claims_or_imports(self):
        @contextmanager
        def deny(request):
            self.assertEqual(request['scope'],'original205-artificial-only')
            self.assertTrue(request['native']); self.assertFalse(request['science'])
            self.assertEqual(request['argv'],['concrete','argv'])
            raise d.Refusal('host denied')
            yield
        with tempfile.TemporaryDirectory(prefix='dev54-guards-') as tmp:
            root=Path(tmp).resolve()/'future'
            with patch.object(d,'load_native',side_effect=AssertionError('native')):
                with self.assertRaisesRegex(d.Refusal,'host denied'): d.execute_after_admission(d.REPOSITORY,root,deny,['concrete','argv'])
            self.assertFalse(root.exists())

    def test_provider_factory_uses_exact_attributes_and_shared_local(self):
        # Entirely new fake modules, never baseline imports/callbacks.
        local=threading.local(); ns={'_owner':local}
        exec('def owner_session(): pass\nclass GridClock:\n def check(self): return _owner\n',ns)
        modules={}
        for api in d.APIS:
            module,name=api.split(':');m=modules.setdefault(module,SimpleNamespace())
            parts=name.split('.');obj=m
            for part in parts:
                if not hasattr(obj,part): setattr(obj,part,type(part,(),{}) if part[0].isupper() else lambda:None)
                obj=getattr(obj,part)
        b=modules['application.walking_pilot_budget']; b._owner=local;b.owner_session=ns['owner_session'];b.GridClock=ns['GridClock']
        apis,owner=d.bind_modules(modules)
        self.assertEqual(set(apis),set(d.APIS));self.assertEqual(len(apis),21);self.assertIs(owner,local)
        self.assertIs(apis['infrastructure.research_store:ResearchStore.artifact'],modules['infrastructure.research_store'].ResearchStore.artifact)
        b._owner=threading.local()
        with self.assertRaisesRegex(d.Refusal,'owner local'): d.bind_modules(modules)

    def test_provider_set_matches_pinned_graph_without_import(self):
        graph=json.loads((d.REPOSITORY/'qualification_preparation/dev50/invocation-spec.json').read_bytes())
        expected={c['callable'] for c in graph['calls']}|{
            'application.research_budget:BudgetSpec','application.walking_pilot_budget:GridClock',
            'application.walking_pilot_budget:GridOwnerMeter','application.walking_stage:prepare_package'}
        self.assertEqual(set(d.APIS),expected)
        self.assertEqual(len(graph['calls']),29)
        self.assertEqual(graph['calls'][-1]['id'],'terminal')
        self.assertEqual(len(graph['config_template']['sources']),11)
        self.assertEqual(len(graph['study_template']['cells']),32)

    def test_future_native_loader_requires_matching_cwd_before_claim(self):
        @contextmanager
        def context(request): yield
        with tempfile.TemporaryDirectory(prefix='dev54-guards-') as tmp:
            root=Path(tmp).resolve()/'future'
            with patch.object(d.Path,'cwd',return_value=root.parent),patch.object(d,'load_native',side_effect=AssertionError('native')):
                with self.assertRaisesRegex(d.Refusal,'worker cwd'): d.execute_after_admission(d.REPOSITORY,root,context,[])
            self.assertFalse(root.exists())

    def test_ast_no_top_level_native_execution_or_replacement_controller(self):
        tree=ast.parse((HERE/'driver.py').read_text(encoding='utf-8'))
        for node in tree.body:
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module]
                self.assertFalse(any(n.split('.')[0] in {'application','infrastructure','models'} for n in names))
        functions={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        native_calls=[n for n in ast.walk(functions['execute_after_admission']) if isinstance(n,ast.Call)]
        self.assertEqual(sum(isinstance(n.func,ast.Attribute) and n.func.attr=='run' for n in native_calls),1)
        source=ast.get_source_segment((HERE/'driver.py').read_text(),functions['execute_after_admission'])
        self.assertLess(source.index('with admission(request)'),source.index('with claim_root(root)'))
        self.assertLess(source.index('with claim_root(root)'),source.index('load_native(repository)'))
        self.assertNotIn('result_validator',source)

if __name__=='__main__': unittest.main(verbosity=2)
