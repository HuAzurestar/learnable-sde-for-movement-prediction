"""DEV-only stdlib checks. No retained tests/product imports or constructors."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import builder as b

ROOT=Path(__file__).resolve().parents[2]

class BuilderChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=b.build(ROOT)

    def test_exact_source_and_no_product_import(self):
        self.assertEqual(self.plan['original_source_count'],650)
        self.assertEqual(self.plan['code_identity17'],'83ddf2026cdcd2318410812d111296501430506a3ea53d34e4faef62c8d8238a')
        self.assertFalse(any(n.split('.')[0] in {'application','infrastructure','models','torch','numpy'} for n in sys.modules))

    def test_complete_matrix_and_rejected_source_preserved(self):
        matrix=self.plan['cells']
        self.assertEqual(len(matrix),32)
        self.assertEqual(len({b.digest(c) for c in matrix}),32)
        self.assertEqual([sum(c['pilot_stage']==s for c in matrix) for s in ('source','prepare','train','predict')],[11,1,10,10])
        last=self.plan['config_template']['sources'][-1]
        self.assertIsNone(last['artifact'])
        self.assertEqual(last['declared_status'],'CONTRACT_REJECTED')
        self.assertEqual(self.plan['selected_cell']['pilot_arm'],'A-S')
        self.assertEqual(self.plan['selected_cell']['source_id'],'train-0')

    def test_artifact_prediction_matches_actual_input_bytes(self):
        self.assertEqual(len(self.plan['artifact_predictions']),7)
        for p in self.plan['artifact_predictions']:
            raw=(b.HERE/'fixtures'/p['file']).read_bytes();m=p['predicted_metadata']
            self.assertEqual(m['artifact_id'],hashlib.sha256(raw).hexdigest())
            self.assertEqual(m['sha256'],m['artifact_id'])
            self.assertEqual(m['size_bytes'],len(raw))
            self.assertNotIn('DRAFT_metadata_incomplete',m)
            self.assertEqual(m['media_type'],'application/json')
        self.assertEqual(self.plan['geometry']['source_inventory_sha256'],b.digest(self.plan['inventory']))
        self.assertEqual(self.plan['inert_grants_metadata']['source_inventory_sha256'],b.digest(self.plan['inventory']))
        for s in self.plan['config_template']['sources']:
            if s['artifact']:
                grant=next(g for g in self.plan['inert_grants_metadata']['grants'] if s['purpose'] in g['purposes'])
                self.assertIn(s['artifact']['artifact_id'],grant['artifact_ids'])

    def test_live_always_fails_closed_and_placeholders_explained(self):
        for inputs in ({},{'retained_prefix':{'event_id':'UNKNOWN','hash':'0'*64}},
                       {k:'plausible-but-unverified' for k in b.EXTERNALS}):
            with self.assertRaisesRegex(b.BindingError,'SOURCE_ONLY'):
                b.reject_live('live',inputs)
        with self.assertRaisesRegex(b.BindingError,'SOURCE_ONLY'):
            b.build(ROOT,mode='LIVE')
        self.assertTrue(b.placeholder({'hash':'0'*64}))
        self.assertTrue(b.placeholder({'$external':'live_owner_capability'}))

    def test_api_binding_rejects_unknown_keyword(self):
        g=b.InvocationGraph(ROOT)
        with self.assertRaisesRegex(b.BindingError,'unsupported binding'):
            g.add('wrong','infrastructure.research_store','ResearchStore.artifact',[b''],{'imaginary_argument':True})
        with self.assertRaisesRegex(b.BindingError,'unsupported source API'):
            b.symbol(ROOT,'infrastructure.research_store','ResearchStore.imaginary')

    def test_budget_dataclass_rejects_wrong_field(self):
        with self.assertRaisesRegex(b.BindingError,'unknown dataclass fields'):
            b.dataclass_pin(ROOT,'application.research_budget','BudgetSpec',{'wall_seconds':1})
        call=next(c for c in self.plan['calls'] if c['id']=='supervisor_run')
        self.assertEqual(set(call['args'][2]['kwargs']),{'job_seconds','category','pilot_id','pilot_phase'})
        self.assertNotIn('result_validator',call['kwargs'])
        self.assertNotIn('partial_result_validator',call['kwargs'])

    def test_graph_rejects_forward_result(self):
        g=b.InvocationGraph(ROOT)
        g.add('wrong','application.research_budget','BudgetLedger.__init__',[b.ref('future')])
        with self.assertRaisesRegex(b.BindingError,'forward/missing result'):
            g.validate()

    def test_changed_source_bytes_rejected(self):
        first=json.loads((b.HERE/'source-manifest.json').read_bytes())[0]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/first['path'];path.parent.mkdir(parents=True);path.write_bytes(b'changed')
            with self.assertRaisesRegex(b.BindingError,'source bytes changed'):
                b.verify_sources(Path(d))

    def test_changed_fixture_rejected(self):
        import shutil
        with tempfile.TemporaryDirectory() as d:
            target=Path(d);shutil.copyfile(b.HERE/'fixture-manifest.json',target/'fixture-manifest.json')
            shutil.copytree(b.HERE/'fixtures',target/'fixtures')
            first=json.loads((target/'fixture-manifest.json').read_bytes())[0]['file']
            (target/'fixtures'/first).write_bytes(b'changed')
            with patch.object(b,'HERE',target):
                with self.assertRaisesRegex(b.BindingError,'fixture changed'):
                    b.verify_fixtures()

    def test_determinism_and_external_authority(self):
        self.assertEqual(b.encode(self.plan),b.encode(b.build(ROOT)))
        self.assertEqual(self.plan['execution_template']['previous_execution_ref'],b.ext('previous_execution_ref'))
        self.assertEqual(self.plan['config_template']['qualification'],b.ext('qualification'))
        self.assertEqual(self.plan['prepare_callback']['returns'],['package_path','verify','deadline'])
        self.assertEqual(self.plan['mode'],'INERT_ARTIFICIAL_NOT_REGISTERED')
        self.assertEqual(self.plan['closure_inputs']['static_closure_count'],54)
        self.assertEqual(self.plan['closure_inputs']['selected_count'],51)
        terminal=next(c for c in self.plan['calls'] if c['id']=='terminal')
        self.assertIn('settlement_pending=True',terminal['contract'])

if __name__=='__main__':
    unittest.main(verbosity=2)