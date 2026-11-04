"""Synthetic DEV checks only: no product imports, children, native or science."""
from contextlib import contextmanager
from dataclasses import dataclass
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

import adapter as a

ROOT = Path(__file__).resolve().parents[2]
GRAPH = json.loads((ROOT/'qualification_preparation/dev50/invocation-spec.json').read_bytes())
_owner = threading.local()


class Case:
    trace = []
    fault = None
    terminal_fault = None
    exit_fault = None
    metadata_fault = False
    bad_prefix = False
    active = False
    result = None
    prepared = None
    clock_calls = 0


def record(name):
    Case.trace.append(name)


def clock():
    Case.clock_calls += 1
    return 12.0


class ResearchStore:
    def __init__(self, runtime_root, store_id, *, initialize=False, allow_corrupt=False, lock_wait_seconds=300):
        record('store')
        assert initialize is False
        self.store_id = store_id
        self.events = [{'event_id':'retained-history','hash':'a'*64}]
        self.opened = None

    def artifact(self, content, *, role, visibility, block_ids, study_id, media_type='application/json'):
        record('artifact')
        sha = hashlib.sha256(content).hexdigest()
        return {'artifact_id':sha,'sha256':sha,'size_bytes':len(content),'study_id':study_id,
                'media_type':media_type,'role':role,'visibility':visibility,'block_ids':sorted(set(block_ids)),
                **({'unexpected':True} if Case.metadata_fault else {})}

    def register(self, spec, expected_hash):
        record('study'); assert a.digest(spec)==expected_hash
        return {'study_id':spec['study_id'],'spec_hash':expected_hash}

    def authorize(self, authorization):
        record('authorize')

    def publish(self, object_id, manifest):
        record('config_manifest'); return a.digest(manifest)

    def register_run(self, study_id, cell):
        record('run'); assert Case.active; assert cell==GRAPH['selected_cell']
        return 'synthetic-run'

    def new_attempt(self, run_id, *, parent_attempt_id=None, reason=None, source_transition_ref=None):
        record('attempt'); assert run_id=='synthetic-run'; return 'synthetic-attempt'

    @contextmanager
    def _read_transaction(self):
        record('transaction-enter')
        try:
            yield
        finally:
            record('transaction-exit')

    def _events(self):
        return list(self.events)


class BudgetLedger:
    def __init__(self, store):
        record('ledger'); self.store=store


@dataclass
class BudgetSpec:
    job_seconds: float = 7200
    arm_seconds: int = 86400
    category: str = 'job'
    allocation_only: bool = False
    pilot_id: str | None = None
    pilot_phase: str | None = None


class GridClock:
    def __init__(self, origin_ns, clock_ns, invocation_id):
        self.origin_ns=origin_ns; self.clock_ns=clock_ns; self.invocation_id=invocation_id
        self.pid=os.getpid(); self.thread=threading.get_ident(); self.scope=None

    def check(self, ledger, pilot_id):
        record('grid-check')
        assert self.scope==(ledger.store,pilot_id)
        assert _owner.capability==(os.getpid(),ledger.store,pilot_id,self.invocation_id)

    def state(self, ledger):
        self.check(ledger,self.scope[1])
        return {'open':ledger.store.opened}


class GridOwnerMeter:
    # Original meter inherits a variadic __new__; __init__ is the source contract.
    def __new__(cls, *args, grid=None, **kwargs):
        return object.__new__(cls)

    def __init__(self, ledger, config, start, clock, invocation_id, *, grid):
        record('meter-construct'); self.ledger=ledger; self.grid=grid
        assert start==grid.origin_ns/1000000000

    def checkpoint(self):
        record('checkpoint'); assert Case.active

    def terminal(self, report, save):
        record('terminal'); assert Case.active and hasattr(_owner,'capability')
        raw=save(); assert type(raw) is bytes
        record('terminal-physical-exit')
        if Case.terminal_fault:
            raise Case.terminal_fault
        return {'synthetic_receipt':hashlib.sha256(raw).hexdigest()}


@contextmanager
def owner_session(ledger, pilot_id, execution_id, invocation_id, *, config=None, grid=None):
    record('owner-enter'); Case.active=True
    prior=ledger.store.events[-1]
    ledger.store.opened={'event_id':'owner-open-'+invocation_id,'hash':'b'*64,
        'payload':{'billing':{'retained_prefix':{'event_id':prior['event_id'],'hash':prior['hash']}}}}
    if Case.bad_prefix:
        ledger.store.opened['payload']['billing']['retained_prefix']['hash']='c'*64
    ledger.store.events.append(ledger.store.opened)
    grid.scope=(ledger.store,pilot_id)
    _owner.capability=(os.getpid(),ledger.store,pilot_id,invocation_id); _owner.grid=grid
    try:
        yield
    finally:
        del _owner.capability; del _owner.grid
        grid.scope=None; Case.active=False; record('owner-exit')
        if Case.exit_fault:
            raise Case.exit_fault


def policy(ledger, pilot_id, *, live=True):
    record('policy'); return {'synthetic':True}


def rebind_grant(original, study_id, version, *, internal=False):
    record('rebind'); return {**original,'version':version,'pilot_study_id':study_id}


def reference(store, ref):
    record('reference'); return ref


def activate_execution(store, binding):
    record('activation'); return binding


def prepare_package(store, config, attempt_id, inputs, output, deadline):
    record('prepare'); assert attempt_id=='synthetic-attempt' and inputs=={}
    def verify(result=None):
        record('verify-none' if result is None else 'verify-result')
    Case.prepared=(Path(output).with_name('inert-package.json'),verify,deadline-.001)
    return Case.prepared


class ResearchSupervisor:
    def __init__(self, store, *, monotonic=time.monotonic):
        record('supervisor'); self.store=store

    def run(self, attempt_id, command_builder, budget, *, result_validator=None, resource_plan=None,
            checkpoint_handler=None, resource_sharing=None, partial_result_validator=None,
            execution_deadline_monotonic=None, walking_prepare=None, walking_clock=None,
            walking_allocation_ms=None, execution_deadline_ns=None):
        record('supervisor_run'); assert Case.active
        assert isinstance(budget,BudgetSpec) and budget.job_seconds==.1 and budget.arm_seconds==86400
        assert result_validator is partial_result_validator is None
        assert resource_plan=={'maximum_result_bytes':4194304,'tree_memory_bytes':2147483648}
        assert command_builder(Path('inert-result.json'))==['SYNTHETIC_ONLY']
        result=walking_prepare(Path('inert-result.json'),execution_deadline_monotonic)
        assert result is Case.prepared
        record('synthetic-stop'); result[1](); result[1]({'stage_binding':'SYNTHETIC'})
        record('synthetic-publication'); record('synthetic-known-settle')
        if Case.fault:
            raise Case.fault
        # Deliberately return an unsuccessful result. Adapter must not manufacture success.
        Case.result={'attempt_id':attempt_id,'state':'FAILED','artifact_id':None,'elapsed_ms':1,
                     'exit_code':2,'error_code':'SYNTHETIC_FAILURE'}
        return Case.result


def providers():
    classes={a.STORE:ResearchStore,a.LEDGER:BudgetLedger,a.BUDGET:BudgetSpec,
             a.GRID:GridClock,a.METER:GridOwnerMeter,a.SUPERVISOR:ResearchSupervisor}
    functions={a.OWNER:owner_session,a.PREPARE:prepare_package,
        'application.walking_pilot_budget:policy':policy,
        'application.walking_execution:reference':reference,
        'application.walking_execution:rebind_grant':rebind_grant,
        'application.walking_execution:activate_execution':activate_execution}
    apis={**classes,**functions}
    for c in GRAPH['calls']:
        api=c['callable']
        if api not in apis:
            cls,attr=api.rsplit('.',1); apis[api]=getattr(classes[cls],attr)
    return a.Bindings(apis,_owner,clock)


def inputs(tmp):
    def command(output): return ['SYNTHETIC_ONLY']
    def save():
        record('save-zero-args')
        raw=a.encode(report); (Path(tmp)/'inert-report.json').write_bytes(raw); return raw
    report={'status':'SYNTHETIC_ONLY','report_path':str(Path(tmp)/'inert-report.json'),'rows':[]}
    result={k:{'synthetic_input':k} for k in set(GRAPH['external_inputs'])-a.DERIVED}
    result.update(runtime_root=str(Path(tmp).resolve()),store_id='synthetic-existing',
        invocation_id='synthetic-invocation',live_grid=GridClock(1000000000,lambda:1000000000,'synthetic-invocation'),
        allocation_ms=100,allocation_seconds=.1,deadline=12.,deadline_ns=12000000000,
        report=report,save_report_bytes=save,command_builder=command,
        qualification={'manifest_id':'synthetic-qualification','sha256':'a'*64},
        decision={'manifest_id':'synthetic-decision','sha256':'b'*64},
        billing_decision={'manifest_id':'synthetic-billing','sha256':'c'*64},
        previous_execution_ref={'policy_hash':'d'*64,'config_hash':'e'*64,'study_hash':'f'*64},
        original_study={'arms':[],'experiment_id':'synthetic-experiment','comparison_family':'synthetic-family',
                        'feature_hash':'d'*64,'selection_hash':'e'*64},
        original_policy={'expires_at':'2026-11-03T23:59:00+08:00'},v1_source_grants=[{},{}],v1_internal_grant={},
        metadata_files={'inventory_path':str(Path(tmp)/'inventory.json'),'grants_path':str(Path(tmp)/'grants.json'),
                        'geometry_path':str(Path(tmp)/'geometry.json')})
    return result


class SyntheticDEV(unittest.TestCase):
    def setUp(self):
        Case.trace=[]; Case.fault=Case.terminal_fault=Case.exit_fault=None
        Case.metadata_fault=Case.bad_prefix=Case.active=False
        Case.clock_calls=0; Case.result=Case.prepared=None
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.e=inputs(self.tmp.name); self.b=providers()

    def adapter(self): return a.Adapter(ROOT,self.b,self.e)

    def rejected(self):
        with self.assertRaises((a.BoundaryError,TypeError,ValueError,KeyError)):
            self.adapter()
        self.assertEqual([],Case.trace)
        self.assertEqual(0,Case.clock_calls)

    def test_order_real_results_and_callbacks(self):
        obj=self.adapter(); self.assertEqual([],Case.trace)
        result=obj.run()
        self.assertEqual(tuple(result),tuple(c['id'] for c in GRAPH['calls']))
        self.assertIs(result['supervisor_run'],Case.result)
        self.assertEqual('FAILED',result['supervisor_run']['state'])
        self.assertIsNone(result['owner'])
        positions=[Case.trace.index(x) for x in ('store','study','activation','ledger','policy','meter-construct',
            'owner-enter','checkpoint','run','attempt','supervisor','supervisor_run','prepare','synthetic-stop',
            'verify-none','verify-result','synthetic-publication','synthetic-known-settle','terminal',
            'save-zero-args','terminal-physical-exit','owner-exit')]
        self.assertEqual(sorted(positions),positions)
        self.assertEqual(7,Case.trace.count('artifact'))
        self.assertEqual(3,Case.trace.count('authorize'))
        self.assertFalse(Case.active); self.assertFalse(hasattr(_owner,'capability'))
        self.assertIsNone(self.e['live_grid'].scope)
        self.assertNotIn('live_owner_capability',result)
        with self.assertRaises(TypeError): result['owner']='forged'
        with self.assertRaises(a.BoundaryError): obj.run()

    def test_missing_external(self): del self.e['qualification']; self.rejected()
    def test_unknown_external(self): self.e['mystery']=1; self.rejected()
    def test_placeholder_external(self): self.e['store_id']='UNKNOWN'; self.rejected()
    def test_serialized_deferred_external(self): self.e['qualification']={'$external':'x'}; self.rejected()
    def test_serialized_owner(self): self.e['live_owner_capability']={}; self.rejected()
    def test_serialized_grid(self): self.e['live_grid']={}; self.rejected()
    def test_malformed_immutable_reference(self): self.e['qualification']={}; self.rejected()
    def test_malformed_predecessor(self): self.e['previous_execution_ref']={}; self.rejected()
    def test_occupied_owner_thread(self):
        _owner.capability=('existing',)
        try: self.rejected()
        finally: del _owner.capability
    def test_bad_metadata_path(self): self.e['metadata_files']['inventory_path']=None; self.rejected()
    def test_default_type_conformance(self):
        class WrongStore(ResearchStore):
            def __init__(self, runtime_root, store_id, *, initialize=0, allow_corrupt=False, lock_wait_seconds=300): pass
        self.b.apis[a.STORE]=WrongStore; self.rejected()
    def test_missing_api(self): del self.b.apis[a.PREPARE]; self.rejected()
    def test_wrong_signature(self):
        self.b.apis[a.PREPARE]=lambda store: None; self.rejected()
    def test_wrong_method_receiver_binding(self):
        def artifact(self, content, *, role, visibility, block_ids, study_id, media_type='application/json'): pass
        self.b.apis[a.STORE+'.artifact']=artifact; self.rejected()
    def test_owner_local_provenance(self): self.b=a.Bindings(self.b.apis,threading.local(),clock); self.rejected()
    def test_wrong_thread(self): self.e['live_grid'].thread=-1; self.rejected()
    def test_wrong_pid(self): self.e['live_grid'].pid=-1; self.rejected()
    def test_wrong_allocation(self): self.e['allocation_seconds']=.2; self.rejected()
    def test_wrong_deadline(self): self.e['deadline_ns']=1; self.rejected()
    def test_missing_static_expression_field(self): del self.e['original_study']['arms']; self.rejected()
    def test_missing_static_expression_item(self): self.e['v1_source_grants']=[]; self.rejected()
    def test_graph_mutation_before_call(self):
        obj=self.adapter(); obj.graph['calls'][0]['args'][0]={'$external':'unknown'}
        with self.assertRaises(a.BoundaryError): obj.run()
        self.assertEqual([],Case.trace)
    def test_metadata_mismatch_stops_binding(self):
        obj=self.adapter(); Case.metadata_fault=True
        with self.assertRaises(a.BoundaryError): obj.run()
        self.assertEqual(['store','artifact'],Case.trace)
    def test_body_error_terminal_and_owner_cleanup(self):
        obj=self.adapter(); error=RuntimeError('synthetic body'); Case.fault=error
        with self.assertRaises(RuntimeError) as caught: obj.run()
        self.assertIs(error,caught.exception)
        self.assertIn('terminal-physical-exit',Case.trace)
        self.assertEqual('owner-exit',Case.trace[-1]); self.assertFalse(hasattr(_owner,'capability'))
    def test_terminal_error_propagates(self):
        obj=self.adapter(); error=OSError('synthetic terminal exit'); Case.terminal_fault=error
        with self.assertRaises(OSError) as caught: obj.run()
        self.assertIs(error,caught.exception); self.assertEqual('owner-exit',Case.trace[-1])
    def test_body_primary_survives_terminal_and_unlock_errors(self):
        obj=self.adapter(); primary=RuntimeError('synthetic primary'); Case.fault=primary
        Case.terminal_fault=OSError('synthetic terminal'); Case.exit_fault=LookupError('synthetic unlock')
        with self.assertRaises(RuntimeError) as caught: obj.run()
        self.assertIs(primary,caught.exception); self.assertFalse(hasattr(_owner,'capability'))
    def test_owner_exit_failure_not_success(self):
        obj=self.adapter(); error=OSError('synthetic unlock'); Case.exit_fault=error
        with self.assertRaises(OSError) as caught: obj.run()
        self.assertIs(error,caught.exception)
    def test_retained_prefix_not_caller_fabrication(self):
        obj=self.adapter(); Case.bad_prefix=True
        with self.assertRaises(a.BoundaryError): obj.run()
        self.assertNotIn('checkpoint',Case.trace); self.assertIn('terminal',Case.trace)
        self.assertEqual('owner-exit',Case.trace[-1])
    def test_typed_refs_and_errors(self):
        obj=object(); r=a.Resolver({'x':{'field':[7]}},{'store':obj},{'f':b'inert'})
        self.assertIs(obj,r.resolve({'$result':'store','path':[]}))
        self.assertEqual(7,r.resolve({'$expression':'item','args':[{'$expression':'field','args':[{'$external':'x'},'field']},0]}))
        for value in ({'$result':'forward','path':[]},{'$result':'store','path':['attr']},
                      {'$expression':'eval','args':[]},{'$fixture_bytes':'unknown'},
                      {'$external':'x','extra':1},{'$callback_arg':'command'}):
            with self.assertRaises(a.BoundaryError): r.resolve(value)
    def test_baseline_bytes_and_source_default_contract(self):
        self.assertEqual(686,a.verify_baseline(ROOT))
        sig=a.source_signature(ROOT,a.BUDGET)
        self.assertEqual(7200,sig.parameters['job_seconds'].default)
        self.assertEqual(86400,sig.parameters['arm_seconds'].default)
        self.assertEqual(time.monotonic,a.source_signature(ROOT,a.SUPERVISOR+'.__init__').parameters['monotonic'].default)
    def test_cli_remains_disabled(self): self.assertEqual(2,a.main()); self.assertEqual([],Case.trace)
    def test_no_product_imports(self):
        self.assertFalse(any(k=='torch' or k.startswith(('application.','infrastructure.','models.','estimation.','evaluation.')) for k in sys.modules))


if __name__=='__main__':
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout,verbosity=2))