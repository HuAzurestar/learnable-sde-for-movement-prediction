"""Build an inert Original205 constructor invocation graph; stdlib only.

This module never imports a product module, opens a ResearchStore, invokes a
constructor, or claims a registration. Output refs denote future original API
return values. External refs cannot be satisfied by the DEV49 guard shapes.
"""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path

BASE = 'fe5cc04efc75dc41674f5cde5af47d9078ffc0c5'
PARENT = '67ff7a776bc207eb900e7ea23c7af1246a322a3a'
HERE = Path(__file__).resolve().parent
ARMS = ('M0-UO2','M2-UO2','A-S','A-N','B-S','B-N','C-T-S','C-T-N','C-R-S','C-R-N')
GRID = {'billing_basis':'walking-invocation-grid-v1',
        'limit_basis':'retained-upper-plus-exact-ns-v1'}
EXTERNALS = {
    'runtime_root':'Absolute exclusive non-Git research root; real filesystem/owner evidence required.',
    'store_id':'Exact existing authorized store identity; no automatic initialization.',
    'original_study':'Verified original spec including exact registered arm family tuples; preserve M0/M2 accounts.',
    'original_policy':'Actual immutable policy with approval ref, original budget/expiry and arms; no replenishment.',
    'original_config':'Actual original config; source/numeric scope must equal new config under _validate.',
    'previous_execution_ref':'Actual latest activation ref or original legacy_ref; never guessed genesis.',
    'retained_prefix':'Actual immediately preceding non-genesis event_id/hash under original lock, read by owner_session.',
    'qualification':'Actual independent manifest and exact ref for this code and GRID_FIELDS; draft passed is not evidence.',
    'decision':'Actual B1 decision manifest/ref, user azurestarhu and exact DECISION-03/SOL-014@v2/B1.',
    'billing_decision':'Actual grid decision manifest/ref, not copied user/approved guard fields.',
    'v1_source_grants':'Exact original source authorization objects; rebind_grant preserves all scope/expiry/evidence.',
    'v1_internal_grant':'Exact original internal authorization; only version/pilot study/internal study may change.',
    'metadata_files':'Actual immutable inventory/grants/geometry file paths and raw byte hashes, written outside Git only after admission.',
    'execution_binding':'Exact binding computed after original register/publish; pinned by _validate/activate_execution.',
    'invocation_id':'Fresh invocation identity from actual admitted owner context.',
    'live_owner_capability':'Non-JSON process/thread-local capability installed by original owner_session with real OS lock.',
    'live_grid':'Original GridClock with actual origin/clock; not DEV49 arithmetic example.',
    'meter':'Original GridOwnerMeter inside original owner_session.',
    'allocation_seconds':'Actual min of remaining account/global/training/stage/expiry/deadline caps; never fixture clock.',
    'allocation_ms':'Actual integer grid allocation from original remaining_ns and deadline; no quota reset.',
    'deadline':'Actual immutable narrowed absolute deadline, including original expiries and Nov03 ceiling.',
    'deadline_ns':'Same deadline in original integer clock domain.',
    'command_builder':'Original source adapter command from walking_pilot._run_pilot.execute.',
    'prepare_callback':'Original lambda output,deadline: prepare_package(store,config,attempt,{},output,deadline).',
    'report':'Minimal205 status/report_path/rows; not full209 report. Original terminal mutates settlement fields.',
    'save_report_bytes':'Original callback writes finite encoded report and returns exact raw bytes before t-star.',
}

class BindingError(ValueError):
    pass

def require(ok, message):
    if not ok:
        raise BindingError(message)

def encode(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')

def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def ext(name):
    require(name in EXTERNALS, 'unknown external '+name)
    return {'$external':name}

def ref(name, *fields):
    return {'$result':name,'path':list(fields)}

def expr(operation, *args):
    return {'$expression':operation,'args':list(args)}

def reject_live(mode='inert', external=None):
    if mode != 'inert':
        supplied=external or {}
        missing=[k for k in EXTERNALS if k not in supplied or placeholder(supplied[k])]
        raise BindingError('SOURCE_ONLY: live construction disabled; missing/placeholder external inputs: '+', '.join(missing)
                           + '; real owner capability cannot be serialized; separate admission and native evidence required')

def placeholder(value):
    if value is None or value == '' or value == {}:
        return True
    if isinstance(value,str):
        return any(x in value.lower() for x in ('unknown','draft','replace','inert','proposed','<','>')) or value == '0'*64
    if isinstance(value,dict):
        return any(str(k).startswith('$') or placeholder(v) for k,v in value.items())
    if isinstance(value,list):
        return not value or any(placeholder(x) for x in value)
    return False

def cells(config):
    """Source projection walking_pilot.cells:42-57, checked by own regression."""
    blocks=[s['block_id'] for s in config['sources']]
    common={'execution_id':config['execution_id'],'seed':2601,'resource_class':'cpu','pilot_id':config['pilot_id']}
    prep=config['preparation_arm']
    rows=[{**common,'arm_id':config['arms'][prep],'pilot_arm':prep,'pilot_stage':'source',
           'source_id':s['source_id'],'block_id':s['block_id']} for s in config['sources']]
    rows.append({**common,'arm_id':config['arms'][prep],'pilot_arm':prep,'pilot_stage':'prepare',
                 'block_id':blocks[0],'lineage_block_ids':blocks})
    for arm in ARMS:
        for stage in ('train','predict'):
            rows.append({**common,'arm_id':config['arms'][arm],'pilot_arm':arm,'pilot_stage':stage,
                         'block_id':blocks[0],'lineage_block_ids':blocks})
    return rows

def symbol(root, module, name):
    path=module.replace('.','/')+'.py'
    raw=(root/path).read_bytes(); tree=ast.parse(raw,filename=path)
    node=tree
    for part in name.split('.'):
        found=[n for n in node.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==part]
        require(len(found)==1, f'unsupported source API {module}:{name}, missing {part}')
        node=found[0]
    require(isinstance(node,ast.FunctionDef),f'not a function: {name}')
    return path,raw,node

def signature(node):
    args=node.args; params=[]
    regular=[*args.posonlyargs,*args.args]; default_start=len(regular)-len(args.defaults)
    for i,a in enumerate(regular):
        if i==0 and a.arg in ('self','cls'):
            continue
        kind=inspect.Parameter.POSITIONAL_ONLY if i<len(args.posonlyargs) else inspect.Parameter.POSITIONAL_OR_KEYWORD
        params.append(inspect.Parameter(a.arg,kind,default=None if i>=default_start else inspect.Parameter.empty))
    if args.vararg: params.append(inspect.Parameter(args.vararg.arg,inspect.Parameter.VAR_POSITIONAL))
    for a,d in zip(args.kwonlyargs,args.kw_defaults):
        params.append(inspect.Parameter(a.arg,inspect.Parameter.KEYWORD_ONLY,
                                        default=inspect.Parameter.empty if d is None else None))
    if args.kwarg: params.append(inspect.Parameter(args.kwarg.arg,inspect.Parameter.VAR_KEYWORD))
    return inspect.Signature(params)

def dataclass_pin(root, module, name, kwargs):
    path=module.replace('.','/')+'.py';raw=(root/path).read_bytes()
    classes=[n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name==name]
    require(len(classes)==1, 'unsupported dataclass '+name)
    node=classes[0]
    require(any(isinstance(d.func if isinstance(d,ast.Call) else d,ast.Name) and (d.func if isinstance(d,ast.Call) else d).id=='dataclass' for d in node.decorator_list),'not a dataclass')
    fields={n.target.id:n for n in node.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name)}
    require(set(kwargs)<=set(fields),'unknown dataclass fields: '+str(set(kwargs)-set(fields)))
    require(all(n.value is not None or k in kwargs for k,n in fields.items()),'missing dataclass field')
    return {'source':path,'sha256':sha(raw),'lines':[node.lineno,node.end_lineno],
            'fields':{k:ast.unparse(n.annotation) for k,n in fields.items()},'bound_kwargs':kwargs}

def verify_sources(root):
    manifest=json.loads((HERE/'source-manifest.json').read_bytes())
    require(len(manifest)==650 and len({x['path'] for x in manifest})==650,'source manifest cardinality')
    for x in manifest:
        raw=(root/x['path']).read_bytes()
        require(sha(raw)==x['sha256'] and len(raw)==x['bytes'],'source bytes changed: '+x['path'])
        oid=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        require(oid==x['oid'],'source Git blob changed: '+x['path'])
    return manifest

def verify_fixtures():
    manifest=json.loads((HERE/'fixture-manifest.json').read_bytes())
    require(len(manifest)==22 and len({x['file'] for x in manifest})==22,'fixture manifest cardinality')
    for x in manifest:
        require(Path(x['file']).name==x['file'],'fixture path')
        raw=(HERE/'fixtures'/x['file']).read_bytes()
        require(sha(raw)==x['sha256'] and len(raw)==x['size_bytes'],'fixture changed: '+x['file'])
    return manifest

class InvocationGraph:
    def __init__(self,root):
        self.root=root; self.calls=[]; self.pins={}
    def add(self,key,module,name,args,kwargs=None,*,receiver=None,phase='future-original',contract=''):
        path,raw,node=symbol(self.root,module,name)
        kwargs=kwargs or {}
        try:
            signature(node).bind(*args,**kwargs)
        except TypeError as error:
            raise BindingError(f'unsupported binding {key}: {error}') from error
        pin={'module':module,'symbol':name,'source':path,'sha256':sha(raw),
             'lines':[node.lineno,node.end_lineno], 'binding_shape':str(signature(node)), 'source_arguments':ast.unparse(node.args)}
        self.pins[module+':'+name]=pin
        self.calls.append({'id':key,'api':module+':'+name,'receiver':receiver,
                           'operation':'construct' if name.endswith('.__init__') else 'call',
                           'callable':module+':'+(name[:-9] if name.endswith('.__init__') else name),
                           'args':args,'kwargs':kwargs,'phase':phase,'contract':contract})
        return ref(key)
    def validate(self):
        seen=set()
        def walk(v):
            if isinstance(v,dict):
                if '$external' in v: require(v['$external'] in EXTERNALS,'unknown external')
                if '$result' in v: require(v['$result'] in seen,'forward/missing result '+v['$result'])
                for x in v.values():walk(x)
            elif isinstance(v,list):
                for x in v:walk(x)
        for call in self.calls:
            require(call['id'] not in seen,'duplicate invocation')
            walk(call);seen.add(call['id'])
        return True

def build(root, *, mode='inert'):
    reject_live(mode); manifest=verify_sources(root); fixtures=verify_fixtures()
    load=lambda name:json.loads((HERE/'fixtures'/name).read_bytes())
    config=load('config.DRAFT.json');inventory=load('inventory.DRAFT.json');geometry=load('geometry.DRAFT.json')
    require(set(config['arms'])==set(ARMS) and len(set(config['arms'].values()))==10,'ten original arms required')
    require(config['arms']['M0-UO2']=='pirc26-m0-o1' and config['arms']['M2-UO2']=='pirc26-m2-o1','original arm accounts')
    require(len(config['sources'])==11 and len({s['source_id'] for s in config['sources']})==11,'eleven sources required')
    require(sum(s['role']=='train' for s in config['sources'])==4,'four training sources')
    require(sum(s['role']=='selection' for s in config['sources'])==7,'seven development sources')
    graph=InvocationGraph(root)
    store=graph.add('store','infrastructure.research_store','ResearchStore.__init__',
                    [ext('runtime_root'),ext('store_id')],{'initialize':False},
                    contract='Existing actual store only. No implicit genesis or initialization; inert graph does not call it.')
    predictions=[]; metadata_by_id={}
    for i in range(7):
        filename='population-pooled.json' if i==0 else f'population-dev-{i-1}.json'
        raw=(HERE/'fixtures'/filename).read_bytes()
        old=inventory['training_artifact'] if i==0 else inventory['development_candidates'][i-1]['artifact']
        predicted={'artifact_id':sha(raw),'sha256':sha(raw),'size_bytes':len(raw),'study_id':old['study_id'],
                   'media_type':'application/json','role':old['role'],'visibility':old['visibility'],
                   'block_ids':sorted(set(old['block_ids']))}
        metadata_by_id[old['artifact_id']]=predicted;predictions.append({'file':filename,'predicted_metadata':predicted})
        graph.add(f'artifact{i}','infrastructure.research_store','ResearchStore.artifact',
                  [{'$fixture_bytes':filename}],{k:predicted[k] for k in ('role','visibility','block_ids','study_id','media_type')},
                  receiver=store,contract='Original returns actual metadata; predicted value must equal that return before further binding.')
    inventory['training_artifact']=metadata_by_id[inventory['training_artifact']['artifact_id']]
    for c in inventory['development_candidates']:
        if c['artifact'] is not None:c['artifact']=metadata_by_id[c['artifact']['artifact_id']]
    inventory_hash=digest(inventory)
    geometry['source_inventory_sha256']=inventory_hash
    geometry['training_artifact_id']=inventory['training_artifact']['artifact_id']
    grants=load('grants-metadata.DRAFT.json')
    grants['source_inventory_sha256']=inventory_hash
    for grant in grants['grants']:
        grant['artifact_ids']=[metadata_by_id[x]['artifact_id'] for x in grant['artifact_ids']]
        grant['source_inventory_sha256']=inventory_hash
    # Guard dictionaries remain artificial data. They are never supplied as live grants.
    config['source_inventory']={'path':expr('field',ext('metadata_files'),'inventory_path'),'sha256':inventory_hash}
    config['source_geometry']={'path':expr('field',ext('metadata_files'),'geometry_path'),'sha256':digest(geometry)}
    config['source_grants']={'path':expr('field',ext('metadata_files'),'grants_path'),'sha256':digest(grants)}
    config['qualification']=ext('qualification');config['billing_decision_ref']=ext('billing_decision')
    for s in config['sources']:
        if s['artifact'] is not None:s['artifact']=metadata_by_id[s['artifact']['artifact_id']]
    matrix=cells(config)
    require(len(matrix)==32 and len({digest(x) for x in matrix})==32,'32 unique cells')
    selected=next(c for c in matrix if c.get('source_id')=='train-0')
    code_paths=[]
    _,_,identity_node=symbol(root,'application.walking_pilot','code_identity')
    tree=ast.parse((root/'application/walking_pilot.py').read_bytes())
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CODE_FILES' for t in n.targets):
            code_paths=list(ast.literal_eval(n.value))
    require(len(code_paths)==17,'source CODE_FILES changed')
    code_hash=digest({p:sha((root/p).read_bytes()) for p in code_paths})
    study={'schema_version':'pirc25-contract-v1','study_id':config['study_id'],
           'experiment_id':expr('field',ext('original_study'),'experiment_id'),
           'comparison_family':expr('field',ext('original_study'),'comparison_family'),
           'protocol_hash':digest({'inventory':inventory_hash,'geometry':digest(geometry)}),
           'code_hash':code_hash,'data_hash':digest(config['sources']),
           'feature_hash':expr('field',ext('original_study'),'feature_hash'),
           'selection_hash':expr('field',ext('original_study'),'selection_hash'),
           'arms':expr('field',ext('original_study'),'arms'),'cells':matrix}
    registered=graph.add('study','infrastructure.research_store','ResearchStore.register',[study,expr('digest',study)],receiver=store,
                          contract='Requires original accurate arm family tuples, 24h cap, all32cells; preserve budget identities.')
    # Each source grant is an external original object; original rebind function alone generates versions.
    for i in range(2):
        rebound=graph.add(f'source_rebind{i}','application.walking_execution','rebind_grant',
                          [expr('item',ext('v1_source_grants'),i),config['study_id'],'v2'])
        graph.add(f'authorize_source{i}','infrastructure.research_store','ResearchStore.authorize',[rebound],receiver=store,
                  contract='Explicit future operator side effect; no current authority or registration.')
    internal=graph.add('internal_rebind','application.walking_execution','rebind_grant',
                        [ext('v1_internal_grant'),config['study_id'],'v2'],{'internal':True})
    graph.add('authorize_internal','infrastructure.research_store','ResearchStore.authorize',[internal],receiver=store)
    for key,value in [('qualification',ext('qualification')),('decision',ext('decision')),('billing_decision',ext('billing_decision'))]:
        graph.add('resolve_'+key,'application.walking_execution','reference',[store,value],
                  contract='Exact externally published manifest_id/sha256, never synthetic passed/user guard.')
    config_hash=graph.add('config_manifest','infrastructure.research_store','ResearchStore.publish',
                          ['dev50-config',config],receiver=store)
    binding={'schema':'walking-execution-grid-v1','execution_id':config['execution_id'],
             'budget_pilot_id':config['pilot_id'],'previous_execution_ref':ext('previous_execution_ref'),
             'study_id':config['study_id'],'config_ref':{'manifest_id':'dev50-config','sha256':config_hash},
             'code_hash':code_hash,'qualification_ref':ext('qualification'),
             'source_inventory_hash':inventory_hash,'source_geometry_hash':digest(geometry),
             'arms':config['arms'],'expires_at':expr('field',ext('original_policy'),'expires_at'),
             'decision_ref':ext('decision'),'billing_decision_ref':ext('billing_decision'),**GRID}
    activation=graph.add('activation','application.walking_execution','activate_execution',[store,binding],
                         contract='Original validation includes original_config/policy/qualification/decision/billing/grant exactness and pending/stop checks. No live call here.')
    ledger=graph.add('ledger','application.research_budget','BudgetLedger.__init__',[store])
    graph.add('policy','application.walking_pilot_budget','policy',[ledger,config['pilot_id']],
              contract='Resolve actual immutable original policy. Never construct new balance or publish reset policy.')
    graph.add('owner','application.walking_pilot_budget','owner_session',
              [ledger,config['pilot_id'],config['execution_id'],ext('invocation_id')],{'config':config,'grid':ext('live_grid')},
              contract='Context manager must remain entered through terminal and physical transaction exit. Original reads retained_prefix itself, installs live capability. No JSON injection.')
    graph.add('checkpoint','application.walking_pilot_budget','GridOwnerMeter.checkpoint',[],receiver=ext('meter'),phase='inside-owner')
    run=graph.add('run','infrastructure.research_store','ResearchStore.register_run',[config['study_id'],selected],receiver=store,phase='inside-owner')
    attempt=graph.add('attempt','infrastructure.research_store','ResearchStore.new_attempt',[run],receiver=store,phase='inside-owner')
    # BudgetSpec is a dataclass: bind its annotated constructor fields structurally below.
    graph.add('supervisor','application.research_supervisor','ResearchSupervisor.__init__',[store],phase='inside-owner')
    graph.add('supervisor_run','application.research_supervisor','ResearchSupervisor.run',
              [attempt,ext('command_builder'),{'$dataclass':'application.research_budget:BudgetSpec',
               'kwargs':{'job_seconds':ext('allocation_seconds'),'category':'pilot','pilot_id':config['pilot_id'],'pilot_phase':'prepare'}}],
              {'resource_plan':{'maximum_result_bytes':4*1024**2,'tree_memory_bytes':2*1024**3},
               'walking_prepare':ext('prepare_callback'),'walking_clock':ext('live_grid'),
               'walking_allocation_ms':ext('allocation_ms'),'execution_deadline_monotonic':ext('deadline'),
               'execution_deadline_ns':ext('deadline_ns')},receiver=ref('supervisor'),phase='inside-owner',
              contract='Original reserves/enqueues/claims once; opens charged stage before prepare, native stop/postcheck/publication then unique known SETTLE. Source result_validator/partial_result_validator absent.')
    graph.add('terminal','application.walking_pilot_budget','GridOwnerMeter.terminal',
              [ext('report'),ext('save_report_bytes')],receiver=ext('meter'),phase='inside-owner',
              contract='All reservations require unique known SETTLE. Raw save before t-star, settlement_pending=True, original append and full verified transaction exit. No209 claim.')
    graph.validate()
    budget_call=next(c for c in graph.calls if c['id']=='supervisor_run')['args'][2]
    graph.pins['application.research_budget:BudgetSpec']=dataclass_pin(root,'application.research_budget','BudgetSpec',budget_call['kwargs'])
    for module,name in [('application.research_budget','BudgetLedger.reserve'),
                        ('application.research_budget','BudgetLedger.settle'),
                        ('application.walking_pilot_budget','GridClock.__init__'),
                        ('application.walking_pilot_budget','GridOwnerMeter.__init__'),
                        ('infrastructure.research_store','ResearchStore.append'),
                        ('infrastructure.research_store','ResearchStore.manifest'),
                        ('application.walking_stage','result_binding')]:
        path,raw,n=symbol(root,module,name)
        graph.pins[module+':'+name]={'source':path,'sha256':sha(raw),'lines':[n.lineno,n.end_lineno],
                                    'binding_shape':str(signature(n)), 'source_arguments':ast.unparse(n.args)}

    # Callback map is separate because Supervisor supplies output/deadline only after RUNNING.
    path,raw,prepare=symbol(root,'application.walking_stage','prepare_package')
    signature(prepare).bind(store,config,attempt,{},object(),object())
    graph.pins['application.walking_stage:prepare_package']={'source':path,'sha256':sha(raw),
        'lines':[prepare.lineno,prepare.end_lineno],'binding_shape':str(signature(prepare)), 'source_arguments':ast.unparse(prepare.args)}
    return {'schema':'dev50-source-bound-invocations-v1','mode':'INERT_ARTIFICIAL_NOT_REGISTERED',
            'candidate':BASE,'parent':PARENT,'original_source_count':len(manifest),'code_identity17':code_hash,
            'fixture_manifest':fixtures,'external_inputs':EXTERNALS,'api_pins':graph.pins,'calls':graph.calls,
            'artifact_predictions':predictions,'inventory':inventory,'geometry':geometry,'inert_grants_metadata':grants,
            'config_template':config,'study_template':study,'execution_template':binding,'cells':matrix,'selected_cell':selected,
            'prepare_callback':{'api':'application.walking_stage:prepare_package','args':[store,config,attempt,{},
                                  {'$callback_arg':'output'},{'$callback_arg':'deadline'}],
                                'returns':['package_path','verify','deadline'],
                                'result_binding':['execution_id','attempt_id','cell_hash','input_hash=sha256(package_bytes)']},
            'closure_inputs':{'dev48_closure_sha256':sha((HERE/'dev48-closure.json').read_bytes()),
                              'dynamic_sites_sha256':sha((HERE/'dev48-dynamic-import-resolution.json').read_bytes()),
                              'static_closure_count':len(json.loads((HERE/'dev48-closure.json').read_bytes())['static_closure']),
                              'selected_count':len(json.loads((HERE/'dev48-closure.json').read_bytes())['selected51']),
                              'note':'static closure only; Torch/numpy before source branch;17 identity is not module closure'},
            'unproven':['real non-genesis predecessor','authority and native owner capability','independent qualification',
                        'native imports/WindowsJob/fuse/stop','paid reads/clock/unique settlement/terminal/verified exit',
                        'original205/207/209/SOL022','missing historical gridtest/f220/full651'],
            'event_contracts':[
                {'kind':'MANIFEST','producer':'ResearchStore.publish/register/artifact/register_run',
                 'payload':['object_id','sha256'],'replay':'manifest- + digest([object_id, sha])'},
                {'kind':'WALKING_EXECUTION_ACTIVATED','producer':'activate_execution',
                 'payload':['budget_pilot_id','execution_ref'],'id':'walking-activation- + execution_id'},
                {'kind':'PILOT_OWNER_OPEN','producer':'owner_session',
                 'payload':['pilot_id','execution_id','invocation_id','billing'],
                 'retained_prefix':'original _ref(store._events()[-1]), non-genesis actual history only'},
                {'kind':'WALKING_STAGE_OPEN','producer':'original Supervisor/_grid_stage_open',
                 'ordering':'reservation before protected read; actual s_ns and immutable deadline'},
                {'kind':'SETTLE','producer':'original BudgetLedger.settle',
                 'ordering':'known native stop; unique reservation settlement; unknown never refunded'},
                {'kind':'PILOT_OWNER_TERMINAL','producer':'GridOwnerMeter.terminal',
                 'ordering':'raw save, SHA, t-star, append, full transaction exit',
                 'report':'settlement_pending=True retained; no full209 implication'}],
            'constraints':['No product imports or constructors executed','pilot1800s/training120s/stage120s',
                           'Nov03 23:59HKT/finaltestclosed','user-onlyCOMPLETION/separate release']}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',default='inert')
    args=parser.parse_args()
    plan=build(args.source_root,mode=args.mode)
    args.output.write_bytes(encode(plan)+b'\n')
    print(json.dumps({'mode':plan['mode'],'calls':len(plan['calls']),'sources':11,'arms':10,'cells':len(plan['cells']),
                      'output_sha256':sha(args.output.read_bytes())}))

if __name__=='__main__':
    main()