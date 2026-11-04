"""Minimal original205 future driver. No original imports at module/CLI preflight time.

Native construction is a FUTURE, separately admitted action. Only new stdlib
code and synthetic injected bindings may be exercised during DEV54.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import stat
import sys
import time
import uuid

BASE = '7f426dfa4fe0f0666ff9cec69301a81f6acb183c'
BASELINE_MANIFEST_SHA256 = '6c5d1acc3c23842391d1ccbc670284d07acfa1c8d5c13b907d346b6bc0be41fa'
EXPIRY = '2026-11-03T23:59:00+08:00'
HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
APIS = (
 'application.research_budget:BudgetLedger','application.research_budget:BudgetSpec',
 'application.research_supervisor:ResearchSupervisor','application.research_supervisor:ResearchSupervisor.run',
 'application.walking_execution:activate_execution','application.walking_execution:rebind_grant','application.walking_execution:reference',
 'application.walking_pilot_budget:GridClock','application.walking_pilot_budget:GridOwnerMeter',
 'application.walking_pilot_budget:GridOwnerMeter.checkpoint','application.walking_pilot_budget:GridOwnerMeter.terminal',
 'application.walking_pilot_budget:owner_session','application.walking_pilot_budget:policy',
 'application.walking_stage:prepare_package','infrastructure.research_store:ResearchStore',
 'infrastructure.research_store:ResearchStore.artifact','infrastructure.research_store:ResearchStore.authorize',
 'infrastructure.research_store:ResearchStore.new_attempt','infrastructure.research_store:ResearchStore.publish',
 'infrastructure.research_store:ResearchStore.register','infrastructure.research_store:ResearchStore.register_run')

class Refusal(ValueError):
    pass

def require(ok, message):
    if not ok:
        raise Refusal(message)

def encode(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')

def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()

def no_redirects(path):
    path=Path(path)
    require(path.is_absolute() and path.absolute()==path.resolve(), 'absolute lexical root required')
    for p in (path,*path.parents):
        if not p.exists() and not p.is_symlink():
            continue
        s=p.lstat()
        require(not stat.S_ISLNK(s.st_mode) and not getattr(s,'st_file_attributes',0)&1024, 'linked/reparse path rejected')
        require(stat.S_ISDIR(s.st_mode), 'root ancestor is not directory')
        require(not (p/'.git').exists(), 'research root must be outside Git')
    return path

def future_root(path):
    path=no_redirects(path)
    require(not path.exists() and not path.is_symlink(), 'existing root rejected; uncertain runs cannot replay')
    require(path.parent.is_dir(), 'explicit existing owned parent required')
    return path

@contextmanager
def claim_root(path):
    """Called only inside external native admission; never on preflight."""
    path=future_root(path)
    path.mkdir(exist_ok=False)
    no_redirects(path)
    token=uuid.uuid4().hex
    marker=path/'.dev54-owner'
    with marker.open('xb') as f:
        f.write(encode({'token':token,'pid':os.getpid(),'root':str(path)})); f.flush(); os.fsync(f.fileno())
    # OS exclusive byte lock, held through original terminal/owner exit.
    with marker.open('r+b') as f:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            identity=marker.stat()
            require(identity.st_nlink==1, 'owner marker linked')
            yield path
            require(marker.stat().st_ino==identity.st_ino and marker.stat().st_nlink==1, 'owner marker replaced')
        finally:
            f.seek(0)
            if os.name=='nt': msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(f,fcntl.LOCK_UN)
    # Deliberately retain all files on success/failure; no deletion or journal repair.

def verify_source(repository):
    repository=Path(repository)
    require(repository.is_absolute() and repository.absolute()==repository.resolve(), 'source root redirected')
    manifest=(HERE/'baseline.json').read_bytes()
    require(hashlib.sha256(manifest).hexdigest()==BASELINE_MANIFEST_SHA256,'baseline manifest differs')
    rows=json.loads(manifest)
    require(len(rows)==704 and len({r['path'] for r in rows})==704,'baseline cardinality differs')
    checked=set()
    for row in rows:
        relative=Path(row['path'])
        require(not relative.is_absolute() and '..' not in relative.parts,'invalid baseline path')
        p=repository/relative
        for directory in p.parents:
            if directory in checked: break
            s=directory.lstat()
            require(stat.S_ISDIR(s.st_mode) and not stat.S_ISLNK(s.st_mode) and not getattr(s,'st_file_attributes',0)&1024,'source directory redirected')
            checked.add(directory)
        s=p.lstat()
        require(stat.S_ISREG(s.st_mode) and s.st_nlink==1 and not getattr(s,'st_file_attributes',0)&1024,'baseline file redirected')
        raw=p.read_bytes()
        require(len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],'baseline bytes differ: '+row['path'])
    return repository

def bind_modules(modules):
    """Map actual module attributes, including unbound methods and shared local.

    Synthetic modules are allowed only in new DEV tests of this mapping function.
    The native loader below obtains modules exclusively from the verified source.
    """
    apis={}
    for api in APIS:
        module,name=api.split(':'); value=modules[module]
        for part in name.split('.'):
            value=getattr(value,part)
        apis[api]=value
    budget=modules['application.walking_pilot_budget']
    require(inspect.unwrap(apis['application.walking_pilot_budget:owner_session']).__globals__.get('_owner') is budget._owner,'owner local differs')
    require(budget.GridClock.check.__globals__.get('_owner') is budget._owner,'grid local differs')
    return apis,budget._owner

def load_native(repository):
    """FUTURE ONLY: caller must already hold separately authorized native scope."""
    repository=verify_source(repository)
    require(os.name=='nt','original 2GiB containment requires Windows')
    require(not any(k.split('.')[0] in {'application','infrastructure','models','estimation','evaluation'} for k in sys.modules),'native context must be clean')
    sys.path.insert(0,str(repository))
    modules={}
    for name in sorted({api.split(':')[0] for api in APIS}|{'application.walking_pilot'}):
        module=importlib.import_module(name)
        require(Path(module.__file__).resolve()==repository/(name.replace('.','/')+'.py'),'foreign module origin: '+name)
        modules[name]=module
    apis,owner=bind_modules(modules)
    p=repository/'qualification_preparation/dev52/adapter.py'
    spec=importlib.util.spec_from_file_location('_dev54_original_adapter',p)
    adapter=importlib.util.module_from_spec(spec); sys.modules[spec.name]=adapter; spec.loader.exec_module(adapter)
    verify_source(repository)
    return modules,adapter,adapter.Bindings(apis,owner,time.monotonic)

def publish_reference(store,name,value):
    sha=store.publish(name,value)
    require(sha==digest(value),'original publication hash differs')
    return {'manifest_id':name,'sha256':sha}

def artificial_baseline(repository,root,modules):
    """FUTURE original-API recipe, not user-provided permission JSON.

    All receipt shapes here are artificial guard inputs in a NEW store. They
    confer no authority over live data/accounts and are never runtime PASS.
    """
    base=Path(repository)/'qualification_preparation/dev50'
    g=json.loads((base/'invocation-spec.json').read_bytes())
    S=modules['infrastructure.research_store']; B=modules['application.research_budget']
    E=modules['application.walking_execution']; P=modules['application.walking_pilot']
    store=S.ResearchStore(root,'dev54-artificial-store',initialize=True)
    # Original artifact constructors publish actual metadata, not a shadow registry.
    for prediction in g['artifact_predictions']:
        meta=prediction['predicted_metadata']
        observed=store.artifact((base/'fixtures'/prediction['file']).read_bytes(),**{k:meta[k] for k in ('role','visibility','block_ids','study_id','media_type')})
        require(observed==meta,'original artifact metadata differs')
    inventory=deepcopy(g['inventory']); geometry=deepcopy(g['geometry'])
    newgrants=deepcopy(g['inert_grants_metadata'])
    oldgrants=deepcopy(newgrants)
    old_id=inventory['study_id']
    for grant in oldgrants['grants']:
        grant.update(version='v1',pilot_study_id=old_id)
        store.authorize(grant)
    oldgrants['pilot_study_id']=old_id
    files={}
    for key,value in [('inventory',inventory),('geometry',geometry),('grants',newgrants),('old-grants',oldgrants)]:
        path=root/(key+'.json'); S.atomic_write(path,S.encode(value)); files[key]=str(path)
    internal=json.loads((base/'fixtures/inert-grants.DRAFT.json').read_bytes())['internal']
    internal.update(version='v1',study_id=old_id,pilot_study_id=old_id,source_inventory_sha256=digest(inventory))
    store.authorize(internal)
    grid_fields={'billing_basis':'walking-invocation-grid-v1','limit_basis':'retained-upper-plus-exact-ns-v1'}
    code=P.code_identity()
    billing=publish_reference(store,'dev54-artificial-billing',{'fixture_only':True,**grid_fields})
    qualification=publish_reference(store,'dev54-artificial-qualification',{'fixture_only':True,'scope':'independent-walking-pilot','status':'passed','code_hash':code,**grid_fields,'billing_decision_ref':billing})
    decision=publish_reference(store,'dev54-artificial-b1',{'fixture_only':True,'user':'azurestarhu','decision_record':'PIRC-26/DECISION-03@v1','solution':'SOL-014@v2/B1','implementation':True})
    old=deepcopy(g['config_template']); old.update(study_id=old_id,schema_version='pirc26-walking-config-v1',qualification=qualification)
    old.pop('execution_id'); old.pop('billing_basis'); old.pop('limit_basis'); old.pop('billing_decision_ref')
    old['source_inventory']={'path':files['inventory'],'sha256':digest(inventory)}
    old['source_geometry']={'path':files['geometry'],'sha256':digest(geometry)}
    old['source_grants']={'path':files['old-grants'],'sha256':digest(oldgrants)}
    old['artifact_grant']={'authorization_id':internal['authorization_id'],'version':'v1'}
    for source in old['sources']: source['authorization_version']='v1'
    arms=[{'arm_id':arm_id,'model_family_id':arm,'method_family_id':'dev54-artificial','objective_id':'uo2','budget_seconds':86400} for arm,arm_id in old['arms'].items()]
    study={'schema_version':'pirc25-contract-v1','study_id':old_id,'experiment_id':'dev54-artificial','comparison_family':'dev54-artificial',
           'protocol_hash':digest({'inventory':digest(inventory),'geometry':digest(geometry)}),'code_hash':code,'data_hash':digest(old['sources']),
           'feature_hash':digest({'fixture_only':'features'}),'selection_hash':digest({'fixture_only':'selection'}),'arms':arms,'cells':P.cells(old)}
    store.register(study,digest(study))
    store.publish('walking-config-'+old['pilot_id'],old)
    approval={'fixture_only':True,'schema_version':'pirc26-user-scoped-pilot-permission-v1','user':'azurestarhu','decision_record':'PIRC-26/DECISION-01@v1',
      'pilot_study_id':old_id,'final_test_authorization':False,'public_export':False,'movement_scope':'walking',
      'walking_speed_assumption':{'U_m_s':old['speed_limit'],'V_m_s':old['speed_limit']},
      'budget':{'deadline_reset':False,'global_scientific_seconds':1800,'per_arm_training_seconds':120,'timeout_policy':'partial-results'}}
    policy={'schema_version':'pirc26-walking-budget-v1','pilot_id':old['pilot_id'],'study_id':old_id,'global_ms':1800000,'training_ms':120000,
      'expires_at':EXPIRY,'motion':'walking','speed_limit':old['speed_limit'],'final_test':False,'arms':old['arms'],'config_hash':digest(old),
      'approval':publish_reference(store,'dev54-artificial-permission',approval)}
    store.publish('walking-pilot-'+old['pilot_id'],policy)
    # Read back authoritative objects and derive the genuine legacy predecessor.
    original=store.manifest('study-'+old_id)['spec']
    original_policy=modules['application.walking_pilot_budget'].policy(B.BudgetLedger(store),old['pilot_id'])
    return dict(store=store,original_study=original,original_config=store.manifest('walking-config-'+old['pilot_id']),original_policy=original_policy,
       v1_source_grants=[store.authorization(x['authorization_id'],version='v1') for x in oldgrants['grants']],
       v1_internal_grant=store.authorization(internal['authorization_id'],version='v1'),
       previous_execution_ref=E.legacy_ref(store,original_policy),qualification=qualification,decision=decision,billing_decision=billing,
       metadata_files={'inventory_path':files['inventory'],'geometry_path':files['geometry'],'grants_path':files['grants']})

def execute_after_admission(repository,runtime_root,admission,argv):
    """FUTURE ONLY. admission is an explicitly supplied in-process context factory.

    A trusted host independently approves the concrete request BEFORE any root,
    native import, constructor or callback. CLI JSON/flags cannot grant admission.
    """
    require(inspect.isfunction(admission),'separate in-process admission context required')
    repository=verify_source(repository); root=future_root(runtime_root)
    request={'scope':'original205-artificial-only','baseline':BASE,'driver':str(HERE/'driver.py'),
             'driver_sha256':hashlib.sha256((HERE/'driver.py').read_bytes()).hexdigest(),'argv':list(argv),'runtime_root':str(root),
             'expiry':EXPIRY,'cwd':str(repository),'native':True,'science':False}
    with admission(request):
        require(Path.cwd().resolve()==repository,'future worker cwd must equal verified repository')
        require(datetime.now(timezone.utc)<datetime.fromisoformat(EXPIRY),'fixed permission ceiling expired')
        with claim_root(root):
            modules,adapter,bindings=load_native(repository)
            values=artificial_baseline(repository,root,modules)
            store=values.pop('store'); pilot=values['original_config']['pilot_id']
            budget=modules['application.walking_pilot_budget']; ledger=modules['application.research_budget'].BudgetLedger(store)
            # Fresh artificial legacy accounting provides a conservative pre-owner
            # request. Original grid reserve/STAGE_OPEN rechecks actual remaining_ns
            # after owner overhead and narrows the same absolute deadline; no reset.
            seconds=min(120.,budget.remaining(ledger,pilot,values['original_config']['arms']['A-S'],'prepare'),
                        (datetime.fromisoformat(EXPIRY)-datetime.now(timezone.utc)).total_seconds())
            ms=math.floor(seconds*1000); require(ms>0,'no positive original allocation')
            invocation=uuid.uuid4().hex; origin=time.monotonic_ns(); grid=budget.GridClock(origin,time.monotonic_ns,invocation)
            deadline=budget.deadline_float(origin+ms*1000000)
            deadline_ns=int(deadline*1000000000)
            report={'status':'MINIMAL205_UNQUALIFIED','report_path':str(root/'205-report.json'),'rows':[]}
            def save():
                raw=modules['infrastructure.research_store'].encode(report)
                modules['infrastructure.research_store'].atomic_write(Path(report['report_path']),raw)
                return raw
            def command(output):
                return [sys.executable,'-m','application.walking_pilot','worker','--runtime-root',str(root),'--store-id',store.store_id,
                        '--pilot-id',pilot,'--attempt',attempt_holder['attempt'],'--worker-stage','source','--inputs','{}','--output',str(output)]
            # Command is built after original new_attempt. Bind its ID from the
            # original store's sole registered selected run, never invent one.
            attempt_holder={}
            def command_bound(output):
                attempts=store.attempts()
                require(len(attempts)==1,'minimal fixture must have one actual attempt')
                attempt_holder['attempt']=next(iter(attempts))
                return command(output)
            values.update(runtime_root=root,store_id=store.store_id,invocation_id=invocation,live_grid=grid,
                          allocation_ms=ms,allocation_seconds=ms/1000,deadline=deadline,deadline_ns=deadline_ns,
                          command_builder=command_bound,report=report,save_report_bytes=save)
            result=adapter.Adapter(repository,bindings,values).run()
            # Do not convert original FAILED results or uncertain settlement to PASS.
            return result

def main(argv=None,*,admission=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('explain','preflight','execute'),default='explain')
    parser.add_argument('--runtime-root',type=Path)
    tokens=list(sys.argv[1:] if argv is None else argv)
    args=parser.parse_args(tokens)
    if args.mode=='execute':
        require(admission is not None,'NATIVE_EXECUTION_NOT_ADMITTED: CLI flags/JSON confer no authority')
        require(args.runtime_root is not None,'explicit fresh root required')
        return execute_after_admission(REPOSITORY,args.runtime_root,admission,[sys.executable,str(HERE/'driver.py'),*tokens])
    if args.mode=='preflight':
        verify_source(REPOSITORY)
        require(args.runtime_root is not None,'explicit prospective root required')
        future_root(args.runtime_root)
    print(json.dumps({'status':'SOURCE_ONLY_NATIVE_UNEXECUTED','driver':str(HERE/'driver.py'),
          'driver_sha256':hashlib.sha256((HERE/'driver.py').read_bytes()).hexdigest(),'baseline':BASE,
          'future_cwd':str(REPOSITORY),'future_argv':[sys.executable,str(HERE/'driver.py'),'--mode','execute','--runtime-root',str(args.runtime_root) if args.runtime_root else '<fresh absolute root>'],
          'entry':'main(argv, admission=trusted_host_context_factory); no CLI admission channel'}))
    return 2

if __name__=='__main__':
    try: raise SystemExit(main())
    except Refusal as error:
        print(str(error)); raise SystemExit(2)
