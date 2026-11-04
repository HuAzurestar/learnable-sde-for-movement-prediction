"""Generate DEV52 structural receipts; reads no live transcript or product module."""
from pathlib import Path
import hashlib
import json
import adapter as a

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PM=Path('E:/Worktable/Project-Management')
INPUT=PM/'feature/PIRC-26/evidence/NEXT1007/dev52-input'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value): (HERE/name).write_bytes(json.dumps(value,indent=2,ensure_ascii=False).encode('utf-8')+b'\n')
index=json.loads((INPUT/'INDEX.json').read_bytes())
assert len(index)==341
assert sha(INPUT/'reader.py')=='19a8d65acc1624af72d512fceb5225006063d2a14bcd2874214d44bf42244843'
assert a.verify_baseline(ROOT)==686
for row in index:
    assert sha(INPUT/row['file'])==row['sha256']
write('read-receipts.json',{
    'revision':'DEV52-read-receipts-v1','kind':'developer receipt; terminal raw audit remains main solecollector responsibility',
    'run':'01a19a0d-8dc6-7923-b341-a1e71a9fd992','session':'01a19a0d-767f-79f1-a932-426e43c52900',
    'initial_assignment':'PIRC-26/DEV-52@v1 only; fresh pureDEV, no subagents or inherited DEV/TEST/REVIEW conversation',
    'protocol':'C:/Users/10655/.agents/skills/long-feature-development/SKILL.md',
    'protocol_sha256':sha('C:/Users/10655/.agents/skills/long-feature-development/SKILL.md'),
    'runtime_profile':'C:/Users/10655/.agents/runtime/long-feature-development/state/project-management.yaml',
    'runtime_profile_sha256':sha('C:/Users/10655/.agents/runtime/long-feature-development/state/project-management.yaml'),
    'index_sha256':sha(INPUT/'INDEX.json'),'reader_sha256':sha(INPUT/'reader.py'),
    'qualified_page_range':'000..340 inclusive; separately displayed full UTF8 BEGIN/END; one functions.exec/one nested exec_command/max_output_tokens4000; actual raw is audit authority',
    'command_template':"$r='E:/Worktable/Project-Management/feature/PIRC-26/evidence/NEXT1007/dev52-input'; $i=Get-Content -Raw -Encoding UTF8 -LiteralPath \"$r/INDEX.json\" | ConvertFrom-Json; python \"$r/reader.py\" \"$r/{page:03d}.txt\" $i[{page}].sha256",
    'task_sha256':'aa401512be2dcac5f59e47929c271e389a03abed9547aec2e2d7284329e9b452',
    'contract_sha256':'2bdf68772084a35469e55393e9abf5bcfc2383ccb64d4a1d44e605106925f0d7',
    'project_sha256':'6f98334c5b3f422606166fa965022103fdbb861a0b351b405d178addc2ddb8f5',
    'status_old_sha256':'99f64222c9964af68b4be3c82af2d5a487a33b9a0c70bc78f90e61fe93031cce',
    'status_snapshot_sha256':sha(HERE/'STATUS-read-snapshot.txt'),
    'status_fullread':'old parts0..23, exact666-char new prepend and proven old suffix SHA, overlapping snapshot parts24..31; part25 qualified reread after oversized path framing',
    'semantic_notes':'read-notes.md; completion notes written before adapter.py or any analysis helper/selftest',
    'pages':index})
g=json.loads((ROOT/'qualification_preparation/dev50/invocation-spec.json').read_bytes())
apis={c['callable'] for c in g['calls']}|{a.BUDGET,a.GRID,a.METER,a.PREPARE}
constructors={a.STORE,a.LEDGER,a.SUPERVISOR,a.GRID,a.METER}
write('api-contract.json',{
    'revision':'DEV52-api-input-v1','baseline':a.BASE,'graph_sha256':a.GRAPH_SHA,
    'entry':'Adapter(repository_root, Bindings(apis, owner_local, monotonic), externals).run()',
    'cli':'always disabled, exit2; no default provider/import path',
    'providers':{api:str(a.source_signature(ROOT,api+'.__init__' if api in constructors else api)) for api in sorted(apis)},
    'external_inputs':{k:v for k,v in g['external_inputs'].items() if k not in a.DERIVED},
    'derived_inputs':{'meter':'GridOwnerMeter(ledger,resolved_config,grid.origin_ns/1e9,explicit_monotonic,invocation_id,grid=grid) before owner enter',
        'prepare_callback':'closure from pinned prepare graph; delegates prepare_package(store,config,attempt,{},output,deadline), returns exact tuple unchanged',
        'live_owner_capability':'actual shared original _owner thread-local inside owner; current PID/store/pilot/invocation identity and original grid.check delegated',
        'retained_prefix':'actual original grid.state open billing ref checked against immediately preceding original _events entry inside original read transaction',
        'execution_binding':'resolved activation argument; original activate_execution remains validator'},
    'call_order':[c['id'] for c in g['calls']],
    'guarantee_boundary':'preflight validates binding shapes/static refs and source signatures, not original authority/resource/native/scientific qualification or callable implementation authenticity',
    'unknown':['actual original APIs imported/frozen/invoked','actual OS owner lock/native tree/fuse/stop','original205/207/209/SOL022 qualification',
               'actual resource plan/admission/positive balance/live grant/read/expiry','missingf220/full651/gridtest','independent TEST28/REVIEW14','user COMPLETION and separate release']})
write('baseline-check.json',{'revision':'DEV52-baseline-v1','physical_and_manifest_git_blob_digests':a.verify_baseline(ROOT),
    'baseline':a.BASE,'parent':'fe5cc04efc75dc41674f5cde5af47d9078ffc0c5','manifest_sha256':a.MANIFEST_SHA,
    'product_imports_or_execution':False,'scope':'mechanical immutable bytes, sizes, SHA256 and Git blob SHA1 only'})
print('341 page physical hashes rechecked; structural read receipt, API contract and 686-byte/blob-digest audit written')