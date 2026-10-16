"""Synthetic paired-package controls; never the shared PIRC-27 evidence root."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from application.propagation_resource_summary import validate_resource_summaries
from application.research_computation import comparison_plan, paper_identity
from application.research_evidence import expected_metrics_csv, expected_paper_index
from application.research_figures import validate_figure_package
from application.research_comparison import ComparisonRunner
from experiments.pirc25.affine import code_hash
from infrastructure.research_store import ResearchError, digest


PAPER = Path(__file__).resolve().parents[2]/'TSDE-SDE'


def paired(mutation='none'):
    script = '''
import json,sys
sys.path.insert(0,'tests')
from test_resource_comparison import failed_resource_bundle
from scripts.pirc25.comparison import compare_package
from scripts.pirc25.computation_plan import comparison_package_plan
from scripts.pirc25.aggregate import fingerprint,csv_bytes,evidence_index
from scripts.pirc25.figures import comparison_figures
b=failed_resource_bundle()
change=sys.argv[1]
if change in {'unknown-memory','missing-observation','unattempted'}:
 row=b['cells'][0]; source=row['resource_usage']['attempts'][0]
 if change=='unknown-memory':
  packet=source['packet']; packet['memory'].update(status='UNAVAILABLE',resident_counter=None,peak_resident_bytes=None,reason='NATIVE_COUNTER_UNAVAILABLE')
  h=fingerprint(packet); source['artifact'].update(artifact_id=h,sha256=h,size_bytes=len(json.dumps(packet,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()))
  source['publication_event']['payload'].update(artifact_id=h,packet_hash=h)
 elif change=='missing-observation':
  source.update(status='UNAVAILABLE',reason='MISSING_OBSERVATION',artifact=None,packet=None)
  source['publication_event']['payload'].update(status='UNAVAILABLE',reason='MISSING_OBSERVATION',artifact_id=None,packet_hash=None)
 else:
  row.update(status='MISSING',attempt_id=None,run_id=None,history=[])
  row['resource_usage']['attempts']=[]
  row['cost'].update(charged_ms=0,reserved_ms=0,measured_ms=0,sources=[])
 if change!='unattempted':
  event=source['publication_event']; event['hash']=fingerprint({k:v for k,v in event.items() if k!='hash'})
 b['bundle_hash']=fingerprint({k:v for k,v in b.items() if k!='bundle_hash'})
p=compare_package(b,{'manifest_id':'synthetic-resource'},include_resource_summaries=True)
a=p['aggregate']
if change not in {'none','unknown-memory','missing-observation','unattempted'}:
 if change=='drop':a['arms'][0].pop('resource_usage_summary')
 else:a['arms'][0]['resource_usage_summary'][change]+=1
 a['aggregate_hash']=fingerprint({k:v for k,v in a.items() if k!='aggregate_hash'})
 t=csv_bytes(a); f=comparison_figures(a)
 p.update(metrics_csv=t.decode(),paper_index={**evidence_index(a,t),'figure_index_hash':fingerprint(f['figure_index'])},**f)
print(json.dumps({'bundle':b,'package':p,'plan':comparison_package_plan(b,20000000,include_resource_summaries=True)}))
'''
    result = subprocess.run([sys.executable,'-B','-c',script,mutation],cwd=PAPER,
        capture_output=True,check=True,timeout=30)
    return json.loads(result.stdout)


@pytest.fixture(scope='module')
def frozen():
    return paired()


def owner_result(source):
    package = source['package']
    request = {'computation_ref':package['aggregate']['computation_ref'], 'formal':False,
        'runtime_code_hash':code_hash(),'paper_identity':paper_identity(PAPER),
        'resource_plan':comparison_plan(source['bundle'],20_000_000,include_resource_summaries=True),
        'include_resource_summaries':True}
    value = {'schema_version':'pirc25-computation-result-v1','request_hash':digest(request),
             'computation_ref':request['computation_ref'],**package}
    return value,request


def test_independent_resource_plan_serializers_and_figures_match(frozen):
    value = deepcopy(frozen['package'])
    aggregate = value['aggregate']
    assert validate_resource_summaries(aggregate,enabled=True) is True
    runtime = comparison_plan(frozen['bundle'],20_000_000,include_resource_summaries=True)
    assert runtime['planned_operation_bound']==frozen['plan']['planned_operation_bound']
    assert runtime['graph_operation_bound']==frozen['plan']['graph_operation_bound']
    assert expected_metrics_csv(aggregate)==value['metrics_csv'].encode()
    assert expected_paper_index(aggregate,value['metrics_csv'].encode(),value['figure_index'])==value['paper_index']
    validate_figure_package(aggregate,value['figure_index'],value['figures'])
    result,request=owner_result(frozen)
    ComparisonRunner(None,PAPER)._validate_result(result,request,frozen['bundle'])


@pytest.mark.parametrize('mutation',['maximum_observed_self_peak_bytes','pipeline_wall_ns','drop'])
def test_fully_resealed_fake_worker_resource_package_rejected(mutation):
    source=paired(mutation)
    value,request=owner_result(source)
    with pytest.raises(ResearchError):
        ComparisonRunner(None,PAPER)._validate_result(value,request,source['bundle'])


@pytest.mark.parametrize('flag',[1,None,'true'])
def test_opt_in_requires_boolean_before_source_reads(flag,monkeypatch):
    owner=ComparisonRunner(None,PAPER)
    monkeypatch.setattr(owner,'_prepare',lambda *a,**k:pytest.fail('source read before flag refusal'))
    with pytest.raises(ResearchError,match='boolean'):
        owner.run('synthetic','0'*64,authorization_id='none',include_resource_summaries=flag)
    with pytest.raises(ResearchError,match='boolean'):
        comparison_plan({'cells':[]},20_000_000,include_resource_summaries=flag)


def test_new_resource_work_never_fits_the_legacy_allowance(frozen):
    old=comparison_plan(frozen['bundle'],20_000_000)
    with pytest.raises(ResearchError,match='RESOURCE_PLAN_REJECTED'):
        comparison_plan(frozen['bundle'],old['planned_operation_bound'],include_resource_summaries=True)


@pytest.mark.parametrize('variant',['unknown-memory','missing-observation','unattempted'])
def test_partial_and_unattempted_sources_survive_independent_owner_bridge(variant):
    source=paired(variant)
    aggregate=source['package']['aggregate']; summary=aggregate['arms'][0]['resource_usage_summary']
    assert validate_resource_summaries(aggregate,enabled=True)
    assert summary['maximum_observed_self_peak_bytes'] is None
    assert summary['memory_complete'] is False
    if variant=='unknown-memory':
        assert summary['pipeline_wall_ns']==100 and summary['duration_complete'] is True
    elif variant=='missing-observation':
        assert summary['pipeline_wall_ns'] is None and summary['unavailable_attempt_ids']
    else:
        assert summary['pipeline_wall_ns'] is None and summary['unattempted_cell_count']==1
    value,request=owner_result(source)
    ComparisonRunner(None,PAPER)._validate_result(value,request,source['bundle'])


def test_missing_request_mode_is_not_inferred_from_untrusted_output(frozen):
    value,request=owner_result(frozen)
    request['include_resource_summaries']=None
    value['request_hash']=digest(request)
    with pytest.raises(ResearchError,match='CONTRACT_MISMATCH'):
        ComparisonRunner(None,PAPER)._validate_result(value,request,frozen['bundle'])


@pytest.mark.parametrize('attempts',['not-an-array',[None]*100_001])
def test_resource_history_allocation_refuses_before_processing(frozen,attempts):
    bundle=deepcopy(frozen['bundle'])
    bundle['cells'][0]['resource_usage']['attempts']=attempts
    with pytest.raises(ResearchError,match='RESOURCE_PLAN_REJECTED'):
        comparison_plan(bundle,20_000_000,include_resource_summaries=True)


def test_unmanaged_summary_cannot_bypass_paid_worker(frozen,tmp_path):
    from tests.test_research_comparison import comparison_source
    from application.research_evidence import accept_aggregate
    store,_,grant,_,package=comparison_source(tmp_path)
    aggregate=json.loads(store.read_artifact(package['aggregate_id'],purpose='export',authorization=grant))
    aggregate['arms'][0]['resource_usage_summary']=deepcopy(frozen['package']['aggregate']['arms'][0]['resource_usage_summary'])
    aggregate['aggregate_hash']=digest({k:v for k,v in aggregate.items() if k!='aggregate_hash'})
    # Everything except the newly inserted summary is a genuinely accepted source.
    with pytest.raises(ResearchError,match='managed computation'):
        accept_aggregate(store,aggregate,'synthetic')


@pytest.mark.parametrize('default',['False','True','None','0'])
def test_paper_opt_in_api_preflight_does_not_execute_source(tmp_path,default):
    from application.research_computation import require_resource_summary_support
    directory=tmp_path/'scripts'/'pirc25';directory.mkdir(parents=True)
    for filename,name in [('comparison.py','compare_package'),('computation_plan.py','comparison_package_plan')]:
        (directory/filename).write_text(
            f'raise RuntimeError("must never execute")\ndef {name}(*,include_resource_summaries={default}): pass\n',encoding='utf-8')
    if default=='False':
        require_resource_summary_support(tmp_path)
    else:
        with pytest.raises(ResearchError,match='Paper lacks'):
            require_resource_summary_support(tmp_path)


def test_actual_owned_worker_resource_comparison_and_exact_reuse(tmp_path,monkeypatch):
    from application.research_budget import BudgetLedger,BudgetSpec
    from application.research_evidence import export_evidence,accept_evidence_package
    from application.research_query import ResearchQuery
    from infrastructure.research_store import encode
    from tests.test_propagation_resource_evidence import run
    store,spec,_,grant,source=run(tmp_path)
    assert source['state']=='SUCCEEDED'
    bundle=export_evidence(store,spec['study_id'],grant)
    path=tmp_path/'input.json';path.write_bytes(encode(bundle))
    subprocess.run([sys.executable,'-B',str(PAPER/'scripts/pirc25/aggregate.py'),str(path),
        '--expected-hash',bundle['bundle_hash'],'--output',str(tmp_path/'bootstrap')],
        check=True,capture_output=True,timeout=30)
    aggregate=json.loads((tmp_path/'bootstrap'/'aggregate.json').read_bytes())
    package=accept_evidence_package(store,tmp_path/'bootstrap',aggregate['aggregate_hash'])
    owner=ComparisonRunner(store,PAPER);arm=spec['arms'][0]['arm_id']
    before=BudgetLedger(store).balance(arm)['committed_ms']
    options={'authorization_id':grant['authorization_id'],'budget':BudgetSpec(30),'include_resource_summaries':True}
    result=owner.run(spec['study_id'],package['aggregate_hash'],**options)
    assert result['state']=='SUCCEEDED',result
    view=ResearchQuery(store,grant['authorization_id']).comparison(result['comparison']['aggregate_hash'])
    saved=view['aggregate'];summary=saved['arms'][0]['resource_usage_summary']
    assert summary['observed_attempt_count']==1
    assert summary['source_artifact_ids']==[bundle['cells'][0]['resource_usage']['attempts'][0]['artifact']['artifact_id']]
    request=store.manifest(saved['computation_ref']['request_manifest_id'])
    assert request['include_resource_summaries'] is request['resource_plan']['include_resource_summaries'] is True
    derived_id='study-'+request['computation_study_id']
    origin=store.manifest(derived_id)['spec']['computation_origin']
    assert origin['include_resource_summaries'] is True and origin['resource_plan']==request['resource_plan']
    proof=store.manifest(saved['computation_ref']['manifest_id'])
    assert BudgetLedger(store).balance(arm)['committed_ms']==before+proof['cost']['charged_ms']
    count=len(store.attempts());balance=BudgetLedger(store).balance(arm)
    again=owner.run(spec['study_id'],package['aggregate_hash'],**options)
    assert again['reused'] and again['attempt_id']==result['attempt_id']
    assert len(store.attempts())==count and BudgetLedger(store).balance(arm)==balance
    from application.research_computation import verified_computation
    original_manifest=store._manifest
    for where in ('origin','request','plan'):
        object_id=derived_id if where=='origin' else saved['computation_ref']['request_manifest_id']
        document=deepcopy(original_manifest(object_id))
        if where=='origin':document['spec']['computation_origin']['include_resource_summaries']=False
        elif where=='request':document['include_resource_summaries']=None
        else:document['resource_plan'].pop('include_resource_summaries')
        forged={object_id:document}
        if where!='origin':
            # Even rehashing the request and its metadata receipt is no proof
            # that the mode agrees with the frozen origin/worker result.
            receipt_id=saved['computation_ref']['manifest_id']
            forged[receipt_id]=deepcopy(original_manifest(receipt_id))
            forged[receipt_id]['request_hash']=digest(document)
        with monkeypatch.context() as patch:
            patch.setattr(store,'_manifest',lambda key,_docs=forged: _docs[key] if key in _docs else original_manifest(key))
            with pytest.raises(ResearchError,match='CORRUPT_ARTIFACT'):
                verified_computation(store,saved['computation_ref'],aggregate=saved)
    assert len(store.attempts())==count and BudgetLedger(store).balance(arm)==balance
