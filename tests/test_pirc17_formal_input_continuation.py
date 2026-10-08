"""Semantic projection negatives, not real input qualification or approval."""
from copy import deepcopy

import pytest

from experiments.pirc17 import formal_input_interruption as module
from experiments.pirc17.protocol_core import digest,envelope,unpack


@pytest.fixture
def context(monkeypatch):
    transport=dict(execution_sha256=digest('OLD SOFTWARE EXECUTION'),approval_sha256=digest('NOT HUMAN'),
        population_sha256=digest('SOFTWARE POPULATION'),access_started_sha256=digest('SOFTWARE ACCESS'))
    population=envelope(dict(execution_sha256=transport['execution_sha256'],eligibility_sha256=digest('REPORT'),
        full_population_identity_sha256=digest('SOURCE POPULATION'),
        selection=dict(selected=[dict(sample_id=digest('SAMPLE'),independent_block_id='BLOCK')],secondary_selected=['SAMPLE']),
        prior_performance_reads=0))
    report=envelope(dict(execution_sha256=transport['execution_sha256'],rows=[dict(sample_id=digest('SAMPLE'),eligible=True)],
        protocol_sha256=digest('PROTOCOL')))
    value=envelope(dict(execution_sha256=transport['execution_sha256'],protocol_sha256=digest('PROTOCOL'),matrix_sha256=digest('MATRIX'),
        input_identity=envelope(dict(**transport,training_policy_sha256=digest('TRAINING POLICY'),sample_counts=dict(train=328,adapt=76,validation=81))),
        map_catalog=envelope(dict(**transport,asset_sha256={'map.tif':digest('MAP BYTES')},query_policy=dict(step=5))),
        population=population,prior_evidence=envelope(dict(training_rows=[dict(velocity=[1.,2.])])),
        scoring_inputs=envelope(dict(execution_sha256=transport['execution_sha256'],population_sha256=transport['population_sha256'],
            positions_access_started_sha256=digest('ACCESS'),targets=[dict(elapsed_seconds=[60,300,900,1800],positions_m=[[1.,2.]])])),
        causal_prefixes=[dict(population_sha256=transport['population_sha256'],window_sha256=digest('WINDOW'),
            condition_sha256=digest('RAW SOURCE'),origin_epoch_ns=123,terrain_prefix_positions_m=[[0.,0.]],score_seconds=[60,300,900,1800])],
        truth_passed_to_predictors=False,raw_sources_independently_reloaded=False))
    monkeypatch.setattr(module,'REGISTERED_CONTEXT_SEMANTICS',module.context_semantic_identity(value))
    monkeypatch.setattr(module,'REGISTERED_ELIGIBILITY_SEMANTICS',module.eligibility_semantic_identity(report,population))
    return value,report,population


def test_only_explicit_execution_approval_access_and_population_transport_may_change(context):
    record,report,population=context
    v=deepcopy(unpack(record));v['execution_sha256']=digest('NEW SOFTWARE EXECUTION')
    for key,fields in [('input_identity',['execution_sha256','approval_sha256','population_sha256','access_started_sha256']),
        ('map_catalog',['execution_sha256','approval_sha256','population_sha256','access_started_sha256']),
        ('population',['execution_sha256','eligibility_sha256']),
        ('scoring_inputs',['execution_sha256','population_sha256','positions_access_started_sha256'])]:
        p=unpack(v[key])
        for name in fields:p[name]=digest('NEW '+name)
        v[key]=envelope(p)
    v['causal_prefixes'][0]['population_sha256']=digest('NEW POPULATION SEAL')
    module.verify_continued_context(envelope(v))
    p=deepcopy(unpack(population));p['execution_sha256']=digest('NEW SOFTWARE EXECUTION');p['eligibility_sha256']=digest('NEW REPORT')
    r=deepcopy(unpack(report));r['execution_sha256']=digest('NEW SOFTWARE EXECUTION')
    module.verify_continued_population(envelope(r),envelope(p))


@pytest.mark.parametrize('fault',['protocol','matrix','sample','order','window','source','clock','prefix','score_time','target',
    'prior','map_asset','map_policy','training_policy','training_count','truth','extra_science'])
def test_new_execution_never_allows_changed_input_science(context,fault):
    record,_,_=context;v=deepcopy(unpack(record))
    if fault in {'protocol','matrix'}:v[fault+'_sha256']=digest('CHANGED '+fault)
    elif fault in {'sample','order'}:
        p=unpack(v['population'])
        if fault=='sample':p['selection']['selected'][0]['sample_id']=digest('REPLACEMENT SAMPLE')
        else:p['selection']['secondary_selected']=['REORDERED']
        v['population']=envelope(p)
    elif fault in {'window','source','clock','prefix','score_time'}:
        p=v['causal_prefixes'][0]
        if fault=='window':p['window_sha256']=digest('OTHER WINDOW')
        elif fault=='source':p['condition_sha256']=digest('OTHER SOURCE')
        elif fault=='clock':p['origin_epoch_ns']+=1
        elif fault=='prefix':p['terrain_prefix_positions_m'][0][0]+=1
        else:p['score_seconds'][-1]=300
    elif fault=='target':
        p=unpack(v['scoring_inputs']);p['targets'][0]['positions_m'][0][0]+=1;v['scoring_inputs']=envelope(p)
    elif fault=='prior':
        p=unpack(v['prior_evidence']);p['training_rows'][0]['velocity'][0]+=1;v['prior_evidence']=envelope(p)
    elif fault.startswith('map_'):
        p=unpack(v['map_catalog'])
        if fault=='map_asset':p['asset_sha256']['map.tif']=digest('OTHER MAP')
        else:p['query_policy']['step']=10
        v['map_catalog']=envelope(p)
    elif fault.startswith('training_'):
        p=unpack(v['input_identity'])
        if fault=='training_policy':p['training_policy_sha256']=digest('OTHER POLICY')
        else:p['sample_counts']['train']+=1
        v['input_identity']=envelope(p)
    elif fault=='truth':v['truth_passed_to_predictors']=True
    else:v['unregistered_scientific_option']=True
    with pytest.raises(ValueError,match='original.*semantics'):module.verify_continued_context(envelope(v))


@pytest.mark.parametrize('fault',['row','selection','identity','extra'])
def test_eligibility_change_is_detected_before_position_loading(context,fault):
    _,report,population=context;r=deepcopy(unpack(report));p=deepcopy(unpack(population))
    if fault=='row':r['rows'][0]['eligible']=False
    elif fault=='selection':p['selection']['selected'].append(dict(sample_id=digest('NEW'),independent_block_id='OTHER'))
    elif fault=='identity':p['full_population_identity_sha256']=digest('OTHER SOURCE POPULATION')
    else:r['unregistered_eligibility_policy']=True
    with pytest.raises(ValueError,match='ordered population'):module.verify_continued_population(envelope(r),envelope(p))
