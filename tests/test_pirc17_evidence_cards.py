"""Review-card software on synthetic full inventory, NOT empirical evidence."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.pirc17 import evidence_cards as module
from experiments.pirc17.evidence_catalog import project as catalog_project
from experiments.pirc17.protocol_core import canonical, digest, envelope, publish
from tests.test_pirc17_checkpoint_export import full_export
from tests.test_pirc17_evidence_catalog import catalog_inputs
from tests.test_pirc17_formal_forecasts import prepared
from tests.test_pirc17_formal_scoring import scored


@pytest.fixture(scope='module')
def card_inputs(full_export,catalog_inputs):
    catalog = catalog_project(*catalog_inputs)
    # SOFTWARE public transport fixture: unify test scope labels only. This
    # does not assert actual provenance/qualification of an empirical catalog.
    for k in ('protocol_sha256','execution_sha256'):
        catalog[k]=full_export['public'][k]
    return envelope(full_export['public']),envelope(catalog)


def test_complete_conservative_review_cards_and_no_private_reader(card_inputs,monkeypatch):
    from experiments.pirc17 import checkpoint_saved,checkpoint_audit,formal_scoring
    def forbidden(*a,**kw): pytest.fail('cards constructed a private reader/scorer')
    monkeypatch.setattr(checkpoint_saved,'CheckpointSavedForecasts',forbidden)
    monkeypatch.setattr(checkpoint_audit,'sources',forbidden)
    monkeypatch.setattr(formal_scoring.ScoringConsumers,'__init__',forbidden)
    value=module.project(*card_inputs)
    assert len(value['terrain_cards'])==4 and len(value['input_cards'])==13
    assert len(value['variant_inventory'])==35
    assert len(value['method_slot_cards'])==36 and len(value['method_arm_cards'])==22
    assert sum(r['definition']['disposition']=='EXCLUDED' for r in value['method_slot_cards'])==8
    assert sum(r['comparison_reference'] is not None for r in value['method_slot_cards'])==21
    assert len(value['kernel_costs'])==38
    assert all(r['expected_forecasts']==290 for r in value['kernel_costs'].values())
    assert value['feature_query_accounting_scope'] == checkpoint_audit.QUERY_ACCOUNTING_SCOPE
    assert all(r['feature_query_accounting']['expected_forecasts']==290 for r in value['kernel_costs'].values())
    assert sum(r['feature_query_accounting']['expected_forecasts'] for r in value['kernel_costs'].values()) == 11020
    assert all(r['feature_query_accounting']['counts_by_status']==r['counts_by_status'] for r in value['kernel_costs'].values())
    assert all(r['verdict']['verdict']=='unavailable' for r in value['terrain_cards'])
    assert value['history']['terrain_verdict'] is None
    terrain_ids={r['card_id'] for r in value['terrain_cards']}
    assert all(set(r['verdict_references'])<=terrain_ids for r in value['input_cards'])
    history=next(r for r in value['input_cards'] if r['disposition']=='baseline-diagnostic')
    assert history['verdict'] is None and history['verdict_references']==[]
    assert value['snapshot_binding']['snapshot_counts']==card_inputs[1]['payload']['snapshot_counts']
    screened=[r for r in value['input_cards'] if r['disposition']=='screened-out']
    assert len(screened)==8
    assert {r['verdict'] for r in screened}=={'unavailable','inconclusive'}
    assert sum(r['disposition']=='not-selected' for r in value['variant_inventory'])==28
    assert value['method_fidelity']['paper_equivalent'] is False
    for key in ('test02_qualification_asserted','scientific_claim_authorized','numerically_qualified','human_accepted'):
        assert value[key] is False
    assert value['new_forecasts']==value['new_fits']==0
    assert value['source_export_sha256']==card_inputs[0]['sha256']
    assert value['source_catalog_sha256']==card_inputs[1]['sha256']
    for token in (b'center_m',b'positions_m',b'loaded_threadpools',b'sample_id'):
        assert token not in canonical(value)


def test_partial_cost_metadata_is_not_complete_latency_or_accuracy():
    rows=[dict(forecast_axes=dict(matrix='terrain',subject='base'),status='failed',
        kernel_prediction_seconds=1.5 if i==0 else 2.5 if i==1 else None) for i in range(290)]
    cost=module._costs(rows,dict(runtime_replay=dict(runtime_subjects=[])))['terrain:base']
    assert cost['expected_forecasts']==290 and cost['counts_by_status']=={'failed':290}
    assert cost['recorded_timing_count']==2 and cost['missing_timing_count']==288
    assert cost['recorded_kernel_total_seconds']==4 and cost['recorded_kernel_mean_seconds']==2
    assert cost['isolated_runtime_references']==[]


@pytest.mark.parametrize('flag',['retired_bootstrap_gate_asserted','raw_trajectories_exported',
    'original_identifiers_exported','scientific_claim_authorized','numerically_qualified','human_accepted'])
def test_source_public_status_cannot_fabricate_authorization(flag):
    scope=dict(protocol_sha256=digest('SOFTWARE protocol'),execution_sha256=digest('SOFTWARE execution'))
    export=dict(schema_version=module.EXPORT_VERSION,**scope)
    export.update({k:False for k in ('retired_bootstrap_gate_asserted','raw_trajectories_exported',
        'original_identifiers_exported','scientific_claim_authorized','numerically_qualified','human_accepted')})
    export[flag]=True
    catalog=dict(schema_version=module.CATALOG_VERSION,**scope,scientific_claim_authorized=False,human_accepted=False)
    with pytest.raises(ValueError,match='unaccepted public'):
        module.project(envelope(export),envelope(catalog))


@pytest.mark.parametrize('change',['verdict','mechanism_flag','score_axes','short_inventory','duplicate_axes',
                                  'short_scores','negative_cost','scope','exclusion','table_value','duplicate_table',
                                  'query_scope','missing_query','query_counter'])
def test_rejects_numeric_lies_scope_changes_and_partial_denominators(card_inputs,change):
    export,catalog=deepcopy(card_inputs[0]['payload']),deepcopy(card_inputs[1]['payload'])
    if change=='verdict': export['recomputed']['factor_conclusions']['road']['verdict']='retain'
    elif change=='mechanism_flag':
        gates=export['recomputed']['modes']['causal_prefix']['mechanism_gates']
        row=next(iter(gates.values()))
        row['passed']=not row['passed']
    elif change=='score_axes': export['inputs']['score_rows'][0]['origin_rank']+=1
    elif change=='short_inventory': export['audit_summary']['work_inventory'].pop()
    elif change=='duplicate_axes':
        rows=[r for r in export['audit_summary']['work_inventory'] if r['kind']=='scientific_forecast']
        rows[0]['forecast_axes']=deepcopy(rows[1]['forecast_axes'])
    elif change=='short_scores': export['inputs']['score_rows'].pop()
    elif change=='negative_cost':
        next(r for r in export['audit_summary']['work_inventory'] if r['kind']=='scientific_forecast')['kernel_prediction_seconds']=-1
    elif change=='query_scope':
        export['audit_summary']['feature_query_accounting_scope']['per_column_validity_rates_available']=True
    elif change in {'missing_query','query_counter'}:
        row=next(r for r in export['audit_summary']['work_inventory'] if r['kind']=='scientific_forecast')
        if change=='missing_query': row.pop('feature_query_accounting')
        else: row['feature_query_accounting']['feature_query_rows']=-1
    elif change=='scope': catalog['protocol_sha256']=digest('another scope')
    elif change=='exclusion': next(r for r in catalog['method_slots'].values() if r['disposition']=='EXCLUDED')['disposition']='REQUIRED'
    elif change=='table_value': export['recomputed']['metric_tables'][0]['expected_forecasts']+=1
    else: export['recomputed']['metric_tables'][0]=deepcopy(export['recomputed']['metric_tables'][1])
    with pytest.raises(ValueError): module.project(envelope(export),envelope(catalog))


def test_pinned_public_cli_reuses_content_and_rejects_wrong_pin(card_inputs,tmp_path):
    export_path,export=publish(tmp_path/'export',card_inputs[0]['payload'])
    catalog_path,catalog=publish(tmp_path/'catalog',card_inputs[1]['payload'])
    args=(export_path,export['sha256'],catalog_path,catalog['sha256'],tmp_path/'cards')
    assert module.generate(*args)==module.generate(*args)
    with pytest.raises(ValueError,match='identity'):
        module.generate(export_path,digest('not the pinned source'),catalog_path,catalog['sha256'],tmp_path/'wrong')


def test_tsde_consumer_reads_actual_producer_card_schema(card_inputs,tmp_path):
    tsde=Path(__file__).resolve().parents[2]/'TSDE-SDE'
    if not (tsde/'scripts/aggregate_pirc17.py').is_file():
        pytest.skip('Cross-repository consumer check requires the declared TSDE checkout')
    value=module.project(*card_inputs)
    path,record=publish(tmp_path/'producer-public-cards',value)
    environment=dict(os.environ,PYTHONPATH=str(tsde),PYTHONIOENCODING='utf-8')
    result=subprocess.run([sys.executable,'-m','scripts.aggregate_pirc17','--cards',str(path),
        '--sha256',record['sha256'],'--output-directory',str(tmp_path/'consumer')],
        cwd=tsde,env=environment,capture_output=True,text=True,encoding='utf-8',timeout=60)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    receipt=json.loads(result.stdout)
    assert receipt['table_row_counts']==dict(comparisons=90,accuracy=120,terrain=4,methods=36,arms=22,
        factors=13,variants=35,kernel_costs=38,terrain_configurations=10,mechanisms=84,
        diagnostic_witnesses=15,variance_witnesses=3,replays=38,runtime_trials=165,runtime_conditions=30,accuracy_times=480)
    manifest=json.loads((Path(receipt['output_directory'])/'manifest.json').read_text(encoding='utf-8'))['payload']
    assert manifest['source_cards_sha256']==record['sha256']
    assert manifest['source_export_sha256']==card_inputs[0]['sha256']
    assert manifest['isolated_runtime_metadata']['registered_counts']==dict(replays=38,cold=75,warmup=15,warm=75)
    assert manifest['runner_cost_snapshot']==value['runner_cost_snapshot']
    assert not manifest['human_accepted'] and not manifest['manuscript_written']
    figures=subprocess.run([sys.executable,'-m','scripts.plot_pirc17','--cards',str(path),
        '--sha256',record['sha256'],'--output-directory',str(tmp_path/'figure-consumer'),'--software-fixture'],
        cwd=tsde,env=environment,capture_output=True,text=True,encoding='utf-8',timeout=120)
    assert figures.returncode==0,figures.stdout+'\n'+figures.stderr
    receipt=json.loads(figures.stdout)
    assert receipt['figure_count']==16 and receipt['image_count']==32
    manifest=json.loads((Path(receipt['output_directory'])/'manifest.json').read_text(encoding='utf-8'))['payload']
    assert manifest['identity']['source_cards_sha256']==record['sha256']
    assert manifest['identity']['software_fixture_only'] is True and not manifest['human_accepted']
    assert len(manifest['files'])==33
