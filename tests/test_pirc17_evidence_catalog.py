"""Synthetic metadata only; no private snapshot, maps or empirical claims."""
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.nex326.specification import SPEC_PATH
from experiments.pirc17 import evidence_catalog as module
from experiments.pirc17.formal_matrix import load_protocol
from experiments.pirc17.protocol_core import digest, envelope, read_json, unpack


@pytest.fixture(scope='module')
def catalog_inputs():
    spec = read_json(Path(__file__).resolve().parents[2]/'DSDE-SDE/registry/pirc21_feature_spec.contract.json')
    p = deepcopy(unpack(load_protocol()))
    counts = dict(points=30,segments=6,files=3,independent_blocks=3)
    binding = dict(snapshot_id='SOFTWARE metadata only',counts=counts,feature_spec_content_sha256=digest(spec),
        manifest_sha256=digest('SOFTWARE manifest'),feature_spec_file_sha256=digest('SOFTWARE spec file'))
    p['dataset_inputs']['snapshot'] = binding
    protocol = envelope(p)
    execution = envelope(dict(protocol_sha256=protocol['sha256'],fixture='SOFTWARE no execution authorization'))
    manifest = dict(snapshot_id=binding['snapshot_id'],counts=counts,feature_spec_sha256=digest(spec),
                    coverage_report_sha256=digest('SOFTWARE coverage file'))
    levels = dict(point='points',segment='segments',file='files',independent_block='independent_blocks')
    def coverage(divisor,absent=False):
        return {level:dict(valid=0 if absent else counts[key]//divisor,denominator=counts[key]//divisor,
            ratio=0. if absent else 1.) for level,key in levels.items()}
    factors = {}
    for i,factor in enumerate(spec['factors']):
        absent = i==0
        factors[factor['factor_id']] = {name:coverage(1,absent) for name in ('source_coverage','materialized_coverage')}
        factors[factor['factor_id']]['splits'] = {split:{name:coverage(3,absent) for name in
            ('source_coverage','materialized_coverage')} for split in ('train','validation','final_eval')}
    report = dict(snapshot_id=binding['snapshot_id'],counts=counts,factors=factors)
    return protocol,execution,spec,manifest,report,read_json(SPEC_PATH)


def test_fixed_catalog_all_factors_variants_methods_and_default_steps(catalog_inputs):
    value = module.project(*catalog_inputs)
    assert len(value['factors'])==13 and len(value['variants'])==35
    assert len(value['selected_variant_ids'])==7 and len(value['method_arms'])==22
    assert len(value['method_slots'])==36
    assert sum(r['disposition']=='EXCLUDED' for r in value['method_slots'].values())==8
    assert value['method_arms'][0]['training_steps']==catalog_inputs[-1]['arm_defaults']['training_steps']
    assert value['factors'][0]['availability']=='unavailable'
    assert value['factors'][1]['availability']=='observed_in_snapshot'
    assert not value['scientific_claim_authorized'] and not value['human_accepted']
    assert value['new_forecasts']==value['new_fits']==0


@pytest.mark.parametrize('change',['ratio','counts','split_missing','split_sum','execution','spec','duplicate_factor'])
def test_catalog_rejects_changed_fixed_metadata(catalog_inputs,change):
    args = list(deepcopy(catalog_inputs))
    row = next(iter(args[4]['factors'].values()))
    if change=='ratio': row['materialized_coverage']['point']['ratio']=.5
    elif change=='counts': args[3]['counts']['points']+=1
    elif change=='split_missing': row['splits'].pop('train')
    elif change=='split_sum': row['splits']['train']['materialized_coverage']['point']['denominator']+=1
    elif change=='execution': args[1]=envelope(dict(protocol_sha256=digest('another protocol')))
    elif change=='spec': args[2]['variants'][0]['output_dim']+=1
    else: args[2]['factors'].append(deepcopy(args[2]['factors'][0]))
    with pytest.raises(ValueError): module.project(*args)
