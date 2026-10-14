"""Public fixed factor/method metadata for cards, not a restart qualification.

Read three snapshot JSON metadata files, never Parquet/map/trajectory arrays.
The frozen protocol binds every source. Coverage is snapshot availability,
NOT coverage of simulated forecasts or proof that an input improves accuracy.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

from experiments.nex326.specification import SPEC_PATH
from experiments.pirc22.consumer import FACTOR_GROUPS
from .checkpoint_resume import load
from .protocol_core import digest, file_hash, publish, read_json, unpack

VERSION = 'pirc17-public-evidence-catalog-v1'


def coverage_projection(value):
    result = {}
    for level in ('point','segment','file','independent_block'):
        row = value[level]
        valid,denominator = row['valid'],row['denominator']
        if (type(valid) is not int or type(denominator) is not int
                or not 0 <= valid <= denominator):
            raise ValueError('integer snapshot coverage counts required')
        ratio = row['ratio']
        expected = valid/denominator if denominator else None
        if ((ratio is None) != (expected is None)
                or ratio is not None and (type(ratio) not in (int,float) or not math.isfinite(ratio)
                                         or abs(ratio-expected)>1e-12)):
            raise ValueError('snapshot coverage ratio differs from its counts')
        result[level] = dict(valid=valid,denominator=denominator,ratio=ratio)
    return result


def project(protocol,execution,spec,manifest,coverage,method_spec):
    p = unpack(protocol)
    if unpack(execution)['protocol_sha256'] != protocol['sha256']:
        raise ValueError('catalog execution belongs to another protocol')
    binding = p['dataset_inputs']['snapshot']
    if (digest(spec) != binding['feature_spec_content_sha256']
            or manifest['feature_spec_sha256'] != digest(spec)
            or manifest['snapshot_id'] != binding['snapshot_id']
            or manifest['counts'] != binding['counts']
            or coverage['snapshot_id'] != manifest['snapshot_id']
            or coverage['counts'] != manifest['counts']):
        raise ValueError('catalog metadata differs from frozen snapshot')
    factors = {r['factor_id']:r for r in spec['factors']}
    variants = {r['variant_id']:r for r in spec['variants']}
    if (len(factors)!=len(spec['factors']) or len(variants)!=len(spec['variants'])
            or set(coverage['factors'])!=set(factors)):
        raise ValueError('all unique declared PIRC21 factors/coverage required')
    full = p['components']['terrain_configurations']['all-terrain']
    selected = set(full['variant_ids'])
    if not selected <= variants.keys():
        raise ValueError('selected variant missing from fixed catalog')
    output = []
    for fid,factor in factors.items():
        source = coverage['factors'][fid]
        cov = {name:coverage_projection(source[name]) for name in ('source_coverage','materialized_coverage')}
        cov['splits'] = {split:{name:coverage_projection(row[name]) for name in
            ('source_coverage','materialized_coverage')} for split,row in source['splits'].items()}
        if set(cov['splits']) != {'train','validation','final_eval'}:
            raise ValueError('all three frozen snapshot coverage splits required')
        count_keys = dict(point='points',segment='segments',file='files',independent_block='independent_blocks')
        for name in ('source_coverage','materialized_coverage'):
            for level,key in count_keys.items():
                if cov[name][level]['denominator'] != binding['counts'][key]:
                    raise ValueError('coverage denominator differs from frozen snapshot counts')
                for field in ('valid','denominator'):
                    if sum(row[name][level][field] for row in cov['splits'].values()) != cov[name][level][field]:
                        raise ValueError('split coverage does not sum to full snapshot coverage')
        used = [vid for vid,v in variants.items() if v['factor_id']==fid and vid in selected]
        groups = sorted(g for g,ids in FACTOR_GROUPS.items() if set(ids)&set(used))
        output.append(dict(factor_id=fid,family=factor['family'],source_id=factor['source_id'],
            rollout_mode=factor['rollout_mode'],value_columns=deepcopy(factor['value_columns']),
            selected_variant_ids=used,selected_owner_groups=groups,
            all_variant_ids=[vid for vid,v in variants.items() if v['factor_id']==fid],coverage=cov,
            availability='observed_in_snapshot' if cov['materialized_coverage']['point']['valid'] else 'unavailable',
            coverage_scope='Frozen full snapshot/split counts; not forecast-origin/map-query success or predictive effect.'))
    methods = p['components']['method_comparisons']['causal_prefix']
    defaults = method_spec.get('arm_defaults',{})
    arms = []
    for row in method_spec['arms']:
        expanded = dict(defaults,**row)
        expanded['control'] = dict(defaults.get('control',{}),**row.get('control',{}))
        arms.append({k:deepcopy(expanded[k]) for k in ('arm_id','group','slot','variant','unique_change','control',
                                                     'training_steps','inference_steps')})
    if {r['arm_id'] for r in arms} != set(range(1,23)) or len(arms)!=22:
        raise ValueError('all22 frozen method arms required')
    return dict(schema_version=VERSION,protocol_sha256=protocol['sha256'],execution_sha256=execution['sha256'],
        snapshot_id=binding['snapshot_id'],snapshot_counts=deepcopy(binding['counts']),
        source_sha256={k:binding[k] for k in ('manifest_sha256','feature_spec_file_sha256','feature_spec_content_sha256')},
        coverage_report_file_sha256=manifest['coverage_report_sha256'],
        method_spec_file_sha256=p['source_sha256']['PSDE-SDE/experiments/nex326/experiment.json'],
        factors=output,variants=[{k:deepcopy(r[k]) for k in ('variant_id','factor_id','transform',
            'fit_scope','output_dim')} for r in spec['variants']],
        selected_variant_ids=full['variant_ids'],selected_composition_ids=full['composition_ids'],
        method_arms=arms,method_slots=deepcopy(methods['slots']),method_fidelity=deepcopy(methods['fidelity']),
        history_role='baseline-only; history-relative-road is not an independently tested history factor',
        scientific_claim_authorized=False,human_accepted=False,new_forecasts=0,new_fits=0)


def catalog(directory,output_directory):
    settings,_ = load(directory)
    bundle = unpack(read_json(settings['bundle']),expected_sha256=settings['bundle_sha256'])
    protocol,execution = bundle['protocol'],bundle['execution']
    p = unpack(protocol); binding = p['dataset_inputs']['snapshot']
    root = Path(settings['input_paths']['snapshot'])
    manifest = read_json(root/'manifest.json',expected_file_sha256=binding['manifest_sha256'])
    spec = read_json(root/'feature_spec.json',expected_file_sha256=binding['feature_spec_file_sha256'])
    coverage = read_json(root/'coverage_report.json',expected_file_sha256=manifest['coverage_report_sha256'])
    method_spec = read_json(SPEC_PATH,expected_file_sha256=p['source_sha256']['PSDE-SDE/experiments/nex326/experiment.json'])
    value = project(protocol,execution,spec,manifest,coverage,method_spec)
    output_directory = Path(output_directory)
    candidate = output_directory/(digest(value)+'.json')
    if candidate.is_file():
        record = read_json(candidate)
        if unpack(record,expected_sha256=digest(value)) != value:
            raise ValueError('existing catalog changed')
        path = candidate
    else:
        path,record = publish(output_directory,value)
    return dict(path=str(path),content_sha256=record['sha256'],file_sha256=file_hash(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output-directory',type=Path,required=True)
    args = parser.parse_args(argv)
    print(json.dumps(catalog(args.directory,args.output_directory)),flush=True)


if __name__=='__main__':
    main()
