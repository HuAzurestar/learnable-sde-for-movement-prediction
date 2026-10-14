"""Deterministic review cards from PINNED PUBLIC aggregates/catalog only.

No saved forecast reader, maps, private targets, fits or prediction entrypoint
is constructed. Repeat the original public arithmetic; do not trust verdict
or mechanism-PASS flags. This is a review candidate, not TEST02 qualification,
scientific permission, manuscript completion or human acceptance.
"""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path

from .checkpoint_audit import COUNTS, QUERY_ACCOUNTING_SCOPE, validate_query_accounting
from .comparison_registry import GROUPS, ORIGIN_MODES
from .decision_policy import CONFIG, PREDICTIVE_SCOPE, registered_contrasts
from .evidence_catalog import VERSION as CATALOG_VERSION
from .formal_export import MAX_BYTES, VERSION as EXPORT_VERSION, _statistical_view, replay_public
from .formal_paired import FAMILIES, SEED_ROLE
from .inference import SEEDS
from .method_comparisons import FAMILY_DEFINITIONS
from .protocol_core import canonical, digest, file_hash, publish, read_json, unpack

VERSION = 'pirc17-public-review-evidence-cards-v1'


def _replay_view(value):
    """Table row ORDER is not evidence; every registered row/value still is.

    JSON sorts the terrain-ownership mapping keys. The original public-table
    helper iterates that mapping, so a saved aggregate may replay its identical
    table in a different order. Normalize only this order, never numbers,
    missingness, verdicts, intervals, mechanism flags or duplicate table axes.
    """
    value = deepcopy(value)
    key = lambda r:(r['origin_mode'],r['matrix'],r['configuration'])
    rows = value['metric_tables']
    if len(rows)!=120 or len({key(r) for r in rows})!=120:
        raise ValueError('all120 unique public descriptive metric table axes required')
    value['metric_tables'] = sorted(rows,key=key)
    return value


def _inventory(export, catalog):
    audit = export['audit_summary']
    if audit.get('feature_query_accounting_scope') != QUERY_ACCOUNTING_SCOPE:
        raise ValueError('saved query counter scope required; no per-column or robustness promotion')
    rows = audit['work_inventory']
    works = {r['work_id']:r for r in rows}
    scores = export['inputs']['score_rows']
    if (audit['full_work_count']!=11659 or len(works)!=11659 or len(rows)!=11659
            or audit['counts_by_kind']!=COUNTS or Counter(r['kind'] for r in rows)!=COUNTS
            or audit['raw_score_rows_recomputed']!=11368 or audit['common_blocks_recomputed']!=58
            or audit['producer_cache_used_for_numbers'] is not False
            or len(scores)!=11368 or len({r['forecast_work_id'] for r in scores})!=11368):
        raise ValueError('cards require full11659-work/11368-score audited denominator')
    required = {k for k,v in catalog['method_slots'].items() if v['disposition']=='REQUIRED'}
    terrain = {'base','all-terrain',*[prefix+g for prefix in ('loo-','lio-') for g in GROUPS]}
    subjects = {('NEX326-methods',s) for s in required}|{('terrain',s) for s in terrain}
    science = [r for r in rows if r['kind']=='scientific_forecast']
    expected = {(m,s,mode,rank,seed) for m,s in subjects for mode in ORIGIN_MODES
                for rank in range(46 if mode=='causal_prefix' else 6) for seed in SEEDS}
    keys = []
    for row in science:
        validate_query_accounting(row.get('feature_query_accounting'))
        axes = row['forecast_axes']
        keys.append(tuple(axes[k] for k in ('matrix','subject','origin_mode','origin_rank','seed')))
        if axes['repetition'] is not None or row['status'] not in {'success','failed','NOT_ADMITTED'}:
            raise ValueError('all registered scientific work must have an honest terminal disposition')
    if len(expected)!=11020 or len(keys)!=len(set(keys)) or set(keys)!=expected:
        raise ValueError('full28 method/10 terrain matched mode/rank/seed axes required')
    for row in scores:
        source = works.get(row['forecast_work_id'])
        if source is None or source['kind'] not in {'scientific_forecast','same_grid_reference','inertial_path'}:
            raise ValueError('public score does not cite an audited forecast work item')
        axes = source['forecast_axes']
        if any(row[k]!=axes[v] for k,v in dict(matrix='matrix',configuration='subject',origin_mode='origin_mode',
                                             origin_rank='origin_rank',seed='seed').items()):
            raise ValueError('public score changed its audited forecast axes')
    return science


def _query_counts(rows, matrix):
    """Recorded scalar arithmetic only; no new queries or predictor validity."""
    if matrix not in {'terrain', 'NEX326-methods'}:
        raise ValueError('registered scientific matrix required for query counter interpretation')
    values = [validate_query_accounting(row.get('feature_query_accounting', dict(
        feature_query_rows=None, invalid_feature_rows=None,
        raw_map_query_rows_this_forecast=None))) for row in rows]
    if matrix == 'NEX326-methods' and any(value['invalid_feature_rows'] not in (None, 0) for value in values):
        raise ValueError('ordinary-method invalid counter must remain contractual zero or absent')
    counters = {}
    for field in ('feature_query_rows', 'invalid_feature_rows', 'raw_map_query_rows_this_forecast'):
        recorded = [(row, value[field]) for row, value in zip(rows, values) if value[field] is not None]
        counters[field] = dict(recorded_forecasts=len(recorded), missing_forecasts=len(rows)-len(recorded),
            row_total=sum(n for _, n in recorded) if recorded else None,
            recorded_counts_by_status=dict(Counter(row['status'] for row, _ in recorded)))
    paired = [(row, value) for row, value in zip(rows, values) if value['invalid_feature_rows'] is not None]
    denominator = sum(value['feature_query_rows'] for _, value in paired) if paired else None
    numerator = sum(value['invalid_feature_rows'] for _, value in paired) if paired else None
    return dict(expected_forecasts=len(rows), counts_by_status=dict(Counter(row['status'] for row in rows)),
        counters=counters, paired_invalid_counter_forecasts=len(paired),
        paired_invalid_counter_missing_forecasts=len(rows)-len(paired),
        paired_invalid_counts_by_status=dict(Counter(row['status'] for row, _ in paired)),
        paired_invalid_row_numerator=numerator, paired_feature_row_denominator=denominator,
        recorded_selected_feature_invalid_fraction=numerator/denominator
            if matrix == 'terrain' and denominator is not None and denominator > 0 else None,
        invalid_counter_role='selected-history-and-map-union' if matrix == 'terrain' else 'contractual-zero-not-map-validity',
        scope='All registered origin modes, seeds and terminal statuses retained; sums over recorded particle-integration rows only, not unique locations, full-cohort validity, per-column map rates or robustness.')


def _costs(science, export):
    output = {}
    for matrix,subject in sorted({(r['forecast_axes']['matrix'],r['forecast_axes']['subject']) for r in science}):
        rows = [r for r in science if (r['forecast_axes']['matrix'],r['forecast_axes']['subject'])==(matrix,subject)]
        times = [r['kernel_prediction_seconds'] for r in rows if r['kernel_prediction_seconds'] is not None]
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in times):
            raise ValueError('finite nonnegative recorded kernel time required')
        output[matrix+':'+subject] = dict(matrix=matrix,subject=subject,expected_forecasts=len(rows),
            counts_by_status=dict(Counter(r['status'] for r in rows)),recorded_timing_count=len(times),
            missing_timing_count=len(rows)-len(times),recorded_kernel_total_seconds=sum(times),
            recorded_kernel_mean_seconds=sum(times)/len(times) if times else None,
            feature_query_accounting=_query_counts(rows, matrix),
            scope='Available recorded kernel times only, NOT complete total cost, isolated latency or an accuracy estimate.',
            isolated_runtime_references=[i for i,r in enumerate(export['runtime_replay']['runtime_subjects'])
                                         if r['matrix']==matrix and r['subject']==subject])
    return output


def project(export_record, catalog_record):
    export,catalog = unpack(export_record),unpack(catalog_record)
    if (export['schema_version']!=EXPORT_VERSION or catalog['schema_version']!=CATALOG_VERSION
            or any(export[k]!=catalog[k] for k in ('protocol_sha256','execution_sha256'))):
        raise ValueError('pinned public export/catalog scientific scope differs')
    if (any(export.get(k) is not False for k in ('retired_bootstrap_gate_asserted','raw_trajectories_exported',
            'original_identifiers_exported','scientific_claim_authorized','numerically_qualified','human_accepted'))
            or any(catalog.get(k) is not False for k in ('scientific_claim_authorized','human_accepted'))):
        raise ValueError('exact unaccepted public sources required; cards do not grant scientific permission')
    factors = {r['factor_id']:r for r in catalog['factors']}
    variants = {r['variant_id']:r for r in catalog['variants']}
    arms = {r['arm_id']:r for r in catalog['method_arms']}
    slots = catalog['method_slots']
    ledger = {r['slot_id']:r for r in export['method_ledger']}
    if (len(factors)!=13 or len(catalog['factors'])!=13 or len(variants)!=35 or len(catalog['variants'])!=35
            or set(arms)!=set(range(1,23)) or len(catalog['method_arms'])!=22
            or len(slots)!=36 or len(ledger)!=36 or len(export['method_ledger'])!=36 or set(ledger)!=set(slots)
            or sum(r['disposition']=='EXCLUDED' for r in slots.values())!=8
            or any(ledger[k]['disposition']!=r['disposition'] for k,r in slots.items())):
        raise ValueError('all13 factors/35variants/22arms/36slots required, including8 exclusions')
    science = _inventory(export,catalog)
    costs = _costs(science,export)
    replayed = _replay_view(replay_public(export['inputs']))
    if canonical(replayed)!=canonical(_replay_view(export['recomputed'])):
        raise ValueError('public verdict/effect/mechanism tables differ from deterministic replay')
    owners = unpack(export['inputs']['terrain_ownership'])
    families = {mode:{f:_statistical_view(replayed['modes'][mode]['families'][f]) for f in FAMILIES}
                for mode in ORIGIN_MODES}
    groups = []
    for group in GROUPS:
        configurations = ['base','all-terrain','loo-'+group,'lio-'+group]
        groups.append(dict(card_id='terrain:'+group,evidence_class='predictive-factor-effect',
            verdict=deepcopy(replayed['factor_conclusions'][group]),
            primary=dict(family_id='weighted-es-primary',comparison=group),
            supporting=dict(family_id='weighted-es-lio',comparison=group),
            ownership={name:deepcopy(owners['configurations'][name]) for name in configurations},
            kernel_cost_references=['terrain:'+name for name in configurations],
            raw_factor_ids=[fid for fid,r in factors.items() if group in r['selected_owner_groups']],
            attribution_scope='Selected group and its dependent/joint interactions; not every raw variant or unique causal contribution.'))
    raw_cards = []
    for fid,factor in factors.items():
        history = 'history.direction' in factor['selected_variant_ids']
        groups_used = [g for g in factor['selected_owner_groups'] if g in GROUPS]
        absent = factor['availability']=='unavailable'
        raw_cards.append(dict(card_id='input:'+fid,factor=deepcopy(factor),
            disposition='baseline-diagnostic' if history else 'evaluated-selected-group' if groups_used else 'screened-out',
            verdict=None if history or groups_used else 'unavailable' if absent else 'inconclusive',
            verdict_references=['terrain:'+g for g in groups_used],
            reason='History is baseline context, not terrain.' if history else
                'Only the selected variants participate; consult incremental group evidence.' if groups_used else
                'Required materialized snapshot data absent.' if absent else
                'Not included in the frozen final comparison; no predictive rejection inferred from selection.',
            eligible_forecast_coverage_established_by_snapshot_counts=False))
    variant_cards = []
    for vid,variant in variants.items():
        selected = vid in catalog['selected_variant_ids']
        factor = factors[variant['factor_id']]
        variant_cards.append(dict(variant=deepcopy(variant),input_card_id='input:'+variant['factor_id'],
            disposition='baseline-only' if vid=='history.direction' else 'selected-group' if selected else 'not-selected',
            verdict=None if selected else 'unavailable' if factor['availability']=='unavailable' else 'inconclusive',
            independent_variant_effect_tested=False,
            limitation='Group evidence does not identify a standalone transform effect; unselected variants are not rejected.'))
    methods = []
    for slot,definition in slots.items():
        family = definition['family_id']
        candidate = definition['role']=='predictive-contrast-candidate'
        if candidate and (family not in FAMILY_DEFINITIONS or slot not in registered_contrasts(family)):
            raise ValueError('method candidate lacks its sealed comparison')
        methods.append(dict(card_id='method:'+slot,slot_id=slot,definition=deepcopy(definition),
            ledger=deepcopy(ledger[slot]),
            predictive_result=deepcopy(families['causal_prefix'][family]['results'][slot]) if candidate else None,
            comparison_reference=dict(family_id=family,comparison=slot) if candidate else None,
            mechanism_reference=slot if definition['disposition']=='REQUIRED' else None,
            kernel_cost_reference='NEX326-methods:'+slot if definition['disposition']=='REQUIRED' else None,
            scope='Method-family effect, never a terrain-factor verdict; aliases/anchors are not extra independent replications.',
            scientific_rejection_implied_by_exclusion=False))
    arm_cards = [dict(card_id=f'arm:{arm_id:02d}',definition=deepcopy(arm),
        slot_card_ids=['method:'+slot for slot,r in slots.items() if r['arm_id']==arm_id],
        disposition='EXCLUDED' if all(r['disposition']=='EXCLUDED' for r in slots.values() if r['arm_id']==arm_id) else 'REQUIRED',
        scientific_rejection_implied_by_exclusion=False) for arm_id,arm in arms.items()]
    return dict(schema_version=VERSION,source_export_sha256=export_record['sha256'],source_catalog_sha256=catalog_record['sha256'],
        scope={k:export[k] for k in ('protocol_sha256','execution_sha256','matrix_sha256','population_sha256','audit_sha256','analysis_sha256')},
        snapshot_binding={k:deepcopy(catalog[k]) for k in ('snapshot_id','snapshot_counts','source_sha256',
                                                          'coverage_report_file_sha256')},
        review_state='candidate-not-human-accepted',predictive_estimand=PREDICTIVE_SCOPE,
        primary_origin_mode='causal_prefix',partition='final_eval',
        fixed_forecast=dict(particles=512,max_step_seconds=5.,nominal_horizon_seconds=1800.,
            scoring_seconds=[60.,300.,900.,1800.],time_weights=[.25]*4,target_tolerance_seconds=30.,
            target_rule='Nearest original observation, earlier ties; no interpolated targets.'),
        inference_config=asdict(CONFIG),seed_ids=list(SEEDS),seed_role=SEED_ROLE,
        multiplicity='Holm zero-mean tests and Bonferroni bootstrap-t intervals within each named family; no global FWER claim.',
        uncertainty='Independent-block resampling conditional on fixed finite-budget predictors and forecast seeds; secondary modes descriptive only.',
        family_evidence=families,metric_tables=deepcopy(replayed['metric_tables']),
        overall_terrain=dict(card_id='terrain:all-vs-base',family_id='weighted-es-primary',comparison='all-vs-base'),
        terrain_cards=groups,input_cards=raw_cards,variant_inventory=variant_cards,
        history=dict(card_id='baseline:history',role='baseline-only',terrain_verdict=None,
            selected_variant_id='history.direction',input_card_id='input:historical_motion',
            kernel_cost_references=['terrain:base','terrain:all-terrain'],
            limitation='No no-history formal ablation; history.relative_road is unselected terrain-dependent context.'),
        method_slot_cards=methods,method_arm_cards=arm_cards,method_fidelity=deepcopy(catalog['method_fidelity']),
        mechanism_evidence={mode:deepcopy(replayed['modes'][mode]['mechanism_gates']) for mode in ORIGIN_MODES},
        numerical_score_and_variance_witnesses=deepcopy(export['inputs']['mechanism_details']),
        kernel_costs=costs,feature_query_accounting_scope=deepcopy(export['audit_summary']['feature_query_accounting_scope']),
        isolated_runtime_and_replay=deepcopy(export['runtime_replay']),
        runner_cost_snapshot=deepcopy(export['budget_snapshot']),limitations=deepcopy(export['limitations']),
        paper_wording_rule='Review candidates only. Any eventual accepted effect wording must name the comparison, exposed cohort, finite algorithm and 30-minute nominal horizon; no causal, untouched-holdout,72-hour or precision-invariant claim.',
        test02_qualification_asserted=False,scientific_claim_authorized=False,numerically_qualified=False,
        human_accepted=False,new_forecasts=0,new_fits=0)


def generate(export_path,export_sha256,catalog_path,catalog_sha256,output_directory):
    export = read_json(export_path,max_bytes=MAX_BYTES)
    catalog = read_json(catalog_path)
    unpack(export,expected_sha256=export_sha256); unpack(catalog,expected_sha256=catalog_sha256)
    value = project(export,catalog)
    path = Path(output_directory)/(digest(value)+'.json')
    if path.is_file():
        record = read_json(path,max_bytes=MAX_BYTES)
        if unpack(record,expected_sha256=digest(value))!=value:
            raise ValueError('existing review cards differ from pinned sources')
    else:
        path,record = publish(output_directory,value)
    return dict(path=str(path),content_sha256=record['sha256'],file_sha256=file_hash(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export',type=Path,required=True); parser.add_argument('--export-sha256',required=True)
    parser.add_argument('--catalog',type=Path,required=True); parser.add_argument('--catalog-sha256',required=True)
    parser.add_argument('--output-directory',type=Path,required=True)
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.export,args.export_sha256,args.catalog,args.catalog_sha256,args.output_directory)),flush=True)


if __name__=='__main__':
    main()
