"""Complete-population adapter over the unchanged sealed statistical engine.

Pure arithmetic, not raw-evidence verification or final-eval authorization.
The formal analysis consumer supplies verified score rows and recomputed
mechanism evidence. No sample/seed/B/horizon tuning, successful intersections,
cross-family selection or secondary-mode hypothesis tests occur here.
"""
from copy import deepcopy
import math

import numpy as np

from .comparison_registry import GROUPS, ORIGIN_MODES
from .decision_policy import (CONFIG, PREDICTIVE_SCOPE, infer_qualified_family, numerical_applicability,
    planning_gate, registered_contrasts, terrain_factor_conclusion)
from .inference import SEEDS, paired_blocks
from .method_comparisons import FAMILY_DEFINITIONS
from .protocol_core import unpack

VERSION = 'pirc17-formal-paired-analysis-v1'
FAMILIES = ('weighted-es-primary', 'weighted-es-lio', *FAMILY_DEFINITIONS)
SEED_ROLE = 'Five fixed forecast RNG seeds per deterministic fitted configuration; not five fitted models or additional independent blocks.'


def _validate_rows(rows, family_id, population, mode):
    family = registered_contrasts(family_id)
    if (mode not in ORIGIN_MODES or not isinstance(population, dict)
            or len(population) > (46 if mode == 'causal_prefix' else 6)
            or any(not isinstance(b, str) or not b or not isinstance(v, (list, tuple)) or len(v) != 1 for b, v in population.items())):
        raise ValueError('registered one-origin-per-block population/mode required')
    origins = [v[0] for v in population.values()]
    if any(not isinstance(o, str) or not o for o in origins) or len(origins) != len(set(origins)):
        raise ValueError('unique frozen origin identities required')
    blocks = {o: b for b, values in population.items() for o in values}
    configurations = {c for pair in family.values() for c in pair}
    expected = {(o, seed, c) for o in origins for seed in SEEDS for c in configurations}
    matrix = 'terrain' if family_id.startswith('weighted-es-') else 'NEX326-methods'
    seen, failed = set(), []
    for row in rows:
        key = row['origin_id'], row['seed'], row['configuration']
        if (key not in expected or key in seen or type(row['seed']) is not int
                or row['independent_block_id'] != blocks[row['origin_id']] or row['partition'] != 'final_eval'
                or row['matrix'] != matrix or row['origin_mode'] != mode):
            raise ValueError('duplicate, extra, wrong-block/partition/matrix/mode score row')
        seen.add(key)
        if row['status'] != 'success':
            if row['status'] not in {'failed', 'unavailable'} or not isinstance(row.get('reason'), str) or not row['reason'].strip():
                raise ValueError('failed required score needs an explicit disposition')
            failed.append(list(key))
        elif type(row['score_m']) not in (int, float) or not math.isfinite(row['score_m']):
            raise ValueError('finite real common score required')
    return dict(schema_version=VERSION, family_id=family_id, matrix=matrix, partition='final_eval', origin_mode=mode,
        predictive_scope=PREDICTIVE_SCOPE, expected_origins_by_block=deepcopy(population), expected_rows=len(expected),
        missing_rows=[list(k) for k in sorted(expected-seen)], failed_rows=failed,
        independent_block_count=len(population), results={}, inference=None, scientific_claim_authorized=False)


def describe_paired(rows, family_id):
    """Complete pairs only; descriptive standardized effect, never post-hoc power."""
    family = registered_contrasts(family_id)
    blocks, names, values, counts = paired_blocks(rows, family)
    means = values.mean(axis=1)
    estimates = means.mean(axis=0)  # Match the sealed engine's reduction order.
    results = {}
    for j, name in enumerate(names):
        delta = float(estimates[j])
        # Exactly identical block values have zero spread even if their mean
        # accumulates roundoff. This is an equality check, not an SD floor.
        sd = (0. if np.all(means[:, j] == means[0, j]) else float(means[:, j].std(ddof=1))) if len(blocks) >= 2 else None
        standardized = delta/sd if sd is not None and sd > 0 else None
        results[name] = dict(candidate=family[name][0], control=family[name][1], delta_estimate_m=delta,
            improvement_m=-delta, improvement_in_registered_margins=-delta/CONFIG.delta_m,
            paired_block_sd_m=sd, standardized_paired_delta=standardized,
            standardized_effect_status='computed' if standardized is not None else 'unavailable',
            standardized_effect_reason=None if standardized is not None else 'fewer than two blocks or zero paired block SD; never floored',
            seed_delta_m=values[:, :, j].mean(axis=0).tolist(), block_seed_delta_m=values[:, :, j].tolist())
    return dict(role='descriptive; not a new hypothesis family or observed-power calculation', block_ids=blocks,
        origin_count_by_block=counts, seed_ids=list(SEEDS), results=results,
        sign_convention='candidate_ES_minus_control_ES; negative favours candidate; positive improvement favours candidate',
        weight_rule='equal independent blocks, equal matched origins within block, equal five forecast seeds')


def paired_family(rows, *, family_id, expected_origins_by_block, origin_mode, mechanisms):
    """Preserve raw engine intervals while applying missing-evidence precedence.

    `mechanisms` is evidence supplied by the raw-source owner, not a public
    boolean authorization API. Validation/recomputation of those sources is
    deliberately outside this pure statistical adapter.
    """
    rows = list(rows)
    report = _validate_rows(rows, family_id, expected_origins_by_block, origin_mode)
    family = registered_contrasts(family_id)
    if (set(mechanisms) != set(family) or any(not isinstance(g, dict) or g.get('status') not in {'computed', 'unavailable'}
            or (type(g.get('passed')) is not bool if g['status'] == 'computed' else g.get('passed') is not None)
            for g in mechanisms.values())):
        raise ValueError('explicit per-comparison mechanism evidence required, not bare pass flags')
    complete = bool(expected_origins_by_block) and not report['missing_rows'] and not report['failed_rows']
    descriptive = describe_paired(rows, family_id) if complete else None
    if complete and len(expected_origins_by_block) >= 2:
        report = infer_qualified_family(rows, family_id=family_id, expected_origins_by_block=expected_origins_by_block,
            mechanism_passed={k: g['status'] == 'computed' and g['passed'] for k, g in mechanisms.items()},
            evidence_partition='final_eval', origin_mode=origin_mode)
    else:
        reason = ('no_admitted_independent_blocks' if not expected_origins_by_block else
                  'incomplete_prespecified_paired_family' if not complete else
                  'secondary_origin_descriptive_only' if origin_mode != 'causal_prefix' else
                  'one_independent_block_no_uncertainty_estimate')
        report['results'] = {name: dict(verdict='inconclusive' if complete else 'unavailable', reason=reason) for name in family}
    for name, result in report['results'].items():
        # Missing mechanisms differ from computed failures. Keep interval/raw
        # diagnostics when scores themselves were complete; never fabricate a
        # bootstrap for zero/one block. Secondary modes remain descriptive.
        if complete and origin_mode == 'causal_prefix' and mechanisms[name]['status'] == 'unavailable':
            result.update(verdict='unavailable', reason='required_mechanism_evidence_unavailable')
        if 'planning' not in result:
            result['planning'] = (planning_gate(family_id, name, blocks=len(expected_origins_by_block), origin_mode=origin_mode)
                                  if expected_origins_by_block else None)
        if 'numerical_applicability' not in result:
            result['numerical_applicability'] = numerical_applicability(family_id, name, origin_mode=origin_mode)
        result['precision_invariant_verdict'] = 'unavailable' if result['verdict'] == 'unavailable' else 'inconclusive'
        result['mechanism_evidence'] = deepcopy(mechanisms[name])
    report.update(adapter_version=VERSION, descriptive=descriptive, registered_seed_role=SEED_ROLE,
        legacy_engine_label_note='The unchanged engine uses legacy fitted/training-seed wording; the actual registered seeds are forecast RNG streams, not retraining replications.',
        hypothesis_tests_performed=report['inference'] is not None,
        multiplicity_scope='This registered family only; no global across-family FWER claim.',
        uncertainty_scope='Independent-block resampling conditional on the registered finite-budget predictors and fixed forecast seeds.')
    return report


def terrain_conclusions(primary, supporting, *, ownership):
    """Owner-verified incremental factor conclusions; no unique interaction attribution."""
    owners = unpack(ownership)
    for report, family in ((primary, 'weighted-es-primary'), (supporting, 'weighted-es-lio')):
        if (report['family_id'] != family or report['matrix'] != 'terrain' or report['partition'] != 'final_eval'
                or report['origin_mode'] != 'causal_prefix' or report['predictive_scope'] != PREDICTIVE_SCOPE):
            raise ValueError('primary-mode final terrain evidence required')
    if primary['expected_origins_by_block'] != supporting['expected_origins_by_block']:
        raise ValueError('LOO and LIO populations differ')
    results = {}
    for group in GROUPS:
        configs = ['all-terrain', 'base', 'loo-'+group, 'lio-'+group]
        absent = [name for name in configs if owners['configurations'][name]['status'] != 'computed']
        if absent or any(r['results'][group]['verdict'] == 'unavailable' for r in (primary, supporting)):
            result = dict(verdict='unavailable', reason='required_LOO_LIO_or_independently_fitted_owner_evidence_unavailable',
                precision_invariant_verdict='unavailable', scientific_claim_authorized=False)
        elif primary['inference'] is None or supporting['inference'] is None:
            result = dict(verdict='inconclusive', reason='one_independent_block_no_uncertainty_estimate',
                precision_invariant_verdict='inconclusive', scientific_claim_authorized=False)
        else:
            result = terrain_factor_conclusion(primary, supporting, group, conflict_unresolved=False,
                correlated_group_rule_passed=all(owners['configurations'][name]['owner_closure_verified']
                                               and owners['configurations'][name]['independent_fit_verified'] for name in configs))
        result.update(predictive_scope=PREDICTIVE_SCOPE, primary_comparison=group, supporting_comparison=group,
            primary_family='weighted-es-primary', supporting_family='weighted-es-lio', missing_fit_configurations=absent,
            ownership_sha256=ownership['sha256'],
            joint_owned_compositions={name: groups for name, groups in owners['composition_owners'].items() if group in groups and len(groups) > 1},
            attribution_scope='Incremental group contribution in this full predictor; shared interactions remain jointly owned. No unique/additive/synergy or resolution-invariant attribution.')
        results[group] = result
    return results
