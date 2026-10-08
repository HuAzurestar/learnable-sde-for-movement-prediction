"""Complete synthetic score populations, not empirical final-eval evidence."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.pirc17 import decision_policy as policy
from experiments.pirc17 import formal_paired as module
from experiments.pirc17.inference import SEEDS
from experiments.pirc17.protocol_core import envelope
from tests.test_pirc17_decision_policy import rows_for


def fixture(family='weighted-es-primary', *, gate=True, **kwargs):
    rows, population = rows_for(family, **kwargs)
    mechanisms = {name: dict(status='computed', passed=gate, evidence='SYNTHETIC arithmetic fixture, not verified raw evidence')
                  for name in policy.registered_contrasts(family)}
    return rows, dict(family_id=family, expected_origins_by_block=population,
                     origin_mode=kwargs.get('mode', 'causal_prefix'), mechanisms=mechanisms)


def report(family='weighted-es-primary', **kwargs):
    rows, args = fixture(family, **kwargs)
    return module.paired_family(rows, **args)


def owners(*, missing=None, closure=True):
    names = ['base', 'all-terrain', *['loo-'+g for g in policy.GROUPS], *['lio-'+g for g in policy.GROUPS]]
    return envelope(dict(fixture='SYNTHETIC owner decision fixture; actual owner checks tested separately',
        configurations={name: dict(status='unavailable' if name == missing else 'computed',
            owner_closure_verified=closure, independent_fit_verified=name != missing) for name in names},
        composition_owners={'synthetic-joint-term': ['history', 'road', 'river']}))


@pytest.mark.parametrize('family', module.FAMILIES)
def test_all_seven_registered_families_preserve_exact_core_intervals_and_fixed_parameters(family):
    rows, args = fixture(family)
    result = module.paired_family(rows, **args)
    original = policy.infer_qualified_family(rows, family_id=family, expected_origins_by_block=args['expected_origins_by_block'],
        mechanism_passed={name: True for name in args['mechanisms']}, evidence_partition='final_eval')
    assert result['inference'] == original['inference']
    assert result['inference']['config'] == vars(policy.CONFIG)
    assert result['hypothesis_tests_performed'] and result['independent_block_count'] == 46
    assert result['registered_seed_role'] == module.SEED_ROLE
    assert result['scientific_claim_authorized'] is False
    for name, desc in result['descriptive']['results'].items():
        assert desc['delta_estimate_m'] == result['inference']['results'][name]['delta_estimate_m']
        assert desc['improvement_m'] == -desc['delta_estimate_m']
        assert desc['paired_block_sd_m'] == 0.
        assert desc['standardized_paired_delta'] is None  # Constant effect, not fake infinite precision/power.
        assert len(desc['block_seed_delta_m']) == 46 and len(desc['block_seed_delta_m'][0]) == 5


def test_descriptive_paired_standardization_is_not_used_as_planning_power():
    rows, args = fixture(blocks=2)
    for row in rows:
        row['score_m'] = 500. - (2. if row['independent_block_id'] == 'block-0' else 4.) if row['configuration'] == 'all-terrain' else 500.
    result = module.paired_family(rows, **args)
    desc = result['descriptive']['results']['road']
    assert desc['delta_estimate_m'] == -3. and desc['paired_block_sd_m'] == pytest.approx(np.sqrt(2.))
    assert desc['standardized_paired_delta'] == pytest.approx(-3./np.sqrt(2.))
    assert result['results']['road']['planning']['paired_block_sd_m'] != desc['paired_block_sd_m']
    assert result['results']['road']['verdict'] == 'inconclusive'


def test_nonzero_descriptive_spread_is_not_replaced_by_a_numerical_floor():
    rows, _ = fixture(blocks=2)
    for row in rows:
        if row['configuration'] == 'all-terrain' and row['independent_block_id'] == 'block-0':
            row['score_m'] += 1.e-9
    desc = module.describe_paired(rows, 'weighted-es-primary')['results']['road']
    assert 0. < desc['paired_block_sd_m'] < 1.e-8
    assert desc['standardized_effect_status'] == 'computed'
    assert desc['standardized_paired_delta'] == desc['delta_estimate_m']/desc['paired_block_sd_m']


@pytest.mark.parametrize('blocks', [0, 1])
def test_zero_and_one_block_never_fabricate_bootstrap_uncertainty(blocks, monkeypatch):
    monkeypatch.setattr(policy, 'infer_guarded', lambda *a, **kw: pytest.fail('zero/one block reached bootstrap'))
    result = report(blocks=blocks)
    assert not result['hypothesis_tests_performed'] and result['inference'] is None
    assert (result['descriptive'] is None) == (blocks == 0)
    assert {r['verdict'] for r in result['results'].values()} == ({'unavailable'} if blocks == 0 else {'inconclusive'})
    factors = module.terrain_conclusions(result, report('weighted-es-lio', blocks=blocks), ownership=owners())
    assert len(factors) == 4
    assert {r['verdict'] for r in factors.values()} == ({'unavailable'} if blocks == 0 else {'inconclusive'})


@pytest.mark.parametrize('mode', ['known_velocity', 'point_only'])
def test_secondary_modes_have_complete_descriptions_without_hypothesis_tests(mode, monkeypatch):
    monkeypatch.setattr(policy, 'infer_guarded', lambda *a, **kw: pytest.fail('secondary mode reached bootstrap'))
    result = report(mode=mode, blocks=6)
    assert result['descriptive'] is not None and result['inference'] is None
    assert not result['hypothesis_tests_performed']
    assert all(r['reason'] == 'secondary_origin_descriptive_only' for r in result['results'].values())
    assert all(not r['planning']['qualified'] for r in result['results'].values())


@pytest.mark.parametrize('change', ['seed', 'origin', 'failure'])
def test_incomplete_population_is_not_successfully_intersected(change):
    rows, args = fixture()
    if change == 'seed': rows.pop()
    elif change == 'origin': rows = [r for r in rows if r['origin_id'] != 'origin-0']
    else: rows[0].update(status='failed', score_m=None, reason='SOFTWARE timeout')
    result = module.paired_family(rows, **args)
    assert result['independent_block_count'] == 46 and result['descriptive'] is None
    assert result['inference'] is None and all(r['verdict'] == 'unavailable' for r in result['results'].values())


@pytest.mark.parametrize('field,value', [('partition', 'validation'), ('matrix', 'NEX326-methods'),
    ('origin_mode', 'point_only'), ('independent_block_id', 'wrong'), ('seed', True), ('score_m', float('nan'))])
def test_wrong_scope_and_nonfinite_common_rows_rejected(field, value):
    rows, args = fixture()
    rows[0][field] = value
    with pytest.raises(ValueError): module.paired_family(rows, **args)


def test_duplicate_origin_and_population_growth_rejected():
    rows, args = fixture()
    with pytest.raises(ValueError): module.paired_family([*rows, rows[0]], **args)
    args['expected_origins_by_block']['block-1'] = ['origin-0']
    with pytest.raises(ValueError, match='unique'): module.paired_family(rows, **args)
    rows, args = fixture(blocks=47)
    with pytest.raises(ValueError, match='population'): module.paired_family(rows, **args)
    rows, args = fixture(blocks=7, mode='point_only')
    with pytest.raises(ValueError, match='population'): module.paired_family(rows, **args)


def test_unavailable_mechanism_differs_from_computed_failure_and_keeps_raw_intervals():
    rows, args = fixture('method-model-structure')
    name = next(iter(args['mechanisms']))
    args['mechanisms'][name] = dict(status='unavailable', passed=None, reason='SOFTWARE missing required diagnostic')
    missing = module.paired_family(rows, **args)
    assert missing['results'][name]['verdict'] == 'unavailable'
    assert missing['results'][name]['precision_invariant_verdict'] == 'unavailable'
    assert missing['inference']['results'][name]['simultaneous_interval_m'] is not None
    args['mechanisms'][name] = dict(status='computed', passed=False, evidence='SOFTWARE actual failed predicate')
    failed = module.paired_family(rows, **args)
    assert failed['results'][name]['verdict'] == 'inconclusive' and failed['results'][name]['reason'] == 'mechanism_gate_failed'
    args['mechanisms'][name] = True
    with pytest.raises(ValueError, match='bare pass flags'): module.paired_family(rows, **args)


def test_strong_final_effect_cannot_override_frozen_low_or_missing_power():
    r = report('method-observation-interval')['results']['arm-06/dt600']
    assert r['raw_statistical_verdict'] == 'beneficial'
    assert r['verdict'] == 'inconclusive' and r['planning']['scenarios'][0]['required_blocks_normal_approximation'] == 47
    assert not r['planning']['automatic_expansion']
    assert all(r['verdict'] == 'inconclusive' for r in report('weighted-es-lio')['results'].values())


@pytest.mark.parametrize('delta,verdict', [(-3*policy.CONFIG.delta_m, 'retain'), (3*policy.CONFIG.delta_m, 'harmful'), (0., 'redundant')])
def test_all_four_factor_conclusions_keep_incremental_joint_ownership_scope(delta, verdict):
    primary, lio = report(delta=delta), report('weighted-es-lio', delta=delta)
    result = module.terrain_conclusions(primary, lio, ownership=owners())
    assert set(result) == set(policy.GROUPS) and {r['verdict'] for r in result.values()} == {verdict}
    assert result['road']['joint_owned_compositions'] == {'synthetic-joint-term': ['history', 'road', 'river']}
    assert all('No unique/additive/synergy' in r['attribution_scope'] for r in result.values())
    assert all(r['precision_invariant_verdict'] == 'inconclusive' and not r['scientific_claim_authorized'] for r in result.values())


def test_raw_opposite_LIO_is_not_hidden_by_its_missing_power():
    lio = report('weighted-es-lio', delta=3*policy.CONFIG.delta_m)
    assert lio['results']['road']['verdict'] == 'inconclusive'
    result = module.terrain_conclusions(report(), lio, ownership=owners())
    assert all(r['verdict'] == 'inconclusive' for r in result.values())


def test_missing_owner_or_failed_owner_rule_cannot_become_terrain_benefit():
    primary, lio = report(), report('weighted-es-lio')
    assert module.terrain_conclusions(primary, lio, ownership=owners(missing='lio-road'))['road']['verdict'] == 'unavailable'
    assert all(r['verdict'] == 'inconclusive' for r in module.terrain_conclusions(primary, lio, ownership=owners(closure=False)).values())
    bad = deepcopy(primary); bad['origin_mode'] = 'point_only'
    with pytest.raises(ValueError): module.terrain_conclusions(bad, lio, ownership=owners())
