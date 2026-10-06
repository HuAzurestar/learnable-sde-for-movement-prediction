"""Approximation metadata/refusal checks, never scientific error estimates."""

import copy
from dataclasses import FrozenInstanceError

import pytest

from experiments.pirc28.approximation import (ApproximationSpec, CHECKS, RECIPE_SCHEMA,
    pending_qualification, withhold_scientific_pass)
from experiments.pirc28.pipeline import execute_recipe, validate_recipe
from experiments.pirc28.plan import load_plan, REFERENCE_PLAN
from experiments.pirc28.study import SCHEMA_V2, SyntheticStudyDefinition
from inference.switching import InferenceSpec
from infrastructure.research_store import ResearchError, encode
from tests.test_pirc28_runtime import recipe
from tests.test_pirc28_study import definition_value


def explicit_spec():
    return load_plan(REFERENCE_PLAN)['approximation_spec']


def v2_recipe():
    value = recipe()
    # Legacy disposable model allows rates up to 100, not the production 5.
    # Give this metadata-only fixture its explicit 1-second prefix quota;
    # never relax the actual scientific plan's 400-step limit.
    approximation = explicit_spec()
    approximation['step_halving']['maximum_substeps'] = 4000
    value.update(schema_version=RECIPE_SCHEMA, approximation_spec=approximation)
    return value


def v2_master():
    value = definition_value()
    approximation = v2_recipe()['approximation_spec']
    value.update(schema_version=SCHEMA_V2, approximation_spec=approximation)
    for item in value['recipes'].values():
        item.update(schema_version=RECIPE_SCHEMA, approximation_spec=copy.deepcopy(approximation))
    return value


def test_canonical_spec_owns_detached_bytes_and_has_no_defaults():
    value = explicit_spec()
    frozen = ApproximationSpec.from_dict(value)
    before = frozen.spec_hash
    value['step_halving']['thresholds']['mode_total_variation'] = 0.15
    frozen.to_dict()['event_driven_reference']['particles'] = 4096
    assert frozen.spec_hash == before
    assert ApproximationSpec(frozen.content).to_dict() == explicit_spec()
    with pytest.raises(FrozenInstanceError):
        frozen.content = b'{}'
    with pytest.raises(ResearchError):
        ApproximationSpec(encode(explicit_spec())+b' ')


@pytest.mark.parametrize('damage', ['missing', 'unknown', 'truth', 'future', 'budget-alias', 'optional-stopping',
    'failure-drop', 'zero-tolerance', 'nonfinite', 'boolean-tolerance', 'missing-error', 'vacuous-tv',
    'time-only-halving', 'boolean-factor', 'substep-quota', 'exact-claim', 'few-batches', 'particle-quota',
    'jump-quota', 'ess', 'confidence', 'uncertain-reference', 'missing-boundary', 'loose-boundary'])
def test_unfrozen_vacuous_or_unbounded_approximation_is_refused(damage):
    value = explicit_spec()
    half, reference = value['step_halving'], value['event_driven_reference']
    if damage == 'missing': value.pop('budget_rule')
    elif damage == 'unknown': value['qualified'] = True
    elif damage == 'truth': value['observation_role'] = 'observed-truth-modes-and-velocity'
    elif damage == 'future': value['scope'] = 'full-test-smoother'
    elif damage == 'budget-alias': value['budget_rule'] = 'new-independent-reference-arm'
    elif damage == 'optional-stopping': value['aggregation_rule'] = 'keep-successful-batches'
    elif damage == 'failure-drop': value['failure_state'] = 'PASSED'
    elif damage == 'zero-tolerance': half['thresholds']['mode_total_variation'] = 0
    elif damage == 'nonfinite': reference['confidence'] = float('nan')
    elif damage == 'boolean-tolerance': half['thresholds']['mode_total_variation'] = True
    elif damage == 'missing-error': half['thresholds'].pop('position_mean_max_abs_m')
    elif damage == 'vacuous-tv': half['thresholds']['mode_total_variation'] = 1
    elif damage == 'time-only-halving': half['factor'] = 1
    elif damage == 'boolean-factor': half['factor'] = True
    elif damage == 'substep-quota': half['maximum_substeps'] = 100_001
    elif damage == 'exact-claim': reference['algorithm'] = 'exact-joint-switching-likelihood'
    elif damage == 'few-batches': reference['independent_batches'] = 1
    elif damage == 'particle-quota': reference['particles'] = 4097
    elif damage == 'jump-quota': reference['maximum_jumps_per_particle_interval'] = 10_001
    elif damage == 'ess': reference['minimum_ess_fraction'] = 0
    elif damage == 'confidence': reference['confidence'] = 0.5
    elif damage == 'uncertain-reference': reference['maximum_mc_half_width_fraction'] = 1
    elif damage == 'missing-boundary': value.pop('known_mode_boundary')
    else: value['known_mode_boundary']['absolute_tolerance'] = 0.01
    with pytest.raises(ResearchError):
        ApproximationSpec.from_dict(value)


def test_halving_reduces_both_active_limits_without_altering_fit_policy():
    spec = ApproximationSpec.from_dict(explicit_spec())
    coarse = InferenceSpec(max_step_seconds=0.125, max_rate_step=0.05, max_substeps=1600)
    fine = spec.fine_inference(coarse)
    assert fine.max_step_seconds == 0.0625 and fine.max_rate_step == 0.025
    assert fine.max_substeps == 400
    for name in coarse.__dataclass_fields__:
        if name not in ('max_step_seconds', 'max_rate_step', 'max_substeps'):
            assert getattr(fine, name) == getattr(coarse, name)


def test_fine_prefix_preflight_uses_allowed_rate_and_exact_observation_prefix():
    spec = ApproximationSpec.from_dict(explicit_spec())
    inference = InferenceSpec(max_step_seconds=0.125, max_rate_step=0.05)
    times = [i*0.25 for i in range(65)]
    result = spec.prefix_preflight(inference, times, 8, 5)
    assert result['maximum_prefix_substeps'] == 400
    assert result['resource_admitted'] is result['executor_implemented'] is False
    changed = explicit_spec()
    changed['step_halving']['maximum_substeps'] = 399
    with pytest.raises(ResearchError, match='allowed fitted rate'):
        ApproximationSpec.from_dict(changed).prefix_preflight(inference, times, 8, 5)


def test_v2_recipe_is_metadata_valid_but_cannot_generate_or_fit(monkeypatch):
    value = v2_recipe()
    validate_recipe(value, 2)
    monkeypatch.setattr('experiments.pirc28.pipeline.generate_scenario', lambda *a, **k: pytest.fail('generated before adapter exists'))
    monkeypatch.setattr('experiments.pirc28.pipeline.fit_switching', lambda *a, **k: pytest.fail('fit before adapter exists'))
    with pytest.raises(ResearchError, match='NEEDS_IMPLEMENTATION'):
        execute_recipe(value, 2, 123, check_budget=lambda: None)


def test_master_binds_complete_recipe_contract_without_budget_change():
    value = v2_master()
    frozen = SyntheticStudyDefinition.from_dict(value)
    assert frozen.to_dict()['approximation_spec'] == v2_recipe()['approximation_spec']
    assert frozen.to_dict()['arms']['2']['arm_id'] == 'pirc28-k2'
    assert frozen.to_dict()['arms']['2']['budget_seconds'] == 86400
    changed = copy.deepcopy(value)
    first = next(iter(changed['recipes'].values()))
    first['approximation_spec']['step_halving']['thresholds']['mode_total_variation'] = 0.03
    with pytest.raises(ResearchError, match='complete specifications differ'):
        SyntheticStudyDefinition.from_dict(changed)


@pytest.mark.parametrize('damage', ['missing-master', 'missing-recipe', 'master-downgrade', 'recipe-downgrade'])
def test_master_and_recipe_schemas_cannot_silently_drop_approximation(damage):
    value = v2_master()
    first = next(iter(value['recipes'].values()))
    if damage == 'missing-master': value.pop('approximation_spec')
    elif damage == 'missing-recipe': first.pop('approximation_spec')
    elif damage == 'master-downgrade': value['schema_version'] = 'pirc28-synthetic-study-definition-v1'
    else: first['schema_version'] = 'pirc28-synthetic-recipe-v1'
    with pytest.raises(ResearchError):
        SyntheticStudyDefinition.from_dict(value)


@pytest.mark.parametrize('v2', [False, True])
def test_mathematical_pass_has_no_scientific_promotion_without_managed_proof(v2):
    studies = [{'switching_recipe': v2_recipe() if v2 else recipe()}]
    gate = {'state': 'PASSED', 'applicable_scenarios': ['required'], 'applicable_ranges': [{'scenario_id': 'required'}]}
    status = withhold_scientific_pass(gate, studies, fixture=False)
    assert status['state'] == ('NEEDS_IMPLEMENTATION' if v2 else 'NEEDS_PREREGISTRATION')
    assert status['qualified'] is status['managed_evidence_bound'] is False
    assert status['required_checks'] == list(CHECKS)
    assert gate['state'] == 'NEEDS_APPROXIMATION_QUALIFICATION'
    assert gate['identifiability_only_state'] == 'PASSED'
    assert gate['applicable_ranges'] == gate['applicable_scenarios'] == []


@pytest.mark.parametrize('state', ['INCOMPLETE', 'BUDGET_EXHAUSTED', 'UNIDENTIFIABLE', 'CONTROL_GATE_FAILED'])
def test_budget_failure_and_negative_conclusions_are_not_hidden_by_pending_adapter(state):
    gate = {'state': state}
    withhold_scientific_pass(gate, [{'switching_recipe': recipe()}], fixture=False)
    assert gate == {'state': state}


def test_fixture_math_and_mixed_plan_refusal_are_explicit():
    gate = {'state': 'PASSED'}
    status = withhold_scientific_pass(gate, [{'switching_recipe': recipe()}], fixture=True)
    assert gate == {'state': 'PASSED'} and status['qualified'] is False
    with pytest.raises(ResearchError, match='complete approximation specification'):
        pending_qualification([{'switching_recipe': recipe()}, {'switching_recipe': v2_recipe()}])
