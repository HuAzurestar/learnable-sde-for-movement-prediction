"""Synthetic joins only; these fixtures never certify empirical fit gates."""
from copy import deepcopy

import pytest

from experiments.pirc17.checkpoint_method_preview import VERSION as STATISTICS_VERSION
from experiments.pirc17.checkpoint_method_qualification_preview import join_qualification
from experiments.pirc17.inference import InferenceConfig
from experiments.pirc17.method_comparisons import DELTA_M, FAMILY_DEFINITIONS


def fixture():
    config = vars(InferenceConfig(delta_m=DELTA_M))
    slots = {s for f in FAMILY_DEFINITIONS.values() for pair in f.items() for s in pair} | {'arm-06/dt60', 'arm-16/full'}
    ledger = [dict(slot_id=s, disposition='REQUIRED') for s in slots]
    policy = dict(inference_config=config, family_rules={name: dict(
        contrasts={c: [c, control] for c, control in definition.items()},
        planning_at46={c: dict(qualified=False, reason='SYNTHETIC planning not qualified') for c in definition},
        numerical_applicability={c: dict(global_qualification=False) for c in definition})
        for name, definition in FAMILY_DEFINITIONS.items()})
    stats = dict(schema_version=STATISTICS_VERSION, primary_rows=6440, independent_blocks=46,
        families={name: dict(config=config, results={c: dict(candidate=c, control=control)
            for c, control in definition.items()}) for name, definition in FAMILY_DEFINITIONS.items()})
    gates = {s: dict(status='computed', passed=True, synthetic=True) for s in slots}
    gates['arm-10/d2_mc'] = dict(status='pending_original_saved_forecast_diagnostics', passed=None)
    return stats, policy, ledger, gates


def test_pending_gate_is_not_a_pass_or_a_computed_failure():
    stats, policy, ledger, gates = fixture()
    result = join_qualification(stats, policy, ledger, gates)
    pending = result['families']['method-objective-and-score']['arm-10/d2_mc']
    assert pending['mechanism_passed'] is None
    assert pending['mechanism_status'] == 'pending_original_saved_forecast_diagnostics'
    assert result['families']['method-model-structure']['arm-02/pointwise']['mechanism_passed']
    assert not result['scientific_claim_authorized']
    assert not result['registered_full_mechanism_work_completed']
    assert not result['independent_raw_output_replay_completed']
    assert all(not c['verdict_authorized'] for f in result['families'].values() for c in f.values())


def test_computed_gate_failure_does_not_hide_statistics_or_planning():
    stats, policy, ledger, gates = fixture()
    gates['arm-02/pointwise']['passed'] = False
    result = join_qualification(stats, policy, ledger, gates)
    contrast = result['families']['method-model-structure']['arm-02/pointwise']
    assert contrast['mechanism_status'] == 'computed'
    assert contrast['mechanism_passed'] is False
    assert contrast['statistical_numbers'] == stats['families']['method-model-structure']['results']['arm-02/pointwise']
    assert contrast['planning']['qualified'] is False


@pytest.mark.parametrize('change', ['missing_gate', 'family', 'contrast', 'config', 'rows', 'blocks', 'participant'])
def test_frozen_whole_primary_scope_cannot_be_redefined(change):
    stats, policy, ledger, gates = deepcopy(fixture())
    if change == 'missing_gate': gates.pop('arm-01/full')
    elif change == 'family': stats['families'].pop('method-model-structure')
    elif change == 'contrast': policy['family_rules']['method-model-structure']['contrasts'].pop('arm-02/pointwise')
    elif change == 'config': policy['inference_config'] = {**policy['inference_config'], 'bootstrap_iterations': 100}
    elif change == 'rows': stats['primary_rows'] = 6439
    elif change == 'blocks': stats['independent_blocks'] = 30
    elif change == 'participant': stats['families']['method-model-structure']['results']['arm-02/pointwise']['control'] = 'arm-07/full'
    with pytest.raises(ValueError):
        join_qualification(stats, policy, ledger, gates)
