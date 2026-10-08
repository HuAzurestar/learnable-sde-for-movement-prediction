"""Synthetic arithmetic fixtures only; never empirical paper evidence."""
from copy import deepcopy

import pytest

from experiments.pirc17.checkpoint_method_preview import statistics
from experiments.pirc17.inference import SEEDS, InferenceConfig
from experiments.pirc17.inference_guard import infer_guarded
from experiments.pirc17.method_comparisons import DELTA_M, FAMILY_DEFINITIONS


def fixture():
    slots = {slot for family in FAMILY_DEFINITIONS.values()
             for pair in family.items() for slot in pair} | {'arm-06/dt60', 'arm-16/full'}
    population = [dict(sample_id=f'sample-{i}', independent_block_id=f'block-{i}') for i in range(46)]
    rows = [dict(origin_rank=i, sample_id=f'sample-{i}', independent_block_id=f'block-{i}',
                 origin_id=f'origin-{i}', seed=seed, configuration=slot, scientific=True,
                 matrix='NEX326-methods', origin_mode='causal_prefix', partition='final_eval',
                 status='success', score_m=500.+i*.2+((j%4)-2)*i*.1)
            for i in range(46) for seed in SEEDS for j, slot in enumerate(sorted(slots))]
    return rows, population, slots


def test_all_families_use_original_numeric_engine_without_authorizing_verdicts():
    rows, population, slots = fixture()
    result = statistics(rows, population, slots)
    assert result['primary_rows'] == 6440
    assert result['independent_blocks'] == 46
    assert not result['scientific_claim_authorized']
    assert not result['independent_raw_output_replay_completed']
    for name, family in FAMILY_DEFINITIONS.items():
        contrasts = {candidate: (candidate, control) for candidate, control in family.items()}
        selected = [r for r in rows if r['configuration'] in {s for p in contrasts.values() for s in p}]
        core = infer_guarded(selected, config=InferenceConfig(delta_m=DELTA_M),
                             family=contrasts, mechanism_passed={c: False for c in contrasts})
        for candidate, comparison in result['families'][name]['results'].items():
            assert 'verdict' not in comparison and 'reason' not in comparison
            for field in ('delta_estimate_m', 'simultaneous_interval_m', 'holm_adjusted_p_zero', 'seed_delta_m'):
                assert comparison[field] == core['results'][candidate][field]


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'failed', 'block', 'sample', 'secondary', 'nonfinite', 'origin'])
def test_entire_primary_grid_is_required_without_successful_subset_selection(change):
    rows, population, slots = fixture()
    rows = deepcopy(rows)
    if change == 'missing': rows.pop()
    elif change == 'duplicate': rows.append(deepcopy(rows[0]))
    elif change == 'failed': rows[0]['status'] = 'failed'
    elif change == 'block': rows[0]['independent_block_id'] = 'foreign-block'
    elif change == 'sample': rows[0]['sample_id'] = 'foreign-sample'
    elif change == 'secondary': rows[0]['origin_mode'] = 'known_velocity'
    elif change == 'nonfinite': rows[0]['score_m'] = float('nan')
    elif change == 'origin': rows[0]['origin_id'] = 'foreign-origin'
    with pytest.raises(ValueError):
        statistics(rows, population, slots)
