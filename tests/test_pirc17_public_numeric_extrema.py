"""Heterogeneous synthetic witnesses; no empirical data or qualification."""
from copy import deepcopy

import pytest

from experiments.pirc17 import formal_export as module
from experiments.pirc17.protocol_core import digest


def witnesses():
    population = {'block-a': ['origin-a'], 'block-b': ['origin-b']}
    diagnostic_slots = module.NUMERICAL_SLOTS | module.SCORE_SLOTS
    definitions = module.mechanism_registry()['slots']
    gates = {}
    for slot, definition in definitions.items():
        if definition['disposition'] != 'REQUIRED':
            continue
        count = len(population) * len(module.SEEDS) if slot in diagnostic_slots else 1
        gates[slot] = dict(
            slot_id=slot, definition_sha256=digest(definition),
            statistic=definition['statistic'], operator=definition['operator'],
            threshold=definition['threshold'], units=definition['units'],
            status='computed', value=definition['threshold'],
            expected_count=count, available_count=count, rows_sha256='e' * 64)
    for slot in module.VARIANCE_SLOTS:
        gates[slot].update(status='unavailable', value=None, available_count=0)
    details = dict(per_origin_diagnostics={}, variance=None)
    for slot in diagnostic_slots:
        rows, values = [], []
        for block, origins in population.items():
            for origin in origins:
                for seed in module.SEEDS:
                    magnitude = .5 + .02 * len(rows)
                    if slot in module.NUMERICAL_SLOTS:
                        errors = [.1, .2, .3, magnitude]
                        measurement = dict(per_time_gaussian_coupling_rms_m=errors)
                    else:
                        a, b = 10., 10. + magnitude
                        error = abs(a - b) / max(abs(a), 1e-9)
                        errors = [error] * 4
                        measurement = dict(diagnostic_draws_per_time=256, quadrature_order=12,
                            per_time=[dict(moment_gaussian_quadrature_es_m=a,
                                gaussian_mc_es_m=b, relative_error=error) for _ in range(4)])
                    values.append(max(errors))  # Same frozen four-time row statistic.
                    rows.append(dict(origin_id=origin, independent_block_id=block,
                        seed=seed, gate_sha256='e' * 64, measurements=measurement))
        aggregate = max if gates[slot]['operator'] == 'le' else min
        gates[slot]['value'] = aggregate(values)
        details['per_origin_diagnostics'][slot] = rows
    return population, gates, details


@pytest.mark.parametrize('slot', sorted(module.NUMERICAL_SLOTS | module.SCORE_SLOTS))
@pytest.mark.parametrize('fault', ['none', 'wrong_extreme', 'missing'])
def test_public_witness_extreme_matches_registered_operator(slot, fault):
    population, gates, details = witnesses()
    rows = details['per_origin_diagnostics'][slot]
    if fault == 'wrong_extreme':
        if slot in module.NUMERICAL_SLOTS:
            values = [max(r['measurements']['per_time_gaussian_coupling_rms_m']) for r in rows]
        else:
            values = [max(t['relative_error'] for t in r['measurements']['per_time']) for r in rows]
        assert min(values) != max(values)
        opposite = min if gates[slot]['operator'] == 'le' else max
        gates[slot]['value'] = opposite(values)
    elif fault == 'missing':
        rows[0].update(measurements=None, gate_sha256=None)
        gates[slot].update(status='unavailable', value=None, available_count=len(rows) - 1)
    replayed = module._replay_gates(deepcopy(gates), len(population))
    if fault == 'wrong_extreme':
        with pytest.raises(ValueError, match='worst-case mechanism'):
            module._verify_numeric_witnesses(replayed, details, population)
    else:
        module._verify_numeric_witnesses(replayed, details, population)
        if fault == 'missing':
            assert replayed[slot]['passed'] is None
