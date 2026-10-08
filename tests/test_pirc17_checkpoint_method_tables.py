"""Synthetic transport fixtures, not qualified scientific paper evidence."""
from copy import deepcopy
import csv
import io

import pytest

from experiments.pirc17.checkpoint_method_tables import tables, csv_bytes, export
from experiments.pirc17.checkpoint_method_qualification_preview import join_qualification
from tests.test_pirc17_checkpoint_method_qualification_preview import fixture


def evidence():
    stats, policy, ledger, gates = fixture()
    for family in policy['family_rules'].values():
        for numeric in family['numerical_applicability'].values():
            numeric.update(global_numerically_qualified=False, precision_invariant_effect_qualified=False)
    for gate in gates.values():
        if gate['status'] == 'computed':
            gate['receipt'] = dict(passed=gate['passed'], statistic='SYNTHETIC', value=3., operator='ge', threshold=1., units='fixture')
        else:
            gate['required_source'] = 'SYNTHETIC pending raw diagnostics'
    for family in stats['families'].values():
        for raw in family['results'].values():
            raw.update(delta_estimate_m=3., simultaneous_interval_m=[-2., 1.],
                       holm_adjusted_p_zero=.125, tail_check_passed=False, seed_delta_m=[3.]*5)
    value = join_qualification(stats, policy, ledger, gates)
    value.update(statistics_sha256='a'*64, settings_id='b'*64, matrix_sha256='c'*64)
    return value


def test_exact_whole_tables_preserve_endpoints_and_unknown_not_zero():
    rows = tables(evidence())
    assert len(rows['method-comparisons.csv']) == 21
    assert len(rows['method-mechanisms.csv']) == 28
    row = rows['method-comparisons.csv'][0]
    assert row['delta_candidate_minus_control_m'] == 3.
    assert [row['simultaneous_lower_m'], row['simultaneous_upper_m']] == [-2., 1.]
    parsed = list(csv.DictReader(io.StringIO(csv_bytes(rows['method-mechanisms.csv']).decode())))
    pending = next(r for r in parsed if r['status'] != 'computed')
    assert pending['value'] == pending['passed'] == ''
    assert all(r['scientific_claim_authorized'] == 'false' for r in parsed)


@pytest.mark.parametrize('change', ['missing', 'reversed', 'seed', 'nonfinite', 'claim'])
def test_transport_refuses_changed_scope_or_invalid_numeric_evidence(change):
    value = deepcopy(evidence())
    family = value['families']['method-model-structure']
    raw = family['arm-02/pointwise']['statistical_numbers']
    if change == 'missing': family.pop('arm-02/pointwise')
    elif change == 'reversed': raw['simultaneous_interval_m'] = [2., 1.]
    elif change == 'seed': raw['seed_delta_m'].pop()
    elif change == 'nonfinite': raw['delta_estimate_m'] = float('nan')
    elif change == 'claim': value['scientific_claim_authorized'] = True
    with pytest.raises(ValueError): tables(value)


def test_bundle_is_deterministic_missing_only_and_never_repairs_corruption(tmp_path):
    value = evidence()
    target, first = export(value, 'd'*64, tmp_path/'one')
    files = {p.name: p.read_bytes() for p in target.iterdir()}
    assert export(value, 'd'*64, tmp_path/'one') == (target, first)
    second, other = export(value, 'd'*64, tmp_path/'two')
    assert first == other
    assert files == {p.name: p.read_bytes() for p in second.iterdir()}
    (target/'method-comparisons.csv').write_bytes(b'corrupt fixture')
    with pytest.raises(ValueError, match='corrupt'): export(value, 'd'*64, tmp_path/'one')
    assert (target/'method-comparisons.csv').read_bytes() == b'corrupt fixture'
