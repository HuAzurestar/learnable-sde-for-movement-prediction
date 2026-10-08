"""Saved scalar transport checks, not new forecasting or validity experiments."""
from copy import deepcopy

import pytest

from experiments.pirc17 import checkpoint_audit as module


@pytest.mark.parametrize('payload,expected', [
    (None, (None, None, None)),
    ({}, (None, None, None)),
    ({'feature_query_rows': 0, 'invalid_feature_rows': 0, 'maps': None}, (0, 0, None)),
    ({'feature_query_rows': 100, 'invalid_feature_rows': 7,
      'maps': {'query_rows_this_forecast': 100, 'observation': {'attempted_query_rows': 999999}}}, (100, 7, 100)),
    ({'feature_query_rows': 100, 'invalid_feature_rows': 0,
      'maps': {'query_rows_this_forecast': 0}}, (100, 0, 0)),
    ({'feature_query_rows': None, 'invalid_feature_rows': None,
      'maps': {'query_rows_this_forecast': 17}}, (None, None, 17)),
])
def test_copy_retains_missing_zero_failed_attempt_and_noncumulative_counts(payload, expected):
    before = deepcopy(payload)
    row = module.forecast_query_accounting(payload)
    assert tuple(row[k] for k in ('feature_query_rows', 'invalid_feature_rows',
                                  'raw_map_query_rows_this_forecast')) == expected
    assert payload == before
    assert set(row) == {'feature_query_rows', 'invalid_feature_rows', 'raw_map_query_rows_this_forecast'}


@pytest.mark.parametrize('field,value', [
    ('feature_query_rows', -1), ('feature_query_rows', True),
    ('invalid_feature_rows', 1.2), ('invalid_feature_rows', 101),
    ('raw_map_query_rows_this_forecast', float('inf')),
])
def test_malformed_or_impossible_counters_rejected(field, value):
    row = dict(feature_query_rows=100, invalid_feature_rows=7, raw_map_query_rows_this_forecast=100)
    row[field] = value
    with pytest.raises(ValueError):
        module.validate_query_accounting(row)


def test_invalid_rows_need_denominator_and_unknown_schema_is_not_accepted():
    with pytest.raises(ValueError):
        module.validate_query_accounting(dict(feature_query_rows=None, invalid_feature_rows=1,
                                              raw_map_query_rows_this_forecast=None))
    with pytest.raises(ValueError):
        module.validate_query_accounting({})


def test_scope_does_not_claim_per_column_map_only_or_missingness_robustness():
    scope = module.QUERY_ACCOUNTING_SCOPE
    assert module.VERSION == 'pirc17-checkpoint-independent-audit-v2'
    for key in ('per_column_validity_rates_available', 'history_and_map_missingness_separated',
                'missing_input_robustness_established', 'independent_samples_implied'):
        assert scope[key] is False
    assert 'union' in scope['invalid_rows']
    assert 'not exported or summed' in scope['raw_map_rows']
    assert 'null' in scope['absence']
    assert 'contractually zero' in scope['ordinary_method_scope']
    assert 'no cross-matrix mask comparison' in scope['ordinary_method_scope']
