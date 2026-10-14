"""Synthetic saved-counter arithmetic, not actual missingness/robustness evidence."""
from copy import deepcopy
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.pirc17 import evidence_cards as module


def row(status='success', feature=None, invalid=None, maps=None):
    return dict(status=status, feature_query_accounting=dict(feature_query_rows=feature,
        invalid_feature_rows=invalid, raw_map_query_rows_this_forecast=maps))


def test_unknown_counts_are_not_zero_or_successfully_filtered():
    rows = [row(), row('failed'), row('NOT_ADMITTED')]
    value = module._query_counts(rows, 'terrain')
    assert value['expected_forecasts'] == 3
    assert value['counts_by_status'] == dict(success=1, failed=1, NOT_ADMITTED=1)
    for counter in value['counters'].values():
        assert counter == dict(recorded_forecasts=0, missing_forecasts=3,
            row_total=None, recorded_counts_by_status={})
    assert value['paired_invalid_counter_forecasts'] == 0
    assert value['paired_feature_row_denominator'] is None
    assert value['paired_invalid_row_numerator'] is None
    assert value['recorded_selected_feature_invalid_fraction'] is None


def test_paired_denominator_partial_counters_and_failures_are_retained():
    rows = [row(feature=100, invalid=7, maps=100), row('failed', feature=20, invalid=2, maps=20),
        row(feature=900, maps=900), row('failed', maps=17), row('NOT_ADMITTED')]
    before = deepcopy(rows)
    value = module._query_counts(rows, 'terrain')
    assert rows == before
    assert value['expected_forecasts'] == 5
    assert value['counts_by_status'] == dict(success=2, failed=2, NOT_ADMITTED=1)
    assert value['counters']['feature_query_rows']['row_total'] == 1020
    assert value['counters']['feature_query_rows']['missing_forecasts'] == 2
    assert value['counters']['invalid_feature_rows']['row_total'] == 9
    assert value['counters']['invalid_feature_rows']['missing_forecasts'] == 3
    assert value['counters']['raw_map_query_rows_this_forecast']['row_total'] == 1037
    assert value['counters']['raw_map_query_rows_this_forecast']['recorded_counts_by_status'] == dict(success=2, failed=2)
    assert value['paired_feature_row_denominator'] == 120  # NOT the unrelated1020 rows.
    assert value['paired_invalid_row_numerator'] == 9
    assert value['paired_invalid_counts_by_status'] == dict(success=1, failed=1)
    assert value['paired_invalid_counter_missing_forecasts'] == 3
    assert value['recorded_selected_feature_invalid_fraction'] == 9/120


@pytest.mark.parametrize('matrix', ['terrain', 'NEX326-methods'])
def test_recorded_zero_is_distinct_from_absent_and_zero_denominator(matrix):
    value = module._query_counts([row(feature=0, invalid=0, maps=0), row('failed')], matrix)
    for counter in value['counters'].values():
        assert counter['recorded_forecasts'] == counter['missing_forecasts'] == 1
        assert counter['row_total'] == 0
    assert value['paired_feature_row_denominator'] == value['paired_invalid_row_numerator'] == 0
    assert value['recorded_selected_feature_invalid_fraction'] is None


def test_ordinary_contractual_zero_is_not_a_measured_map_validity_rate():
    value = module._query_counts([row(feature=100, invalid=0)], 'NEX326-methods')
    assert value['paired_feature_row_denominator'] == 100
    assert value['paired_invalid_row_numerator'] == 0
    assert value['recorded_selected_feature_invalid_fraction'] is None
    assert value['invalid_counter_role'] == 'contractual-zero-not-map-validity'
    assert value['counters']['raw_map_query_rows_this_forecast']['row_total'] is None


def test_nonzero_ordinary_invalid_counts_or_unknown_matrix_are_not_relabelled():
    with pytest.raises(ValueError, match='contractual zero'):
        module._query_counts([row(feature=100, invalid=1)], 'NEX326-methods')
    with pytest.raises(ValueError, match='registered scientific matrix'):
        module._query_counts([row(feature=100, invalid=1)], 'Terrain')


@pytest.mark.parametrize('bad', [row(feature=True, invalid=0), row(feature=1, invalid=2),
    row(feature=1, invalid=-1), row(invalid=1), row(maps=float('nan'))])
def test_invalid_saved_counter_is_not_hidden_by_aggregation(bad):
    with pytest.raises(ValueError):
        module._query_counts([row(feature=100, invalid=0), bad], 'terrain')


def test_existing_tsde_kernel_table_transports_exact_counter_summary():
    """Real CSV serializer, tiny software records only; no extra table."""
    rows = [dict(row(feature=100, invalid=7, maps=100),
        forecast_axes=dict(matrix='terrain', subject='base'), kernel_prediction_seconds=1.5),
        dict(row('failed'), forecast_axes=dict(matrix='terrain', subject='base'),
            kernel_prediction_seconds=None)]
    cost = module._costs(rows, dict(runtime_replay=dict(runtime_subjects=[])))['terrain:base']
    tsde = Path(__file__).resolve().parents[2]/'TSDE-SDE'
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    command = 'import json,sys; from scripts.aggregate_pirc17 import csv_bytes; sys.stdout.write(csv_bytes(json.load(sys.stdin)).decode("utf-8"))'
    result = subprocess.run([sys.executable, '-X', 'utf8', '-c', command], cwd=tsde, env=environment,
        input=json.dumps([dict(cost_id='terrain:base', **cost)]), capture_output=True,
        text=True, encoding='utf-8', timeout=60, check=True)
    exported = list(csv.DictReader(io.StringIO(result.stdout)))
    assert len(exported) == 1
    assert exported[0]['cost_id'] == 'terrain:base'
    assert json.loads(exported[0]['feature_query_accounting']) == cost['feature_query_accounting']
    assert json.loads(exported[0]['counts_by_status']) == dict(success=1, failed=1)
    assert exported[0]['missing_timing_count'] == '1'
