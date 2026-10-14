"""Only metadata and AST fixtures; no real data or SDE prediction."""
import pyarrow as pa
import pytest

from experiments.pirc17.clock_description import clock_literals, footer_description, reconciliation_description, COUNTS


def test_literal_clock_policy_does_not_import_or_call_module():
    source = '''
SOLAR_POLICY = 'software-fixture'
SOLAR_REFERENCE = 'https://example.invalid/reference'
NANOSECONDS = 1_000_000_000
def forbidden_training():
    raise AssertionError('not imported or called')
'''
    result = clock_literals(source)
    assert result['NANOSECONDS'] == 1_000_000_000
    assert result['SOLAR_POLICY'] == 'software-fixture'


def test_complete_original_time_unit_required():
    with pytest.raises(ValueError, match='complete'):
        clock_literals('NANOSECONDS = 1_000_000_000')
    with pytest.raises(ValueError, match='nanosecond'):
        clock_literals("SOLAR_POLICY='x'\nSOLAR_REFERENCE='x'\nNANOSECONDS=1000")


def test_naive_clock_footer_not_silently_certified_utc():
    assert footer_description(pa.field('t', pa.timestamp('ns')), 7)['clock_timezone'] is None
    assert footer_description(pa.field('t', pa.timestamp('ns', tz='UTC')), 7)['clock_timezone'] == 'UTC'
    with pytest.raises(ValueError):
        footer_description(pa.field('wrong', pa.timestamp('ns')), 7)


def test_reported_denominators_and_counts():
    counts = dict.fromkeys(COUNTS, 0)
    counts.update(source_condition_points=10, aligned_refined_points=7, total_condition_points_not_aligned=3)
    assert reconciliation_description({'reconciliation': counts}) == counts
    counts['total_condition_points_not_aligned'] = 2
    with pytest.raises(ValueError, match='denominator'):
        reconciliation_description({'reconciliation': counts})


@pytest.mark.parametrize('value', [-1, 1.5, True])
def test_invalid_reported_count_rejected(value):
    counts = dict.fromkeys(COUNTS, 0)
    counts['duplicate_timestamp_intervals'] = value
    with pytest.raises(ValueError, match='integer'):
        reconciliation_description({'reconciliation': counts})
