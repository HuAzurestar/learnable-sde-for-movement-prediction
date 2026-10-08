"""Metadata-only regression; no saved real predictions, fits or authority."""
from collections import Counter
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from experiments.pirc17 import formal_partial_inputs as partial, formal_eligibility as eligibility
from experiments.pirc17 import formal_inputs, final_eval_guard as guard
from experiments.pirc17.formal_restoration_costs import RestorationCosts, MAX_EVENTS
from experiments.pirc17.protocol_core import canonical, digest, envelope, publish, read_json, unpack
from tests.test_pirc17_formal_eligibility import metadata_fixture


def metadata_source(root, count):
    samples = [metadata_fixture(str(index)) for index in range(count)]
    identities = ('sample_id', 'segment_id', 'independent_block_id')
    payload = dict(dataset_inputs=dict(dataset_id='SYNTHETIC METADATA ONLY', sha256=digest('synthetic inputs'),
        partitions=dict(counts={'final_eval': count},
            split_identity_sha256={'final_eval': digest({k: sorted({s[k] for s, _ in samples}) for k in identities})},
            split_row_identity_sha256={'final_eval': digest(sorted([s[k] for k in identities] for s, _ in samples))})),
        eligibility_contract={'fixture': 'SYNTHETIC metadata only'})
    protocol = envelope(payload)
    execution = envelope(dict(fixture='NO REAL AUTHORITY', protocol_sha256=protocol['sha256']))
    successor = envelope(dict(unpack(execution), fixture='NO REAL SUCCESSOR AUTHORITY'))
    rows, windows = [], {}
    for sample, points in samples:
        row, window = eligibility.qualify_window(sample, points, input_sha256=payload['dataset_inputs']['sha256'],
                                                 rule_sha256=digest(payload['eligibility_contract']))
        assert row['eligible']
        path, _ = publish(root/'source'/'windows', unpack(window))
        rows.append(row)
        windows[sample['sample_id']] = path.relative_to(root/'source').as_posix()
    report = envelope(dict(schema_version=guard.ELIGIBILITY_VERSION, protocol_sha256=protocol['sha256'],
        execution_sha256=execution['sha256'], dataset_id=payload['dataset_inputs']['dataset_id'],
        eligibility_rule_sha256=digest(payload['eligibility_contract']), prior_performance_reads=0, rows=rows))
    population = envelope(guard.population_contract(report, protocol=protocol, execution=execution))
    report_path, _ = publish(root/'source'/'eligibility', unpack(report))
    population_path, _ = publish(root/'source'/'population', unpack(population))
    approval = digest('SOFTWARE ONLY approval')
    _, event = publish(root/'access', dict(schema_version='pirc17-final-access-event-v1', event='started',
        attempt_id='1'*32, at_utc='2026-10-04T00:00:00+00:00', access_kind='final_eval_eligibility',
        protocol_sha256=protocol['sha256'], execution_sha256=execution['sha256'],
        approval_sha256=approval, population_sha256=None))
    qpath, qualified = publish(root/'source', dict(schema_version=eligibility.RESULT_VERSION,
        protocol_sha256=protocol['sha256'], execution_sha256=execution['sha256'], approval_sha256=approval,
        access_started_sha256=event['sha256'], position_feature_value_prediction_metric_reads=0,
        eligibility_path=report_path.relative_to(root/'source').as_posix(), eligibility_sha256=report['sha256'],
        population_path=population_path.relative_to(root/'source').as_posix(), population_sha256=population['sha256'],
        denominator_samples=count, eligible_samples=count, windows=windows))
    return dict(protocol=protocol, execution=successor, report=report, report_path=report_path,
        access=SimpleNamespace(access_kind='final_eval_eligibility', population_sha256=None,
            approval_sha256=digest('SOFTWARE ONLY successor'), access_started_sha256=digest('SOFTWARE ONLY access')),
        source=dict(root=root, original=dict(execution=execution), costs=dict(approval_sha256=approval),
            item=dict(directory=root/'source', manifest=dict(qualification_path=str(qpath), qualification_sha256=qualified['sha256']))))


@pytest.mark.parametrize('count', [1, 8, 106])
def test_fixed_report_hashing_is_constant_and_every_window_is_still_validated(tmp_path, monkeypatch, count):
    fixture = metadata_source(tmp_path/'original', count)
    hashes, checked = Counter(), []
    original_unpack, original_validate = partial.unpack, formal_inputs.validate_window
    def counted(record, **kwargs):
        hashes[record.get('sha256')] += 1
        return original_unpack(record, **kwargs)
    def window(record, **kwargs):
        checked.append(record['sha256'])
        return original_validate(record, **kwargs)
    monkeypatch.setattr(partial, 'unpack', counted)
    monkeypatch.setattr(formal_inputs, 'validate_window', window)
    restored = partial._qualification(fixture['access'], source=fixture['source'], protocol=fixture['protocol'],
                                     execution=fixture['execution'], output=tmp_path/'restored')
    assert len(checked) == count and len(set(checked)) == count
    assert hashes[fixture['report']['sha256']] == 2  # No per-window complete-report hash.
    assert hashes[fixture['protocol']['sha256']] == 0  # Owned snapshot constructor validates it.
    result = unpack(restored['qualification'])
    assert result['eligible_samples'] == result['denominator_samples'] == count
    for relative in result['windows'].values():
        assert read_json(tmp_path/'restored'/relative) == read_json(tmp_path/'original'/'source'/relative)


@pytest.mark.parametrize('fault', ['protocol-rehashed', 'protocol-unhashed', 'report-file', 'window'])
def test_mutation_cannot_escape_at_qualification_handoff(tmp_path, monkeypatch, fault):
    fixture = metadata_source(tmp_path/'original', 3)
    actual, changed = formal_inputs.validate_window, []
    def corrupt(record, **kwargs):
        result = actual(record, **kwargs)
        if not changed:
            changed.append(True)
            if fault.startswith('protocol'):
                fixture['protocol']['payload']['dataset_inputs']['sha256'] = digest('changed input')
                if fault == 'protocol-rehashed':
                    fixture['protocol']['sha256'] = digest(fixture['protocol']['payload'])
            elif fault == 'report-file':
                # Only this test's synthetic file; same content hash cannot
                # conceal changed physical bytes at the closing boundary.
                with fixture['report_path'].open('ab') as stream:
                    stream.write(b' ')
            else:
                result = deepcopy(result)
                result['eligibility_rule_sha256'] = digest('changed window rule')
        return result
    monkeypatch.setattr(formal_inputs, 'validate_window', corrupt)
    with pytest.raises(ValueError, match='changed'):
        partial._qualification(fixture['access'], source=fixture['source'], protocol=fixture['protocol'],
                               execution=fixture['execution'], output=tmp_path/'restored')
    assert not (tmp_path/'restored'/'eligibility').exists()


def test_diagnostic_records_stage_counts_times_and_never_admits_science(tmp_path):
    clock = iter(range(0, 100000, 100))
    costs = RestorationCosts(tmp_path, 'worker', _clock=lambda: next(clock))
    costs.mark('input-restoration')
    costs.mark('saved-result-restoration', total=501)
    for index in range(1, 502):
        costs.progress(index)
    costs.close('returned')
    costs.close('failed')  # No replacement terminal diagnostic.
    rows = [json.loads(line) for line in (tmp_path/'restoration-cost-worker.jsonl').read_text().splitlines()]
    assert len(rows) == 7 and rows[-1]['status'] == 'returned'
    assert [r['completed'] for r in rows if r['event'] == 'progress'] == [250, 500, 501]
    assert all(r['elapsed_ns'] >= r['stage_elapsed_ns'] >= 0 for r in rows)
    assert all(r['diagnostic_only'] and not r['authorizes_execution'] and not r['scientific_admission'] for r in rows)
    assert not (tmp_path/'bootstrap').exists()


def test_missing_or_exhausted_optional_sink_does_not_change_the_control_path(tmp_path):
    costs = RestorationCosts(tmp_path/'missing', 'controller')
    costs.mark('source-validation')
    costs.close('failed')
    assert costs.closed and not (tmp_path/'missing').exists()
    costs = RestorationCosts(tmp_path, 'controller')
    costs.mark('source-validation', total=MAX_EVENTS*250)
    for index in range(1, MAX_EVENTS+1):
        costs.progress(index*250)
    costs.close('returned')
    lines = (tmp_path/'restoration-cost-controller.jsonl').read_text().splitlines()
    assert len(lines) == MAX_EVENTS


def test_optional_sink_io_failure_cannot_mask_a_validator_exception(tmp_path):
    costs = RestorationCosts(tmp_path/'missing', 'worker')
    class BrokenSink:
        def write(self, value):
            raise OSError('synthetic log write failure')
        def close(self):
            raise OSError('synthetic log close failure')
    costs.stream = BrokenSink()
    costs.mark('source-validation')
    assert costs.stream is None
    with pytest.raises(ValueError, match='original domain rejection'):
        try:
            raise ValueError('original domain rejection')
        finally:
            costs.close('failed')


def test_existing_diagnostic_is_never_overwritten(tmp_path):
    first = RestorationCosts(tmp_path, 'worker')
    first.mark('source-validation')
    first.close('failed')
    path = tmp_path/'restoration-cost-worker.jsonl'
    before = path.read_bytes()
    second = RestorationCosts(tmp_path, 'worker')
    second.mark('source-validation')
    second.close('returned')
    assert path.read_bytes() == before


@pytest.mark.parametrize('count', [-1, True, 2])
def test_invalid_progress_is_rejected_without_authority(tmp_path, count):
    costs = RestorationCosts(tmp_path, 'worker')
    try:
        costs.mark('model-restoration', total=1)
        with pytest.raises(ValueError):
            costs.progress(count)
    finally:
        costs.close('failed')
