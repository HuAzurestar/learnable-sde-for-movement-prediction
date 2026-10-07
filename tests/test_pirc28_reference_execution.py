"""Actual completed reference batches and native bindings, never SCI acceptance."""

import copy
import json

import pytest

from experiments.pirc28 import approximation_checks as core
from experiments.pirc28 import reference_execution as proof
from experiments.pirc28.computation import validate_computation_result
from infrastructure.research_store import ResearchError, digest
from models.switching import SwitchingError
from tests.test_pirc28_approximation_checks import observations, run
from tests.test_pirc28_managed_evidence import computed_prefix


def test_sink_observes_actual_completed_batches_and_exact_published_summaries(monkeypatch):
    batch, model = observations(paths=2)
    original, completed, trace = core._reference_batch, [], []
    def sample(*args, **kwargs):
        value = original(*args, **kwargs)
        completed.append(value)
        return value
    def sink(record):
        assert len(completed) == len(trace)+1
        trace.append(record)
    monkeypatch.setattr(core, '_reference_batch', sample)
    report = run(batch, model, execution_callback=sink)
    expected = [proof.batch_completion_record(path['path_index'], item)
                for path in report['paths'] for item in path['reference_batch_summaries']]
    assert trace == expected and len(trace) == 8
    assert report['resource_plan']['completed_batch_trace_elements'] >= len(trace)*6


def test_noncallable_sink_refuses_before_sampling(monkeypatch):
    batch, model = observations()
    monkeypatch.setattr(core, '_reference_batch', lambda *a, **k: pytest.fail('sample before contract'))
    with pytest.raises(SwitchingError, match='completion sink'):
        run(batch, model, execution_callback=False)


def test_sink_failure_propagates_without_further_sampling(monkeypatch):
    batch, model = observations()
    class Sentinel(Exception):
        pass
    calls, original = [], core._reference_batch
    def sample(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    def refuse(record):
        raise Sentinel('original native evidence sink unavailable')
    monkeypatch.setattr(core, '_reference_batch', sample)
    with pytest.raises(Sentinel):
        run(batch, model, execution_callback=refuse)
    assert len(calls) == 1


def test_numerical_sampler_failure_retains_only_completed_batches(monkeypatch):
    batch, model = observations()
    trace, original = [], core._reference_batch
    def sample(*args, **kwargs):
        if trace:
            raise SwitchingError('NUMERICAL_FAILURE', 'second fixed batch failed')
        return original(*args, **kwargs)
    monkeypatch.setattr(core, '_reference_batch', sample)
    with pytest.raises(SwitchingError, match='second fixed batch failed'):
        run(batch, model, execution_callback=trace.append)
    assert len(trace) == 1 and trace[0]['batch_index'] == 0


def source(computed_prefix):
    store, study, _, grant, cell, executed, _, value, request, _ = computed_prefix
    result = json.loads(store.read_artifact(executed['artifact_id'], purpose='evaluate', authorization=grant))
    attempt = store.attempts()[executed['attempt_id']]
    return store, study, cell, attempt, result, value, request


def test_native_receipt_binds_actual_original_attempt_and_complete_sampling_without_science(computed_prefix):
    store, study, cell, attempt, result, value, request = source(computed_prefix)
    fit = result['fit']
    record = fit['reference_execution']
    assert record['attempt_id'] == attempt['attempt_id'] and record['run_id'] == attempt['run_id']
    assert record['runtime_binding'] == study['runtime_binding']
    assert proof.validate_reference_execution(record, study, cell, attempt, fit['model'],
                                               fit['approximation_checks']) is True
    assert len(record['completed_batches']) == 8
    receipt, = value['report']['cell_approximation_evidence']
    assert receipt['reference_execution_hash'] == digest(record)
    assert receipt['reference_sampling_verified'] is True
    assert receipt['managed_evidence_bound'] is receipt['qualified_identifiability'] is False
    assert value['report']['state'] == 'FIXTURE_ONLY'
    validate_computation_result(store, value, request)


@pytest.mark.parametrize('damage', ['missing-batch', 'duplicate-batch', 'order', 'batch-hash', 'seed',
    'attempt', 'run', 'source', 'model', 'report', 'runtime', 'boolean-complete', 'unknown', 'science'])
def test_complete_reference_metadata_cannot_be_self_rehashed_as_execution(computed_prefix, damage):
    _, study, cell, attempt, result, *_ = source(computed_prefix)
    fit = result['fit']
    record = copy.deepcopy(fit['reference_execution'])
    batches = record['completed_batches']
    if damage == 'missing-batch': batches.pop()
    elif damage == 'duplicate-batch': batches.append(copy.deepcopy(batches[0]))
    elif damage == 'order': batches.reverse()
    elif damage == 'batch-hash': batches[0]['summary_hash'] = '0'*64
    elif damage == 'seed': batches[0]['seed_entropy'][0] += 1
    elif damage == 'runtime': record['runtime_binding']['store_id'] += '-fresh-budget'
    elif damage == 'unknown': record['independent_science'] = True
    else:
        key = {'attempt': 'attempt_id', 'run': 'run_id', 'source': 'source_code_hash',
               'model': 'selected_model_hash', 'report': 'reference_report_hash',
               'boolean-complete': 'complete', 'science': 'qualified_identifiability'}[damage]
        record[key] = 1 if damage == 'boolean-complete' else True if damage == 'science' else 'forged'
    with pytest.raises(ResearchError):
        proof.validate_reference_execution(record, study, cell, attempt, fit['model'],
                                            fit['approximation_checks'])


def test_missing_report_is_partial_not_complete_sampling_proof(computed_prefix):
    _, study, cell, attempt, result, *_ = source(computed_prefix)
    fit = result['fit']
    record = proof.execution_record(study, cell, attempt, fit['model'], None,
                                    fit['reference_execution']['completed_batches'][:1])
    assert record['complete'] is False
    assert proof.validate_reference_execution(record, study, cell, attempt, fit['model'], None) is False
    record['complete'] = True
    with pytest.raises(ResearchError):
        proof.validate_reference_execution(record, study, cell, attempt, fit['model'], None)
