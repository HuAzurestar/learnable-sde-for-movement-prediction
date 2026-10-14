"""Real integration on software data; synthetic closures are not OS certificates."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.pirc17 import particle_suffix_session as session
from experiments.pirc17.rollout import rollout as real_rollout
from tests.test_pirc17_particle_suffix_contract import (
    base_inputs, inputs as contract_inputs, candidate_bytes, evidence, reference, case, ready,
    closed_parent, make_source, publish, read_rows)
from tests.test_pirc17_particle_extension_execution import interrupt_after
from tests.test_pirc17_particle_extension_lineage import publish_process

execution, lineage, science, maps = session.execution, session.lineage, session.science, session.maps
engine = execution.engine


@pytest.fixture
def inputs(contract_inputs, tmp_path, monkeypatch):
    args, tracker = contract_inputs
    old_maps, modules = engine.resolve_map_backend('multicell')
    original_asset = tmp_path/'software-map-asset.bin'
    extra_asset = tmp_path/'software-extra-map-asset.bin'
    extra_asset.write_bytes(b'extra existing frozen software asset')
    assets = {p.name: {'checksum': {'value': lineage.native._hash(p)}} for p in (original_asset, extra_asset)}
    tracker.update(suffix_calls=[], whole_calls=[], include_extra_map=False, extra_asset=extra_asset)

    class Maps(old_maps):
        def __init__(self, *a):
            super().__init__(*a)
            self.assets, self.seen = deepcopy(assets), set()

        def __call__(self, positions):
            self.seen.add(original_asset.name)
            if tracker['include_extra_map']:
                self.seen.add(extra_asset.name)
            return super().__call__(positions)

        @property
        def identity(self):
            return {**super().identity, 'verified_assets': {p: self.assets[p]['checksum']['value'] for p in sorted(self.seen)}}

    def full(*a, **kwargs):
        tracker['whole_calls'].append(kwargs['particles'])
        return real_rollout(*a, **kwargs)

    original = execution.rollout_suffix
    def suffix(*a, **kwargs):
        tracker['suffix_calls'].append((kwargs['prefix_particles'], kwargs['driver'].particles,
                                       kwargs['stream_id'], kwargs['max_step_seconds']))
        return original(*a, **kwargs)

    monkeypatch.setattr(engine, 'resolve_map_backend', lambda _: (Maps, modules))
    monkeypatch.setattr(engine, 'rollout', full)
    monkeypatch.setattr(execution, 'rollout_suffix', suffix)
    return args, tracker


@pytest.fixture
def source(closed_parent, tmp_path):
    reservation, result = closed_parent
    return make_source(reservation, tmp_path, science_sha=result['whole_science_sha256'])


@pytest.fixture
def oracle(source, reference, tmp_path):
    args = dict(reference[1], particles=[32], output=tmp_path/'full-suffix-oracle.jsonl')
    assert engine.run(**args)['status'] == 'complete'
    return source, read_rows(args['output'])[2:-1], args['output'].parent, reference[-1]


def reserve(source):
    return lineage.reserve(source=source)


def run(reservation, **kwargs):
    return execution.run(**{k: reservation[k] for k in ('directory', 'root_sha256', 'index', 'start_sha256')}, **kwargs)


def seal(reservation, **changes):
    # These storage fixtures deliberately publish simulated process facts.
    publish_process(reservation, **changes)
    return lineage.seal(reservation)


def assert_complete(reservation, result, oracle):
    _, expected, folder, _ = oracle
    assert result['status'] == 'computed_pending_supervisory_closure', result
    work = reservation['attempt']/'work'
    saved = lineage.storage._snapshot(work, result['journal_tip'])
    rows = execution.assembled(work, saved)
    assert len(rows) == result['expected_run_count'] == 24
    assert result['new_failure_count'] == result['missing_completed_record_count'] == 0
    assert set(result) == execution.RESULT_FIELDS
    for row, full in zip(rows, expected):
        assert all(np.array_equal(a, b) for a, b in zip(execution.load_particle_evidence(row, work),
                                                       execution.load_particle_evidence(full, folder)))
        for key in ('scores', 'particle_precision', 'brownian_identity', 'model_identity_sha256'):
            assert row[key] == full[key]
        account = row['suffix_accounting']
        assert account['computed_particles'] == account['particle_start'] == 16 and account['particle_stop'] == 32
        assert account['new_feature_query_rows'] == account['inherited_feature_query_rows']
        assert account['ensemble_feature_rows'] == full['feature_query_rows']
    for key, value in science.UNQUALIFIED.items():
        assert result[key] == value
    assert not list(work.glob('*.jsonl'))
    return rows


def test_three_attempts_compute_only_missing_particle_suffixes_with_exact_full_oracle(oracle):
    source, _, _, tracker = oracle
    whole_calls = len(tracker['whole_calls'])
    parent_before = lineage.storage.inventory(Path(source['parent_directory']))
    charged = 0.
    for count in (5, 7):
        current = reserve(source)
        assert current['start']['prior_elapsed_seconds'] == charged
        with pytest.raises(KeyboardInterrupt):
            run(current, progress=interrupt_after(count))
        closure = seal(current)
        assert closure['status'] == 'interrupted' and closure['continuable']
        charged += closure['elapsed_seconds']
    final = reserve(source)
    assert final['start']['inherited_count'] == 12
    assert final['start']['cooperative_remaining_seconds'] == 1000.-charged
    result = run(final)
    assert result['inherited_success_count'] == result['new_committed_count'] == 12
    assert_complete(final, result, oracle)
    assert seal(final, worker_returncode=0, interrupted=False)['status'] == 'computed_closed'
    report = session.inspect_closed(final['directory'], final['root_sha256'])
    assert report['attempt_count'] == 3 and report['expected_run_count'] == report['new_committed_count'] == 24
    assert report['parent_runs_covered'] == report['exact_parent_prefix_checks'] == 24
    assert report['reference_runs_covered'] == 48
    assert report['reference_resource_charges'] == result['reference_resource_charges']
    assert len(tracker['whole_calls']) == whole_calls
    assert len(tracker['suffix_calls']) == 24 and all(c[:2] == (16, 32) for c in tracker['suffix_calls'])
    assert all(m.closed for m in tracker['maps'])
    assert lineage.storage.inventory(Path(source['parent_directory'])) == parent_before
    with pytest.raises(ValueError, match='no automatic retry'):
        reserve(source)


def test_audit_only_continuation_has_no_new_forecasts_or_map_use(oracle):
    source, _, _, tracker = oracle
    first = reserve(source)
    with pytest.raises(KeyboardInterrupt):
        run(first, progress=interrupt_after(24))
    assert seal(first)['continuable']
    calls, queries = len(tracker['suffix_calls']), len(tracker['queries'])
    second = reserve(source)
    result = run(second)
    assert_complete(second, result, oracle)
    assert result['inherited_success_count'] == 24 and result['new_committed_count'] == 0
    assert result['new_attempt_maps'] is None
    assert result['map_observations']['committed_observations'] == []
    assert result['map_observations']['observed_asset_sha256'] == {}
    seal(second, worker_returncode=0, interrupted=False)
    assert session.inspect_closed(second['directory'], second['root_sha256'])['attempt_count'] == 2
    assert len(tracker['suffix_calls']) == calls and len(tracker['queries']) == queries


@pytest.mark.parametrize('exception', [ValueError, MemoryError])
def test_failure_is_retained_with_full_denominator_and_never_retried(source, monkeypatch, exception):
    original, calls = execution.rollout_suffix, []
    def fail_once(*a, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise exception('software failed suffix forecast')
        return original(*a, **kwargs)
    monkeypatch.setattr(execution, 'rollout_suffix', fail_once)
    current = reserve(source)
    result = run(current)
    assert result['status'] == 'failed' and result['new_failure_count'] == 1
    assert result['expected_run_count'] == 24
    assert result['new_committed_count'] == (1 if exception is MemoryError else 24)
    assert result['missing_completed_record_count'] == (23 if exception is MemoryError else 0)
    assert not (current['attempt']/'work/whole-science.json').exists()
    assert not seal(current, worker_returncode=1, interrupted=False)['continuable']
    with pytest.raises(ValueError, match='no automatic retry'):
        reserve(source)


@pytest.mark.parametrize('fault', ['targets', 'times', 'block', 'runtime'])
def test_changed_actual_inputs_are_rejected_before_any_suffix(source, inputs, monkeypatch, fault):
    tracker = inputs[1]
    current = reserve(source)
    window = tracker['windows']['validation'][0]
    if fault == 'targets': window.target_positions_m = window.target_positions_m+1
    if fault == 'times': window.horizon_seconds = window.horizon_seconds+1
    if fault == 'block': window.block_id = 'changed'
    if fault == 'runtime':
        original = science.science.runtime_identity()
        monkeypatch.setattr(science.science, 'runtime_identity', lambda: {**original, 'torch_intraop_threads': 999})
    with pytest.raises(ValueError):
        run(current)
    assert not tracker['suffix_calls'] and not (current['attempt']/'work').exists()


def test_unclosed_and_existing_attempts_cannot_be_reopened(source):
    current = reserve(source)
    with pytest.raises(FileExistsError, match='unclosed'):
        reserve(source)
    with pytest.raises(KeyboardInterrupt):
        run(current, progress=interrupt_after(1))
    with pytest.raises(FileExistsError, match='never reopens'):
        run(current)
    with pytest.raises(FileExistsError, match='unclosed'):
        session.inspect_closed(current['directory'], current['root_sha256'])


def test_fresh_low_memory_stops_after_durable_progress(source, monkeypatch):
    current = reserve(source)
    def lower(event):
        if event['phase'] == 'run_committed' and event['new_committed'] == 2:
            monkeypatch.setattr(lineage.native.physical_memory, 'available_physical_bytes', lambda: 0)
    result = run(current, progress=lower)
    assert result['status'] == 'failed' and result['new_committed_count'] == 2
    assert result['missing_completed_record_count'] == 22 and result['new_failure_count'] == 0
    assert result['observer']['minimum_observed_available_bytes'] == 0
    assert not seal(current, worker_returncode=1, interrupted=False)['continuable']


def test_orphan_observation_and_particle_do_not_become_committed_work(oracle, monkeypatch):
    source, _, _, tracker = oracle
    current = reserve(source)
    original = lineage.journal._publish_json
    def interrupt_record(path, value):
        if path.parent.name == 'records' and path.name == '000005.json':
            raise KeyboardInterrupt('software cut after particle and map publication')
        return original(path, value)
    with monkeypatch.context() as cut:
        cut.setattr(lineage.journal, '_publish_json', interrupt_record)
        with pytest.raises(KeyboardInterrupt):
            run(current)
    closure = seal(current)
    assert closure['new_committed_count'] == 5 and closure['continuable']
    orphan = current['attempt']/'work/map-observations/000005.json'
    assert orphan.exists()
    old_inventory = lineage.storage.inventory(current['attempt'])
    second = reserve(source)
    result = run(second)
    assert result['inherited_success_count'] == 5 and result['new_committed_count'] == 19
    assert_complete(second, result, oracle)
    seal(second, worker_returncode=0, interrupted=False)
    assert session.inspect_closed(second['directory'], second['root_sha256'])['new_committed_count'] == 24
    assert len(tracker['suffix_calls']) == 25  # One computed-but-uncommitted forecast is honestly repeated.
    assert old_inventory == lineage.storage.inventory(current['attempt'])


def test_additional_catalog_bound_map_asset_and_late_mutation(oracle, monkeypatch):
    source, _, _, tracker = oracle
    tracker['include_extra_map'] = True
    current = reserve(source)
    result = run(current)
    assert_complete(current, result, oracle)
    assert tracker['extra_asset'].name in result['new_attempt_maps']['verified_assets']
    seal(current, worker_returncode=0, interrupted=False)
    assert session.inspect_closed(current['directory'], current['root_sha256'])['expected_run_count'] == 24
    original = science.whole_audit
    def mutate(*a, **kwargs):
        result = original(*a, **kwargs)
        tracker['extra_asset'].write_bytes(b'late change to extra current-attempt-only asset')
        return result
    monkeypatch.setattr(science, 'whole_audit', mutate)
    with pytest.raises(ValueError, match='observed suffix map asset changed'):
        session.inspect_closed(current['directory'], current['root_sha256'])


@pytest.mark.parametrize('fault', ['science', 'observation', 'map_summary', 'charge', 'memory'])
def test_internally_rehashed_closed_results_still_require_independent_science(source, fault):
    current = reserve(source)
    result = run(current)
    assert result['status'] == 'computed_pending_supervisory_closure', result
    work = current['attempt']/'work'
    if fault == 'science':
        path = work/'whole-science.json'
        value = json.loads(path.read_text())
        value['original_numerical']['out_of_tolerance'] += 1
        result['whole_science_sha256'] = publish(path, value)
    if fault == 'observation':
        path = work/'map-observations/000002.json'
        value = json.loads(path.read_text())
        value['row_sha256'] = 'f'*64
        publish(path, value)
    if fault == 'map_summary': result['new_attempt_maps'] = None
    if fault == 'charge': result['reference_resource_charges']['parent_extension_charged_seconds'] = 0.
    if fault == 'memory': result['observer']['minimum_observed_available_bytes'] = 0
    publish(work/'core-result.json', result)
    assert seal(current, worker_returncode=0, interrupted=False)['status'] == 'computed_closed'
    with pytest.raises(ValueError):
        session.inspect_closed(current['directory'], current['root_sha256'])


def test_stop_request_and_direct_worker_entry_do_not_claim_termination(source, inputs):
    current = reserve(source)
    assert not session._stop_requested(current)
    session.request_stop(current['directory'], current['root_sha256'])
    assert session._stop_requested(current)
    assert not (current['attempt']/'closure.json').exists()
    if os.name == 'nt':
        with pytest.raises(PermissionError, match='live reserved process job'):
            session.worker(**{k: current[k] for k in ('directory', 'root_sha256', 'index', 'start_sha256')})
    assert not inputs[1]['suffix_calls'] and not (current['attempt']/'work').exists()


@pytest.mark.skipif(os.name != 'nt', reason='actual Windows owned suffix CLI')
def test_actual_cold_suffix_cli_closes_owned_job_on_bad_parent_without_forecasts(source):
    source = deepcopy(source)
    spec = json.loads(source['plan'].read_text())
    spec['parent_root_sha256'] = 'f'*64
    source['plan_sha256'] = publish(source['plan'], spec)
    command = [sys.executable, '-u', '-m', 'experiments.pirc17.particle_suffix_session', 'run']
    for name, value in source.items():
        command.extend(['--'+name.replace('_', '-'), str(value)])
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', timeout=90,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode == 1, (result.stdout, result.stderr)
    assert 'closed evidence hash mismatch' in result.stderr
    directory = science.directory_for(source, spec)
    digest = lineage.native._hash(directory/'root.json')
    _, closed, _, _ = lineage.history(directory, digest)
    assert len(closed) == 1 and closed[0]['closure']['status'] == 'failed'
    assert closed[0]['closure']['process_tree_closed'] and not closed[0]['closure']['continuable']
    assert closed[0]['closure']['new_committed_count'] == 0
    process = json.loads((closed[0]['directory']/'work-process.json').read_text())
    assert process['accounting']['active_processes'] == 0 and process['accounting']['total_processes'] >= 2
