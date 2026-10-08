"""Closed-parent and complete suffix science fixtures, not OS/empirical proof."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.pirc17 import particle_suffix_contract as contract
from tests.test_pirc17_direct_linear_rollout import read_rows
from tests.test_pirc17_particle_extension import publish
from tests.test_pirc17_particle_extension_execution import (
    candidate_bytes, evidence, inputs as base_inputs, reference, case, ready, reserve, run, seal, interrupt_after)

engine, lineage = contract.engine, contract.lineage


@pytest.fixture
def inputs(base_inputs, tmp_path, monkeypatch):
    old_maps, modules = engine.resolve_map_backend('multicell')
    receipt = tmp_path/'software-map-receipt.json'
    receipt_sha = publish(receipt, {'software_only': True})
    asset = tmp_path/'software-map-asset.bin'
    asset.write_bytes(b'software-only map asset')
    asset_sha = lineage.native._hash(asset)

    class Maps(old_maps):
        @property
        def identity(self):
            return {**super().identity, 'receipt_sha256': {str(receipt): receipt_sha},
                    'verified_assets': {asset.name: asset_sha}}

    monkeypatch.setattr(engine, 'resolve_map_backend', lambda _: (Maps, modules))
    return base_inputs


@pytest.fixture
def closed_parent(ready):
    reservation = reserve(ready)
    result = run(reservation)
    assert result['status'] == 'computed_pending_supervisory_closure', result
    seal(reservation, worker_returncode=0, interrupted=False)
    return reservation, result


def make_source(reservation, folder, *, science_sha=None):
    root = lineage.read_root(reservation['directory'], reservation['root_sha256'])
    spec = dict(root['plan'], schema_version=contract.PLAN_VERSION, purpose=contract.PURPOSE,
        output_name='software-particle-suffix', prefix_particles=root['plan']['particles'][0],
        particles=[2*root['plan']['particles'][0]], parent_root_sha256=reservation['root_sha256'],
        parent_closure_sha256=(lineage.native._hash(reservation['attempt']/'closure.json')
                              if (reservation['attempt']/'closure.json').exists() else 'f'*64),
        parent_whole_science_sha256=science_sha or 'f'*64)
    path = folder/'suffix-plan.json'
    return dict(plan=path, plan_sha256=publish(path, spec), parent_directory=reservation['directory'])


@pytest.fixture
def admitted(closed_parent, tmp_path):
    reservation, result = closed_parent
    source = make_source(reservation, tmp_path, science_sha=result['whole_science_sha256'])
    return contract.prepare(source=source)


@pytest.fixture
def completed(admitted, reference, tmp_path):
    # Full software oracle supplies arrays. The explicit counters below are
    # synthetic accounting fixtures, not measurements of a suffix executor.
    args = dict(reference[1], particles=admitted.spec['particles'], output=tmp_path/'suffix-oracle.jsonl')
    assert engine.run(**args)['status'] == 'complete'
    rows = read_rows(args['output'])[2:-1]
    for index, (row, original) in enumerate(zip(rows, admitted.parent_rows)):
        full_features, full_invalid = row['feature_query_rows'], row['invalid_feature_rows']
        for name in contract.MEASURED_FIELDS:
            row.pop(name)
        new_features = full_features-original['feature_query_rows']
        row.update(schema_version=contract.VERSION+'-run', prefix_binding=contract.prefix_binding(admitted, index),
            suffix_accounting=dict(particle_start=16, particle_stop=32, computed_particles=16,
                new_feature_query_rows=new_features, new_invalid_feature_rows=full_invalid-original['invalid_feature_rows'],
                new_raw_map_query_rows=0 if row['configuration']=='base' else new_features,
                inherited_feature_query_rows=original['feature_query_rows'], inherited_invalid_feature_rows=original['invalid_feature_rows'],
                ensemble_feature_rows=full_features, ensemble_invalid_rows=full_invalid,
                new_rollout_wall_seconds_including_lazy_map_initialization=.25))
    return admitted, rows, args['output'].parent


def test_entire_closed_parent_is_read_only_and_denominators_remain_distinct(admitted, ready):
    before = lineage.storage.inventory(Path(admitted.source['parent_directory']))
    calls = len(ready[-1]['calls'])
    assert len(admitted.parent_rows) == len(admitted.keys) == 24
    assert len(admitted.original_reference)-3 == 48
    assert admitted.parent_inspection['status'] == 'verified_closed_whole_workload'
    assert admitted.parent_science['reference_numerical']['out_of_tolerance'] > 0
    assert admitted.spec['prefix_particles'] == 16 and all(k[3] == 32 for k in admitted.keys)
    assert not contract.directory_for(admitted.source, admitted.spec).exists()
    assert admitted.reference_resource_charges['original_native_reference_seconds'] == 1.
    assert admitted.reference_resource_charges['parent_extension_charged_seconds'] > 0
    admitted.recheck()
    assert len(ready[-1]['calls']) == calls
    assert before == lineage.storage.inventory(Path(admitted.source['parent_directory']))


def test_live_or_interrupted_prefix_never_becomes_eligible(ready, tmp_path):
    reservation = reserve(ready)
    source = make_source(reservation, tmp_path)
    with pytest.raises(FileExistsError, match='unclosed'):
        contract.prepare(source=source)
    with pytest.raises(KeyboardInterrupt):
        run(reservation, progress=interrupt_after(2))
    seal(reservation)
    source = make_source(reservation, tmp_path)
    with pytest.raises(ValueError, match='entire successfully closed'):
        contract.prepare(source=source)


def test_every_fixed_parent_field_and_hash_remains_bound(closed_parent, tmp_path):
    reservation, result = closed_parent
    source = make_source(reservation, tmp_path, science_sha=result['whole_science_sha256'])
    original = json.loads(source['plan'].read_text())
    mutations = {
        'schema_version': 'old-ledger', 'purpose': 'seed_extension', 'particles': [16],
        'prefix_particles': 8, 'configurations': original['configurations'][:-1],
        'steps': [5., 1.25], 'seeds': [engine.SEEDS[1]], 'limit_origins': 1,
        'expected_run_count': 12, 'selection_policy': 'lexical_origins', 'map_backend': 'batched',
        'tolerance_m': 100., 'wall_seconds': 1001, 'outer_wall_seconds': 1051,
        'minimum_free_bytes': 0, 'fit_sha256': 'f'*64, 'fit_ledger_sha256': 'f'*64,
        'training_policy_sha256': 'f'*64, 'eligibility_sha256': 'f'*64,
        'reference_ledger_sha256': 'f'*64, 'reference_audit_sha256': 'f'*64,
        'reference_envelope_sha256': 'f'*64, 'reference_supervisor_sha256': 'f'*64,
        'parent_root_sha256': 'f'*64, 'parent_closure_sha256': 'f'*64,
        'parent_whole_science_sha256': 'f'*64, 'output_name': '../escape',
        'certified': True, 'formal_training_accepted': True, 'final_eval_authorized': True,
    }
    for name, value in mutations.items():
        spec = dict(original, **{name: value})
        source['plan_sha256'] = publish(source['plan'], spec)
        with pytest.raises((ValueError, FileExistsError), match='.'):
            contract.prepare(source=source)
    for bad in ([16, 32], [17], [True]):
        source['plan_sha256'] = publish(source['plan'], dict(original, particles=bad))
        with pytest.raises(ValueError):
            contract.prepare(source=source)


def test_complete_new_science_retains_all_old_failures_and_reconciles_actual_work(completed):
    prepared, rows, directory = completed
    protected = {p: p.read_bytes() for p in prepared.bound if p.is_file()}
    assert contract.validate_rows(prepared, rows[:7], [directory]*7) == 7
    assert contract.validate_rows(prepared, rows[7:], [directory]*17, offset=7) == 17
    report = contract.whole_audit(prepared, rows, directory)
    assert report['new_run_count'] == report['parent_runs_covered'] == report['exact_parent_prefix_checks'] == 24
    assert report['reference_runs_covered'] == 48
    assert report['original_numerical'] == prepared.parent_science['reference_numerical']
    assert report['parent_numerical'] == prepared.parent_science['new_numerical']
    assert report['parent_particle_sensitivities'] == prepared.parent_science['cross_source_adjacent_particle_sensitivities']
    assert len(report['new_numerical']['sensitivities']) == 12
    assert len(report['new_particle_precision']['runs']) == 24
    assert len(report['cross_parent_particle_precision']) == len(report['cross_parent_particle_sensitivities']) == 24
    assert len(report['cross_parent_aggregate_precision']) == 12
    accounting = report['execution_accounting']
    assert accounting['new_feature_query_rows'] == accounting['inherited_feature_query_rows']
    assert accounting['ensemble_feature_rows'] == 2*accounting['new_feature_query_rows']
    assert accounting['new_raw_map_query_rows'] < accounting['new_feature_query_rows']
    assert accounting['new_rollout_wall_seconds_including_lazy_map_initialization'] == 6.
    assert report['reference_resource_charges'] == prepared.reference_resource_charges
    for key, expected in contract.UNQUALIFIED.items():
        assert report[key] == expected
    for row, old, comparison in zip(rows, prepared.parent_rows, report['cross_parent_particle_sensitivities']):
        expected = row['scores']['time_weighted_energy_score_m']-old['scores']['time_weighted_energy_score_m']
        assert comparison['second_minus_first_ES_m'] == expected
        assert comparison['settings'] == [16, 32]
    assert all(p.read_bytes() == raw for p, raw in protected.items())
    assert not contract.directory_for(prepared.source, prepared.spec).exists()


def test_missing_reordered_duplicate_failed_or_reclassified_work_is_rejected(completed):
    prepared, rows, directory = completed
    with pytest.raises(ValueError, match='every registered'):
        contract.whole_audit(prepared, rows[:-1], directory)
    for fault in ('reordered', 'duplicate', 'failed', 'old_schema', 'legacy_counter', 'binding', 'model', 'block', 'driver', 'score', 'precision'):
        bad = deepcopy(rows)
        if fault == 'reordered': bad[0], bad[1] = bad[1], bad[0]
        if fault == 'duplicate': bad[1] = deepcopy(bad[0])
        if fault == 'failed': bad[0]['status'] = 'failure'
        if fault == 'old_schema': bad[0]['schema_version'] = 'pirc17-development-rollout-v1'
        if fault == 'legacy_counter': bad[0]['feature_query_rows'] = 1
        if fault == 'binding': bad[0]['prefix_binding']['row_sha256'] = 'f'*64
        if fault == 'model': bad[0]['model_identity_sha256'] = 'f'*64
        if fault == 'block': bad[0]['independent_block_id'] = 'changed'
        if fault == 'driver': bad[0]['brownian_identity']['seed'] += 1
        if fault == 'score': bad[0]['scores']['time_weighted_energy_score_m'] += 1
        if fault == 'precision': bad[0]['particle_precision']['time_weighted_standard_error_m'] += 1
        with pytest.raises(ValueError):
            contract.validate_rows(prepared, bad, [directory]*len(bad))


def test_every_execution_accounting_field_is_checked_without_combining_costs(completed):
    prepared, rows, directory = completed
    for field in contract.ACCOUNT_FIELDS:
        bad = deepcopy(rows[:1])
        if 'seconds' in field:
            bad[0]['suffix_accounting'][field] = -1.
        else:
            bad[0]['suffix_accounting'][field] += 1
        # An arbitrary nonnegative missing suffix count is an observation, but
        # it cannot change without the separate whole-ensemble count changing.
        with pytest.raises(ValueError, match='accounting'):
            contract.validate_rows(prepared, bad, [directory])
    for field, value in [('new_feature_query_rows', True), ('new_invalid_feature_rows', -1),
                         ('new_rollout_wall_seconds_including_lazy_map_initialization', float('inf'))]:
        bad = deepcopy(rows[:1])
        bad[0]['suffix_accounting'][field] = value
        with pytest.raises(ValueError, match='accounting'):
            contract.validate_rows(prepared, bad, [directory])


def test_rehashed_particle_prefix_mutation_cannot_be_inherited(completed):
    prepared, rows, directory = completed
    bad = deepcopy(rows[:1])
    arrays = list(contract.load_particle_evidence(bad[0], directory))
    arrays[0] = arrays[0].copy()
    arrays[0][0, 0, 0] += 1.
    path = directory/bad[0]['particle_artifact']['path']
    np.savez(path, positions_m=arrays[0], target_positions_m=arrays[1], elapsed_seconds=arrays[2])
    bad[0]['particle_artifact']['sha256'] = lineage.native._hash(path)
    with pytest.raises(ValueError, match='saved parent particle'):
        contract.validate_rows(prepared, bad, [directory])


def test_parent_storage_closure_does_not_replace_its_independent_science(ready, tmp_path):
    reservation = reserve(ready)
    result = run(reservation)
    path = reservation['attempt']/'work/whole-science.json'
    whole = json.loads(path.read_text())
    whole['reference_numerical']['out_of_tolerance'] += 1
    result['whole_science_sha256'] = publish(path, whole)
    publish(reservation['attempt']/'work/core-result.json', result)
    assert seal(reservation, worker_returncode=0, interrupted=False)['status'] == 'computed_closed'
    source = make_source(reservation, tmp_path, science_sha=result['whole_science_sha256'])
    with pytest.raises(ValueError, match='does not independently reproduce'):
        contract.prepare(source=source)


@pytest.mark.parametrize('fault', ['changed', 'missing_receipts', 'missing_assets'])
def test_rehashed_parent_cannot_change_or_omit_map_identity(ready, tmp_path, fault):
    reservation = reserve(ready)
    result = run(reservation)
    if fault == 'changed':
        result['new_attempt_maps']['software_only'] = 'changed'
    else:
        result['new_attempt_maps'].pop('receipt_sha256' if fault == 'missing_receipts' else 'verified_assets')
    publish(reservation['attempt']/'work/core-result.json', result)
    assert seal(reservation, worker_returncode=0, interrupted=False)['status'] == 'computed_closed'
    source = make_source(reservation, tmp_path, science_sha=result['whole_science_sha256'])
    with pytest.raises(ValueError, match='unchanged parent map identity'):
        contract.prepare(source=source)


@pytest.mark.parametrize('field', ['receipt_sha256', 'verified_assets'])
def test_late_map_evidence_change_invalidates_admission(admitted, field):
    maps = admitted.original_reference[-1]['maps']
    saved_path = next(iter(maps[field]))
    path = (Path(saved_path) if field == 'receipt_sha256'
            else Path(admitted.parent_root['data_locations']['data_root'])/saved_path)
    assert path in admitted.bound
    path.write_bytes(path.read_bytes()+b'late software map mutation')
    with pytest.raises(ValueError, match='maps or reference evidence changed'):
        admitted.recheck()


def test_late_child_array_change_and_bound_source_change_fail_closed(completed, monkeypatch):
    prepared, rows, directory = completed
    original = contract.replay
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        path = directory/rows[-1]['particle_artifact']['path']
        path.write_bytes(path.read_bytes()+b'late fixture mutation')
        return result
    monkeypatch.setattr(contract, 'replay', mutate)
    with pytest.raises(ValueError, match='hash mismatch'):
        contract.whole_audit(prepared, rows, directory)
    monkeypatch.setattr(contract, 'source_hashes', lambda: {**prepared.sources, 'particle_suffix.py': 'f'*64})
    with pytest.raises(ValueError, match='source'):
        prepared.recheck()


def test_resource_guard_is_not_swallowed_and_no_forecast_is_made(closed_parent, ready, tmp_path):
    source = make_source(closed_parent[0], tmp_path, science_sha=closed_parent[1]['whole_science_sha256'])
    before = len(ready[-1]['calls'])
    def refuse():
        raise TimeoutError('software cumulative admission budget exhausted')
    with pytest.raises(TimeoutError, match='budget exhausted'):
        contract.prepare(source=source, guard=refuse)
    assert len(ready[-1]['calls']) == before
