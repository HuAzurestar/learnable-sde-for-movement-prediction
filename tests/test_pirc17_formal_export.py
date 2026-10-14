"""Public aggregates from actual native-closed synthetic scientific artifacts.

The source fixture explicitly replaces human approval for positive transport
tests. No raw final data, real authorization or formal timing claim is made.
"""
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pytest

from experiments.pirc17 import formal_export as module
from experiments.pirc17 import formal_forecasts, formal_training
from experiments.pirc17.formal_closed import ClosedOutputs
from experiments.pirc17.protocol_core import canonical, digest, envelope, publish, read_json, unpack
from tests.test_pirc17_formal_forecasts import prepared
from tests.test_pirc17_formal_results import verified, metric_access
from tests.test_pirc17_formal_reanalysis import deliver
from tests.test_pirc17_formal_closed import changed


@pytest.fixture(scope='module')
def exported(verified):
    source, results = verified, verified['results']
    ledger = results.ledger
    args = dict(protocol=results.protocol, execution=results.execution, matrix=results.matrix,
        scope=unpack(results.context['input_identity']), access_journal=results.authority['journal_directory'])
    work = next(w for w in results.works.values() if w['kind'] == 'aggregate_export')
    directory = source['root']/'export-source'
    fixtures = source['root']/'export-delivery'; fixtures.mkdir()
    holder = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(results, '_check_authority', lambda phase: None)  # Explicit synthetic authorization only.
        def authorize(w):
            authority = results.authorize(w)
            closed = ClosedOutputs(ledger.directory, contract=ledger._state.contract, tip=ledger.tip)
            exporter = module.ExportConsumers(closed=closed, **args)
            access = metric_access(results, 104)
            output = exporter._execute(access, work, directory)
            holder.update(exporter=exporter, access=access, original=read_json(output['artifact_path']))
            (fixtures/(work['work_id']+'.json')).write_bytes(canonical(dict(source_directory=str(directory), manifest=output)))
            return authority
        deliver(ledger, results.matrix, fixtures,
            {work['work_id']: dict(artifact_path=str(directory/'placeholder.json'), artifact_sha256='0'*64)},
            authorize_work=authorize, validate_result=results.validate)
    final = ClosedOutputs(ledger.directory, contract=ledger._state.contract, tip=ledger.tip)
    item = final.read(work)
    return dict(source=source, work=work, final=final, item=item, args=args, **holder)


def test_real_native_export_replays_audited_sources_without_private_identifiers(exported, monkeypatch):
    def forbidden(*a, **kw): pytest.fail('public export must not fit or predict')
    monkeypatch.setattr(formal_training.FitConsumers, 'execute', forbidden)
    monkeypatch.setattr(formal_forecasts.ForecastConsumers, 'execute', forbidden)
    value = unpack(read_json(exported['item']['manifest']['artifact_path']))
    assert value == unpack(exported['original'])
    assert not value['audit_summary']['all_predecessors_verified']  # Tiny fixture, not full science.
    assert len(value['audit_summary']['work_inventory']) == 11659
    assert len(value['score_work_dispositions']) == 58 and len(value['method_ledger']) == 36
    assert sum(r['disposition'] == 'EXCLUDED' for r in value['method_ledger']) == 8
    assert len(value['inputs']['score_rows']) == 196
    assert len(value['recomputed']['metric_tables']) == 120  # 38 science +reference +inertial, three modes.
    inertial = next(r for r in value['recomputed']['metric_tables'] if r['origin_mode'] == 'causal_prefix' and r['matrix'] == 'inertial')
    assert inertial['status'] == 'computed' and inertial['expected_forecasts'] == 1
    assert [r['level'] for r in inertial['means']['by_time'][0]['coverage']] == [.5, .8, .9, .95]
    assert all(r['verdict'] == 'unavailable' for r in value['recomputed']['factor_conclusions'].values())
    assert module.replay_public(value['inputs']) == value['recomputed']
    assert not any(value[k] for k in ('raw_trajectories_exported', 'original_identifiers_exported',
        'scientific_claim_authorized', 'numerically_qualified', 'human_accepted'))
    assert not value['budget_snapshot']['includes_export_completion']
    assert value['new_forecasts'] == value['new_fits'] == 0
    raw = canonical(value)
    assert b'center_m' not in raw and b'positions_m' not in raw and b'causal_prefixes' not in raw
    assert b'loaded_threadpools' not in raw and b'sample_id' not in raw
    assert str(exported['source']['root']).encode() not in raw
    for prefix in exported['source']['results'].context['positions'].prefixes:
        assert canonical(prefix.independent_block_id) not in raw
        assert canonical(prefix.sample_id) not in raw
    assert len(raw) < module.MAX_BYTES


@pytest.mark.parametrize('fault', ['claim', 'verdict', 'score', 'denominator', 'raw_coordinate', 'cost', 'audit'])
def test_export_domain_verifier_rejects_rehashed_changes(exported, fault):
    payload = deepcopy(unpack(exported['original']))
    if fault == 'claim': payload['scientific_claim_authorized'] = True
    elif fault == 'verdict': payload['recomputed']['factor_conclusions']['road']['verdict'] = 'retain'
    elif fault == 'score': payload['inputs']['score_rows'][0]['score_m'] = -999.
    elif fault == 'denominator': payload['score_work_dispositions'].pop()
    elif fault == 'raw_coordinate': payload['inputs']['score_rows'][0]['latitude'] = 1.
    elif fault == 'cost': payload['budget_snapshot']['includes_export_completion'] = True
    else: payload['audit_summary']['all_predecessors_verified'] = True
    with pytest.raises(ValueError, match='differs from actual audited sources'):
        exported['exporter'].verify(envelope(payload), work=exported['work'],
            access_started_sha256=exported['access'].access_started_sha256)


def test_public_replay_recomputes_predicates_and_rejects_false_pass_inputs(exported):
    value = deepcopy(unpack(exported['original'])['inputs'])
    gates = value['mechanisms']['causal_prefix']
    key = next(k for k, g in gates.items() if g['status'] == 'computed')
    gates[key]['passed'] = True
    with pytest.raises(ValueError): module.replay_public(value)
    gates[key].pop('passed')
    gates[key]['threshold'] += 1
    with pytest.raises(ValueError, match='definition'): module.replay_public(value)


def test_no_second_export_or_unclosed_audit_substitution(exported, tmp_path):
    with pytest.raises(ValueError, match='already attempted'):
        exported['exporter']._execute(exported['access'], exported['work'], tmp_path)
    closed = exported['exporter'].closed
    audit_work = exported['source']['audit_work']
    audit_item = closed.read(audit_work)
    with changed(Path(audit_item['manifest']['artifact_path']), b'changed'), pytest.raises(ValueError):
        exported['exporter'].compute(exported['work'])


def test_missing_closed_audit_is_not_an_empty_success(exported):
    closed = exported['exporter'].closed
    audit_item = closed.entries[exported['source']['audit_work']['work_id']]
    before = dict(root_sha256=closed.root_sha256, event_count=audit_item['reservation_event_index'],
        last_event_sha256=unpack(read_json(closed.directory/'events'/f"{audit_item['reservation_event_index']:06d}.json"))['previous_sha256'])
    view = ClosedOutputs(closed.directory, contract=closed.contract, tip=before)
    exporter = module.ExportConsumers(closed=view, **exported['args'])
    with pytest.raises(ValueError, match='required closed export source unavailable'):
        exporter.compute(exported['work'])


def test_public_entry_keeps_real_approval_barrier(exported, tmp_path, monkeypatch):
    def forbidden(*a, **kw): pytest.fail('unapproved public export reached scientific outputs')
    exporter = exported['exporter']
    monkeypatch.setattr(exporter, '_execute', forbidden)
    results = exported['source']['results']
    with pytest.raises((ValueError, FileNotFoundError, TypeError)):
        module.export_formal_work(exporter=exporter, work=exported['work'], output_directory=tmp_path,
            protocol=results.protocol, execution=results.execution, **results.authority, **results.population_paths)


@pytest.mark.parametrize('fault', [None, 'error', 'score', 'absolute_variance', 'ratio', 'bootstrap', 'missing_row', 'zero_denominator'])
def test_public_numeric_witnesses_replay_actual_arithmetic(exported, fault):
    inputs = unpack(exported['original'])['inputs']
    population = inputs['populations']['causal_prefix']
    gates = deepcopy(inputs['mechanisms']['causal_prefix'])
    details = dict(per_origin_diagnostics={}, variance=None)
    for slot in module.NUMERICAL_SLOTS | module.SCORE_SLOTS:
        rows = []
        for block, origins in population.items():
            for origin in origins:
                for seed in module.SEEDS:
                    numeric = slot in module.NUMERICAL_SLOTS
                    measurement = (dict(per_time_gaussian_coupling_rms_m=[.2, .3, .4, .5]) if numeric else
                        dict(diagnostic_draws_per_time=256, quadrature_order=12,
                            per_time=[dict(moment_gaussian_quadrature_es_m=10., gaussian_mc_es_m=10.1,
                                relative_error=abs(10.-10.1)/10.) for _ in range(4)]))
                    rows.append(dict(origin_id=origin, independent_block_id=block, seed=seed,
                        gate_sha256='e'*64, measurements=measurement))
        details['per_origin_diagnostics'][slot] = rows
        gates[slot].update(status='computed', value=.5 if slot in module.NUMERICAL_SLOTS else abs(10.-10.1)/10.,
            expected_count=len(rows), available_count=len(rows))
    zero = fault == 'zero_denominator'
    differences = [[float(i), 0. if zero else i/2] for i in range(5)]
    variances = np.var(differences, axis=0, ddof=1).tolist()
    blocks = [dict(block_id=b, origin_count=1, paired_seed_differences_m=differences,
        mc_minus_fp_variance_m2=variances[0], crn_minus_fp_variance_m2=variances[1]) for b in sorted(population)]
    details['variance'] = dict(partition='final_eval', blocks=blocks, independent_block_count=len(blocks),
        seed_count=5, variance_ddof=1, numerator_m2=variances[0], denominator_m2=variances[1],
        bootstrap_iterations=2000, bootstrap_seed=20260926, descriptive_block_bootstrap_interval=None if zero else [4., 4.],
        undefined_bootstrap_replicates=2000 if zero else 0)
    for slot in module.VARIANCE_SLOTS:
        gates[slot].update(status='unavailable' if zero else 'computed', value=None if zero else 4., available_count=0 if zero else 1)
    if fault == 'error': details['per_origin_diagnostics']['arm-18/full'][0]['measurements']['per_time_gaussian_coupling_rms_m'][-1] += 1.
    elif fault == 'score': details['per_origin_diagnostics']['arm-10/d2_mc'][0]['measurements']['per_time'][0]['relative_error'] += 1.
    elif fault == 'absolute_variance': details['variance']['blocks'][0]['mc_minus_fp_variance_m2'] += 1.
    elif fault == 'ratio': gates['arm-21/mc']['value'] += 1.
    elif fault == 'bootstrap': details['variance']['descriptive_block_bootstrap_interval'] = [0., 0.]
    elif fault == 'missing_row': details['per_origin_diagnostics']['arm-18/full'].pop()
    def check(): module._verify_numeric_witnesses(module._replay_gates(gates, len(population)), details, population)
    if fault in (None, 'zero_denominator'): check()
    else:
        with pytest.raises(ValueError): check()


def test_actual_offline_cli_replays_without_private_sources(exported, tmp_path):
    record = exported['original']
    # Only the public file is copied into this new directory. No private source,
    # trajectory, ledger, model, approval or native session path is supplied.
    source = tmp_path/'aggregate.json'; source.write_bytes(canonical(record))
    completed = subprocess.run([sys.executable, '-m', 'experiments.pirc17.formal_export', '--input', str(source),
        '--sha256', record['sha256'], '--output-directory', str(tmp_path/'replayed')],
        capture_output=True, text=True, check=True, timeout=45)
    response = json.loads(completed.stdout)
    result = unpack(read_json(response['path']))
    assert response['status'] == 'verified' and result['source_sha256'] == record['sha256']
    assert result['result'] == unpack(record)['recomputed'] and not result['raw_data_read']
    with pytest.raises(ValueError, match='sealed content identity'):
        module.main(['--input', str(source), '--sha256', '0'*64, '--output-directory', str(tmp_path/'wrong')])
    assert not (tmp_path/'wrong').exists()


@pytest.mark.parametrize('missing', [False, True])
def test_complete_public_statistical_arithmetic_retains_missing_pairs(exported, missing):
    # Fabricated scalar test data, explicitly NOT empirical performance. Exercise
    # the complete three-block bootstrap path absent from the sparse fixture.
    inputs = deepcopy(unpack(exported['original'])['inputs'])
    source = next(r for r in inputs['score_rows'] if r['status'] == 'success' and r['scientific'])
    population = {f'block-{i:03d}': [f'origin-{i:03d}'] for i in range(3)}
    inputs['populations']['causal_prefix'] = population
    inputs['score_rows'] = [r for r in inputs['score_rows'] if r['origin_mode'] != 'causal_prefix']
    for slot in module.NUMERICAL_SLOTS | module.SCORE_SLOTS:
        inputs['mechanisms']['causal_prefix'][slot].update(expected_count=15, available_count=0, status='unavailable', value=None)
        inputs['mechanism_details']['causal_prefix']['per_origin_diagnostics'][slot] = [dict(origin_id=o,
            independent_block_id=b, seed=s, gate_sha256=None, measurements=None)
            for b, origins in population.items() for o in origins for s in module.SEEDS]
    configs = sorted(unpack(inputs['terrain_ownership'])['configurations'])
    for bi, (block, origins) in enumerate(population.items()):
        for ci, config in enumerate(configs):
            for si, seed in enumerate(module.SEEDS):
                if missing and bi == ci == si == 0: continue
                score = 100.+bi*10+si+ci*(1.+bi*.1)
                row = deepcopy(source)
                row.update(matrix='terrain', configuration=config, independent_block_id=block, origin_id=origins[0],
                    origin_mode='causal_prefix', origin_rank=bi, seed=seed, score_m=score,
                    forecast_work_id=digest([block, config, seed]))
                for key in module.SCORE_SCALARS: row['scores'][key] = score
                inputs['score_rows'].append(row)
    output = module.replay_public(inputs)
    primary = output['modes']['causal_prefix']['families']['weighted-es-primary']
    if missing:
        assert primary['inference'] is None and primary['failed_rows']
        table = next(r for r in output['metric_tables'] if r['origin_mode'] == 'causal_prefix' and r['configuration'] == configs[0])
        assert table['means'] is None and table['missing_score_rows'] == 1
    else:
        assert primary['inference'] is not None and primary['independent_block_count'] == 3
        assert primary['descriptive']['seed_ids'] == list(module.SEEDS)
        tables = [r for r in output['metric_tables'] if r['origin_mode'] == 'causal_prefix' and r['matrix'] == 'terrain']
        assert len(tables) == 10 and all(r['status'] == 'computed' and r['expected_forecasts'] == 15 for r in tables)
        # Independent original-label replay: ordinal relabeling must preserve
        # actual bootstrap output, not just agree with the public copy itself.
        aliases = {b: f'private-block-{i}' for i, b in enumerate(sorted(population))}
        aliases.update({origins[0]: f'private-origin-{i}' for i, origins in enumerate(population.values())})
        family = 'weighted-es-primary'
        selected_configs = {c for pair in module.registered_contrasts(family).values() for c in pair}
        private_rows = module._rename([r for r in inputs['score_rows'] if r['origin_mode'] == 'causal_prefix'
            and r['matrix'] == 'terrain' and r['configuration'] in selected_configs], aliases)
        original = module.paired_family(private_rows, family_id=family, origin_mode='causal_prefix',
            expected_origins_by_block=module._rename(population, aliases),
            mechanisms=module._comparison_gates(family, method_gates={},
                terrain_configs=unpack(inputs['terrain_ownership'])['configurations'], mechanism_sha256='f'*64,
                ownership_sha256=inputs['terrain_ownership']['sha256']))
        assert module._rename(module._statistical_view(original), {v: k for k, v in aliases.items()}) == module._statistical_view(primary)
    assert all(r['verdict'] in {'inconclusive', 'unavailable'} for r in output['factor_conclusions'].values())


@pytest.mark.parametrize('field', ['configuration', 'origin_mode'])
def test_unknown_metric_subjects_are_not_silently_dropped(exported, field):
    inputs = deepcopy(unpack(exported['original'])['inputs'])
    inputs['score_rows'][0][field] = 'unregistered'
    with pytest.raises(ValueError, match='unregistered mode or subject'):
        module.public_tables(inputs)


def test_full_capacity_public_file_roundtrip_uses_export_bound_not_ipc_or_default_bound(exported, tmp_path):
    """11368 FABRICATED aggregate rows; no new fitted model or prediction.

    This is file/aggregate arithmetic capacity, not full native science proof.
    The genuine independently audited sparse source fixture above is separate.
    """
    from experiments.pirc17.formal_input_work import _owned_record
    from experiments.pirc17.formal_session import MAX_MESSAGE_BYTES
    assert MAX_MESSAGE_BYTES == 64*1024
    value = deepcopy(unpack(exported['original']))
    inputs = value['inputs']
    stochastic = next(r for r in inputs['score_rows'] if r['status'] == 'success' and r['scientific'])
    inertial = next(r for r in inputs['score_rows'] if r['status'] == 'success' and r['matrix'] == 'inertial')
    ranks = dict(causal_prefix=46, known_velocity=6, point_only=6)
    for mode, count in ranks.items():
        population = {f'block-{i:03d}': [f'{mode}:origin-{i:03d}'] for i in range(count)}
        inputs['populations'][mode] = population
        for slot in module.NUMERICAL_SLOTS | module.SCORE_SLOTS:
            inputs['mechanisms'][mode][slot].update(expected_count=count*5, available_count=0, status='unavailable', value=None)
            inputs['mechanism_details'][mode]['per_origin_diagnostics'][slot] = [dict(origin_id=o,
                independent_block_id=b, seed=s, gate_sha256=None, measurements=None)
                for b, origins in population.items() for o in origins for s in module.SEEDS]
        inputs['mechanism_details'][mode]['variance'] = None
        for slot in module.VARIANCE_SLOTS:
            inputs['mechanisms'][mode][slot].update(available_count=0, status='unavailable', value=None)
    rows = []
    for work in unpack(exported['source']['results'].matrix)['workloads']:
        if work['kind'] not in {'scientific_forecast', 'same_grid_reference', 'inertial_path'}: continue
        row = dict(inertial if work['kind'] == 'inertial_path' else stochastic)
        row.update(forecast_work_id=work['work_id'], matrix=work['matrix'], configuration=work['subject'],
            origin_mode=work['origin_mode'], origin_rank=work['origin_rank'], seed=work['seed'], scientific=work['scientific'],
            independent_block_id=f"block-{work['origin_rank']:03d}",
            origin_id=f"{work['origin_mode']}:origin-{work['origin_rank']:03d}")
        rows.append(row)
    assert len(rows) == 11368
    inputs['score_rows'] = rows
    value['recomputed'] = module.replay_public(inputs)
    value['synthetic_capacity_only'] = True
    path, record = publish(tmp_path/'capacity', value)
    assert 32*1024*1024 < path.stat().st_size < module.MAX_BYTES
    assert len(value['recomputed']['metric_tables']) == 120
    with pytest.raises(ValueError, match='bounded regular JSON'):
        read_json(path)  # All non-export scientific readers retain this bound.
    manifest = dict(artifact_path=str(path), artifact_sha256=record['sha256'])
    _, restored = _owned_record(manifest, path.parent, 'artifact_path', 'artifact_sha256', max_bytes=module.MAX_BYTES)
    assert restored['sha256'] == record['sha256']
    assert module.main(['--input', str(path), '--sha256', record['sha256'],
        '--output-directory', str(tmp_path/'capacity-replay')]) == 0
    assert not value['audit_summary']['all_predecessors_verified'] and not value['human_accepted']
