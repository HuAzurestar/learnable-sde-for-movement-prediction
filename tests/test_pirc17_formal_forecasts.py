"""Real kernels/fits on synthetic inputs; NOT formal execution authorization.

Full N512, h5 and 30min clocks are retained. Real frozen feature transforms
consume a clearly synthetic static provider, not private raw maps or targets.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import io
import zipfile

import numpy as np
import pytest

from experiments.pirc17 import formal_forecasts as module
from experiments.pirc17 import formal_forecast_records as records
from experiments.pirc17.configurations import configuration_encoder
from experiments.pirc17.features import CanonicalEncoder
from experiments.pirc17.formal_maps import RegisteredMaps
from experiments.pirc17.formal_matrix import build_matrix, load_protocol
from experiments.pirc17.formal_origins import build_origin_cases
from experiments.pirc17.formal_training import FitConsumers
from experiments.pirc17.inference import SEEDS
from experiments.pirc17.method_mechanisms import EXACT_REFERENCE, integration_diagnostic
from experiments.pirc17.protocol_core import canonical, digest, envelope, file_hash, read_json, unpack
from tests import test_pirc21_adapter as adapter_fixture
from tests.test_pirc17_formal_origins import final_fixture
from tests.test_pirc17_formal_training import fit_fixture


class SyntheticMap:
    def __init__(self, row): self.row, self.rows, self.closed = row, 0, False
    @property
    def identity(self): return {'query_version': 'SOFTWARE STATIC MAP', 'receipt_sha256': {}, 'verified_assets': {}}
    def __call__(self, xy):
        assert not self.closed and xy.shape[1] == 2 and np.isfinite(xy).all()
        self.rows += len(xy)
        return [self.row for _ in xy]
    def close(self): self.closed = True


def fixture_maps(scope, row):
    query = SyntheticMap(row)
    catalog = envelope({**{k: scope[k] for k in ('protocol_sha256', 'execution_sha256', 'approval_sha256', 'population_sha256')},
        'static_identity': {k: v for k, v in query.identity.items() if k != 'verified_assets'}, 'asset_sha256': {}})
    return RegisteredMaps(query, catalog, root=None, policy={}, parents=(), execution={})


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp('formal-forecast-software')
    args = fit_fixture()
    # Real full public transform specification in a wholly synthetic snapshot.
    spec = read_json(Path(__file__).resolve().parents[2]/'DSDE-SDE/registry/pirc21_feature_spec.contract.json')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(adapter_fixture, '_spec', lambda: deepcopy(spec))
        snapshot = adapter_fixture._write_snapshot(root)
    manifest = read_json(snapshot/'manifest.json'); manifest['history_window_points'] = 3
    (snapshot/'manifest.json').write_bytes(canonical(manifest))
    full = CanonicalEncoder.frozen_pirc22(snapshot)
    encoders = {name: configuration_encoder(full, name) for name in args['inputs'].encoders}
    public = load_protocol()
    p = deepcopy(unpack(args['protocol']))
    p['forecast_contract']['forecast'] = deepcopy(unpack(public)['forecast_contract']['forecast'])
    p['resource_contract'] = deepcopy(unpack(public)['resource_contract'])
    p['components']['finite_delivery'] = deepcopy(unpack(public)['components']['finite_delivery'])
    p['components']['decision_policy'] = deepcopy(unpack(public)['components']['decision_policy'])
    protocol = envelope(p)
    matrix = deepcopy(unpack(build_matrix(public))); matrix['protocol_sha256'] = protocol['sha256']
    matrix = envelope(matrix)
    execution = envelope({'fixture': 'NO ACTUAL APPROVAL/EXECUTION SEAL', 'protocol_sha256': protocol['sha256'], 'matrix_sha256': matrix['sha256']})
    prefixes, population = final_fixture(n=1)
    identity = deepcopy(unpack(args['inputs'].identity))
    identity.update(protocol_sha256=protocol['sha256'], execution_sha256=execution['sha256'], population_sha256=population['sha256'],
                    configuration_columns={k: list(v.columns) for k, v in encoders.items()})
    inputs = replace(args['inputs'], encoders=encoders, identity=envelope(identity))
    fits = FitConsumers(protocol=protocol, execution=execution, matrix=matrix, inputs=inputs)
    for work in fits.work.values(): fits.execute(work, output_directory=root/'fits'/work['work_id'])
    cases = build_origin_cases(prefixes, population, prior=inputs.prior)
    row = {column['name']: 1. for factor in spec['factors'] for column in factor['value_columns']}
    row.update({factor['status_column']: 'valid' for factor in spec['factors']}); row['worldcover_class'] = 10.
    return dict(fits=fits, cases=cases, population=population, row=row, root=root, prefixes=prefixes)


def consumer(prepared, **changes):
    fits = changes.pop('fits', prepared['fits'])
    maps = fixture_maps(unpack(fits.inputs.identity), prepared['row'])
    return module.ForecastConsumers(fits=fits, cases=changes.pop('cases', prepared['cases']),
        population=prepared['population'], maps=maps, **changes)


def select(owner, subject='arm-01/full', *, kind='scientific_forecast', mode='causal_prefix', rank=0, repeat=None):
    return next(w for w in owner.work.values() if w['subject'] == subject and w['kind'] == kind
        and w['origin_mode'] == mode and w['origin_rank'] == rank and w['repetition'] == repeat
        and (w['seed'] is None or w['seed'] == SEEDS[0]))


def restore(owner, work, output):
    group = owner.cases[work['origin_mode']]
    return records.restore_forecast(read_json(output['artifact_path']), directory=Path(output['artifact_path']).parent,
        work=work, protocol=owner.protocol, execution=owner.execution, matrix=owner.matrix,
        input_identity=owner.input_identity, case=group[work['origin_rank']] if work['origin_rank'] < len(group) else None,
        fit_receipt=owner.receipts.get(work['fit_identity']), model=owner.models.get(work['fit_identity']), map_catalog=owner.maps.catalog)


@pytest.fixture(scope='module')
def completed(prepared):
    owner = consumer(prepared)
    outputs = {}
    selected = [w for w in owner.work.values() if w['origin_mode'] == 'causal_prefix' and w['origin_rank'] == 0
        and w['seed'] in (None, SEEDS[0]) and w['kind'] in ('scientific_forecast', 'same_grid_reference', 'inertial_path')]
    for work in selected:
        output = owner.execute(work, output_directory=prepared['root']/'predictions'/work['work_id'])
        assert output['status'] == 'success', read_json(output['artifact_path'])
        outputs[work['subject']] = (work, output)
    return owner, outputs


def test_all_28_methods_ten_terrain_reference_and_inertial_use_actual_full_clocks(completed):
    owner, outputs = completed
    assert len(outputs) == 40
    for subject, (work, output) in outputs.items():
        score, native = restore(owner, work, output)
        assert score.positions_m.shape == (1 if work['kind'] == 'inertial_path' else 512, 4, 2)
        np.testing.assert_array_equal(score.elapsed_seconds, [60., 300., 900., 1800.])
        assert not score.positions_m.flags.writeable
        record = unpack(read_json(output['artifact_path']))
        assert record['scientific_claim_authorized'] is False and record['numerically_qualified'] is False
        with np.load(Path(output['artifact_path']).parent/'forecast.npz', allow_pickle=False) as saved:
            assert not any('target' in key or 'truth' in key for key in saved.files)
        if native is not None:
            assert native.diagnostics['max_step_seconds'] == 5.
            assert native.diagnostics['integration_steps'] <= native.diagnostics['registered_max_steps']
        if work['matrix'] == 'terrain':
            assert record['maps']['query_rows_this_forecast'] == (0 if subject == 'base' else score.feature_query_rows)


def test_paired_aliases_and_saved_full_horizon_integration_remain_recomputable(completed):
    owner, outputs = completed
    full = restore(owner, *outputs['arm-01/full'])[0]
    same = restore(owner, *outputs['arm-06/dt60'])[0]
    np.testing.assert_array_equal(full.positions_m, same.positions_m)
    reference = restore(owner, *outputs[EXACT_REFERENCE])[1]
    for slot in ('arm-18/full', 'arm-19/em', 'arm-19/euler'):
        result = restore(owner, *outputs[slot])[1]
        gate = integration_diagnostic(slot, result, reference,
            approximate_context_sha256=digest('same synthetic scope'), reference_context_sha256=digest('same synthetic scope'))
        assert gate['status'] == 'computed' and gate['value'] >= 0


@pytest.mark.parametrize('mode', ['known_velocity', 'point_only'])
def test_secondary_modes_and_deterministic_inertial_use_only_visible_case(prepared, tmp_path, mode):
    owner = consumer(prepared)
    for subject, kind in [('arm-01/full', 'scientific_forecast'), ('base', 'scientific_forecast'), ('all', 'inertial_path')]:
        work = select(owner, subject, kind=kind, mode=mode)
        result = owner.execute(work, output_directory=tmp_path/work['work_id'])
        score, _ = restore(owner, work, result)
        assert score is not None
    case = owner.cases[mode][0]
    assert records.origin_stream_id(case) != records.origin_stream_id(owner.cases['causal_prefix'][0])
    changed = replace(case, window_sha256=digest('different hidden-file provenance'))
    assert records.origin_stream_id(case) == records.origin_stream_id(changed)
    assert len(case.method_origin.history_times_seconds) == 1


def test_missing_ranks_and_fit_dependency_preserve_dispositions_without_prediction(prepared, tmp_path, monkeypatch):
    owner = consumer(prepared)
    monkeypatch.setattr(module, 'forecast_method', lambda *a, **kw: pytest.fail('must not forecast unavailable work'))
    missing_rank = select(owner, rank=1)
    output = owner.execute(missing_rank, output_directory=tmp_path/'rank')
    assert output['status'] == 'NOT_ADMITTED' and output['generated_forecasts_attempted'] == 0
    assert restore(owner, missing_rank, output) == (None, None)
    work = select(owner)
    owner.models.pop(work['fit_identity']); owner.receipts.pop(work['fit_identity'])
    output = owner.execute(work, output_directory=tmp_path/'fit')
    assert output['status'] == 'DEPENDENCY_UNAVAILABLE' and output['generated_forecasts_attempted'] == 0
    assert restore(owner, work, output) == (None, None)
    with pytest.raises(ValueError, match='already attempted'): owner.execute(work, output_directory=tmp_path/'retry')


def test_numerical_failure_record_is_not_scientific_success_or_retry(prepared, tmp_path, monkeypatch):
    owner = consumer(prepared); work = select(owner)
    def fail(*a, **kw): raise ValueError('synthetic nonfinite path')
    monkeypatch.setattr(module, 'forecast_method', fail)
    result = owner.execute(work, output_directory=tmp_path)
    assert result['status'] == 'failed' and result['generated_forecasts_attempted'] == 1
    assert restore(owner, work, result) == (None, None)
    assert not list(tmp_path.glob('*.npz'))
    with pytest.raises(ValueError, match='already attempted'): owner.execute(work, output_directory=tmp_path)


@pytest.mark.parametrize('kind', ['forecast_replay', 'runtime_cold', 'runtime_warmup', 'runtime_warm'])
def test_registered_replay_and_runtime_work_physically_forecast(prepared, tmp_path, monkeypatch, kind):
    owner = consumer(prepared)
    fresh = []
    def cold():
        maps = fixture_maps(unpack(owner.input_identity), prepared['row']); fresh.append(maps)
        return maps
    monkeypatch.setattr(owner.maps, 'fresh_provider', cold)
    for subject in ('arm-01/full', 'base'):
        if kind == 'runtime_warm':
            warmup = select(owner, subject, kind='runtime_warmup', repeat=0)
            owner.execute(warmup, output_directory=tmp_path/warmup['work_id'])
        work = select(owner, subject, kind=kind, repeat=None if kind == 'forecast_replay' else 0)
        output = owner.execute(work, output_directory=tmp_path/work['work_id'])
        assert output['status'] == 'success' and output['generated_forecasts_attempted'] == 1
        restore(owner, work, output)
    owner.close()
    assert len(fresh) == int(kind in {'runtime_cold', 'runtime_warmup', 'runtime_warm'})
    assert all(m.closed for m in fresh)


@pytest.mark.parametrize('field,value', [('protocol_sha256', '0'*64), ('parameter_identity', '0'*64),
    ('case_sha256', '0'*64), ('scientific_claim_authorized', True), ('numerically_qualified', True),
    ('generated_forecasts_attempted', 0), ('invalid_feature_rows', -1), ('prediction_seconds', -1),
    ('forecast_version', 'fake'), ('status', 'NOT_ADMITTED')])
def test_rehashed_result_changes_are_rejected(completed, tmp_path, field, value):
    owner, outputs = completed; work, output = outputs['arm-01/full']
    record = deepcopy(unpack(read_json(output['artifact_path']))); record[field] = value
    path = tmp_path/'forged.json'; path.write_bytes(canonical(envelope(record)))
    # Array directory remains bound separately from the forged JSON path.
    with pytest.raises(ValueError):
        records.restore_forecast(envelope(record), directory=Path(output['artifact_path']).parent, work=work,
            protocol=owner.protocol, execution=owner.execution, matrix=owner.matrix, input_identity=owner.input_identity,
            case=owner.cases['causal_prefix'][0], fit_receipt=owner.receipts[work['fit_identity']], model=owner.models[work['fit_identity']], map_catalog=owner.maps.catalog)


def test_saved_arrays_reject_changed_bytes_and_restore_without_generation(completed, tmp_path, monkeypatch):
    owner, outputs = completed; work, output = outputs['arm-01/full']
    monkeypatch.setattr(module, 'forecast_method', lambda *a, **kw: pytest.fail('restore generated a forecast'))
    score, native = restore(owner, work, output)
    assert score is not None and native is not None
    (tmp_path/'forecast.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='array bytes'):
        records.restore_forecast(read_json(output['artifact_path']), directory=tmp_path, work=work,
            protocol=owner.protocol, execution=owner.execution, matrix=owner.matrix, input_identity=owner.input_identity,
            case=owner.cases['causal_prefix'][0], fit_receipt=owner.receipts[work['fit_identity']], model=owner.models[work['fit_identity']], map_catalog=owner.maps.catalog)


@pytest.mark.parametrize('change', ['stream', 'clock', 'step-count', 'case-frame', 'covariance', 'header', 'extra-array'])
def test_rehashed_diagnostics_and_array_headers_cannot_forge_valid_domain(completed, tmp_path, monkeypatch, change):
    owner, outputs = completed; work, output = outputs['arm-01/full']
    record = deepcopy(unpack(read_json(output['artifact_path'])))
    source = Path(output['artifact_path']).parent/'forecast.npz'
    with np.load(source, allow_pickle=False) as saved: arrays = {k: saved[k] for k in saved.files}
    if change == 'stream': record['diagnostics']['random_stream_sha256'] = '0'*64
    elif change == 'clock': record['diagnostics']['history_step_seconds'] = 5.
    elif change == 'step-count': record['diagnostics']['integration_steps'] = 0
    elif change == 'case-frame': arrays['positions_m'] += 100.
    elif change == 'covariance': arrays['conditional_covariances_m2'][:, :, 0, 0] = -1e10
    elif change == 'extra-array': arrays['target_positions_m'] = np.zeros((4, 2))
    path = tmp_path/'forecast.npz'
    if change == 'header':
        with zipfile.ZipFile(source) as original, zipfile.ZipFile(path, 'w') as forged:
            for entry in original.infolist():
                value = original.read(entry.filename)
                if entry.filename == 'positions_m.npy':
                    header = io.BytesIO()
                    np.lib.format.write_array_header_1_0(header, {'descr': '<f8', 'fortran_order': False, 'shape': (10**9, 4, 2)})
                    value = header.getvalue()
                forged.writestr(entry.filename, value)
        monkeypatch.setattr(np, 'load', lambda *a, **kw: pytest.fail('unbounded header reached numpy allocation'))
    else: np.savez(path, **arrays)
    record['arrays'].update(sha256=file_hash(path), bytes=path.stat().st_size)
    with pytest.raises(ValueError):
        records.restore_forecast(envelope(record), directory=tmp_path, work=work, protocol=owner.protocol,
            execution=owner.execution, matrix=owner.matrix, input_identity=owner.input_identity,
            case=owner.cases['causal_prefix'][0], fit_receipt=owner.receipts[work['fit_identity']],
            model=owner.models[work['fit_identity']], map_catalog=owner.maps.catalog)


def test_rehashed_map_use_cannot_claim_unadmitted_asset(completed):
    owner, outputs = completed; work, output = outputs['all-terrain']
    record = deepcopy(unpack(read_json(output['artifact_path'])))
    record['maps']['observation']['identity']['verified_assets']['invented'] = '0'*64
    with pytest.raises(ValueError, match='raw-map identity'):
        records.restore_forecast(envelope(record), directory=Path(output['artifact_path']).parent, work=work,
            protocol=owner.protocol, execution=owner.execution, matrix=owner.matrix, input_identity=owner.input_identity,
            case=owner.cases['causal_prefix'][0], fit_receipt=owner.receipts[work['fit_identity']],
            model=owner.models[work['fit_identity']], map_catalog=owner.maps.catalog)


def test_wrong_case_population_and_unregistered_work_are_rejected(prepared, tmp_path):
    cases = deepcopy(prepared['cases'])
    cases['point_only'] = (replace(cases['point_only'][0], population_sha256='0'*64),)
    with pytest.raises(ValueError, match='population'): consumer(prepared, cases=cases)
    owner = consumer(prepared)
    work = dict(select(owner), seed=1)
    with pytest.raises(ValueError, match='exact registered'): owner.execute(work, output_directory=tmp_path)
