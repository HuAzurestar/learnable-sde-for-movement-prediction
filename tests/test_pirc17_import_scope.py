"""Real typed model/array restoration on SYNTHETIC saved inputs.

No formal source/approval is simulated as genuine. The partial evidence below
is constructed as software evidence, while production bridge/domain readers
are unmodified. Existing synthetic fits are reused, not refit in these tests.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np
import pytest

from experiments.pirc17 import formal_budget as budget, formal_import_scope as module
from experiments.pirc17 import formal_partial_costs as costs, formal_partial_science as science
from experiments.pirc17 import formal_partial_predecessor as predecessor
from experiments.pirc17.formal_fit_records import restore_registered_fit
from experiments.pirc17.formal_forecast_records import restore_forecast
from experiments.pirc17.formal_input_work import restore_input_context
from experiments.pirc17.formal_inputs import FinalPositionInputs, ScoringTargets
from experiments.pirc17.formal_reanalysis import input_context
from experiments.pirc17.formal_saved import SavedForecasts, ScoringInputs
from experiments.pirc17.protocol_core import canonical, digest, envelope, file_hash, publish, read_json, unpack
from tests.test_pirc17_formal_forecasts import prepared, consumer, select


def rebind(context, execution, approval):
    """Fixture-only transport projection; NOT a guarded production input API."""
    value = deepcopy(unpack(context))
    value['execution_sha256'] = execution['sha256']
    pop = deepcopy(unpack(value['population']))
    pop.update(execution_sha256=execution['sha256'], eligibility_sha256=digest('SYNTHETIC new eligibility seal'))
    value['population'] = envelope(pop)
    for key in ('input_identity', 'map_catalog'):
        row = deepcopy(unpack(value[key]))
        row.update(execution_sha256=execution['sha256'], approval_sha256=approval,
                   population_sha256=value['population']['sha256'], access_started_sha256=digest(['SYNTHETIC new access', key]))
        value[key] = envelope(row)
    for row in value['causal_prefixes']:
        row['population_sha256'] = value['population']['sha256']
    row = deepcopy(unpack(value['scoring_inputs']))
    row.update(execution_sha256=execution['sha256'], population_sha256=value['population']['sha256'],
               positions_access_started_sha256=digest('SYNTHETIC new position access'))
    value['scoring_inputs'] = envelope(row)
    return envelope(value)


@pytest.fixture(scope='module')
def source(prepared, tmp_path_factory):
    root = tmp_path_factory.mktemp('import-scope-SYNTHETIC').resolve()
    fits = prepared['fits']
    positions = FinalPositionInputs(prepared['prefixes'], tuple(ScoringTargets(p.sample_id, p.window_sha256,
        p.score_seconds, p.terrain_origin.position_m[None, :]+p.score_seconds[:, None]*p.terrain_origin.velocity_mps[None, :])
        for p in prepared['prefixes']), prepared['population']['sha256'], digest('SYNTHETIC source position access'))
    owner = consumer(prepared)
    old_args = dict(protocol=fits.protocol, execution=fits.execution, matrix=fits.matrix,
        input_identity=fits.inputs.identity, population=prepared['population'], positions=positions,
        prior=fits.inputs.prior, map_catalog=owner.maps.catalog)
    context_path, context = publish(root/'contexts', unpack(input_context(**old_args)))
    restore_input_context(context, **{k: old_args[k] for k in ('protocol', 'execution', 'matrix')},
                          approval_sha256=unpack(fits.inputs.identity)['approval_sha256'])
    bindings = {}
    for work in fits.work.values():
        source_path = prepared['root']/'fits'/work['work_id']/(fits.receipts[work['fit_identity']]['sha256']+'.json')
        target = root/'fits'/work['work_id']/source_path.name
        target.parent.mkdir(parents=True)
        shutil.copyfile(source_path, target)
        bindings[work['fit_identity']] = dict(path=target.relative_to(root).as_posix(),
            file_sha256=file_hash(target), content_sha256=fits.receipts[work['fit_identity']]['sha256'])
    work = select(owner)
    output = owner.execute(work, output_directory=root/'forecast'/work['work_id'])
    path = Path(output['artifact_path'])
    forecast = dict(path=path.relative_to(root).as_posix(), file_sha256=file_hash(path), content_sha256=output['artifact_sha256'])
    matrix, protocol, execution = fits.matrix, fits.protocol, fits.execution
    runtime = envelope(dict(protocol_sha256=protocol['sha256'], matrix_sha256=matrix['sha256'], ledger_directory=str(root)))
    approval = unpack(fits.inputs.identity)['approval_sha256']
    contract = budget.contract_for_matrix(matrix, protocol_sha256=protocol['sha256'], execution_sha256=execution['sha256'],
        runtime_manifest_sha256=runtime['sha256'], approval_sha256=approval, ledger_directory=root)
    works = unpack(matrix)['workloads']
    input_work = next(w for w in works if w['kind'] == 'input_qualification_and_population')
    completed = [input_work, *fits.work.values(), work]
    pending_work = next(w for w in works if w['kind'] == 'scientific_forecast' and w['work_id'] != work['work_id'])
    states = {w['work_id']: 'success' for w in completed}
    states[pending_work['work_id']] = 'reserved'
    charges = dict.fromkeys(contract['phase_caps_ns'], 0)
    charges['method_forecasts'] = 30*budget.NANOSECONDS
    tip = dict(root_sha256=digest(contract), event_count=60, last_event_sha256=digest('SYNTHETIC tip'))
    cost = envelope(dict(schema_version=costs.VERSION, read_only=True, authorizes_execution=False,
        authorizes_generation_token_transfer=False, process_tree_closed=True, pending_dispatched=False,
        pending_requested=False, old_ledger_modified=False, accounting_categories_reconciled=True,
        ledger_directory=str(root), ledger_root_sha256=digest(contract), committed_tip=tip,
        protocol_sha256=protocol['sha256'], execution_sha256=execution['sha256'], matrix_sha256=matrix['sha256'],
        runtime_manifest_sha256=runtime['sha256'], approval_sha256=approval, work_dispositions=states,
        work_disposition_counts=dict(Counter(states.values())), pending_reservation=dict(work_id=pending_work['work_id'],
            phase=pending_work['phase'], reserved_ns=30*budget.NANOSECONDS, generated_forecasts=1,
            reservation_name='SYNTHETIC NOT DISPATCHED', reservation_sha256=digest('SYNTHETIC reservation')),
        generation_reservations_retained=2, generated_calls_released=0, phase_caps_ns=contract['phase_caps_ns'],
        total_cap_ns=contract['total_cap_ns'], charged_ns_by_phase=charges,
        measured_ns_by_phase=dict.fromkeys(charges, 0), conservatively_charged_ns_by_phase=charges,
        control_charged_ns_by_phase=dict.fromkeys(charges, 0), control_observed_ns_by_phase=dict.fromkeys(charges, 0),
        charged_total_ns=sum(charges.values()), retained_terminal_total_hold_ns=sum(charges.values()),
        head_sha256=digest('SYNTHETIC original head'), terminal_sha256=digest('SYNTHETIC original terminal')))
    def artifact(w):
        return None if w == input_work else forecast if w == work else bindings[w['fit_identity']]
    sources = {w['work_id']: dict(result_sha256=digest(['SYNTHETIC result', w['work_id']]),
        settlement_sha256=digest(['SYNTHETIC settlement', w['work_id']]),
        observation_sha256=digest(['SYNTHETIC observation', w['work_id']]), controller_elapsed_ns=1,
        artifact=artifact(w)) for w in completed}
    saved_science = envelope(dict(schema_version=science.VERSION, read_only=True, authorizes_execution=False,
        cross_execution_admitted=False, new_fits=0, new_forecasts=0, ledger_root_sha256=digest(contract), committed_tip=tip,
        original_protocol_sha256=protocol['sha256'], original_execution_sha256=execution['sha256'],
        original_matrix_sha256=matrix['sha256'], original_approval_sha256=approval, completed_sources=sources,
        successful_work_counts=dict(Counter(w['kind'] for w in completed)), fit_bindings=bindings,
        forecast_index={work['work_id']: forecast}, forecast_kind_counts={'scientific_forecast': 1},
        forecast_status_counts={'success': 1}, context_sha256=context['sha256']))
    refs = {}
    for name, record in [('cost', cost), ('science', saved_science),
                         ('bundle', envelope(dict(protocol=protocol, execution=execution, matrix=matrix, runtime=runtime)))]:
        path, _ = publish(root/name, unpack(record))
        refs[name] = predecessor.reference(path)
    binding = predecessor.build_binding(cost_reference=refs['cost'], science_reference=refs['science'], bundle_reference=refs['bundle'])
    path, _ = publish(root/'binding', unpack(binding))
    yield dict(root=root, context=context, context_ref=predecessor.reference(context_path), predecessor_ref=predecessor.reference(path),
               args=old_args, fits=fits, work=work, forecast=forecast)
    owner.close(); owner.maps.close()


def make_case(source, tmp_path):
    execution = envelope(dict(unpack(source['args']['execution']),
        fixture='SYNTHETIC NONRUNNABLE successor '+digest(str(tmp_path))))
    context = rebind(source['context'], execution,
        digest(['SYNTHETIC distinct successor approval', str(tmp_path)]))
    path, _ = publish(tmp_path/'context', unpack(context))
    scope_args = dict(predecessor_reference=source['predecessor_ref'], source_context_reference=source['context_ref'],
        successor_context_reference=predecessor.reference(path), protocol=source['args']['protocol'],
        execution=execution, matrix=source['args']['matrix'])
    record = module.build_scope(**scope_args)
    path, _ = publish(tmp_path/'bridge', unpack(record))
    args = restore_input_context(context, protocol=scope_args['protocol'], execution=execution, matrix=scope_args['matrix'],
                                approval_sha256=unpack(unpack(context)['input_identity'])['approval_sha256'])
    bridge = module.ScopeBridge(predecessor.reference(path), **{k: args[k] for k in ('protocol', 'execution', 'matrix', 'input_identity')})
    return dict(source=source, bridge=bridge, args=args, context=context, record=record, scope_args=scope_args, root=tmp_path)


@pytest.fixture
def case(source, tmp_path):
    return make_case(source, tmp_path)


@pytest.fixture(scope='module')
def nested(source, tmp_path_factory):
    """Second transport hop of REAL typed SYNTHETIC records, no new fits/arrays."""
    root = tmp_path_factory.mktemp('nested-import-SYNTHETIC').resolve()
    first = make_case(source, root/'first')
    bridge = first['bridge']
    binding, _ = predecessor._load(source['predecessor_ref'])
    _, old_cost, old_science, _, _ = predecessor._history(binding)
    receipts, artifacts = {}, {}
    for work in source['fits'].work.values():
        record = bridge.fit_record(work)
        receipts[work['fit_identity']] = record
        path, _ = publish(first['root']/'fits'/work['work_id'], unpack(record))
        artifacts[work['work_id']] = dict(path=path.relative_to(first['root']).as_posix(),
            file_sha256=file_hash(path), content_sha256=record['sha256'])
    work = source['work']
    causal = bridge.new_cases[work['origin_mode']][work['origin_rank']]
    forecast = bridge.forecast_record(work, case=causal, fit_receipt=receipts[work['fit_identity']])
    path, _ = publish(first['root']/'forecast'/work['work_id'], unpack(forecast))
    artifacts[work['work_id']] = dict(path=path.relative_to(first['root']).as_posix(),
        file_sha256=file_hash(path), content_sha256=forecast['sha256'])
    args = first['args']
    approval = unpack(args['input_identity'])['approval_sha256']
    runtime = envelope(dict(protocol_sha256=args['protocol']['sha256'], matrix_sha256=args['matrix']['sha256'],
        ledger_directory=str(first['root'])))
    contract = budget.contract_for_matrix(args['matrix'], protocol_sha256=args['protocol']['sha256'],
        execution_sha256=args['execution']['sha256'], runtime_manifest_sha256=runtime['sha256'],
        approval_sha256=approval, ledger_directory=first['root'])
    tip = dict(root_sha256=digest(contract), event_count=60, last_event_sha256=digest('SYNTHETIC second tip'))
    cost = deepcopy(old_cost)
    cost.update(ledger_directory=str(first['root']), ledger_root_sha256=digest(contract), committed_tip=tip,
        execution_sha256=args['execution']['sha256'], runtime_manifest_sha256=runtime['sha256'],
        approval_sha256=approval, head_sha256=digest('SYNTHETIC second head'))
    saved = deepcopy(old_science)
    saved.update(ledger_root_sha256=digest(contract), committed_tip=tip,
        original_execution_sha256=args['execution']['sha256'], original_approval_sha256=approval,
        context_sha256=first['context']['sha256'], fit_bindings={w['fit_identity']: artifacts[w['work_id']]
            for w in source['fits'].work.values()}, forecast_index={work['work_id']: artifacts[work['work_id']]})
    for wid, completed in saved['completed_sources'].items():
        completed['artifact'] = artifacts.get(wid)
        # Historical actual completion stays identical across transport hops.
    refs = {}
    for name, value in [('cost', cost), ('science', saved), ('bundle', dict(protocol=args['protocol'],
        execution=args['execution'], matrix=args['matrix'], runtime=runtime))]:
        path, _ = publish(first['root']/name, value)
        refs[name] = predecessor.reference(path)
    binding = predecessor.build_binding(cost_reference=refs['cost'], science_reference=refs['science'],
        bundle_reference=refs['bundle'])
    path, _ = publish(first['root']/'binding', unpack(binding))
    next_source = dict(source, root=first['root'], args=args, context=first['context'],
        context_ref=first['scope_args']['successor_context_reference'], predecessor_ref=predecessor.reference(path))
    second = make_case(next_source, root/'second')
    return dict(first=first, second=second, original=source)


def test_all_26_saved_models_restore_exactly_without_retraining_or_old_record_edits(case, monkeypatch):
    from experiments.pirc17 import formal_training
    def forbidden(*a, **kw): pytest.fail('import invoked training')
    monkeypatch.setattr(formal_training, 'fit_development_method', forbidden)
    monkeypatch.setattr(formal_training, 'fit_direct_dynamics', forbidden)
    bridge, args, fits = case['bridge'], case['args'], case['source']['fits']
    for work in fits.work.values():
        imported = bridge.fit_record(work)
        model = restore_registered_fit(imported, work=work, import_bridge=bridge,
            **{k: args[k] for k in ('protocol', 'execution', 'matrix', 'input_identity')})
        old = fits.models[work['fit_identity']]
        identity = lambda x: x.dynamics.identity() if hasattr(x, 'dynamics') else x.identity
        assert canonical(identity(model)) == canonical(identity(old))
        assert unpack(imported)['parameter_identity'] == unpack(fits.receipts[work['fit_identity']])['parameter_identity']
        assert unpack(imported)['artifact'] == unpack(fits.receipts[work['fit_identity']])['artifact']
    bridge.check_references()
    assert unpack(case['record'])['authorizes_execution'] is False


def test_saved_collection_scores_original_arrays_in_successor_scope_without_forecasting(case, monkeypatch):
    from experiments.pirc17 import formal_forecasts
    from experiments.pirc17.formal_scoring import ScoringConsumers
    monkeypatch.setattr(formal_forecasts, 'forecast_method', lambda *a, **kw: pytest.fail('import invoked forecast'))
    bridge, args, work = case['bridge'], case['args'], case['source']['work']
    receipts = {w['fit_identity']: bridge.fit_record(w) for w in case['source']['fits'].work.values()}
    cases = bridge.new_cases
    record = bridge.forecast_record(work, case=cases[work['origin_mode']][work['origin_rank']], fit_receipt=receipts[work['fit_identity']])
    path, _ = publish(case['root']/'imports', unpack(record))
    saved = SavedForecasts(**{k: args[k] for k in ('protocol', 'execution', 'matrix', 'input_identity', 'population', 'map_catalog')},
        cases=cases, fit_receipts=receipts, root=case['root'], index={work['work_id']: dict(path=path.relative_to(case['root']).as_posix(),
            file_sha256=file_hash(path), content_sha256=record['sha256'])}, import_bridge=bridge)
    restored = saved.read(work)
    source_path = case['source']['root']/case['source']['forecast']['path']
    with np.load(source_path.parent/'forecast.npz', allow_pickle=False) as arrays:
        np.testing.assert_array_equal(restored.forecast.positions_m, arrays['positions_m'])
    assert restored.status == 'success' and not restored.forecast.positions_m.flags.writeable
    assert unpack(restored.record)['execution_sha256'] == args['execution']['sha256']
    scorer = ScoringConsumers(saved=saved, positions=args['positions'])
    assert scorer.inputs.identity['sha256'] == ScoringInputs(args['positions'], saved).identity['sha256']


@pytest.mark.parametrize('fault', ['target', 'clock', 'source', 'map', 'training', 'population', 'flag'])
def test_rehashed_but_scientifically_changed_successor_context_is_rejected(case, fault):
    value = deepcopy(unpack(case['context']))
    if fault == 'target':
        row = deepcopy(unpack(value['scoring_inputs'])); row['targets'][0]['positions_m'][0][0] += 1
        value['scoring_inputs'] = envelope(row)
    elif fault == 'clock': value['causal_prefixes'][0]['prefix_times_seconds'][0] -= 1
    elif fault == 'source': value['causal_prefixes'][0]['window_sha256'] = digest('changed original source')
    elif fault == 'map':
        row = deepcopy(unpack(value['map_catalog'])); row['asset_sha256']['extra'] = digest('different map')
        value['map_catalog'] = envelope(row)
    elif fault == 'training':
        row = deepcopy(unpack(value['input_identity'])); row['configuration_columns']['base'].append('extra')
        value['input_identity'] = envelope(row)
    elif fault == 'population':
        row = deepcopy(unpack(value['population'])); row['selection']['selected'][0]['independent_block_id'] = 'changed'
        value['population'] = envelope(row)
    else: value['raw_sources_independently_reloaded'] = True
    path, _ = publish(case['root']/'altered', value)
    with pytest.raises(ValueError):
        module.build_scope(**dict(case['scope_args'], successor_context_reference=predecessor.reference(path)))


@pytest.mark.parametrize('field', ['parameter_identity', 'source_completion', 'execution_sha256', 'import_scope', 'artifact'])
def test_rehashed_fit_projection_cannot_change_parameters_scope_or_completion(case, field):
    work = next(iter(case['source']['fits'].work.values()))
    value = deepcopy(unpack(case['bridge'].fit_record(work)))
    if field == 'source_completion': value[field]['settlement_sha256'] = digest('not settled')
    elif field == 'import_scope': value[field]['content_sha256'] = digest('other bridge')
    elif field == 'artifact': value[field]['fit_seconds'] += 1
    else: value[field] = digest('changed')
    with pytest.raises(ValueError):
        restore_registered_fit(envelope(value), work=work, import_bridge=case['bridge'],
            **{k: case['args'][k] for k in ('protocol', 'execution', 'matrix', 'input_identity')})


@pytest.mark.parametrize('field', ['case_sha256', 'diagnostics', 'source_completion', 'arrays', 'fit_receipt_sha256'])
def test_rehashed_forecast_projection_is_not_a_new_prediction_or_new_completed_owner(case, field):
    bridge, work = case['bridge'], case['source']['work']
    fit_work = next(w for w in case['source']['fits'].work.values() if w['fit_identity'] == work['fit_identity'])
    fit = bridge.fit_record(fit_work)
    causal = bridge.new_cases[work['origin_mode']][work['origin_rank']]
    value = deepcopy(unpack(bridge.forecast_record(work, case=causal, fit_receipt=fit)))
    if field == 'diagnostics': value[field]['seed'] += 1
    elif field == 'source_completion': value[field]['result_sha256'] = digest('different completed result')
    elif field == 'arrays': value[field]['sha256'] = digest('changed arrays')
    else: value[field] = digest('changed')
    with pytest.raises(ValueError):
        restore_forecast(envelope(value), directory=case['root'], work=work, case=causal, fit_receipt=fit,
            model=bridge.restore_fit(fit, work=fit_work), import_bridge=bridge,
            **{k: case['args'][k] for k in ('protocol', 'execution', 'matrix', 'input_identity', 'map_catalog')})


def test_uncompleted_forecast_and_changed_causal_case_cannot_be_imported(case):
    bridge, work = case['bridge'], case['source']['work']
    pending = next(w for w in bridge.work.values() if w['kind'] == 'scientific_forecast' and w != work)
    with pytest.raises(ValueError, match='successful original work'): bridge._source(pending)
    original_case = bridge.new_cases[work['origin_mode']][work['origin_rank']]
    with pytest.raises(ValueError, match='causal case'):
        bridge._case(work, replace(original_case, window_sha256=digest('changed original window')))


def test_wrong_context_and_reused_authority_are_rejected(case):
    old = case['source']
    with pytest.raises(ValueError, match='original protocol/matrix/context'):
        module.build_scope(**dict(case['scope_args'], source_context_reference=case['scope_args']['successor_context_reference']))
    context = rebind(old['context'], case['args']['execution'], unpack(old['args']['input_identity'])['approval_sha256'])
    path, _ = publish(case['root']/'reuse', unpack(context))
    with pytest.raises(ValueError, match='distinct successor authority'):
        module.build_scope(**dict(case['scope_args'], successor_context_reference=predecessor.reference(path)))


def test_retained_worker_fits_and_prediction_attempts_restore_without_dummy_training(case, monkeypatch):
    from experiments.pirc17 import formal_training, formal_forecasts
    from tests.test_pirc17_formal_forecasts import fixture_maps
    def forbidden(*a, **kw): pytest.fail('already completed work was recomputed')
    monkeypatch.setattr(formal_training, 'fit_development_method', forbidden)
    monkeypatch.setattr(formal_training, 'fit_direct_dynamics', forbidden)
    monkeypatch.setattr(formal_forecasts, 'forecast_method', forbidden)
    fits = formal_training.FitConsumers.restore_all(bridge=case['bridge'], encoders=case['source']['fits'].inputs.encoders)
    assert len(fits.models) == len(fits.receipts) == len(fits.attempted) == 26
    assert not hasattr(fits.inputs, 'methods') and not hasattr(fits.inputs, 'terrain')
    assert isinstance(fits.import_bridge, module.FitBridge)
    for context in (fits.import_bridge.old, fits.import_bridge.new):
        assert set(context) == {'protocol', 'execution', 'matrix', 'input_identity'}
        assert not {'positions', 'targets', 'scoring_inputs', 'causal_prefixes'} & set(context)
    assert not hasattr(fits.import_bridge, 'old_cases') and not hasattr(fits.import_bridge, 'new_cases')
    for work in fits.work.values():
        with pytest.raises(ValueError, match='already attempted'): fits.execute(work, output_directory=case['root']/'forbidden')
    maps = fixture_maps(unpack(fits.inputs.identity), {name: 1. for e in fits.inputs.encoders.values() for name in e.columns})
    # Preserve original synthetic map semantics; no map query executes here.
    maps.catalog = case['args']['map_catalog']
    owner = formal_forecasts.ForecastConsumers(fits=fits, cases=case['bridge'].new_cases,
        population=case['args']['population'], maps=maps)
    try:
        work = case['source']['work']
        assert work['work_id'] in owner.attempted
        assert len(owner.attempted) == 1  # Only completed, never all remaining.
        assert isinstance(owner.import_bridge, module.FitBridge)
        with pytest.raises(ValueError, match='already attempted'):
            owner.execute(work, output_directory=case['root']/'forbidden')
        assert not (case['root']/'forbidden').exists()
    finally:
        owner.close(); maps.close()


def test_cached_fit_projection_avoids_revalidating_same_domain_but_still_checks_source_bytes(case, monkeypatch):
    from experiments.pirc17 import formal_fit_records
    bridge = case['bridge']
    work = next(w for w in bridge.work.values() if w['kind'] == 'method_fit')
    original = formal_fit_records.restore_registered_fit
    calls = []
    def counted(*args, **kwargs):
        calls.append(kwargs['work']['work_id'])
        return original(*args, **kwargs)
    monkeypatch.setattr(formal_fit_records, 'restore_registered_fit', counted)
    first = bridge.fit_record(work)
    assert bridge.fit_record(work) == first and calls == [work['work_id']]
    source_checks = []
    actual_source = bridge._source
    def checked(w):
        source_checks.append(w['work_id'])
        return actual_source(w)
    monkeypatch.setattr(bridge, '_source', checked)
    assert bridge.fit_record(work) == first and source_checks == [work['work_id']]
    # A consumer's reconstructed model is independent, not the producer's
    # mutable fitted object or the bridge's internal forecast validator model.
    one = bridge.restore_fit(first, work=work)
    two = bridge.restore_fit(first, work=work)
    assert one is not two and one.dynamics is not two.dynamics


def test_restoration_batch_reuses_fit_and_owned_scope_but_revalidates_each_array(case, monkeypatch):
    from experiments.pirc17 import formal_forecast_records, formal_pinned_metadata
    from experiments.pirc17.formal_metadata_batch import metadata_batch
    bridge, work = case['bridge'], case['source']['work']
    fit_work = next(w for w in bridge.work.values() if w['fit_identity'] == work['fit_identity']
                    and w['kind'] == 'method_fit')
    for context in (bridge.old, bridge.new):
        assert all(type(context[k]) is formal_pinned_metadata.OwnedRecord
                   for k in ('protocol', 'execution', 'input_identity', 'map_catalog'))
    checks, arrays = [], []
    original_bytes, original_hash = module.PinnedMetadata._bytes, formal_forecast_records.file_hash
    def checked(self, reference):
        checks.append(reference['path'])
        return original_bytes(self, reference)
    def array_hash(path):
        arrays.append(str(path))
        return original_hash(path)
    monkeypatch.setattr(module.PinnedMetadata, '_bytes', checked)
    monkeypatch.setattr(formal_forecast_records, 'file_hash', array_hash)
    causal = bridge.new_cases[work['origin_mode']][work['origin_rank']]
    with metadata_batch(bridge) as batch:
        receipt = bridge.fit_record(fit_work)
        view = bridge.for_fits()
        own_protocol = bridge.old['protocol']
        actual_unpack = formal_pinned_metadata.unpack
        def no_repeated_owned_unpack(record):
            if record is own_protocol: pytest.fail('rehashed immutable protocol per prediction')
            return actual_unpack(record)
        monkeypatch.setattr(formal_pinned_metadata, 'unpack', no_repeated_owned_unpack)
        projected = [bridge.forecast_record(work, case=causal, fit_receipt=receipt) for _ in range(20)]
        assert all(row == projected[0] for row in projected)
        assert len(arrays) == 20  # Every actual synthetic NPZ is still reread/checked.
        fit_path = bridge._source(fit_work)[0]['path']
        assert checks.count(fit_path) == 2  # Constructor and batch entry, not per forecast.
        assert view._metadata_batch is batch
    assert checks.count(fit_path) == 3  # Full closing bytes check.
    assert bridge._metadata_batch is view._metadata_batch is None
    assert all(getattr(b, '_metadata_batch', None) is None for b in bridge._nested_bridges.values())


def test_changed_fit_during_bulk_restoration_prevents_handoff_and_resumes_strict_reads(case):
    from experiments.pirc17.formal_metadata_batch import metadata_batch
    bridge, work = case['bridge'], case['source']['work']
    fit_work = next(w for w in bridge.work.values() if w['fit_identity'] == work['fit_identity']
                    and w['kind'] == 'method_fit')
    path = Path(bridge._source(fit_work)[0]['path'])
    original = path.read_bytes()
    causal = bridge.new_cases[work['origin_mode']][work['origin_rank']]
    handed_off = False
    try:
        with pytest.raises(ValueError, match='bytes changed'):
            with metadata_batch(bridge):
                receipt = bridge.fit_record(fit_work)
                bridge.forecast_record(work, case=causal, fit_receipt=receipt)
                path.write_bytes(original+b' ')  # Only the synthetic fixture's source.
                bridge.forecast_record(work, case=causal, fit_receipt=receipt)
            handed_off = True
        assert not handed_off and bridge._metadata_batch is None
        with pytest.raises(ValueError, match='bytes changed'): bridge.fit_record(fit_work)
    finally:
        path.write_bytes(original)


def test_target_free_fit_scope_cannot_read_predictions_or_retain_private_targets(case):
    fit_scope = case['bridge'].for_fits()
    assert set(fit_scope.work) == {w['work_id'] for w in case['source']['fits'].work.values()}
    assert set(fit_scope.science) == {'completed_sources'}
    assert set(fit_scope.science['completed_sources']) == set(fit_scope.work)
    for method in (fit_scope.forecast_record, fit_scope.restore_forecast):
        with pytest.raises(ValueError, match='target-free'): method(case['source']['work'])
    with pytest.raises(ValueError, match='successful original work'): fit_scope._source(case['source']['work'])


def test_two_hop_models_and_forecast_keep_original_parameters_arrays_and_completions(nested, monkeypatch):
    from experiments.pirc17 import formal_training, formal_forecasts
    def forbidden(*args, **kwargs):
        pytest.fail('nested restoration recomputed completed science')
    monkeypatch.setattr(formal_training, 'fit_development_method', forbidden)
    monkeypatch.setattr(formal_training, 'fit_direct_dynamics', forbidden)
    monkeypatch.setattr(formal_forecasts, 'forecast_method', forbidden)
    second, original = nested['second'], nested['original']
    bridge, args = second['bridge'], second['args']
    receipts = {}
    for work in original['fits'].work.values():
        receipt = bridge.fit_record(work)
        receipts[work['fit_identity']] = receipt
        model = restore_registered_fit(receipt, work=work, import_bridge=bridge,
            **{k: args[k] for k in ('protocol', 'execution', 'matrix', 'input_identity')})
        old_model = original['fits'].models[work['fit_identity']]
        identity = lambda x: x.dynamics.identity() if hasattr(x, 'dynamics') else x.identity
        assert canonical(identity(model)) == canonical(identity(old_model))
        old_record = original['fits'].receipts[work['fit_identity']]
        assert unpack(receipt)['artifact'] == unpack(old_record)['artifact']
        assert unpack(receipt)['parameter_identity'] == unpack(old_record)['parameter_identity']
        intermediate, _ = predecessor._load(unpack(receipt)['source_artifact'])
        assert unpack(intermediate)['schema_version'] == module.FIT_VERSION
        for name in ('result_sha256', 'settlement_sha256', 'observation_sha256', 'controller_elapsed_ns'):
            assert unpack(receipt)['source_completion'][name] == unpack(intermediate)['source_completion'][name]
        assert unpack(receipt)['source_completion']['artifact']['content_sha256'] == intermediate['sha256']
        assert unpack(intermediate)['source_completion']['artifact']['content_sha256'] == old_record['sha256']
    work = original['work']
    record = bridge.forecast_record(work, case=bridge.new_cases[work['origin_mode']][work['origin_rank']],
        fit_receipt=receipts[work['fit_identity']])
    path, _ = publish(second['root']/'predictions', unpack(record))
    saved = SavedForecasts(**{k: args[k] for k in
        ('protocol', 'execution', 'matrix', 'input_identity', 'population', 'map_catalog')},
        cases=bridge.new_cases, fit_receipts=receipts, root=second['root'], import_bridge=bridge,
        index={work['work_id']: dict(path=path.relative_to(second['root']).as_posix(),
            file_sha256=file_hash(path), content_sha256=record['sha256'])})
    restored = saved.read(work)
    original_path = original['root']/original['forecast']['path']
    with np.load(original_path.parent/'forecast.npz', allow_pickle=False) as arrays:
        np.testing.assert_array_equal(restored.forecast.positions_m, arrays['positions_m'])
    intermediate, _ = predecessor._load(unpack(record)['source_artifact'])
    assert unpack(intermediate)['schema_version'] == module.FORECAST_VERSION
    for name in ('result_sha256', 'settlement_sha256', 'observation_sha256', 'controller_elapsed_ns'):
        assert unpack(record)['source_completion'][name] == unpack(intermediate)['source_completion'][name]
    assert unpack(record)['source_completion']['artifact']['content_sha256'] == intermediate['sha256']
    assert unpack(intermediate)['source_completion']['artifact'] == original['forecast']
    assert not (path.parent/'forecast.npz').exists()
    assert len(bridge._nested_bridges) == 1
    bridge.check_references()


def test_nested_predictor_fit_view_never_reopens_full_target_contexts(nested, monkeypatch):
    second, original = nested['second'], nested['original']
    fit_scope = second['bridge'].for_fits()
    def inspect(scope):
        assert isinstance(scope, module.FitBridge)
        for context in (scope.old, scope.new):
            assert set(context) == {'protocol', 'execution', 'matrix', 'input_identity'}
        assert not hasattr(scope, 'old_cases') and not hasattr(scope, 'new_cases')
        assert all(w['kind'] in {'method_fit', 'terrain_fit'} for w in scope.work.values())
        for inner in scope._nested_bridges.values():
            inspect(inner)
    inspect(fit_scope)
    monkeypatch.setattr(module, '_contexts', lambda **kwargs: pytest.fail('predictor loaded scoring contexts'))
    for work in original['fits'].work.values():
        record = fit_scope.fit_record(work)
        restore_registered_fit(record, work=work, import_bridge=fit_scope,
            **{k: second['args'][k] for k in ('protocol', 'execution', 'matrix', 'input_identity')})
    fit_scope.check_references()


def test_cached_nested_scope_does_not_hide_changed_original_artifact_bytes(nested, monkeypatch):
    bridge = nested['second']['bridge']
    work = next(iter(nested['original']['fits'].work.values()))
    receipt = bridge.fit_record(work)
    intermediate, _ = predecessor._load(unpack(receipt)['source_artifact'])
    original_reference = unpack(intermediate)['source_artifact']
    # Fit sources now parse once but recheck bytes through the pinned reader.
    # Inject the same byte failure at its actual per-consumption boundary.
    actual = module.PinnedMetadata._bytes
    def changed(reader, reference):
        if reference == original_reference:
            raise ValueError('original pinned artifact bytes changed')
        return actual(reader, reference)
    monkeypatch.setattr(module.PinnedMetadata, '_bytes', changed)
    with pytest.raises(ValueError, match='original pinned artifact bytes'):
        bridge.fit_record(work)


def test_nested_scope_cache_is_reused_without_reopening_context_per_forecast(nested, monkeypatch):
    bridge, work = nested['second']['bridge'], nested['original']['work']
    fit_work = next(w for w in bridge.work.values() if w['kind'] in {'method_fit', 'terrain_fit'}
        and w['fit_identity'] == work['fit_identity'])
    fit = bridge.fit_record(fit_work)
    causal = bridge.new_cases[work['origin_mode']][work['origin_rank']]
    first = bridge.forecast_record(work, case=causal, fit_receipt=fit)
    monkeypatch.setattr(module, '_contexts', lambda **kwargs: pytest.fail('reparsed nested scope per forecast'))
    assert bridge.forecast_record(work, case=causal, fit_receipt=fit) == first
