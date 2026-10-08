"""Actual closed synthetic artifacts -> independent whole scientific replay.

Incomplete synthetic inventory stays incomplete. No real data, human approval,
new empirical runs or claim that a fixture is the eventual formal executor.
"""
from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from experiments.pirc17 import formal_budget as budget, formal_controller as control, formal_session as native
from experiments.pirc17 import formal_forecasts as forecasts, formal_training as training, formal_reanalysis as module
from experiments.pirc17 import formal_inputs as inputs
from experiments.pirc17 import final_eval_guard as guard, formal_eligibility as eligibility
from experiments.pirc17.formal_analysis import AnalysisConsumers
from experiments.pirc17.formal_closed import ClosedOutputs
from experiments.pirc17.protocol_core import canonical, digest, envelope, publish, read_json, unpack
from tests.test_pirc17_formal_controller import authority, verification
from tests.test_pirc17_formal_forecasts import prepared, select, consumer, fixture_maps
from tests.test_pirc17_formal_scoring import scored, positions, score_work, access_for
from tests.test_pirc17_formal_inputs import qualified_fixture
from tests.test_pirc17_formal_eligibility import metadata_fixture
from tests.test_pirc17_formal_closed import changed


def deliver(ledger, matrix, fixture_directory, outputs, *, validate_result=verification, authorize_work=None):
    """Close a SELECTED software fixture subset with the real native controller.

    Unselected full-matrix work remains unattempted; never shrink its contract.
    Model/scoring computation is fixture setup, not a formal timing claim.
    """
    by_id = {w['work_id']: w for w in unpack(matrix)['workloads']}
    for wid, output in outputs.items():
        artifact = next(output[key] for key in ('artifact_path', 'context_path') if key in output)
        config = dict(source_directory=str(Path(artifact).parent), manifest=output)
        (fixture_directory/(wid+'.json')).write_bytes(canonical(config))
    runner = control.Controller(ledger, [sys.executable, '-u',
        str((Path(__file__).parent/'fixtures/pirc17_session_worker.py').resolve()),
        '--mode', 'deliver_saved_fixture', '--fixture-directory', str(fixture_directory)],
        authorize_work=authorize_work or (lambda w: authority(ledger, w)), validate_result=validate_result)
    groups = {}
    for wid in outputs:
        work = by_id[wid]
        groups.setdefault(work['phase'], []).append(work)
    try:
        for index, (phase, works) in enumerate(groups.items()):
            runner._open_phase(phase)
            for work in works:
                runner._work(ledger._state.work[work['work_id']])
            runner._close_phase(last=index == len(groups)-1)
    finally:
        if runner.meter is not None:
            runner.session.request_termination('operator_stop')
            runner.session.close()
            runner.meter.close()
        runner.session.close()
    assert native._query_job(runner.session.job_name)['exists'] is False


@pytest.fixture(scope='module')
def closed_science(prepared, scored, tmp_path_factory):
    if os.name != 'nt': pytest.skip('actual Windows closed-output chain fixture')
    root = tmp_path_factory.mktemp('independent-closed-science')
    source_config = root/'test-only-delivery'; source_config.mkdir()
    owner, _, outputs, _ = scored
    outputs = deepcopy(outputs)
    for kind, repetition in [('forecast_replay', None), ('runtime_cold', 0), ('runtime_warmup', 0), ('runtime_warm', 0)]:
        work = select(owner, kind=kind, repeat=repetition)
        outputs[work['work_id']] = owner.execute(work, output_directory=prepared['root']/'audit-fixtures'/work['work_id'])
    context = dict(protocol=owner.protocol, execution=owner.execution, matrix=owner.matrix,
        input_identity=owner.input_identity, population=prepared['population'], positions=positions(prepared),
        prior=prepared['fits'].inputs.prior, map_catalog=owner.maps.catalog, access_journal=root/'access')
    binding = budget.contract_for_matrix(owner.matrix, protocol_sha256=owner.protocol['sha256'],
        execution_sha256=owner.execution['sha256'], runtime_manifest_sha256=digest('SOFTWARE RUNTIME ONLY'),
        approval_sha256=unpack(owner.input_identity)['approval_sha256'], ledger_directory=root/'ledger')
    def reader(ledger):
        return module.ReanalysisConsumers(closed=ClosedOutputs(ledger.directory, contract=binding, tip=ledger.tip), **context)
    with budget.Ledger.create(root/'ledger', binding) as ledger:
        fit_outputs = {}
        for work in prepared['fits'].work.values():
            record = prepared['fits'].receipts[work['fit_identity']]
            directory = prepared['root']/'fits'/work['work_id']
            path = directory/(record['sha256']+'.json')
            fit_outputs[work['work_id']] = dict(artifact_path=str(path), artifact_sha256=record['sha256'],
                status='fitted', generated_forecasts_attempted=0)
        deliver(ledger, owner.matrix, source_config, {**fit_outputs, **outputs})
        first = reader(ledger)
        saved, scorer, _, _, _, _, _ = first.sources()
        access = access_for(scorer)
        _, event = publish(root/'access', dict(schema_version='pirc17-final-access-event-v1', event='started',
            access_kind='final_eval_metrics', attempt_id='b'*32, at_utc='2026-09-30T00:00:00+00:00',
            **{k: getattr(access, k) for k in ('protocol_sha256', 'execution_sha256', 'approval_sha256', 'population_sha256')}))
        access = replace(access, access_started_sha256=event['sha256'])
        work = score_work(scorer)
        output = scorer._execute(access, work, root/'score-source')
        deliver(ledger, owner.matrix, source_config, {work['work_id']: output})
        second = reader(ledger)
        before_analysis_tip = deepcopy(ledger.tip)
        _, _, score_sources, _, _, _, _ = second.sources()
        analysis = AnalysisConsumers(scores=score_sources)
        work = next(iter(analysis.work.values()))
        output = analysis._execute(access, work, root/'analysis-source')
        deliver(ledger, owner.matrix, source_config, {work['work_id']: output})
        audit = reader(ledger)
    return dict(audit=audit, work=next(iter(audit.work.values())), access=access, context=context,
                binding=binding, root=root, analysis_work=work, before_analysis_tip=before_analysis_tip)


def test_actual_closed_fit_forecast_score_analysis_and_runtime_recompute_without_generation(closed_science, monkeypatch):
    def forbidden(*args, **kwargs): pytest.fail('saved-output audit must not fit or forecast')
    monkeypatch.setattr(training.FitConsumers, 'execute', forbidden)
    monkeypatch.setattr(forecasts.ForecastConsumers, 'execute', forbidden)
    monkeypatch.setattr(forecasts, 'forecast_method', forbidden)
    monkeypatch.setattr(forecasts, 'rollout', forbidden)
    case = closed_science
    result = case['audit'].compute(case['work'])
    assert len(result['work_inventory']) == 11659 and result['required_predecessors'] == 11657
    assert result['new_forecasts'] == result['new_fits'] == 0
    assert result['analysis']['status'] == 'verified'
    assert result['all_predecessors_verified'] is False and result['input_evidence'] is None
    assert result['restored_forecast_counts'] == {'success': 16, 'failed': 1}
    assert sum(r['domain_status'] == 'verified_fit' for r in result['work_inventory']) == 26
    assert sum(r['domain_status'] == 'verified_scores' for r in result['work_inventory']) == 1
    assert sum(r['role'] == 'current_audit' for r in result['work_inventory']) == 1
    assert sum(r['role'] == 'future_export' for r in result['work_inventory']) == 1
    runtime = unpack(result['runtime_replay'])
    assert runtime['registered_counts'] == dict(replays=38, cold=75, warmup=15, warm=75)
    assert runtime['replay_counts_by_status']['reproduced'] == 1
    measured = next(r for r in runtime['runtime_subjects'] if r['subject'] == 'arm-01/full')
    assert sum(r['runtime'] is not None for r in measured['rows']) == 3
    assert next(r for r in measured['rows'] if r['kind'] == 'runtime_warm' and r['repetition'] == 0)['warm_condition_verified']
    assert measured['conditions']['runtime_cold']['status'] == 'unavailable'
    assert all(r['verdict'] == 'unavailable' for r in result['analysis']['factor_conclusions'].values())


def test_audit_publication_exact_replay_rejects_forged_completeness_and_no_retry(closed_science, tmp_path):
    case = closed_science
    audit = module.ReanalysisConsumers(closed=case['audit'].closed, **case['context'])
    output = audit._execute(case['access'], case['work'], tmp_path)
    record = read_json(output['artifact_path'])
    check = audit.verify(record, work=case['work'], access_started_sha256=case['access'].access_started_sha256)
    assert check['actual_saved_output_reanalysis'] and not check['all_predecessors_verified']
    forged = deepcopy(unpack(record)); forged['all_predecessors_verified'] = True
    with pytest.raises(ValueError, match='actual closed outputs'):
        audit.verify(envelope(forged), work=case['work'], access_started_sha256=case['access'].access_started_sha256)
    with pytest.raises(ValueError, match='already attempted'):
        audit._execute(case['access'], case['work'], tmp_path)


def test_missing_closed_analysis_stays_unavailable_despite_computable_statistics(closed_science):
    case = closed_science
    closed = ClosedOutputs(case['audit'].closed.directory, contract=case['binding'], tip=case['before_analysis_tip'])
    audit = module.ReanalysisConsumers(closed=closed, **case['context'])
    result = audit.compute(case['work'])
    assert result['analysis']['status'] == 'unavailable' and not result['analysis']['missing_saved_analysis_replaced']
    row = next(r for r in result['work_inventory'] if r['work_id'] == case['analysis_work']['work_id'])
    assert row['ledger_status'] == 'unattempted' and row['domain_status'] == 'unavailable'
    assert not result['all_predecessors_verified']


def test_analysis_is_compared_with_recomputed_science_not_just_its_digest(closed_science, monkeypatch):
    case = closed_science
    audit = module.ReanalysisConsumers(closed=case['audit'].closed, **case['context'])
    original = audit.sources
    def changed_sources():
        # Domain-boundary fault injection, not a claim that tampering passed
        # the separate real native/byte provenance checks.
        bundle = original()
        records = bundle[3]
        key = next(iter(records)); forged = deepcopy(unpack(records[key]))
        forged['factor_conclusions']['road']['verdict'] = 'retain'
        records[key] = envelope(forged)
        return bundle
    monkeypatch.setattr(audit, 'sources', changed_sources)
    with pytest.raises(ValueError, match='independent raw-score'):
        audit.compute(case['work'])


@pytest.mark.parametrize('fault', [None, 'population', 'count', 'window', 'prefixes'])
def test_replay_actual_qualification_selection_and_original_windows(tmp_path, monkeypatch, fault):
    args = qualified_fixture(tmp_path, monkeypatch)
    loaded = inputs.load_final_positions(**args)
    record = read_json(args['qualification_path'])
    population = read_json(args['population_path'])
    prefixes = loaded.prefixes
    if fault == 'population': population = envelope({'selection': {'selected': []}})
    if fault == 'count':
        payload = deepcopy(unpack(record)); payload['eligible_samples'] -= 1; record = envelope(payload)
    if fault == 'window':
        payload = unpack(record)
        path = Path(args['qualification_path']).parent/next(iter(payload['windows'].values()))
        window = deepcopy(unpack(read_json(path))); window['origin_epoch_ns'] += 1
        path.write_bytes(canonical(envelope(window)))
    if fault == 'prefixes': prefixes = prefixes[:-1]
    def replay():
        return module.qualification_evidence(record, directory=Path(args['qualification_path']).parent,
            protocol=args['protocol'], execution=args['execution'], approval_sha256=args['approval_sha256'],
            population=population, prefixes=prefixes)
    if fault is None:
        result = replay()
        assert result['original_window_files_verified'] and result['outcome_blind_selection_recomputed']
        assert result['selected_samples'] == len(prefixes) and not result['raw_sources_independently_reloaded']
    else:
        with pytest.raises(ValueError): replay()


def saved_input_fixture(prepared, tmp_path):
    """Tiny saved input fixture; shared prepared state contains synthetic fits.

    This helper adds no fits/forecasts. Real protocol guards still reject its
    deliberately synthetic protocol/approval identities.
    """
    fits = prepared['fits']
    maps = fixture_maps(unpack(fits.inputs.identity), prepared['row'])
    old = dict(protocol=fits.protocol, execution=fits.execution, matrix=fits.matrix,
        input_identity=fits.inputs.identity, prior=fits.inputs.prior, map_catalog=maps.catalog)
    sample, rows = metadata_fixture('closed-input-only')
    p = deepcopy(unpack(old['protocol']))
    p['eligibility_contract'] = {'fixture': 'SYNTHETIC METADATA ONLY'}
    p['dataset_inputs'].update(dataset_id='SYNTHETIC CLOSED INPUT', sha256=digest('synthetic input binding'),
        partitions=dict(counts={'final_eval': 1},
            split_identity_sha256={'final_eval': digest({k: [sample[k]] for k in ('sample_id', 'segment_id', 'independent_block_id')})},
            split_row_identity_sha256={'final_eval': digest([[sample[k] for k in ('sample_id', 'segment_id', 'independent_block_id')]])}))
    protocol = envelope(p)
    m = deepcopy(unpack(old['matrix'])); m['protocol_sha256'] = protocol['sha256']; matrix = envelope(m)
    execution = envelope(dict(fixture='NO HUMAN APPROVAL', protocol_sha256=protocol['sha256'], matrix_sha256=matrix['sha256']))
    disposition, window = eligibility.qualify_window(sample, rows, input_sha256=p['dataset_inputs']['sha256'],
                                                    rule_sha256=digest(p['eligibility_contract']))
    report = envelope(dict(schema_version=guard.ELIGIBILITY_VERSION, protocol_sha256=protocol['sha256'],
        execution_sha256=execution['sha256'], dataset_id=p['dataset_inputs']['dataset_id'],
        eligibility_rule_sha256=digest(p['eligibility_contract']), prior_performance_reads=0, rows=[disposition]))
    population = envelope(guard.population_contract(report, protocol=protocol, execution=execution))
    scope = deepcopy(unpack(old['input_identity']))
    scope.update(protocol_sha256=protocol['sha256'], execution_sha256=execution['sha256'], population_sha256=population['sha256'])
    journal = tmp_path/'access'
    def event(kind, index):
        _, record = publish(journal, dict(schema_version='pirc17-final-access-event-v1', event='started',
            access_kind=kind, attempt_id=f'{index:032x}', at_utc='2026-09-30T00:00:00+00:00',
            **{k: scope[k] for k in ('protocol_sha256', 'execution_sha256', 'approval_sha256')},
            population_sha256=None if kind == 'final_eval_eligibility' else population['sha256']))
        return record['sha256']
    qualification_access = event('final_eval_eligibility', 1)
    position_access = event('final_eval_positions', 2)
    scope['access_started_sha256'] = event('final_eval_features', 3)
    map_access = event('final_eval_features', 4)
    prefix = inputs.bind_final_prefix(window, np.array([[110., 35.], [110.001, 35.001], [110.002, 35.002]]),
        population_sha256=population['sha256'], condition_sha256=digest('SYNTHETIC CONDITION'))
    loaded = positions(dict(prefixes=(prefix,), population=population), access_started_sha256=position_access)
    catalog = deepcopy(unpack(old['map_catalog']))
    catalog.update({k: scope[k] for k in ('protocol_sha256', 'execution_sha256', 'approval_sha256', 'population_sha256')})
    catalog['access_started_sha256'] = map_access
    args = dict(protocol=protocol, execution=execution, matrix=matrix, input_identity=envelope(scope), population=population,
        positions=loaded, prior=old['prior'], map_catalog=envelope(catalog), access_journal=journal)
    source = tmp_path/'source'
    context_path, context = publish(source, unpack(module.input_context(**{k: v for k, v in args.items() if k != 'access_journal'})))
    report_path, _ = publish(source/'eligibility', unpack(report))
    population_path, _ = publish(source/'population', unpack(population))
    window_path, _ = publish(source/'windows', unpack(window))
    qualification_path, qualification = publish(source, dict(schema_version=eligibility.RESULT_VERSION,
        protocol_sha256=protocol['sha256'], execution_sha256=execution['sha256'], approval_sha256=scope['approval_sha256'],
        access_started_sha256=qualification_access, eligibility_path=report_path.relative_to(source).as_posix(),
        eligibility_sha256=report['sha256'], population_path=population_path.relative_to(source).as_posix(),
        population_sha256=population['sha256'], windows={sample['sample_id']: window_path.relative_to(source).as_posix()},
        denominator_samples=1, eligible_samples=1, position_feature_value_prediction_metric_reads=0))
    manifest = dict(context_path=str(context_path), context_sha256=context['sha256'],
                    qualification_path=str(qualification_path), qualification_sha256=qualification['sha256'])
    return dict(args=args, source=source, manifest=manifest, context=context, qualification=qualification,
        map_access=map_access, maps=maps)


def test_real_closed_input_context_qualification_and_access_files_are_connected(prepared, tmp_path):
    if os.name != 'nt': pytest.skip('actual Windows closed input fixture')
    fixture = saved_input_fixture(prepared, tmp_path)
    args, manifest = fixture['args'], fixture['manifest']
    protocol, execution, matrix = (args[k] for k in ('protocol', 'execution', 'matrix'))
    scope = unpack(args['input_identity'])
    loaded, journal, map_access = args['positions'], args['access_journal'], fixture['map_access']
    binding = budget.contract_for_matrix(matrix, protocol_sha256=protocol['sha256'], execution_sha256=execution['sha256'],
        runtime_manifest_sha256=digest('synthetic-only-runtime'), approval_sha256=scope['approval_sha256'], ledger_directory=tmp_path/'ledger')
    work = next(w for w in unpack(matrix)['workloads'] if w['kind'] == 'input_qualification_and_population')
    fixture_directory = tmp_path/'test-only-delivery'; fixture_directory.mkdir()
    with budget.Ledger.create(tmp_path/'ledger', binding) as ledger:
        deliver(ledger, matrix, fixture_directory, {work['work_id']: manifest})
        closed = ClosedOutputs(ledger.directory, contract=binding, tip=ledger.tip)
    audit = module.ReanalysisConsumers(closed=closed, **args)
    audit_work = next(iter(audit.work.values()))
    result = audit.compute(audit_work)
    assert result['input_evidence']['selected_samples'] == 1
    assert result['input_evidence']['outcome_blind_selection_recomputed']
    assert next(r for r in result['work_inventory'] if r['work_id'] == work['work_id'])['domain_status'] == 'verified_guarded_context_and_qualification'
    assert not result['all_predecessors_verified'] and result['restored_forecast_counts'] == {}
    forged = replace(loaded, targets=(replace(loaded.targets[0], positions_m=loaded.targets[0].positions_m+1),))
    with pytest.raises(ValueError, match='closed input context'):
        module.ReanalysisConsumers(closed=closed, **dict(args, positions=forged)).compute(audit_work)
    with changed(journal/(map_access+'.json'), b'not an access record'), pytest.raises(ValueError):
        audit.compute(audit_work)


@pytest.mark.parametrize('fault', ['matrix', 'approval', 'prior', 'positions'])
def test_independent_context_and_complete_matrix_cannot_be_substituted(closed_science, fault):
    case = closed_science
    args = deepcopy(case['context'])
    if fault == 'matrix':
        matrix = deepcopy(unpack(args['matrix'])); matrix['workloads'].pop()
        args['matrix'] = envelope(matrix)
    elif fault == 'approval':
        scope = deepcopy(unpack(args['input_identity'])); scope['approval_sha256'] = '0'*64
        args['input_identity'] = envelope(scope)
    elif fault == 'prior':
        args['prior'] = replace(args['prior'], evidence=envelope({'invalid': True}))
    else:
        args['positions'] = replace(args['positions'], targets=())
    with pytest.raises((ValueError, KeyError)):
        audit = module.ReanalysisConsumers(closed=case['audit'].closed, **args)
        audit.compute(case['work'])


@pytest.mark.parametrize('fault', ['approval', 'population'])
def test_real_guard_denies_before_any_closed_scientific_source_is_read(tmp_path, monkeypatch, fault):
    args = qualified_fixture(tmp_path, monkeypatch)
    keys = ('protocol', 'execution', 'approval_path', 'approval_sha256', 'test_path', 'review_path',
        'journal_directory', 'population_path', 'population_sha256', 'eligibility_path')
    call = {k: args[k] for k in keys}
    class Trap:
        def _execute(self, *args): pytest.fail('unauthorized independent audit read private outputs')
    if fault == 'approval': call['approval_sha256'] = '0'*64
    else: call['population_path'] = call['population_sha256'] = None
    with pytest.raises(ValueError):
        module.reanalyze_formal_work(audit=Trap(), work={}, output_directory=tmp_path/'forbidden', **call)
    assert not (tmp_path/'forbidden').exists()
