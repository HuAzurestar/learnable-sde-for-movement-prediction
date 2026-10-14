"""Saved-input restoration and actual controller wiring on synthetic fixtures.

No human approval or empirical work. Dispatch adapters below test orchestration
only; separate denial cases exercise the real guard before any raw operation.
"""
from copy import deepcopy
from dataclasses import fields, replace
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from experiments.pirc17 import formal_budget as budget, formal_input_work as module
from experiments.pirc17 import formal_controller as control
from experiments.pirc17.formal_closed import ClosedOutputs
from experiments.pirc17.protocol_core import canonical, digest, envelope, publish, read_json, unpack
from tests.test_pirc17_formal_closed import changed
from tests.test_pirc17_formal_forecasts import prepared
from tests.test_pirc17_formal_reanalysis import deliver, saved_input_fixture
from tests.pirc17_predecessor_fixture import software_frozen_metadata


@pytest.fixture
def case(prepared, tmp_path):
    value = saved_input_fixture(prepared, tmp_path)
    value['scope'] = {k: value['args'][k] for k in ('protocol', 'execution', 'matrix')}
    value['scope']['approval_sha256'] = unpack(value['args']['input_identity'])['approval_sha256']
    value['work'] = next(w for w in unpack(value['args']['matrix'])['workloads']
                         if w['kind'] == 'input_qualification_and_population')
    yield value
    value['maps'].close()


def read(case, manifest=None):
    return module.read_input_work(case['manifest'] if manifest is None else manifest,
        directory=case['source'], access_journal=case['args']['access_journal'], **case['scope'])


def test_typed_roundtrip_keeps_causal_coordinates_prior_and_truth_separate(case):
    restored = module.restore_input_context(case['context'], **case['scope'])
    assert canonical(module.input_context(**restored)) == canonical(case['context'])
    prefix = restored['positions'].prefixes[0]
    target = restored['positions'].targets[0]
    assert not {'targets', 'truth', 'future'} & {f.name for f in fields(prefix)}
    np.testing.assert_array_equal(prefix.terrain_origin.velocity_mps, case['args']['positions'].prefixes[0].terrain_origin.velocity_mps)
    assert prefix.condition_at.identity() == case['args']['positions'].prefixes[0].condition_at.identity()
    assert restored['prior'].prior.identity == case['args']['prior'].prior.identity
    for array in (prefix.terrain_origin.history_positions_m, prefix.method_origin.history_times_seconds,
                  prefix.score_seconds, target.elapsed_seconds, target.positions_m):
        assert not array.flags.writeable
    assert not np.shares_memory(target.positions_m, case['args']['positions'].targets[0].positions_m)
    args, paths, details = read(case)
    assert args['positions'].population_sha256 == paths['population_sha256']
    assert details['actual_access_events_verified'] and details['typed_inputs_restored']
    assert details['qualification']['outcome_blind_selection_recomputed']
    assert not details['raw_sources_independently_reloaded']


@pytest.mark.parametrize('fault', ['scope', 'flag', 'origin', 'frame', 'solar', 'history', 'clock', 'target',
                                 'prior_population', 'prior_role', 'training_rows', 'training_count'])
def test_rehashed_context_does_not_bypass_typed_domain_checks(case, fault):
    value = deepcopy(unpack(case['context']))
    prefix = value['causal_prefixes'][0]
    if fault == 'scope': value['execution_sha256'] = '0'*64
    elif fault == 'flag': value['truth_passed_to_predictors'] = 0
    elif fault == 'origin': prefix['terrain_prefix_positions_m'][-1][0] = 1.
    elif fault == 'frame': prefix['scoring_frame'][0] += .1
    elif fault == 'solar': prefix['solar']['condition_names'] = ['invented']
    elif fault == 'history': prefix['terrain_prefix_positions_m'].pop(0)
    elif fault == 'clock': prefix['prefix_times_seconds'][-1] = 1.
    elif fault == 'target':
        scores = deepcopy(unpack(value['scoring_inputs'])); scores['targets'][0]['window_sha256'] = '0'*64
        value['scoring_inputs'] = envelope(scores)
    elif fault.startswith('prior_'):
        prior = deepcopy(unpack(value['prior_evidence']))
        if fault == 'prior_population': prior['population_identity'] = '0'*64
        else: prior['training_rows'][0]['method_role'] = 'validation'
        value['prior_evidence'] = envelope(prior)
    else:
        identity = deepcopy(unpack(value['input_identity']))
        identity['final_eval_numeric_training_rows' if fault == 'training_rows' else 'outer_train_windows'] = True
        value['input_identity'] = envelope(identity)
    with pytest.raises(ValueError): module.restore_input_context(envelope(value), **case['scope'])


def test_self_consistent_prefix_secant_must_still_match_original_window_clock(case):
    value = deepcopy(unpack(case['context']))
    value['causal_prefixes'][0]['prefix_times_seconds'][0] += 1.
    forged = envelope(value)
    # Causal/scoring projection is internally consistent; only original window
    # metadata proves that its observation clock was changed.
    module.restore_input_context(forged, **case['scope'])
    path, record = publish(case['source'], value)
    manifest = dict(case['manifest'], context_path=str(path), context_sha256=record['sha256'])
    with pytest.raises(ValueError, match='history clock'): read(case, manifest)


@pytest.mark.parametrize('fault', ['escape', 'hash', 'unknown', 'access'])
def test_input_manifest_requires_owned_files_and_actual_guard_events(case, tmp_path, fault):
    manifest = dict(case['manifest'])
    if fault == 'escape':
        path, record = publish(tmp_path/'outside', unpack(case['context']))
        manifest.update(context_path=str(path), context_sha256=record['sha256'])
    elif fault == 'hash': manifest['qualification_sha256'] = '0'*64
    elif fault == 'unknown': manifest['verified'] = True
    else:
        path = case['args']['access_journal']/(case['map_access']+'.json')
        with changed(path, b'not an access receipt'), pytest.raises(ValueError): read(case, manifest)
        return
    with pytest.raises(ValueError): read(case, manifest)


def test_input_callback_requires_exact_work_and_only_registered_input_files(case):
    kwargs = dict(directory=case['source'], access_journal=case['args']['access_journal'], **case['scope'])
    validation, args, paths = module.input_verification(case['work'], case['manifest'], **kwargs)
    assert len(validation['artifacts']) == 5 and validation['verified']
    assert args['positions'].population_sha256 == paths['population_sha256']
    wrong = dict(case['work'], generated_forecasts=1)
    with pytest.raises(ValueError, match='exact registered input work'):
        module.input_verification(wrong, case['manifest'], **kwargs)
    (case['source']/'unregistered-private.bin').write_bytes(b'SYNTHETIC EXTRA FILE')
    with pytest.raises(ValueError, match='missing/extra files'):
        module.input_verification(case['work'], case['manifest'], **kwargs)


@pytest.mark.parametrize('corrupt', [False, True])
def test_actual_native_controller_uses_restored_input_domain_not_worker_pass(case, tmp_path, corrupt):
    if os.name != 'nt': pytest.skip('native Windows controller')
    if corrupt:
        value = deepcopy(unpack(case['context'])); value['causal_prefixes'][0]['scoring_frame'][0] += .1
        path = Path(case['manifest']['context_path']); record = envelope(value)
        path.write_bytes(canonical(record)); case['manifest']['context_sha256'] = record['sha256']
    args, scope = case['args'], case['scope']
    contract = budget.contract_for_matrix(args['matrix'], protocol_sha256=args['protocol']['sha256'],
        execution_sha256=args['execution']['sha256'], runtime_manifest_sha256=digest('SYNTHETIC RUNTIME'),
        approval_sha256=scope['approval_sha256'], ledger_directory=tmp_path/'ledger')
    restored = []
    def validate(work, manifest, directory):
        result, inputs, paths = module.input_verification(work, manifest, directory,
            access_journal=args['access_journal'], **scope)
        restored.append((inputs, paths))
        return result
    configs = tmp_path/'fixture-config'; configs.mkdir()
    with budget.Ledger.create(tmp_path/'ledger', contract) as ledger:
        if corrupt:
            with pytest.raises(control.ControllerStopped):
                deliver(ledger, args['matrix'], configs, {case['work']['work_id']: case['manifest']}, validate_result=validate)
            assert ledger.dispositions()[case['work']['work_id']] == 'failure'
            assert ledger.summary()['halted_reason'] and restored == []
        else:
            deliver(ledger, args['matrix'], configs, {case['work']['work_id']: case['manifest']}, validate_result=validate)
            closed = ClosedOutputs(ledger.directory, contract=contract, tip=ledger.tip)
            item = closed.read(case['work'])
            assert item is not None and len(restored) == 1
            assert restored[0][0]['positions'].prefixes[0].identity() == args['positions'].prefixes[0].identity()
            assert Path(restored[0][1]['population_path']).is_relative_to(item['directory'])


def orchestrator(case, tmp_path, *,continuation_binding=None, **authority_changes):
    scope = case['scope']
    authority = dict(approval_path=None, approval_sha256=scope['approval_sha256'], test_path=None,
        review_path=None, journal_directory=tmp_path/'journal')
    authority.update(authority_changes)
    return module.InputWork(**{k: scope[k] for k in ('protocol', 'execution', 'matrix')}, authority=authority,
        continuation_binding=continuation_binding,
        release=tmp_path/'raw-release', snapshot=tmp_path/'raw-snapshot', data_root=tmp_path/'raw-data',
        trajectory_path=tmp_path/'raw-trajectories', development_eligibility_path=tmp_path/'development-eligibility')


def test_actual_input_operation_denies_before_loaders_and_consumes_attempt(case, tmp_path, monkeypatch):
    worker = orchestrator(case, tmp_path)
    def forbidden(*a, **kw): pytest.fail('unapproved protected input operation')
    monkeypatch.setattr(module.formal_eligibility, '_qualify', forbidden)
    monkeypatch.setattr(module.formal_inputs, 'load_final_positions', forbidden)
    monkeypatch.setattr(module.formal_training, 'prepare_formal_training', forbidden)
    monkeypatch.setattr(module.formal_maps, 'prepare_formal_maps', forbidden)
    with pytest.raises(ValueError, match='human approval'):
        worker.execute(case['work'], output_directory=tmp_path/'out')
    assert not (tmp_path/'out').exists() and not (tmp_path/'journal').exists()
    with pytest.raises(ValueError, match='already attempted'):
        worker.execute(case['work'], output_directory=tmp_path/'other')


@pytest.mark.parametrize('break_publication', [False, True])
@pytest.mark.parametrize('continuation',['none','match','population_mismatch','context_mismatch'])
def test_actual_input_orchestration_orders_dispatch_retains_maps_and_closes_on_failure(case, prepared, tmp_path, monkeypatch, break_publication,continuation):
    # Public-call adapters are intentionally ONLY an orchestration test. They
    # do not certify raw-source acquisition or confer real guard authorization.
    from experiments.pirc17 import formal_input_interruption
    gates=[]
    binding=None
    if continuation!='none':
        binding=envelope(dict(software_fixture_only=True,authorizes_execution=False))
        def scope(value):assert value['software_fixture_only'] is True;gates.append('scope')
        monkeypatch.setattr(formal_input_interruption,'validate_binding_scope',scope)
    worker, seen = orchestrator(case, tmp_path,continuation_binding=binding), []
    def common(kwargs):
        assert all(kwargs[k] == v for k, v in worker.authority.items())
        assert kwargs['protocol'] == worker.protocol and kwargs['execution'] == worker.execution
    def qualify(**kwargs):
        common(kwargs); seen.append('qualification')
        assert kwargs['release'] == worker.paths['release'] and kwargs['snapshot'] == worker.paths['snapshot']
        output = kwargs['output_directory']; output.mkdir(parents=True)
        q = unpack(case['qualification'])
        files = [Path(case['manifest']['qualification_path']).name, q['eligibility_path'], q['population_path'], *q['windows'].values()]
        for relative in files:
            destination = output/relative; destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(case['source']/relative, destination)
        return dict(result_path=str(output/files[0]), result_sha256=case['qualification']['sha256'])
    def positions(**kwargs):
        common(kwargs); seen.append('positions')
        assert seen == ['qualification', 'positions']
        assert read_json(kwargs['population_path']) == case['args']['population']
        assert kwargs['qualification_sha256'] == case['qualification']['sha256']
        return case['args']['positions']
    training = replace(prepared['fits'].inputs, identity=case['args']['input_identity'])
    def train(**kwargs):
        common(kwargs); seen.append('training')
        assert seen == ['qualification', 'positions', 'training']
        assert all(kwargs[k] == v for k, v in worker.paths.items())
        assert read_json(kwargs['final_eligibility_path'])['sha256'] == unpack(case['qualification'])['eligibility_sha256']
        return training
    maps = case['maps']; maps.catalog = case['args']['map_catalog']
    def map_inputs(**kwargs):
        common(kwargs); seen.append('maps')
        assert seen == ['qualification', 'positions', 'training', 'maps']
        assert read_json(kwargs['population_path']) == case['args']['population']
        return maps
    monkeypatch.setattr(module.formal_eligibility, 'qualify_final_inputs', qualify)
    monkeypatch.setattr(module.formal_inputs, 'load_final_positions', positions)
    monkeypatch.setattr(module.formal_training, 'prepare_formal_training', train)
    monkeypatch.setattr(module.formal_maps, 'prepare_formal_maps', map_inputs)
    def population_check(report,population):
        assert seen==['qualification'];gates.append('population')
        assert population==case['args']['population']
        if continuation=='population_mismatch':raise ValueError('synthetic population mismatch')
    def context_check(context):
        assert seen==['qualification','positions','training','maps'];gates.append('context')
        if continuation=='context_mismatch':raise ValueError('synthetic context mismatch')
    monkeypatch.setattr(formal_input_interruption,'verify_continued_population',population_check)
    monkeypatch.setattr(formal_input_interruption,'verify_continued_context',context_check)
    if continuation in {'population_mismatch','context_mismatch'}:
        with pytest.raises(ValueError,match='synthetic .* mismatch'):
            worker.execute(case['work'],output_directory=tmp_path/'out')
        if continuation=='population_mismatch':assert seen==['qualification'] and gates==['scope','population']
        else:assert maps.closed and gates==['scope','population','context']
        assert not list((tmp_path/'out').glob('*.json'))
        with pytest.raises(ValueError,match='already attempted'):
            worker.execute(case['work'],output_directory=tmp_path/'retry')
        return
    if break_publication:
        def fail(*a, **kw): raise OSError('synthetic storage failure')
        monkeypatch.setattr(module, 'publish', fail)
        with pytest.raises(OSError): worker.execute(case['work'], output_directory=tmp_path/'out')
        assert maps.closed and not list((tmp_path/'out').glob('*.json'))
    else:
        output = worker.execute(case['work'], output_directory=tmp_path/'out')
        assert output.training is training and output.maps is maps and not maps.closed
        assert len(output.manifest) == 4 and output.context['positions'] is case['args']['positions']
        validation, _, _ = module.input_verification(case['work'], output.manifest, tmp_path/'out',
            access_journal=case['args']['access_journal'], **case['scope'])
        assert validation['verified']
        output.close(); assert maps.closed
    assert seen == ['qualification', 'positions', 'training', 'maps']
    assert gates==([] if continuation=='none' else ['scope','population','context'])
    with pytest.raises(ValueError, match='already attempted'):
        worker.execute(case['work'], output_directory=tmp_path/'retry')
