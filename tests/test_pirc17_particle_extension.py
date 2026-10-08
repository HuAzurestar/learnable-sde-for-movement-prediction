"""Missing-budget contract and replay tests; all observations are software fixtures."""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest

from experiments.pirc17 import particle_extension as extension
from experiments.pirc17 import direct_linear_rollout as engine
from experiments.pirc17.brownian import integration_grid
from tests.test_pirc17_direct_linear_rollout import candidate_bytes, evidence, inputs, read_rows

SOFTWARE_TOLERANCE = 1e-6


def publish(path, value, jsonl=False):
    raw = ''.join(json.dumps(r)+'\n' for r in value) if jsonl else json.dumps(value)
    path.write_text(raw, encoding='utf-8')
    return extension.native._hash(path)


def fixture_reference(args, evidence):
    assert engine.run(**args)['status'] == 'complete'
    ledger = args['output']
    rows = read_rows(ledger)
    audit = ledger.with_suffix('.audit.json')
    report = extension.audited(rows, ledger.parent, extension.native._hash(ledger), SOFTWARE_TOLERANCE, evidence)
    audit_sha = publish(audit, report)
    original = {**{k:args[k] for k in extension.native.AXES}, **{k:v for k,v in evidence.items() if k.endswith('_sha256')},
        'eligibility_sha256':args['eligibility_sha256'], 'tolerance_m':SOFTWARE_TOLERANCE, 'outer_wall_seconds':1050}
    original_sha = publish(ledger.with_suffix('.original-plan.json'), original)
    begun = datetime(2026, 9, 28, tzinfo=timezone.utc)
    supervisor = {'schema_version':'pirc17-native-primary-qualification-v1-supervisor', 'plan_sha256':original_sha,
        'status':'complete', 'worker_returncode':0, 'timed_out':False, 'termination_confirmed':True,
        'started_at':begun.isoformat(), 'finished_at':(begun+timedelta(seconds=1)).isoformat(),
        'deadline_at':(begun+timedelta(seconds=1050)).isoformat(), 'outer_wall_seconds':1050,
        'elapsed_seconds':1., 'certified':False, 'formal_training_accepted':False, 'final_eval_label_prediction_metric_reads':0}
    supervisor_path = ledger.with_suffix('.supervisor.json')
    envelope = [dict(type='initialization', schema_version='pirc17-native-primary-qualification-v1',
        plan_sha256=original_sha, plan=original, certified=False, formal_training_accepted=False,
        final_eval_label_prediction_metric_reads=0), dict(type='reference_verified'),
        dict(type='engine_completed', ledger_sha256=extension.native._hash(ledger), completion=rows[-1]),
        dict(type='audit', sha256=audit_sha, numerical=extension.numerical_summary(report)),
        dict(type='completion', status='complete', **report['counts'], resource_stopped=False,
             engine_terminal_error_count=0, certified=False, formal_training_accepted=False,
             final_eval_label_prediction_metric_reads=0)]
    envelope_path = ledger.with_suffix('.envelope.jsonl')
    spec = {**original, 'schema_version':extension.PLAN_VERSION, 'purpose':extension.PURPOSE,
        'output_name':'software-particle-extension', 'particles':[16],
        'expected_run_count':args['limit_origins']*len(args['configurations'])*len(args['seeds'])*len(args['steps']),
        'minimum_free_bytes':engine.MINIMUM_FREE_BYTES, 'certified':False, 'formal_training_accepted':False,
        'final_eval_authorized':False, 'reference_ledger_sha256':extension.native._hash(ledger),
        'reference_audit_sha256':audit_sha, 'reference_envelope_sha256':publish(envelope_path, envelope, True),
        'reference_supervisor_sha256':publish(supervisor_path, supervisor)}
    plan = ledger.with_suffix('.extension-plan.json')
    return dict(plan=plan, plan_sha256=publish(plan, spec), reference_envelope=envelope_path,
        reference_supervisor=supervisor_path, reference_ledger=ledger, reference_audit=audit,
        fit=evidence['fit'], fit_ledger=evidence['fit_ledger'])


@pytest.fixture
def reference(inputs, evidence, monkeypatch):
    args, tracker = inputs
    args = dict(args, configurations=['base','all-terrain','loo-road','loo-river','loo-worldcover','loo-surface'],
        seeds=[engine.SEEDS[0]], particles=[4,8], steps=[5.,2.5], limit_origins=2, wall_seconds=1000,
        selection_policy='lexical_independent_blocks', map_backend='multicell')
    fake = engine.rollout
    def counted(origin, times, **kwargs):
        result = fake(origin, times, **kwargs)
        count = kwargs['particles']*(len(integration_grid(times, kwargs['max_step_seconds'],5.))-1)
        return replace(result, feature_query_rows=count)
    monkeypatch.setattr(engine, 'rollout', counted)
    return fixture_reference(args, evidence), args, tracker


def test_whole_closed_reference_and_missing_only_denominator_are_preserved(reference):
    source, args, tracker = reference
    before = len(tracker['calls'])
    saved = {p:p.read_bytes() for k,p in source.items() if k != 'plan_sha256'}
    prepared = extension.prepare(source=source)
    assert len(prepared.keys) == 24 and len(prepared.reference)-3 == 48
    assert prepared.reference_report['numerical_audit']['sensitivities']
    assert extension.numerical_summary(prepared.reference_report)['out_of_tolerance'] > 0
    assert all(key[3] == 16 for key in prepared.keys)
    assert prepared.scientific_header['particle_counts'] == [16]
    assert prepared.reference[1]['particle_counts'] == [4,8]
    assert len(tracker['calls']) == before and all(p.read_bytes() == value for p,value in saved.items())
    assert not extension.directory_for(prepared.source, prepared.spec).exists()


def generate_new(source, args):
    prepared = extension.prepare(source=source)
    path = args['output'].with_name('new-particles.jsonl')
    assert engine.run(**dict(args, particles=[16], output=path))['status'] == 'complete'
    return prepared, path, read_rows(path)[2:-1]


def test_slice_and_whole_prefix_score_precision_and_adjacent_sensitivity_oracle(reference):
    source, args, _ = reference
    prepared, path, rows = generate_new(source, args)
    first = extension.validate_rows(prepared, rows[:5], [path.parent]*5)
    rest = extension.validate_rows(prepared, rows[5:], [path.parent]*19, offset=5)
    assert len(first) == 10 and len(rest) == 38 and not first & rest
    report = extension.whole_audit(prepared, rows, path.parent)
    assert report['new_run_count'] == 24 and report['reference_runs_covered'] == report['exact_reference_prefix_checks'] == 48
    assert len(report['new_numerical']['sensitivities']) == 12
    assert len(report['cross_source_adjacent_particle_sensitivities']) == 24
    assert report['reference_numerical']['out_of_tolerance'] > 0
    assert all(report[k] is False for k in ('certified','numerically_qualified','formal_training_accepted'))
    assert report['final_eval_label_prediction_metric_reads'] == 0
    old = {(r['sample_id'],r['configuration'],r['max_step_seconds']):r for r in prepared.reference[2:-1] if r['particles']==8}
    for row, comparison in zip(rows,report['cross_source_adjacent_particle_sensitivities']):
        previous = old[row['sample_id'],row['configuration'],row['max_step_seconds']]
        expected = row['scores']['time_weighted_energy_score_m']-previous['scores']['time_weighted_energy_score_m']
        by_time = [b['energy_score_m']-a['energy_score_m'] for a,b in zip(previous['scores']['by_time'],row['scores']['by_time'])]
        assert comparison['second_minus_first_ES_m'] == expected
        assert comparison['second_minus_first_by_time_ES_m'] == by_time
        assert comparison['all_scoring_times_within_tolerance'] == (max(map(abs,[expected,*by_time])) <= SOFTWARE_TOLERANCE)
    with pytest.raises(ValueError,match='every registered'):
        extension.whole_audit(prepared, rows[:-1], path.parent)


@pytest.mark.parametrize('fault', ['seed_profile','old_budget','two_budgets','missing_family','count','step','origins',
    'seed','tolerance','wall_cap','outer_cap','ram_floor','output_escape','authorized','runtime'])
def test_reference_or_plan_changes_fail_closed(reference, fault):
    source, _, _ = reference
    spec = json.loads(source['plan'].read_text())
    if fault == 'seed_profile': spec['schema_version'] = extension.native.SEED_PLAN_VERSION
    if fault == 'old_budget': spec['particles'] = [8]
    if fault == 'two_budgets': spec['particles'] = [8,16]
    if fault == 'missing_family': spec['configurations'].remove('loo-road')
    if fault == 'count': spec['expected_run_count'] -= 1
    if fault == 'step': spec['steps'] = [5.,1.25]
    if fault == 'origins': spec['limit_origins'] = 1; spec['expected_run_count'] = 12
    if fault == 'seed': spec['seeds'] = [engine.SEEDS[1]]
    if fault == 'tolerance': spec['tolerance_m'] = 2.
    if fault == 'wall_cap': spec['wall_seconds'] = 1010
    if fault == 'outer_cap': spec['outer_wall_seconds'] = 1100
    if fault == 'ram_floor': spec['minimum_free_bytes'] = 0
    if fault == 'output_escape': spec['output_name'] = '../escape'
    if fault == 'authorized': spec['final_eval_authorized'] = True
    if fault == 'runtime':
        rows = read_rows(source['reference_ledger'])
        rows[0]['runtime']['numpy'] = 'changed'
        spec['reference_ledger_sha256'] = publish(source['reference_ledger'],rows,True)
    source['plan_sha256'] = publish(source['plan'],spec)
    with pytest.raises(ValueError):
        extension.prepare(source=source)


@pytest.mark.parametrize('fault', ['live','nonzero','timeout','unconfirmed','late','final_eval','audit_link'])
def test_reference_requires_real_complete_bound_closure(reference, fault):
    source, _, _ = reference
    spec = json.loads(source['plan'].read_text())
    supervisor = json.loads(source['reference_supervisor'].read_text())
    if fault == 'live': supervisor['status'] = 'running'
    if fault == 'nonzero': supervisor['worker_returncode'] = 1
    if fault == 'timeout': supervisor['timed_out'] = True
    if fault == 'unconfirmed': supervisor['termination_confirmed'] = False
    if fault == 'late': supervisor['elapsed_seconds'] = 1051
    if fault == 'final_eval': supervisor['final_eval_label_prediction_metric_reads'] = 1
    if fault == 'audit_link':
        rows = read_rows(source['reference_envelope'])
        rows[3]['sha256'] = 'f'*64
        spec['reference_envelope_sha256'] = publish(source['reference_envelope'],rows,True)
    spec['reference_supervisor_sha256'] = publish(source['reference_supervisor'],supervisor)
    source['plan_sha256'] = publish(source['plan'],spec)
    with pytest.raises(ValueError):
        extension.prepare(source=source)


def test_whole_validator_rejects_changed_new_rows_and_late_bytes(reference, monkeypatch):
    source, args, _ = reference
    prepared, path, rows = generate_new(source,args)
    for key, value in [('configuration','base'),('seed',engine.SEEDS[1]),('independent_block_id','other'),
            ('feature_query_rows',0),('invalid_feature_rows',-1),('scores',{}),('particle_precision',{}),('brownian_identity',{})]:
        bad = deepcopy(rows)
        bad[3][key] = value
        assert bad[3][key] != rows[3][key]
        with pytest.raises(ValueError):
            extension.validate_rows(prepared,bad,[path.parent]*len(bad))
    original = extension.replay
    def mutate(*a, **kw):
        result = original(*a,**kw)
        particle = path.parent/rows[-1]['particle_artifact']['path']
        particle.write_bytes(particle.read_bytes()+b'changed')
        return result
    monkeypatch.setattr(extension,'replay',mutate)
    with pytest.raises(ValueError):
        extension.whole_audit(prepared,rows,path.parent)
