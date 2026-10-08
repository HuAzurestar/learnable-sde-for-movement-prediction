"""Real controller/domain callbacks over closed synthetic scientific files.

Positive native transport fixtures replace authority checking explicitly and
generate their science as fixture setup, NOT a formal timing/approval claim.
The separate no-approval test exercises the actual unmodified authority guard.
"""
from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
import shutil
import sys

import pytest

from experiments.pirc17 import formal_results as module
from experiments.pirc17 import formal_budget as budget, formal_controller as control, formal_session as native
from experiments.pirc17.final_eval_guard import VerifiedAccess
from experiments.pirc17.formal_analysis import AnalysisConsumers, SavedScores
from experiments.pirc17.formal_closed import ClosedOutputs
from experiments.pirc17.formal_forecasts import ForecastConsumers
from experiments.pirc17.formal_origins import build_origin_cases
from experiments.pirc17.formal_reanalysis import ReanalysisConsumers, _artifact
from experiments.pirc17.formal_saved import SavedForecasts
from experiments.pirc17.formal_scoring import ScoringConsumers
from experiments.pirc17.formal_training import FitConsumers
from experiments.pirc17.protocol_core import canonical, digest, envelope, file_hash, publish, read_json, unpack
from tests.test_pirc17_formal_forecasts import prepared, select
from tests.test_pirc17_formal_input_work import case
from tests.test_pirc17_formal_reanalysis import deliver, saved_input_fixture
from tests.test_pirc17_formal_closed import changed


def setup_results(case, directory):
    args = case['args']
    authority = dict(approval_path=None, approval_sha256=unpack(args['input_identity'])['approval_sha256'],
        test_path=None, review_path=None, journal_directory=args['access_journal'])
    contract = budget.contract_for_matrix(args['matrix'], protocol_sha256=args['protocol']['sha256'], execution_sha256=args['execution']['sha256'],
        runtime_manifest_sha256=digest('SYNTHETIC SOURCE DELIVERY'), approval_sha256=authority['approval_sha256'], ledger_directory=directory)
    ledger = budget.Ledger.create(directory, contract)
    results = module.ScientificResults(ledger, **{k: args[k] for k in ('protocol', 'execution', 'matrix')}, authority=authority)
    return ledger, results


def metric_access(results, index):
    scope = unpack(results.context['input_identity'])
    _, event = publish(results.authority['journal_directory'], dict(schema_version='pirc17-final-access-event-v1', event='started',
        access_kind='final_eval_metrics', attempt_id=f'{index:032x}', at_utc='2026-09-30T00:00:00+00:00',
        **{k: scope[k] for k in ('protocol_sha256', 'execution_sha256', 'approval_sha256', 'population_sha256')}))
    return VerifiedAccess(access_kind='final_eval_metrics', legacy_cohort_ack='SYNTHETIC ONLY', access_started_sha256=event['sha256'],
        **{k: scope[k] for k in ('protocol_sha256', 'execution_sha256', 'approval_sha256', 'population_sha256')})


def snapshot_readers(results):
    """Fixture-only reader including the last actually closed tentative result."""
    ledger, context = results.ledger, results.context
    closed = ClosedOutputs(ledger.directory, contract=ledger._state.contract, tip=ledger.tip)
    forecasts, scores = deepcopy(results.saved.index), deepcopy(results.score_index)
    pending = results.pending
    if pending is not None:
        work = pending['work']; item = closed.read(work)
        assert item is not None
        record, binding = _artifact(closed, item)
        if work['kind'] in module.KINDS: forecasts[work['work_id']] = binding
        elif work['kind'] == 'common_scores':
            scores[work['work_id']] = {**binding, 'metrics_access_started_sha256': unpack(record)['metrics_access_started_sha256']}
    saved = SavedForecasts(**{k: context[k] for k in ('protocol', 'execution', 'matrix', 'input_identity', 'population', 'map_catalog')},
        cases=results.saved.cases, fit_receipts=results.receipts, root=ledger.directory, index=forecasts)
    scorer = ScoringConsumers(saved=saved, positions=context['positions'])
    return scorer, SavedScores(scorer=scorer, root=ledger.directory, index=scores, access_journal=results.authority['journal_directory'])


@pytest.fixture(scope='module')
def verified(prepared, tmp_path_factory):
    if os.name != 'nt': pytest.skip('actual Windows closed source delivery')
    root = tmp_path_factory.mktemp('scientific-result-callbacks')
    case = saved_input_fixture(prepared, root)
    args = case['args']
    inputs = replace(prepared['fits'].inputs, identity=args['input_identity'])
    fits = FitConsumers(**{k: args[k] for k in ('protocol', 'execution', 'matrix')}, inputs=inputs)
    input_work = next(w for w in unpack(args['matrix'])['workloads'] if w['kind'] == 'input_qualification_and_population')
    outputs = {input_work['work_id']: case['manifest']}
    for work in fits.work.values(): outputs[work['work_id']] = fits.execute(work, output_directory=root/'science'/work['work_id'])
    case['maps'].catalog = args['map_catalog']
    cases = build_origin_cases(args['positions'].prefixes, args['population'], prior=args['prior'])
    forecasts = ForecastConsumers(fits=fits, cases=cases, population=args['population'], maps=case['maps'])
    chosen = [select(forecasts), select(forecasts, 'base'), select(forecasts, kind='forecast_replay'),
        select(forecasts, kind='runtime_cold', repeat=0), select(forecasts, 'all', kind='inertial_path')]
    for work in chosen: outputs[work['work_id']] = forecasts.execute(work, output_directory=root/'science'/work['work_id'])
    ledger, results = setup_results(case, root/'ledger')
    fixture_dir = root/'delivery'; fixture_dir.mkdir()
    initializations = {'readers': 0, 'audit_views': 0}
    try:
        with pytest.MonkeyPatch.context() as patch:
            # Fixtures are not human authorization. Only replace that boundary;
            # all actual per-item metadata, domain and settlement checks run.
            patch.setattr(results, '_check_authority', lambda phase: None)
            actual_readers, actual_closed = results._readers, module.ClosedOutputs
            def readers():
                initializations['readers'] += 1; return actual_readers()
            def closed(*a, **kw):
                initializations['audit_views'] += 1; return actual_closed(*a, **kw)
            patch.setattr(results, '_readers', readers)
            patch.setattr(module, 'ClosedOutputs', closed)
            def send(items, authorize=None):
                deliver(ledger, args['matrix'], fixture_dir, items, validate_result=results.validate,
                        authorize_work=authorize or results.authorize)
            send(outputs)
            assert initializations == {'readers': 1, 'audit_views': 0}
            scorer, _ = snapshot_readers(results)
            score_work = next(w for w in scorer.work.values() if w['origin_mode'] == 'causal_prefix' and w['origin_rank'] == 0)
            score = scorer._execute(metric_access(results, 101), score_work, root/'science'/score_work['work_id'])
            send({score_work['work_id']: score})
            _, score_sources = snapshot_readers(results)
            analysis = AnalysisConsumers(scores=score_sources); analysis_work = next(iter(analysis.work.values()))
            analysis_result = analysis._execute(metric_access(results, 102), analysis_work, root/'science'/analysis_work['work_id'])
            send({analysis_work['work_id']: analysis_result})
            audit_work = next(w for w in results.works.values() if w['kind'] == 'independent_reanalysis')
            audit_output = root/'science'/audit_work['work_id']
            # Its record binds the actual current audit reservation. Generate
            # this synthetic source during authorization, before the test-only
            # file-delivery worker starts. NOT native scientific timing proof.
            def audit_authority(work):
                authorized = results.authorize(work)
                view = ClosedOutputs(ledger.directory, contract=ledger._state.contract, tip=ledger.tip)
                audit = ReanalysisConsumers(closed=view, **results.context, access_journal=args['access_journal'])
                result = audit._execute(metric_access(results, 103), audit_work, audit_output)
                (fixture_dir/(work['work_id']+'.json')).write_bytes(canonical(dict(source_directory=str(audit_output), manifest=result)))
                return authorized
            # deliver normally prepares existing file fixtures; the placeholder
            # is overwritten by audit_authority after the real reservation.
            send({audit_work['work_id']: dict(artifact_path=str(audit_output/'placeholder.json'), artifact_sha256='0'*64)}, audit_authority)
            assert initializations == {'readers': 1, 'audit_views': 1}
            final = ClosedOutputs(ledger.directory, contract=ledger._state.contract, tip=ledger.tip)
            audit_item = final.read(audit_work)
            audit_record = read_json(audit_item['manifest']['artifact_path'])
            assert not unpack(audit_record)['all_predecessors_verified']
            assert unpack(audit_record)['analysis']['status'] == 'verified'
            result = dict(root=root, case=case, results=results, closed=final, chosen=chosen, input_work=input_work,
                score_work=score_work, analysis_work=analysis_work, audit_work=audit_work,
                initializations=initializations, successful=len(results.completed)+1)
        yield result
    finally:
        forecasts.close(); case['maps'].close(); ledger.close()


def test_actual_controller_validates_closed_science_and_restores_models_only_once(verified, monkeypatch):
    value, results = verified, verified['results']
    assert len(results.receipts) == 26 and value['initializations'] == {'readers': 1, 'audit_views': 1}
    assert len(results.saved.index) == 5 and len(results.score_index) == 1
    assert results.pending['work']['work_id'] == value['audit_work']['work_id']
    assert value['successful'] == 35  # 1 input +26 fits +5 forecast-kind +score +analysis +audit.
    for work in [value['input_work'], *[w for w in results.works.values() if w['kind'] in {'method_fit', 'terrain_fit'}],
                 *value['chosen'], value['score_work'], value['analysis_work'], value['audit_work']]:
        item = value['closed'].read(work)
        assert item is not None
        observer = unpack(read_json(value['closed'].directory/'controls'/(item['observation_sha256']+'.json')))
        proof = control.read_verification(value['closed'], work, observer['scientific_manifest_validation'])
        assert proof['verified'] and 'software_fixture_only' not in proof['details']
    def forbidden(*a, **kw): pytest.fail('scientific validation invoked a generator')
    import experiments.pirc17.formal_forecasts as generated
    monkeypatch.setattr(generated, 'forecast_method', forbidden); monkeypatch.setattr(generated, 'rollout', forbidden)
    item = value['closed'].read(value['chosen'][0])
    proof, _, _ = results._scientific(value['chosen'][0], item['manifest'], item['directory'])
    assert proof['details']['saved_forecast_restored'] and proof['details']['new_forecasts'] == 0


@pytest.mark.parametrize('fault', ['parameter', 'forecast_scope', 'array_bytes', 'extra_file', 'manifest_flag', 'score', 'analysis', 'access'])
def test_actual_domain_callback_rejects_rehashed_or_changed_scientific_artifacts(verified, fault):
    results, closed = verified['results'], verified['closed']
    if fault == 'parameter': work = next(w for w in results.works.values() if w['kind'] == 'method_fit')
    elif fault in {'score', 'access'}: work = verified['score_work']
    elif fault == 'analysis': work = verified['analysis_work']
    else: work = verified['chosen'][0]
    original = closed.read(work)
    output = closed.directory/'domain-negative'/fault; output.mkdir(parents=True)
    for relative in original['artifacts']:
        destination = output/relative; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original['directory']/relative, destination)
    manifest = dict(original['manifest']); path = output/Path(manifest['artifact_path']).relative_to(original['directory'])
    manifest['artifact_path'] = str(path)
    payload = deepcopy(unpack(read_json(path)))
    if fault == 'parameter': payload['parameter_identity'] = '0'*64
    elif fault == 'forecast_scope': payload['scientific_claim_authorized'] = True
    elif fault == 'array_bytes': (output/'forecast.npz').write_bytes(b'not saved arrays')
    elif fault == 'extra_file': (output/'extra.bin').write_bytes(b'extra')
    elif fault == 'manifest_flag': manifest['verified'] = True
    elif fault == 'score': payload['rows'][0]['score_m'] = -1.
    elif fault == 'analysis': payload['factor_conclusions']['road']['verdict'] = 'invented benefit'
    else:
        journal = Path(results.authority['journal_directory'])/(payload['metrics_access_started_sha256']+'.json')
        with changed(journal, b'bad access'), pytest.raises(ValueError): results._scientific(work, manifest, output)
        return
    forged = envelope(payload); path.write_bytes(canonical(forged)); manifest['artifact_sha256'] = forged['sha256']
    with pytest.raises(ValueError): results._scientific(work, manifest, output)


def test_real_authorization_rejects_before_worker_launch_or_scientific_file_read(case, tmp_path, monkeypatch):
    ledger, results = setup_results(case, tmp_path/'ledger')
    before = {p.name: file_hash(p) for p in case['args']['access_journal'].glob('*.json')}
    def forbidden(*a, **kw): pytest.fail('unapproved scientific validation entered')
    monkeypatch.setattr(results, '_scientific', forbidden)
    runner = control.Controller(ledger, [sys.executable, '-u', str((Path(__file__).parent/'fixtures/pirc17_session_worker.py').resolve())],
        authorize_work=results.authorize, validate_result=results.validate)
    try:
        runner._open_phase(case['work']['phase'])
        with pytest.raises(control.ControllerStopped): runner._work(ledger._state.work[case['work']['work_id']])
        assert runner.session.child is None and results.failed
        assert ledger.dispositions()[case['work']['work_id']] == 'failure'
        assert before == {p.name: file_hash(p) for p in case['args']['access_journal'].glob('*.json')}
    finally:
        runner._close_phase(last=True, stopped=True); runner.session.close(); ledger.close()


def test_validation_cannot_skip_authorization_or_adopt_existing_completed_work(case, tmp_path):
    ledger, results = setup_results(case, tmp_path/'ledger')
    try:
        ledger.reserve(case['work']['work_id'])
        with pytest.raises(ValueError, match='must be authorized'):
            results.validate(ledger._state.work[case['work']['work_id']], case['manifest'], case['source'])
        assert results.failed and results.context is None
    finally: ledger.close()
