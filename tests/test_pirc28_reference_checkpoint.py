"""Real 80%-checkpoint delivery during admitted engineering reference work.

Distinct disposable stores are test fixtures, not production budget identities.
No fake worker, replaced sampler, forced checkpoint or owner-created state is used.
"""

import copy
import json

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_recovery import SharedRecovery
from experiments.pirc28.approximation_evidence import reference_seed
from experiments.pirc28.runtime import registries
from infrastructure.research_store import ResearchError, ResearchStore, digest
from tests import test_pirc28_evidence as evidence_tests
from tests.test_pirc28_managed_approximation import engineering_recipe
from tests.test_pirc28_synthetic import fixture_poll


def fail_with_logs(store, result):
    logs = list((store.path / 'artifacts').glob('.attempt-*/worker.log'))
    pytest.fail(str(result)+'\n'+'\n'.join(path.read_text(errors='replace') for path in logs))


def test_reference_phase_checkpoint_reopened_resume_matches_actual_continuous_worker(tmp_path, monkeypatch):
    # Fixed before execution. This exercises real reference work under the
    # original hard limits; elapsed-time observations never choose a new budget.
    monkeypatch.setattr(evidence_tests, 'recipe', engineering_recipe)
    continuous, baseline, _, baseline_grant, baseline_runner = evidence_tests.prepared(tmp_path / 'continuous')
    baseline_cell = next(cell for cell in baseline['cells'] if cell['fit_K'] == 2)
    completed = baseline_runner.run_cell(baseline['study_id'], digest(baseline_cell), budget=BudgetSpec(120))
    if completed['state'] != 'SUCCEEDED':
        fail_with_logs(continuous, completed)
    expected = json.loads(continuous.read_artifact(completed['artifact_id'], purpose='evaluate',
                                                  authorization=baseline_grant))
    store, study, definition, grant, runner = evidence_tests.prepared(tmp_path / 'resumed')
    cell = next(cell for cell in study['cells'] if cell['fit_K'] == 2)
    assert study['switching_recipe'] == baseline['switching_recipe']
    interrupted = runner.run_cell(study['study_id'], digest(cell), budget=BudgetSpec(18))
    if interrupted['state'] != 'FAILED' or store.attempts()[interrupted['attempt_id']]['error_code'] != 'CHECKPOINT_SAVED':
        fail_with_logs(store, interrupted)
    saved = [event['payload'] for event in store.events() if event['event_kind'] == 'CHECKPOINT_SAVED']
    assert len(saved) == 1
    receipt = saved[0]
    assert 14_400 <= receipt['elapsed_ms'] < 18_000
    assert receipt['progress']['eta_seconds'] is None
    frame = json.loads(store.read_artifact(receipt['artifact_id'], purpose='resume', authorization=grant))
    state = frame['state']
    assert state['data_position']['reference_phase'] == 'running; whole-fixed-reference-restart'
    assert state['data_position']['reference_recovery'] == (
        'restart-all-fixed-batches-from-frozen-recipe-cell-seed; no-partial-reuse')
    assert all(fit is not None and fit['complete_reason'] is not None
               for fit in state['method_state']['fit_states'])
    assert 'approximation_checks' not in state['method_state']
    previous_cost = BudgetLedger(store).balance('pirc28-k2')['committed_ms']
    assert previous_cost > 0 and not BudgetLedger(store).balance('pirc28-k2')['closed']

    reopened = ResearchStore(tmp_path / 'resumed', store.store_id)
    registry, adapters = registries()
    recovery = SharedRecovery(reopened, registry, adapters)
    forged_grant = copy.deepcopy(grant)
    forged_grant['purposes'] = ['preview']
    with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
        recovery.prepare(interrupted['attempt_id'], receipt['artifact_id'], authorization=forged_grant)
    assert len(reopened.attempts()) == 1
    assert BudgetLedger(reopened).balance('pirc28-k2')['committed_ms'] == previous_cost
    resumed = recovery.resume(interrupted['attempt_id'], receipt['artifact_id'], authorization=grant,
                              budget=BudgetSpec(120))
    if resumed['state'] != 'SUCCEEDED':
        fail_with_logs(reopened, resumed)
    actual = json.loads(reopened.read_artifact(resumed['artifact_id'], purpose='evaluate', authorization=grant))
    for field in ('metrics', 'fit', 'forecast'):
        assert actual[field] == expected[field]
    report = actual['fit']['approximation_checks']
    assert report['reference_seed'] == reference_seed(study['switching_recipe'], cell['seed'])
    assert all(path['reference_batches'] == 8 for path in report['paths'])
    assert report['managed_evidence_bound'] is report['qualified_identifiability'] is False
    assert reopened.attempts()[resumed['attempt_id']]['parent_attempt_id'] == interrupted['attempt_id']
    events = reopened.events()
    reservations = [event['payload'] for event in events if event['event_kind'] == 'RESERVE']
    settlements = [event['payload'] for event in events if event['event_kind'] == 'SETTLE']
    assert len(reservations) == len(settlements) == 2
    assert all(value['arm_id'] == 'pirc28-k2' for value in reservations + settlements)
    assert [value['outcome'] for value in settlements] == ['FAILED', 'SUCCEEDED']
    assert BudgetLedger(reopened).balance('pirc28-k2')['committed_ms'] == sum(
        value['charged_ms'] for value in settlements) > previous_cost
    assert len([event for event in events if event['event_kind'] == 'WORKER_TREE_STOPPED']) == 2
    collected = evidence_tests.collect_qualification(reopened, definition, [study['study_id']],
        {study['study_id']: grant}, check_budget=fixture_poll, allow_fixture=True)
    history = next(item for item in collected['lineage'] if item['selected_attempt_id'] == resumed['attempt_id'])
    assert [item['state'] for item in history['attempts']] == ['FAILED', 'SUCCEEDED']
    assert collected['state'] == 'FIXTURE_ONLY'
    assert collected['qualified_identifiability'] is collected['real_data_authorized'] is False
