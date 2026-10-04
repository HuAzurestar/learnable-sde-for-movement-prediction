"""Real pending ACK exit precedes fresh terminal authority, without extra grace.

The separate600-step/6s fixture permits a short actual closing interval. Original
300-step/3s acceptance remains unchanged and must still be tested independently.
"""
import time

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_contracts import CapabilityRegistry
from application.research_recovery import RecoveryPlugin, RecoveryRegistry
from experiments.pirc25.runner import SharedRunner
from infrastructure.process_tree import ProcessTree
from infrastructure.research_control import CheckpointExchange
from infrastructure.research_store import ResearchError, ResearchStore, atomic_write, digest, encode
from tests.research_admission_fixtures import synthetic_plugin, admit_fixture
from tests.test_research_checkpoint_phase_work import longer_command, longer_resume_command
from tests.test_research_store import spec


def settling_command(output, value, cell):
    command = longer_command(output, value, cell)
    marker = 'control.save(saved, progress)'
    assert command[2].count(marker) == 1
    closing = ('time.sleep(0.075)' if cell['fixture_close_policy'] == 'settle'
               else 'while True: time.sleep(0.01)')
    command[2] = command[2].replace(marker, marker + '\n        ' + closing)
    return command


def pending_owner(tmp_path, *, level='exact', policy='settle'):
    store = ResearchStore(tmp_path, 'checkpoint-stop-order', initialize=True)
    value = spec()
    value['cells'][0].update(plugin_id='checkpoint-stop-order', capability='generic-rollout',
                            visibility='synthetic', fixture_resume_level=level, fixture_close_policy=policy)
    plugin = synthetic_plugin('checkpoint-stop-order', frozenset({'generic-rollout'}),
        ('x', 'y', 'vx', 'vy'), ('m', 'm', 'm/s', 'm/s'), level, settling_command)
    plugin.registry_entry.resource_contract['counts']['steps'] = {'constant': 600}
    registry, recovery = CapabilityRegistry(), RecoveryRegistry()
    registry.register(plugin)
    recovery.register(RecoveryPlugin(plugin.plugin_id, level, longer_resume_command, '1.0.0'))
    admit_fixture(store, value, plugin, tmp_path, recovery_command_builder=longer_resume_command)
    store.register(value, digest(value))
    return store, value, SharedRunner(store, registry, recovery_registry=recovery)


@pytest.mark.parametrize('level', ['exact', 'chunk'])
def test_real_pending_ack_native_stop_precedes_fresh_terminal_budget(tmp_path, monkeypatch, level):
    store, value, runner = pending_owner(tmp_path, level=level)
    original_ack, original_wait, original_balance = (
        CheckpointExchange.acknowledge, ProcessTree.wait_stopped, BudgetLedger.balance)
    acknowledged, stopped, premature, final = [], [], [], []

    def acknowledge(channel, artifact):
        result = original_ack(channel, artifact)
        acknowledged.append(time.monotonic())
        return result

    def wait(tree, *args, **kwargs):
        result = original_wait(tree, *args, **kwargs)
        if result:
            stopped.append((time.monotonic(), tree.process.poll()))
        return result

    def balance(ledger, arm):
        if acknowledged:
            (final if stopped else premature).append(time.monotonic())
        return original_balance(ledger, arm)

    monkeypatch.setattr(CheckpointExchange, 'acknowledge', acknowledge)
    monkeypatch.setattr(ProcessTree, 'wait_stopped', wait)
    monkeypatch.setattr(BudgetLedger, 'balance', balance)
    result = runner.run_cell('synthetic', digest(value['cells'][0]), budget=BudgetSpec(6))
    observed = {'acknowledged': acknowledged, 'stopped': stopped,
                'premature_budget': premature, 'final_budget': final, 'result': result}
    (tmp_path / 'stop-order-observed.json').write_bytes(encode(observed))
    assert acknowledged and stopped and stopped[0][1] == 85, observed
    assert result['state'] == 'FAILED' and 'checkpoint' in result, observed
    assert not premature, 'fresh budget I/O blocked an actual acknowledged worker close before native stop'
    assert final and stopped[0][0] <= final[0], 'fresh current budget must still precede terminal return'


@pytest.mark.parametrize('change', ['closed-arm', 'recovery-hold'])
def test_real_post_ack_budget_change_still_refuses_checkpoint_terminal(tmp_path, monkeypatch, change):
    store, value, runner = pending_owner(tmp_path)
    original_ack, changed = CheckpointExchange.acknowledge, []

    def acknowledge(channel, artifact):
        result = original_ack(channel, artifact)
        if change == 'closed-arm':
            store.append('ARM_CLOSED', {'arm_id': 'affine', 'reason': 'actual synthetic post-ACK closure'})
        else:
            atomic_write(store.path / 'recovery-hold.json', encode({'reason': 'actual synthetic post-ACK hold'}))
        changed.append(True)
        return result

    monkeypatch.setattr(CheckpointExchange, 'acknowledge', acknowledge)
    result = runner.run_cell('synthetic', digest(value['cells'][0]), budget=BudgetSpec(6))
    assert changed == [True]
    assert result['state'] == 'BUDGET_EXHAUSTED', result
    assert BudgetLedger(store).balance('affine')['closed']
    events = store.events()
    assert [event['payload']['outcome'] for event in events if event['event_kind'] == 'SETTLE'] == ['BUDGET_EXHAUSTED']
    assert len([event for event in events if event['event_kind'] == 'WORKER_TREE_STOPPED']) == 1


def test_real_post_ack_physical_corruption_cannot_escape_final_budget_guard(tmp_path, monkeypatch):
    store, value, runner = pending_owner(tmp_path)
    original_ack, original_wait = CheckpointExchange.acknowledge, ProcessTree.wait_stopped
    changed, stopped = [], []

    def acknowledge(channel, artifact):
        result = original_ack(channel, artifact)
        (store.path / 'events' / '0000000000000001.json').write_bytes(b'{}')
        changed.append(True)
        return result

    def wait(tree, *args, **kwargs):
        result = original_wait(tree, *args, **kwargs)
        if result:
            stopped.append(tree.process.poll())
        return result

    monkeypatch.setattr(CheckpointExchange, 'acknowledge', acknowledge)
    monkeypatch.setattr(ProcessTree, 'wait_stopped', wait)
    with pytest.raises(ResearchError, match='CORRUPT_ARTIFACT'):
        runner.run_cell('synthetic', digest(value['cells'][0]), budget=BudgetSpec(6))
    assert changed == [True] and stopped, 'actual corruption and native containment must be reached'


def test_real_worker_ignoring_ack_is_still_stopped_by_original_hard_fuse(tmp_path, monkeypatch):
    store, value, runner = pending_owner(tmp_path, policy='ignore')
    original_ack, acknowledgements = CheckpointExchange.acknowledge, []

    def acknowledge(channel, artifact):
        result = original_ack(channel, artifact)
        acknowledgements.append(time.monotonic())
        return result

    monkeypatch.setattr(CheckpointExchange, 'acknowledge', acknowledge)
    result = runner.run_cell('synthetic', digest(value['cells'][0]), budget=BudgetSpec(6))
    assert acknowledgements, 'actual worker ACK boundary was not reached'
    assert result['state'] == 'TIMEOUT' and result['elapsed_ms'] >= 6000, result
    assert BudgetLedger(store).balance('affine')['closed']
    events = store.events()
    assert [event['payload']['outcome'] for event in events if event['event_kind'] == 'SETTLE'] == ['TIMEOUT']
    assert len([event for event in events if event['event_kind'] == 'WORKER_TREE_STOPPED']) == 1
