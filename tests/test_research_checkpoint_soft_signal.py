"""The actual 80% control signal must not wait for blocking owner authority I/O.

This separate 600-step/6s engineering control does not replace or relax the
original 300-step/3s interruption and 6s reopened continuation acceptance.
Signalling is not an authorization: fresh closure/hold checks still veto ACK.
"""
import time

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_contracts import CapabilityRegistry
from application.research_recovery import RecoveryPlugin, RecoveryRegistry
import application.research_supervisor as supervisor_module
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_control import CheckpointExchange, read_frame
from infrastructure.research_store import ResearchStore, atomic_write, digest, encode
from tests.research_admission_fixtures import synthetic_plugin, admit_fixture
from tests.test_research_checkpoint_phase_work import longer_command, longer_resume_command
from tests.test_research_store import spec


def owner(tmp_path, level):
    store = ResearchStore(tmp_path, 'soft-signal', initialize=True)
    value = spec()
    value['cells'][0].update(plugin_id='soft-signal', capability='generic-rollout',
                             visibility='synthetic', fixture_resume_level=level)
    plugin = synthetic_plugin('soft-signal', frozenset({'generic-rollout'}),
        ('x', 'y', 'vx', 'vy'), ('m', 'm', 'm/s', 'm/s'), level, longer_command)
    plugin.registry_entry.resource_contract['counts']['steps'] = {'constant': 600}
    registry, recovery = CapabilityRegistry(), RecoveryRegistry()
    registry.register(plugin)
    recovery.register(RecoveryPlugin(plugin.plugin_id, level, longer_resume_command, '1.0.0'))
    admit_fixture(store, value, plugin, tmp_path, recovery_command_builder=longer_resume_command)
    store.register(value, digest(value))
    return store, value, SharedRunner(store, registry, recovery_registry=recovery)


@pytest.mark.parametrize('level', ['exact', 'chunk'])
@pytest.mark.parametrize('change', ['none', 'closed-arm', 'recovery-hold'])
def test_actual_soft_signal_reaches_worker_while_owner_budget_read_is_blocked(
        tmp_path, monkeypatch, level, change):
    store, value, runner = owner(tmp_path, level)
    channels, requests, blocked, acknowledgements = [], [], [], []
    original_init = CheckpointExchange.__init__
    original_request = CheckpointExchange.request
    original_ack = CheckpointExchange.acknowledge
    original_balance = BudgetLedger.balance

    def initialize(channel, *args, **kwargs):
        original_init(channel, *args, **kwargs)
        channels.append(channel)

    def request(channel):
        started = time.monotonic()
        assert started >= channel.deadline - 1.2, 'signal moved before the original80% threshold'
        result = original_request(channel)
        requests.append((started, result))
        return result

    def balance(ledger, arm):
        if not blocked:
            assert len(channels) == 1
            blocked.append(True)
            channel = channels[0]
            # Actually block the owner, not its clock or the native worker. The
            # original ledger read is still forwarded after the real signal.
            time.sleep(max(0, channel.deadline - 1.2 - time.monotonic()) + 0.15)
            frame = read_frame(channel.directory / 'checkpoint-request.json', 16384)
            assert frame is not None, 'actual80% signal waited for blocking owner budget I/O'
            assert frame['request_id'] == channel.request_id
            if change == 'closed-arm':
                store.append('ARM_CLOSED', {'arm_id': arm, 'reason': 'actual blocked-owner close'})
            elif change == 'recovery-hold':
                atomic_write(store.path / 'recovery-hold.json', encode({'reason': 'actual blocked-owner hold'}))
        return original_balance(ledger, arm)

    def acknowledge(channel, artifact):
        acknowledgements.append(artifact)
        assert store._read_snapshot() is None
        return original_ack(channel, artifact)

    monkeypatch.setattr(CheckpointExchange, '__init__', initialize)
    monkeypatch.setattr(CheckpointExchange, 'request', request)
    monkeypatch.setattr(CheckpointExchange, 'acknowledge', acknowledge)
    monkeypatch.setattr(BudgetLedger, 'balance', balance)
    result = runner.run_cell('synthetic', digest(value['cells'][0]), budget=BudgetSpec(6))
    assert blocked == [True] and len(requests) == 1
    events = store.events()
    assert len([e for e in events if e['event_kind'] == 'CHECKPOINT_REQUESTED']) == 1
    if change == 'none':
        assert result['state'] == 'FAILED' and 'checkpoint' in result, result
        assert acknowledgements == [result['checkpoint']['artifact_id']]
        assert result['elapsed_ms'] < 6000
    else:
        assert result['state'] == 'BUDGET_EXHAUSTED' and 'checkpoint' not in result, result
        assert not acknowledgements
        assert not (channels[0].directory / 'checkpoint-ack.json').exists()
        assert BudgetLedger(store).balance('affine')['closed']


def test_soft_signal_rechecks_real_threshold_after_early_timer_wakeup(tmp_path, monkeypatch):
    original_timer = supervisor_module.threading.Timer
    early = []

    def timer(interval, function, *args, **kwargs):
        if function.__name__ == 'signal_checkpoint':
            early.append(interval)
            interval = 0.02  # Actual early scheduling; clocks/frames stay real.
        return original_timer(interval, function, *args, **kwargs)

    monkeypatch.setattr(supervisor_module.threading, 'Timer', timer)
    test_actual_soft_signal_reaches_worker_while_owner_budget_read_is_blocked(
        tmp_path, monkeypatch, 'exact', 'none')
    assert len(early) == 1 and early[0] > 0.02


def test_finished_worker_cancels_early_woken_signal_wait_without_late_control(tmp_path, monkeypatch):
    from tests.test_research_live_checkpoint import prepared
    store, value, registry, adapters, _ = prepared(tmp_path / 'owner', 'exact', 'owner')
    original_timer = supervisor_module.threading.Timer
    monitors = []

    def timer(interval, function, *args, **kwargs):
        if function.__name__ == 'signal_checkpoint':
            monitor = original_timer(0.02, function, *args, **kwargs)
            monitors.append(monitor)
            return monitor
        return original_timer(interval, function, *args, **kwargs)

    monkeypatch.setattr(supervisor_module.threading, 'Timer', timer)
    result = SharedRunner(store, registry, recovery_registry=adapters).run_cell(
        'owner', digest(value['cells'][0]), budget=BudgetSpec(6))
    assert result['state'] == 'SUCCEEDED', result
    assert len(monitors) == 1 and not monitors[0].is_alive()
    assert not [event for event in store.events() if event['event_kind'] == 'CHECKPOINT_REQUESTED']
    work = store.path / 'artifacts' / ('.attempt-' + result['attempt_id'])
    assert not (work / 'checkpoint-request.json').exists()
    assert not (work / 'checkpoint-ack.json').exists()
