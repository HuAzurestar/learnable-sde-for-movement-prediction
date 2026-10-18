"""Original native controls: response-ready work precedes periodic owner I/O.

These separate600-step/6s adversaries do not change or replace the original
300-step/3s/6s acceptance. No IPC frame, native exit, authority or clock is fake.
"""

import time
import threading

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
import application.research_supervisor as supervisor
from infrastructure.research_control import CheckpointExchange
from infrastructure.research_store import ResearchStore, atomic_write, digest, encode
from tests.test_research_checkpoint_budget_phase import (
    gated_owner,
    test_ready_response_budget_and_save_use_two_actual_physical_passes as original_ready_control,
)


def test_actual_inflight_request_enters_idle_before_periodic_owner_journal(tmp_path, monkeypatch):
    channels, order = [], []
    original_init, original_request = CheckpointExchange.__init__, CheckpointExchange.request
    original_append, original_popen = ResearchStore.append, supervisor.subprocess.Popen

    def initialize(channel, *args, **kwargs):
        original_init(channel, *args, **kwargs)
        channels.append(channel)

    def request(channel):
        result = original_request(channel)
        time.sleep(.2)  # Actual sending call completes after the unchanged5s journal boundary.
        return result

    def append(store, kind, *args, **kwargs):
        if kind == 'HEARTBEAT' and channels and channels[0].request_id is not None:
            order.append('periodic-owner-journal')
        return original_append(store, kind, *args, **kwargs)

    def popen(command, *args, **kwargs):
        process = original_popen(command, *args, **kwargs)
        if len(command) > 1 and str(command[1]).endswith('research_worker.py'):
            original_wait = process.wait

            def wait(*wait_args, **wait_kwargs):
                if channels and channels[0].request_id is not None:
                    order.append('native-idle-wait')
                return original_wait(*wait_args, **wait_kwargs)
            monkeypatch.setattr(process, 'wait', wait)
        return process

    monkeypatch.setattr(CheckpointExchange, '__init__', initialize)
    monkeypatch.setattr(CheckpointExchange, 'request', request)
    monkeypatch.setattr(ResearchStore, 'append', append)
    monkeypatch.setattr(supervisor.subprocess, 'Popen', popen)
    # Existing helper releases the REAL worker during owner idle and verifies
    # native stop/checkpoint and both uncached physical passes before returning.
    original_ready_control(tmp_path, monkeypatch)
    assert order and order[0] == 'native-idle-wait', order


@pytest.mark.parametrize('change', ['closed-arm', 'recovery-hold'])
def test_actual_response_after_idle_keeps_fresh_budget_before_ack(tmp_path, monkeypatch, change):
    store, value, runner = gated_owner(tmp_path)
    channels, released, changed, acknowledged = [], [], [], []
    owner_thread = threading.get_ident()
    original_init, original_popen = CheckpointExchange.__init__, supervisor.subprocess.Popen
    original_ack = CheckpointExchange.acknowledge

    def initialize(channel, *args, **kwargs):
        original_init(channel, *args, **kwargs)
        channels.append(channel)

    def popen(command, *args, **kwargs):
        process = original_popen(command, *args, **kwargs)
        if len(command) > 1 and str(command[1]).endswith('research_worker.py'):
            original_wait = process.wait

            def wait(*wait_args, **wait_kwargs):
                timeout = wait_kwargs.get('timeout')
                if (threading.get_ident() == owner_thread and timeout is not None
                        and 0 < timeout <= .025 and channels and not released
                        and channels[0].request_id is not None):
                    # Release only at the actual owner's ordinary idle wait,
                    # before forwarding that wait, as in the existing control.
                    # Never inject into stop_tree's wait(timeout=1) or a timer.
                    released.append(True)
                    atomic_write(channels[0].directory/'release-response', b'release')
                    while channels[0].response() is None:
                        assert time.monotonic() < channels[0].deadline
                        time.sleep(.001)
                    if change == 'closed-arm':
                        store.append('ARM_CLOSED', {'arm_id': 'affine', 'reason': 'actual idle-boundary close'})
                    else:
                        atomic_write(store.path/'recovery-hold.json', encode({'reason': 'actual idle-boundary hold'}))
                    changed.append(True)
                return original_wait(*wait_args, **wait_kwargs)
            monkeypatch.setattr(process, 'wait', wait)
        return process

    def acknowledge(channel, artifact):
        acknowledged.append(artifact)
        return original_ack(channel, artifact)

    monkeypatch.setattr(CheckpointExchange, '__init__', initialize)
    monkeypatch.setattr(supervisor.subprocess, 'Popen', popen)
    monkeypatch.setattr(CheckpointExchange, 'acknowledge', acknowledge)
    result = runner.run_cell('synthetic', digest(value['cells'][0]), budget=BudgetSpec(6))
    assert released == [True], 'actual ordinary owner idle boundary was not reached'
    assert changed == [True]
    assert result['state'] == 'BUDGET_EXHAUSTED' and 'checkpoint' not in result, result
    assert acknowledged == []
    assert not (channels[0].directory/'checkpoint-ack.json').exists()
    assert BudgetLedger(store).balance('affine')['closed']
    assert store._read_snapshot() is None
