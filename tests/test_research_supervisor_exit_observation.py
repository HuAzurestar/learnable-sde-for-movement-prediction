"""Native exit after slow owner I/O must not become a stale-heartbeat failure.

Fresh disposable stores and stdlib toy children only. The paired controls keep
the original 15s heartbeat and real resource observations, not a fake clock.
"""

import hashlib
import json
from pathlib import Path
import sys
import time

import pytest

from application.research_budget import BudgetSpec, ResourceBudgetSpec
from application.research_supervisor import ResearchSupervisor
from infrastructure.research_store import ResearchStore, atomic_write, digest, encode
from tests.test_research_execution_policy import policy
from tests.test_research_store import spec


def isolated_store(root, resource):
    assert not root.exists(), 'fixture must not reuse an existing store'
    store = ResearchStore(root, 'readiness-classification', initialize=True)
    value = spec()
    arm = 'pirc28-k1' if resource else 'affine'
    value['code_hash'] = 'a' * 64
    value['arms'][0]['arm_id'] = arm
    value['cells'] = [{'arm_id': arm, 'seed': 1, 'block_id': 'fixture', 'visibility': 'synthetic'}]
    store.register(value, digest(value))
    attempt = store.new_attempt(store.register_run('synthetic', value['cells'][0]))
    if resource:
        decision = policy(store.store_id)  # Fixture provenance, never production authority.
        reference = {'manifest_id': 'policy-' + digest(decision), 'sha256': digest(decision)}
        store.publish(reference['manifest_id'], decision)
        budget = ResourceBudgetSpec(reference, 'a' * 64)
    else:
        budget = BudgetSpec(60)
    return store, attempt, budget


@pytest.mark.parametrize('resource', [False, True], ids=['finite', 'resource-no-time-cap'])
@pytest.mark.parametrize('slow', [False, True], ids=['fast-owner', 'slow-owner'])
def test_fast_nonzero_child_exit_does_not_become_heartbeat_loss(tmp_path, monkeypatch, resource, slow):
    store, attempt_id, budget = isolated_store(tmp_path / 'isolated store', resource)
    ready, release = tmp_path / 'child-ready', tmp_path / 'child-release'
    supervisor = ResearchSupervisor(store)
    original_balance = supervisor.budget.balance
    observation = {'released': False, 'injected_delay_seconds': 0.0}
    work = store.path / 'artifacts' / ('.attempt-' + attempt_id)

    def observed_balance(arm_id, **kwargs):
        if work.exists() and not observation['released']:
            until = time.monotonic() + 10
            while not ready.exists():
                assert time.monotonic() < until, 'native toy child never reached its barrier'
                time.sleep(0.01)
            release.touch()
            observation['released'] = True
            if slow:
                began = time.monotonic()
                time.sleep(18)  # Real elapsed time; the original15s threshold is unchanged.
                observation['injected_delay_seconds'] = time.monotonic() - began
        return original_balance(arm_id, **kwargs)  # Fresh real authority, never a cached approval.

    monkeypatch.setattr(supervisor.budget, 'balance', observed_balance)
    child = (
        "from pathlib import Path; import sys,time; "
        "ready,release=map(Path,sys.argv[1:]); ready.touch(); "
        "until=time.monotonic()+10; "
        "\nwhile not release.exists():"
        "\n if time.monotonic()>until: raise SystemExit(8)"
        "\n time.sleep(0.01)"
        "\nprint('CONTROLLED_BOOTSTRAP_FAILURE_EXIT7',flush=True)"
        "\nraise SystemExit(7)"
    )
    result = supervisor.run(attempt_id,
        lambda _: [sys.executable, '-B', '-c', child, str(ready), str(release)], budget)
    events = store.events()
    attempt = store.attempts()[attempt_id]
    started = [e['payload'] for e in events if e['event_kind'] == 'WORKER_STARTED']
    stopped = [e['payload'] for e in events if e['event_kind'] == 'WORKER_TREE_STOPPED']
    settled = [e['payload'] for e in events if e['event_kind'] == 'SETTLE']
    log = (work / 'worker.log').read_text(encoding='utf-8')
    assert observation['released'] and 'CONTROLLED_BOOTSTRAP_FAILURE_EXIT7' in log
    assert len(started) == len(stopped) == len(settled) == 1
    assert settled[0]['settled'] and settled[0]['charged_ms'] == result['elapsed_ms']
    assert stopped[0]['confirmation'] == 'native-job-or-process-group-no-running-descendants'
    assert result['artifact_id'] is None
    if slow:
        assert observation['injected_delay_seconds'] >= 18
    source = Path(sys.modules[ResearchSupervisor.__module__].__file__)
    report = {'supervisor_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'scope': 'disposable toy child, not a scientific result or canonical recovery',
        'fixture_root': str(store.path), 'budget_mode': 'resource' if resource else 'finite',
        'slow_owner': slow, **observation, 'attempt_id': attempt_id,
        'state': attempt['state'], 'error_code': attempt['error_code'],
        'native_start': started[0], 'native_stop': stopped[0], 'settlement': settled[0],
        'worker_log': log}
    atomic_write(tmp_path / 'classification.json', encode(report))
    print(json.dumps(report), flush=True)
    assert attempt['state'] == 'FAILED' and attempt['error_code'] == 'WORKER_FAILED', report


@pytest.mark.parametrize('resource,veto', [(False, 'closed'), (True, 'closed'), (False, 'deadline')],
                         ids=['finite-closed', 'resource-closed', 'finite-deadline'])
def test_confirmed_exit_does_not_skip_fresh_authority_or_deadline(tmp_path, monkeypatch, resource, veto):
    import application.research_supervisor as module

    store, attempt_id, budget = isolated_store(tmp_path / 'isolated store', resource)
    if veto == 'deadline':
        budget = BudgetSpec(2)
    supervisor = ResearchSupervisor(store)
    original_popen = module.subprocess.Popen
    original_balance = supervisor.budget.balance
    wrappers, observed = [], []
    work = store.path / 'artifacts' / ('.attempt-' + attempt_id)

    def popen(command, *args, **kwargs):
        process = original_popen(command, *args, **kwargs)
        if len(command) > 1 and str(command[1]).endswith('research_worker.py'):
            wrappers.append(process)
        return process

    def balance(arm_id, **kwargs):
        if work.exists() and not observed:
            assert len(wrappers) == 1
            assert wrappers[0].wait(timeout=5) == 7  # Real native exit, not mocked poll.
            observed.append(True)
            if veto == 'closed':
                store.append('ARM_CLOSED', {'arm_id': arm_id, 'reason': 'fixture closure after native exit'})
            else:
                time.sleep(2.1)  # Cross the unchanged disposable2s hard deadline.
        return original_balance(arm_id, **kwargs)

    monkeypatch.setattr(module.subprocess, 'Popen', popen)
    monkeypatch.setattr(supervisor.budget, 'balance', balance)
    result = supervisor.run(attempt_id,
        lambda _: [sys.executable, '-B', '-c', 'raise SystemExit(7)'], budget)
    expected = 'BUDGET_EXHAUSTED' if veto == 'closed' else 'TIMEOUT'
    assert observed == [True]
    assert result['state'] == expected and result['artifact_id'] is None
    assert store.attempts()[attempt_id]['error_code'] == expected
    events = store.events()
    stops = [e for e in events if e['event_kind'] == 'WORKER_TREE_STOPPED']
    settlements = [e for e in events if e['event_kind'] == 'SETTLE']
    assert len(stops) == len(settlements) == 1
    assert stops[0]['sequence'] < settlements[0]['sequence']
    assert settlements[0]['payload']['outcome'] == expected
    assert settlements[0]['payload']['charged_ms'] == result['elapsed_ms'] > 0
