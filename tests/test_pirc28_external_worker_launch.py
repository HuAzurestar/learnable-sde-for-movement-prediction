"""Full launch/admission/result/reuse in the existing tiny synthetic fixtures.

These are disposable engineering controls, never canonical research or a new
scientific retry. Child source binding must survive an external cwd/PYTHONPATH.
"""

import json
from pathlib import Path
import sys

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from experiments.pirc25.affine import code_hash
from experiments.pirc25.runner import SharedRunner
from experiments.pirc28.runtime import command
from infrastructure.research_store import ResearchStore, digest
from tests.test_pirc28_runtime import (
    prepare,
    test_shared_owner_checkpoint_and_real_worker_resume_preserve_prior_cost as original_resume,
)


def external_launch_directory(tmp_path, monkeypatch, *, shadow):
    outside = tmp_path / 'external launch directory'
    outside.mkdir()
    if shadow:
        package = outside / 'experiments'
        package.mkdir()
        (package / '__init__.py').write_text(
            "raise AssertionError('UNBOUND_CHILD_SOURCE_SHADOW')\n", encoding='utf-8')
        monkeypatch.setenv('PYTHONPATH', str(outside))
    else:
        monkeypatch.delenv('PYTHONPATH', raising=False)
    monkeypatch.chdir(outside)
    original_init = ResearchStore.__init__

    def initialize(store, root, *args, **kwargs):
        assert Path(root).resolve().is_relative_to(tmp_path.resolve()), 'non-fixture store refused'
        return original_init(store, root, *args, **kwargs)

    monkeypatch.setattr(ResearchStore, '__init__', initialize)


@pytest.mark.parametrize('shadow', [False, True], ids=['external-cwd', 'shadow-cwd-and-pythonpath'])
def test_registered_worker_binds_source_outside_repository(tmp_path, monkeypatch, shadow):
    external_launch_directory(tmp_path, monkeypatch, shadow=shadow)
    store, value, registry, recovery, _ = prepare(tmp_path / 'isolated fixture')
    assert value['code_hash'] == code_hash()
    runner = SharedRunner(store, registry, recovery)
    cell = value['cells'][0]
    result = runner.run_cell(value['study_id'], digest(cell), budget=BudgetSpec(45))
    if result['state'] != 'SUCCEEDED':
        logs = list((store.path / 'artifacts').glob('.attempt-*/worker.log'))
        pytest.fail(str(result) + '\n' + '\n'.join(path.read_text(errors='replace') for path in logs))
    payload = json.loads((store.path / 'artifacts' / result['artifact_id']).read_bytes())
    assert payload['qualification'] == 'fixture' and payload['admission_hash']
    assert payload['spec_hash'] == digest(value) and payload['cell_hash'] == digest(cell)
    events = store.events()
    started = [event for event in events if event['event_kind'] == 'WORKER_STARTED']
    stopped = [event for event in events if event['event_kind'] == 'WORKER_TREE_STOPPED']
    settled = [event for event in events if event['event_kind'] == 'SETTLE']
    assert len(started) == len(stopped) == len(settled) == 1
    assert started[0]['sequence'] < stopped[0]['sequence'] < settled[0]['sequence']
    assert stopped[0]['payload']['confirmation'] == 'native-job-or-process-group-no-running-descendants'
    assert settled[0]['payload']['charged_ms'] == result['elapsed_ms'] > 0
    before = BudgetLedger(store).balance('affine')
    reused = runner.run_cell(value['study_id'], digest(cell), budget=BudgetSpec(45))
    assert reused['reused'] and reused['artifact_id'] == result['artifact_id']
    assert BudgetLedger(store).balance('affine') == before
    assert len(store.attempts()) == 1
    assert sum(e['event_kind'] == 'WORKER_STARTED' for e in store.events()) == 1
    # Editable installs must not hide a regression to an unbound -m entry.
    entry = command(store.path / 'unused-output', value, cell)
    assert entry[0] == sys.executable
    assert Path(entry[1]).is_absolute()
    assert Path(entry[1]) == Path(__file__).resolve().parents[1] / 'experiments/pirc28/bootstrap.py'


def test_registered_resume_binds_source_despite_shadow_package(tmp_path, monkeypatch):
    external_launch_directory(tmp_path, monkeypatch, shadow=True)
    # Preserve the existing45s fixture, restored RNG/results, prior costs and
    # exact parent-child lineage assertions; do not substitute a fake worker.
    original_resume(tmp_path)
