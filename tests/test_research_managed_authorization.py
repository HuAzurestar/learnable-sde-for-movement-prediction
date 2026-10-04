"""Actual comparison CLI publication must recheck the frozen export grants."""
import json

import pytest

import application.research_comparison as comparison
import infrastructure.research_store as store_module
from application.research_budget import BudgetLedger
from experiments.pirc25.__main__ import main
from infrastructure.research_store import encode
from tests.test_research_comparison import comparison_source, compute
from tests.test_research_export_authorization import permission_clock


@pytest.mark.parametrize('phase', ['fsync-expiry', 'fsync-grant-change', 'identical-expiry',
                                  'after-final-publication'])
def test_actual_managed_cli_denies_changed_permission_at_publication(
        tmp_path, monkeypatch, capsys, permission_clock, phase):
    source = comparison_source(tmp_path / 'fixture')
    store, _, grant, paper, package = source
    computed = compute(source)
    (tmp_path / 'managed-initial-computation-observed.json').write_bytes(encode(computed))
    assert computed['state'] == 'SUCCEEDED' and computed['exit_code'] == 0
    before, attempts = BudgetLedger(store).balance('affine'), store.attempts()
    directory = tmp_path / 'output'
    directory.mkdir()
    target = directory / ('manifest.json' if phase == 'after-final-publication' else 'aggregate.json')
    old = None
    if phase == 'identical-expiry':
        private = store.path / 'artifacts' / ('.comparison-' + computed['attempt_id'])
        old = (private / target.name).read_bytes()
        target.write_bytes(old)
    original_write, original_fsync, touched = comparison.atomic_write, store_module.os.fsync, []

    def change_permission():
        if not touched:
            touched.append(True)
            if phase == 'fsync-grant-change':
                (store.path / 'manifests/authorization-viewer.json').write_bytes(
                    encode({**grant, 'purposes': []}))
            else:
                permission_clock[0] = True

    def publish(path, content, *args, **kwargs):
        if path != target:
            return original_write(path, content, *args, **kwargs)
        if phase == 'after-final-publication':
            result = original_write(path, content, *args, **kwargs)
            assert path.read_bytes() == content
            change_permission()
            return result

        def fsync(fd):
            original_fsync(fd)
            # Forward the real staging fsync before changing the real grant or
            # clock. Do not invent an error, replace a result, or bypass I/O.
            change_permission()

        with monkeypatch.context() as scoped:
            scoped.setattr(store_module.os, 'fsync', fsync)
            return original_write(path, content, *args, **kwargs)

    monkeypatch.setattr(comparison, 'atomic_write', publish)
    arguments = ['--root', str(store.path.parent), '--store-id', store.store_id,
        'compare', 'synthetic', package['aggregate_hash'], '--paper-root', str(paper),
        '--authorization-id', 'viewer', '--seconds', '10', '--output', str(directory)]
    exit_code = main(arguments)
    response = json.loads(capsys.readouterr().out)
    observed = {'arguments': arguments, 'exit_code': exit_code, 'response': response,
        'phase': phase, 'touched': list(touched), 'target_exists': target.exists(),
        'staging': [path.name for path in directory.glob('.*.staging')],
        'budget_unchanged': BudgetLedger(store).balance('affine') == before,
        'attempts_unchanged': store.attempts() == attempts}
    (tmp_path / 'managed-publication-authorization-observed.json').write_bytes(encode(observed))
    assert touched == [True], observed
    assert exit_code == 1 and response['error']['code'] == 'UNAUTHORIZED_DATA', observed
    assert not observed['staging'] and observed['budget_unchanged'] and observed['attempts_unchanged'], observed
    if phase == 'after-final-publication':
        # Preserve the previously permitted immutable files; deny the result
        # return after permission expires rather than deleting old evidence.
        assert target.exists()
    elif old is not None:
        assert target.read_bytes() == old
    else:
        assert not target.exists(), observed
