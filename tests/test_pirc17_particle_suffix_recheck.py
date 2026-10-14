"""Ancestral map-use boundaries on the actual kernel and software data only."""
import json

import pytest

from tests.test_pirc17_particle_suffix_execution import (
    base_inputs, contract_inputs, inputs, candidate_bytes, evidence, reference,
    case, ready, closed_parent, source, reserve, run, seal, interrupt_after,
    execution, lineage, science)


@pytest.mark.parametrize('committed_before_stop', [4, 24], ids=['new-work', 'audit-only'])
def test_core_rechecks_ancestor_only_map_assets_after_whole_science(
        source, inputs, monkeypatch, committed_before_stop):
    tracker = inputs[1]
    tracker['include_extra_map'] = True
    first = reserve(source)
    with pytest.raises(KeyboardInterrupt):
        run(first, progress=interrupt_after(committed_before_stop))
    assert seal(first)['continuable']
    observation = json.loads((first['attempt']/'work/map-observations/000002.json').read_text())
    extra = tracker['extra_asset']
    assert extra.name in observation['map_identity']['verified_assets']
    inherited_inventory = lineage.storage.inventory(first['attempt'])

    # The next attempt never visits this asset. It remains a dependency of
    # the complete ensemble through the earlier committed child forecasts.
    tracker['include_extra_map'] = False
    second = reserve(source)
    original, replay_calls = science.whole_audit, []

    def mutate_after_replay(*args, **kwargs):
        report = original(*args, **kwargs)
        replay_calls.append(1)
        extra.write_bytes(b'late mutation of an ancestor-only map asset')
        return report

    monkeypatch.setattr(science, 'whole_audit', mutate_after_replay)
    result = run(second)
    assert replay_calls == [1]
    assert result['status'] == 'failed', result
    assert any(error['error_type'] == 'ValueError'
               and 'observed suffix map asset changed' in error['error_message']
               for error in result['errors'])
    assert result['whole_science_sha256'] is None
    assert not (second['attempt']/'work/whole-science.json').exists()
    assert result['inherited_success_count'] == committed_before_stop
    assert result['new_committed_count'] == 24-committed_before_stop
    assert result['new_failure_count'] == result['missing_completed_record_count'] == 0
    assert len(tracker['suffix_calls']) == 24
    if committed_before_stop == 24:
        assert result['new_attempt_maps'] is None
    else:
        assert extra.name not in result['new_attempt_maps']['verified_assets']
    assert lineage.storage.inventory(first['attempt']) == inherited_inventory
    closure = seal(second, worker_returncode=1, interrupted=False)
    assert closure['status'] == 'failed' and not closure['continuable']
    with pytest.raises(ValueError, match='no automatic retry'):
        reserve(source)
