"""Explicit orchestration seams; not domain, native, or empirical evidence."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.pirc17 import formal_partial_science as module
from experiments.pirc17.protocol_core import canonical, digest, envelope, unpack


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root = tmp_path / 'source'
    root.mkdir()
    (root / 'head.json').write_bytes(b'fixture head, not a real ledger')
    works = [dict(work_id='input', kind='input_qualification_and_population')]
    works += [dict(work_id=f'fit{i}', kind='method_fit' if i < 16 else 'terrain_fit',
                   fit_identity=f'owner{i}') for i in range(26)]
    works += [dict(work_id='forecast', kind='scientific_forecast'),
              dict(work_id='reference', kind='same_grid_reference'),
              dict(work_id='remaining', kind='scientific_forecast')]
    statuses = {w['work_id']: 'success' for w in works[:-1]}
    calls = Counter()
    artifacts, items = {}, {}
    for work in works[:-1]:
        directory = root / work['work_id']
        directory.mkdir()
        record = envelope(dict(fixture_only=True, work_id=work['work_id']))
        path = directory / 'artifact.json'
        path.write_bytes(canonical(record))
        artifacts[work['work_id']] = path, record
        items[work['work_id']] = dict(directory=directory,
            manifest=dict(artifact_path=str(path), artifact_sha256=record['sha256']),
            result_sha256=digest(['result',work['work_id']]),
            settlement_sha256=digest(['settlement',work['work_id']]),
            observation_sha256=digest(['observation',work['work_id']]), controller_elapsed_ns=1)
    context = dict(input_identity=envelope(dict(fixture_only=True)), population=envelope(dict(fixture_only=True)),
        positions=SimpleNamespace(prefixes=()), prior=None, map_catalog=envelope(dict(fixture_only=True)))
    records = {k: envelope(dict(fixture_only=True, kind=k)) for k in ('protocol','execution')}
    matrix = envelope(dict(workloads=works))
    context.update(records, matrix=matrix)
    contract = dict(runtime_manifest_sha256=digest('fixture runtime'), approval_sha256=digest('not approval'))

    class Closed:
        def __init__(self, directory, **kwargs):
            self.directory = Path(directory)
            self.root_sha256 = digest('fixture ledger')
        def disposition(self, wid):
            return statuses.get(wid, 'unattempted')
        def read(self, work):
            calls['transport'] += 1
            return items[work['work_id']]

    class Saved:
        def __init__(self, **kwargs):
            self.index = {}
            self.identity = envelope(dict(fixture_only=True))
        def admit(self, work, binding):
            self.index[work['work_id']] = binding
        def read(self, work):
            calls['forecast_domain'] += 1
            return SimpleNamespace(status='success')

    def owned(manifest, directory, *args):
        return artifacts[directory.name]
    def input_check(*args, **kwargs):
        calls['input_domain'] += 1
        return dict(details=dict(fixture_only=True)), context, {}
    def fit_check(*args, **kwargs):
        calls['fit_domain'] += 1
    monkeypatch.setattr(module.budget, 'contract_for_matrix', lambda *args, **kwargs: dict(contract))
    monkeypatch.setattr(module, 'ClosedOutputs', Closed)
    monkeypatch.setattr(module, 'input_verification', input_check)
    monkeypatch.setattr(module, '_owned_record', owned)
    monkeypatch.setattr(module, 'restore_registered_fit', fit_check)
    monkeypatch.setattr(module, 'build_origin_cases', lambda *args, **kwargs: {})
    monkeypatch.setattr(module, 'SavedForecasts', Saved)
    args = dict(directory=root, contract=contract, tip=dict(fixture_only=True), matrix=matrix,
                access_journal=root/'access', **records)
    return SimpleNamespace(args=args, works=works, statuses=statuses, calls=calls, root=root,
                           items=items, artifacts=artifacts)


def test_complete_original_prefix_is_read_without_admission_or_new_work(fixture):
    value = unpack(module.inspect_partial_science(**fixture.args))
    assert fixture.calls == dict(transport=29, input_domain=1, fit_domain=26, forecast_domain=2)
    assert value['successful_work_counts'] == dict(input_qualification_and_population=1,
        method_fit=16, terrain_fit=10, scientific_forecast=1, same_grid_reference=1)
    assert value['forecast_kind_counts'] == dict(scientific_forecast=1, same_grid_reference=1)
    assert value['work_disposition_counts'] == dict(success=29, unattempted=1)
    assert set(value['completed_sources']) == set(fixture.statuses)
    assert value['new_fits'] == value['new_forecasts'] == 0
    assert value['read_only'] is True
    for flag in ('cost_allocation_complete','cross_execution_admitted','authorizes_execution',
                 'scientific_claim_authorized','raw_sources_independently_reloaded'):
        assert value[flag] is False


@pytest.mark.parametrize('status', ['reserved','failure','interrupted','unattempted'])
def test_input_must_actually_have_settled_success(fixture,status):
    fixture.statuses['input'] = status
    with pytest.raises(ValueError, match='settled guarded input'):
        module.inspect_partial_science(**fixture.args)
    assert not fixture.calls


def test_forecasts_require_every_fit_owner_not_just_referenced_fit(fixture):
    fixture.statuses['fit25'] = 'unattempted'
    with pytest.raises(ValueError, match='all26'):
        module.inspect_partial_science(**fixture.args)


def test_missing_forecasts_stay_unattempted_and_partial_fits_are_allowed(fixture):
    for wid in ('fit25','forecast','reference'):
        fixture.statuses.pop(wid)
    value = unpack(module.inspect_partial_science(**fixture.args))
    assert len(value['fit_bindings']) == 25 and value['forecast_index'] == {}
    assert value['work_disposition_counts'] == dict(success=26, unattempted=4)


def test_later_completed_analysis_cannot_be_omitted(fixture):
    fixture.works[-1]['kind'] = 'common_scores'
    fixture.args['matrix'] = envelope(dict(workloads=fixture.works))
    fixture.statuses['remaining'] = 'success'
    with pytest.raises(ValueError, match='omit completed analysis'):
        module.inspect_partial_science(**fixture.args)


def test_wrong_contract_rejected_before_transport_or_domain_read(fixture):
    fixture.args['contract'] = dict(fixture.args['contract'], extra='wrong contract')
    with pytest.raises(ValueError, match='complete contract'):
        module.inspect_partial_science(**fixture.args)
    assert not fixture.calls


@pytest.mark.parametrize('boundary', ['input_verification','restore_registered_fit','_owned_record'])
def test_domain_or_artifact_failure_never_selects_successful_subset(fixture,monkeypatch,boundary):
    def fail(*args, **kwargs):
        raise ValueError('fixture corruption')
    monkeypatch.setattr(module,boundary,fail)
    with pytest.raises(ValueError,match='fixture corruption'):
        module.inspect_partial_science(**fixture.args)


def test_forecast_read_error_propagates(fixture,monkeypatch):
    original = module.SavedForecasts.read
    def read(self,work):
        if work['work_id'] == 'reference':
            raise ValueError('fixture array changed')
        return original(self,work)
    monkeypatch.setattr(module.SavedForecasts,'read',read)
    with pytest.raises(ValueError,match='array changed'):
        module.inspect_partial_science(**fixture.args)


def test_changed_head_invalidates_the_whole_inspection(fixture,monkeypatch):
    original = module.SavedForecasts.read
    def read(self,work):
        (fixture.root/'head.json').write_bytes(b'changed fixture head')
        return original(self,work)
    monkeypatch.setattr(module.SavedForecasts,'read',read)
    with pytest.raises(ValueError,match='head changed'):
        module.inspect_partial_science(**fixture.args)


def test_imported_source_reuses_one_scoped_bridge_for_all_model_and_forecast_readers(fixture, monkeypatch):
    """Explicit orchestration seam, NOT typed/domain/native qualification."""
    from experiments.pirc17 import formal_import_scope
    marker, calls = object(), []
    reference = dict(SOFTWARE_SCOPE_REFERENCE=True)
    original_closed = module.ClosedOutputs
    class Closed(original_closed):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.imports = SimpleNamespace(manifest=dict(import_scope=reference))
        def read(self, work):
            return dict(super().read(work), imported=True)
    monkeypatch.setattr(module, 'ClosedOutputs', Closed)
    def scope(ref, **kwargs):
        assert ref == reference
        calls.append('one_scope')
        return marker
    monkeypatch.setattr(formal_import_scope, 'ScopeBridge', scope)
    def fit(*args, **kwargs):
        assert kwargs['import_bridge'] is marker
        calls.append('fit')
    monkeypatch.setattr(module, 'restore_registered_fit', fit)
    original_saved = module.SavedForecasts
    class Saved(original_saved):
        def __init__(self, **kwargs):
            assert kwargs['import_bridge'] is marker
            calls.append('saved_collection')
            super().__init__(**kwargs)
    monkeypatch.setattr(module, 'SavedForecasts', Saved)
    value = unpack(module.inspect_partial_science(**fixture.args))
    assert Counter(calls) == {'one_scope': 1, 'fit': 26, 'saved_collection': 1}
    assert len(value['completed_sources']) == 29
