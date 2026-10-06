"""Keyed history facts never become permission or replace fresh source I/O."""
import json
import math
from datetime import datetime, timedelta, timezone

import pytest

import application.research_preregistration as core
from infrastructure.research_history_lookup import SourceHistoryLookup
from infrastructure.research_store import ResearchError, digest
from tests.test_research_history_lookup_work import frozen_history
from tests.research_file_observation import observe_file


def report(records):
    evidence = {'records': records}
    return {'schema_version': 'pirc25-exposure-history-v1', 'source': 'synthetic history control',
            'source_evidence': evidence, 'source_evidence_hash': digest(evidence)}


@pytest.mark.parametrize('status', ['unexposed', 'exposed', 'unknown'])
@pytest.mark.parametrize('variant', ['same', 'same-content', 'same-source', 'new-window', 'old-release', 'other'])
def test_merged_source_facts_preserve_original_predicate_and_exact_coverage(status, variant):
    identity = {'dataset_id': 'fixture', 'release_id': 'v1', 'source_block_id': 'block', 'sha256': digest('target')}
    record = {**identity, 'status': status}
    if variant == 'same-content':
        record.update(dataset_id='old-dataset', source_block_id='old-window')
    elif variant == 'same-source':
        record.update(release_id='old-release', sha256=digest('changed content'))
    elif variant == 'new-window':
        record.update(source_block_id='other', sha256=digest('other'))
    elif variant == 'old-release':
        record.update(release_id='old-release')
    elif variant == 'other':
        record.update(dataset_id='other', source_block_id='other', sha256=digest('other'))
    value = report([record])
    core.validate_history(value)
    key = ('exposure-history-' + digest(value), digest(value))
    lookup = SourceHistoryLookup()
    lookup.add(key, value['source_evidence']['records'])
    assert lookup.coverage(key, identity) == (status if all(record[k] == v for k, v in identity.items()) else None)
    assert lookup.prior_exposure(identity) == (core.same_source(record, identity) and status in {'exposed', 'unknown'})


@pytest.mark.parametrize('count', [16, 256])
def test_actual_sorted_coverage_and_global_veto_key_access_is_logarithmic(count):
    identity = {'dataset_id': 'fixture', 'release_id': 'v1', 'source_block_id': 'block', 'sha256': digest('target')}
    records = [{**identity, 'status': 'unexposed'}] + [
        {'dataset_id': 'other-' + str(n), 'release_id': 'v1', 'source_block_id': 'block-' + str(n),
         'sha256': digest('other source ' + str(n)), 'status': 'unknown'} for n in range(count)]
    value = report(records)
    core.validate_history(value)
    key = ('exposure-history-' + digest(value), digest(value))
    lookup = SourceHistoryLookup()
    lookup.add(key, records)
    assert not lookup.prior_exposure(identity)
    visits = []
    class ObservedKeys(list):
        def __getitem__(self, position):
            visits.append(position)
            return super().__getitem__(position)
    keys, statuses = lookup._reports[key]
    lookup._reports[key] = (ObservedKeys(keys), statuses)
    assert lookup.coverage(key, identity) == 'unexposed'
    assert len(visits) <= math.ceil(math.log2(len(keys) + 1)) + 1
    visits.clear()
    lookup._veto_keys = ObservedKeys(lookup._veto_keys)
    assert not lookup.prior_exposure(identity)
    assert len(visits) <= 2 * (math.ceil(math.log2(len(lookup._veto_keys) + 1)) + 1)


def test_real_new_history_manifest_invalidates_owned_facts_and_denies_before_data(tmp_path, monkeypatch):
    store, ledger, _, protocol = frozen_history(tmp_path, 16)
    opened = []
    observe_file(monkeypatch, tmp_path / 'block.bin', before_open=lambda: opened.append(True))
    with store._read_transaction():
        assert ledger._read_authority(protocol, protocol['blocks'][0], 'evaluate', 'evaluate')[0]
        previous = store._event_lookup().source_history()
        value = report([{**core.source_identity(protocol['blocks'][0]), 'status': 'unknown'}])
        core.PreregistrationGate(store).import_history(value, digest(value))
        assert store._event_lookup().source_history() is not previous
        with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
            ledger.read('reserved', 'test-block', purpose='evaluate', authorization_id='evaluate', data_root=tmp_path)
    assert not opened
    assert store.events()[-1]['event_kind'] == 'EXPOSURE_DENIED'


@pytest.mark.parametrize('status', ['exposed', 'unknown'])
@pytest.mark.parametrize('alias', ['same-content', 'same-source'])
def test_actual_provider_rejects_imported_alias_exposure_before_data(tmp_path, monkeypatch, status, alias):
    store, ledger, _, protocol = frozen_history(tmp_path, 16)
    record = {**core.source_identity(protocol['blocks'][0]), 'status': status}
    if alias == 'same-content':
        record.update(dataset_id='previous-dataset', release_id='old', source_block_id='old-window')
    else:
        record.update(release_id='old', sha256=digest('different actual synthetic content'))
    assert core.same_source(record, core.source_identity(protocol['blocks'][0]))
    value = report([record])
    core.PreregistrationGate(store).import_history(value, digest(value))
    opened = []
    observe_file(monkeypatch, tmp_path / 'block.bin', before_open=lambda: opened.append(True))
    with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
        ledger.read('reserved', 'test-block', purpose='evaluate', authorization_id='evaluate', data_root=tmp_path)
    assert not opened
    assert store.events()[-1]['event_kind'] == 'EXPOSURE_DENIED'


@pytest.mark.parametrize('which', ['current', 'prior'])
def test_real_changed_history_bytes_are_not_hidden_by_owned_facts(tmp_path, monkeypatch, which):
    store, ledger, _, protocol = frozen_history(tmp_path, 16)
    opened = []
    observe_file(monkeypatch, tmp_path / 'block.bin', before_open=lambda: opened.append(True))
    with store._read_transaction():
        assert ledger._read_authority(protocol, protocol['blocks'][0], 'evaluate', 'evaluate')[0]
        object_id = ('exposure-history-' + protocol['history_hash'] if which == 'current'
                     else store._manifest_events('exposure-history-')[0]['payload']['object_id'])
        path = store.path / ('manifests/' + object_id + '.json')
        changed = json.loads(path.read_bytes())
        changed['source_evidence']['records'][0]['status'] = 'unknown'
        path.write_text(json.dumps(changed), encoding='utf-8')
        with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
            ledger.read('reserved', 'test-block', purpose='evaluate', authorization_id='evaluate', data_root=tmp_path)
    assert not opened
    assert store.events()[-1]['event_kind'] == 'EXPOSURE_DENIED'


def test_next_physical_phase_has_new_facts_and_revalidates_actual_report_schema(tmp_path, monkeypatch):
    store, ledger, _, protocol = frozen_history(tmp_path, 16)
    validator, validations, owned = core.validate_history, [], []
    def observed(value):
        validations.append(value['source_evidence_hash'])
        return validator(value)
    monkeypatch.setattr(core, 'validate_history', observed)
    for _ in range(2):
        with store._read_transaction():
            assert ledger._read_authority(protocol, protocol['blocks'][0], 'evaluate', 'evaluate')[0]
            owned.append(store._event_lookup().source_history())
        assert not hasattr(store._read_scope, 'event_lookup')
    assert owned[0] is not owned[1]
    assert len(validations) == 4


def test_owned_history_facts_cannot_authorize_a_different_expiring_grant_with_declared_clock(tmp_path, monkeypatch):
    store, ledger, _, protocol = frozen_history(tmp_path, 16)
    grant = store.authorization('evaluate')
    present = datetime.now(timezone.utc)
    grant.update(authorization_id='expired', expires_at=(present + timedelta(days=1)).isoformat())
    store.authorize(grant)
    opened = []
    observe_file(monkeypatch, tmp_path / 'block.bin', before_open=lambda: opened.append(True))
    with store._read_transaction():
        assert ledger._read_authority(protocol, protocol['blocks'][0], 'evaluate', 'evaluate')[0]
        previous = store._event_lookup().source_history()
        # Register an actually valid grant, then explicitly move only the data
        # gate's test clock. Do not forge an admitted already-expired grant.
        import application.research_data as data_core
        class Later(datetime):
            @classmethod
            def now(cls, tz=None):
                return (present + timedelta(days=2)).astimezone(tz)
        monkeypatch.setattr(data_core, 'datetime', Later)
        with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
            ledger.read('reserved', 'test-block', purpose='evaluate', authorization_id='expired', data_root=tmp_path)
        assert store._event_lookup().source_history() is previous
    assert not opened


def test_history_facts_are_not_a_memo_of_protocol_or_study_permission(tmp_path):
    store, ledger, _, protocol = frozen_history(tmp_path, 16)
    with store._read_transaction():
        assert ledger._read_authority(protocol, protocol['blocks'][0], 'evaluate', 'evaluate')[0]
        changed = {**protocol, 'study_id': 'different-study'}
        assert not ledger._read_authority(changed, protocol['blocks'][0], 'evaluate', 'evaluate')[0]


@pytest.mark.parametrize('failure_point', ['veto-update', 'report-publication'])
def test_interrupted_actual_fact_publication_cannot_turn_next_read_into_blind_permission(tmp_path, monkeypatch, failure_point):
    store, ledger, _, protocol = frozen_history(tmp_path, 16)
    identity = core.source_identity(protocol['blocks'][0])
    value = report([
        {'dataset_id': 'aaa-unrelated', 'release_id': 'v1', 'source_block_id': 'other',
         'sha256': digest('unrelated actual synthetic source'), 'status': 'unknown'},
        {**identity, 'status': 'exposed'},
    ])
    object_id = 'exposure-history-' + digest(value)
    actual_add, failed, opened = SourceHistoryLookup.add, [], []
    class ActualUpdates(set):
        def update(self, values):
            super().update(values)  # Actual derived facts, never forged markers.
            if not failed:
                failed.append(True)
                raise OSError('source fact publication interrupted')
    class ActualReports(dict):
        def __setitem__(self, key, entry):
            super().__setitem__(key, entry)
            if key[0] == object_id and not failed:
                failed.append(True)
                raise OSError('source fact publication interrupted')
    def interrupted(lookup, key, records):
        if key[0] == object_id:
            if failure_point == 'veto-update':
                lookup._veto_scopes = ActualUpdates(lookup._veto_scopes)
            else:
                lookup._reports = ActualReports(lookup._reports)
        return actual_add(lookup, key, records)
    monkeypatch.setattr(SourceHistoryLookup, 'add', interrupted)
    observe_file(monkeypatch, tmp_path / 'block.bin', before_open=lambda: opened.append(True))
    with store._read_transaction():
        assert ledger._read_authority(protocol, protocol['blocks'][0], 'evaluate', 'evaluate')[0]
        core.PreregistrationGate(store).import_history(value, digest(value))
        with pytest.raises(OSError, match='source fact publication interrupted'):
            ledger._read_authority(protocol, protocol['blocks'][0], 'evaluate', 'evaluate')
        current = store._event_lookup().source_history()
        assert not current._reports and not current._veto_scopes
        assert current._veto_keys is None
        # The same owner may handle a transient projection failure. It still
        # must not trust a half-published index on the next actual provider read.
        with pytest.raises(ResearchError, match='UNAUTHORIZED_DATA'):
            ledger.read('reserved', 'test-block', purpose='evaluate', authorization_id='evaluate', data_root=tmp_path)
    assert failed == [True]
    assert not opened
