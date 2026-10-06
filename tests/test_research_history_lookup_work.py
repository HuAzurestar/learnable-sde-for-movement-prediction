"""Actual history predicate/shape work, separately from fresh physical hashing."""
import pytest

from application.research_data import EvaluationExposureLedger
import application.research_preregistration as core
from infrastructure.research_store import digest
from tests.test_research_input_gates import final_eval, freeze_evidence, authorize
from tests.research_file_observation import observe_file


def frozen_history(tmp_path, count):
    store, content, protocol = final_eval(tmp_path, None)
    report = {'records': [{'dataset_id': 'other-dataset-' + str(n), 'release_id': 'v1',
                          'source_block_id': 'other-block-' + str(n),
                          'sha256': digest('other synthetic source ' + str(n)), 'status': 'unexposed'}
                         for n in range(count)]}
    history = {'schema_version': 'pirc25-exposure-history-v1', 'source': 'synthetic external ledger',
               'source_evidence': report, 'source_evidence_hash': digest(report)}
    core.PreregistrationGate(store).import_history(history, digest(history))
    protocol, _, _ = freeze_evidence(store, protocol)
    ledger = EvaluationExposureLedger(store)
    ledger.register_protocol(protocol, digest(protocol))
    authorize(store, protocol)
    return store, ledger, content, protocol


@pytest.mark.parametrize('count', [16, 64])
def test_repeated_actual_authority_checks_do_not_rewalk_unchanged_history_records(tmp_path, monkeypatch, count):
    store, ledger, _, protocol = frozen_history(tmp_path, count)
    initial = len(store.events())
    predicate, validator, original_json = core.same_source, core.validate_history, store._json
    comparisons, validations, manifests, event_reads, data_opens = [], [], [], [], []
    def compare(left, right):
        comparisons.append((left['dataset_id'], right['dataset_id']))
        return predicate(left, right)
    def validate(value):
        validations.append(value['source_evidence_hash'])
        return validator(value)
    def metadata(path):
        value = original_json(path)
        if path.parent == store.path / 'manifests':
            manifests.append(path.name)
        elif path.parent == store.path / 'events':
            event_reads.append(path.name)
        return value
    monkeypatch.setattr(core, 'same_source', compare)
    monkeypatch.setattr(core, 'validate_history', validate)
    monkeypatch.setattr(store, '_json', metadata)
    observe_file(monkeypatch, tmp_path / 'block.bin', before_open=lambda: data_opens.append(True))
    with store._read_transaction():
        for _ in range(24):
            allowed, _, _ = ledger._read_authority(protocol, protocol['blocks'][0], 'evaluate', 'evaluate')
            assert allowed
    assert not data_opens
    assert len(manifests) == 192, 'Every original manifest/authority hash check remains physical'
    assert sum(name.startswith('exposure-history-') for name in manifests) == 72
    assert len(event_reads) == 2 * initial, 'Both uncached complete physical prefix validations remain'
    assert len(comparisons) <= 48 and len(validations) <= 2, (
        f'24 checks repeated {len(comparisons)} real comparisons/{len(validations)} full history validations '
        f'for {count} unrelated records; only keyed metadata facts may be shared, not permission')
