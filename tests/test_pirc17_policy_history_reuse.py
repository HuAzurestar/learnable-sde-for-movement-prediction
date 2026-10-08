"""Routing/closing negative software seams; not actual history admission."""
from copy import deepcopy

import pytest

from experiments.pirc17 import formal_resource_predecessor as resource
from experiments.pirc17 import formal_recovery_policy as recovery
from experiments.pirc17 import formal_continuation_policy as continuation
from experiments.pirc17 import formal_resource_policy as policy
from experiments.pirc17.protocol_core import digest, envelope, unpack


@pytest.fixture(params=[(recovery, resource.RECOVERY_VERSION),
                        (continuation, resource.CONTINUATION_VERSION)])
def scoped(request, monkeypatch):
    module, version = request.param
    previous = envelope(dict(ledger_directory='original-science',
        science_snapshot=dict(path='metadata-source'), software_only=True))
    ref = dict(path='software-predecessor', content_sha256=previous['sha256'],
        file_sha256=digest('software bytes'))
    costs = dict(ledger_root_sha256=digest('original scientific root'),
        execution_sha256=digest('original scientific execution'),
        approval_sha256=digest('original scientific approval'))
    p = dict(schema_version=module.VERSION, previous_binding_reference=ref,
        scientific_source_root_sha256=costs['ledger_root_sha256'],
        scientific_source_execution_sha256=costs['execution_sha256'],
        scientific_source_approval_sha256=costs['approval_sha256'])
    history = (unpack(previous), costs, dict(software_science=True),
        dict(software_bundle=True), dict(software_contract=True))
    payload = dict(schema_version=version, ledger_directory='original-science',
        resource_policy=envelope(p), science_snapshot=unpack(previous)['science_snapshot'],
        science_predecessor=ref, read_only=True, authorizes_execution=False)
    calls = []

    def validator(record):
        assert record == payload['resource_policy']
        calls.append('complete-policy-and-history')
        return p, history

    def load(reference):
        assert reference == ref
        calls.append('post-validation-reference-bytes')
        return previous, unpack(previous)

    monkeypatch.setattr(module, '_validate_policy_and_history', validator)
    monkeypatch.setattr(policy, '_load_reference', load)
    # Any recursive reconstruction after the complete validator is a defect.
    def repeat(record):
        pytest.fail('already-validated private history reconstructed again')
    monkeypatch.setattr(resource, '_history', repeat)
    return dict(payload=payload, previous=previous, calls=calls)


def test_reuses_only_this_validation_and_keeps_post_validation_reference_check(scoped):
    result = resource._recovery_history(envelope(scoped['payload']))
    assert result[0] == scoped['payload']
    assert result[1]['ledger_root_sha256'] == digest('original scientific root')
    assert scoped['calls'] == ['complete-policy-and-history', 'post-validation-reference-bytes']


def test_changed_complete_predecessor_is_rejected_after_validation(scoped):
    # Change meaningful data, not just its claimed header checksum.
    # Replace the caller-facing payload without aliasing the validated tuple.
    scoped['previous']['payload'] = dict(scoped['previous']['payload'], extra='changed')
    scoped['previous']['sha256'] = digest(scoped['previous']['payload'])
    with pytest.raises(ValueError, match='cannot relabel'):
        resource._recovery_history(envelope(scoped['payload']))


@pytest.mark.parametrize('field,value', [('ledger_directory', 'latest-accounting-not-science'),
                                       ('science_snapshot', dict(path='wrong-science'))])
def test_checked_history_cannot_relabel_scientific_owner(scoped, field, value):
    payload = deepcopy(scoped['payload'])
    payload[field] = value
    with pytest.raises(ValueError, match='cannot relabel'):
        resource._recovery_history(envelope(payload))
