"""Synthetic floor/routing checks, never source admission or launch authority.

The private history seam is explicit. Complete floor predicates and physical
reference readers are unchanged; full-size historical regressions are separate.
"""
from copy import deepcopy

import pytest

from experiments.pirc17 import formal_entrypoint as entry, formal_partial_launch as launch
from experiments.pirc17 import formal_partial_predecessor as partial
from experiments.pirc17 import formal_resource_predecessor as resource
from experiments.pirc17.protocol_core import digest, envelope, publish, unpack


@pytest.fixture
def sample(tmp_path, monkeypatch):
    protocol, matrix = envelope({'synthetic': 'protocol'}), envelope({'synthetic': 'matrix'})
    inputs = {'trajectory_path': str((tmp_path/'ABSENT-scientific-data').resolve())}
    original = dict(protocol=protocol, matrix=matrix, runtime=envelope(dict(input_paths=inputs)))
    old_contract = dict(workloads=[dict(work_id=digest('synthetic-work'))])
    cost = dict(protocol_sha256=protocol['sha256'], matrix_sha256=matrix['sha256'],
        ledger_directory=str((tmp_path/'ABSENT-original').resolve()),
        execution_sha256=digest('old-execution'), approval_sha256=digest('old-approval'))
    policy = envelope(dict(phase_caps_ns={'input': 7200}, total_cap_ns=7200, effective_generation_limit=2,
                           cost_reference={'synthetic_latest': True}))
    binding = envelope(dict(schema_version=resource.METADATA_VERSION, resource_policy=policy))
    path, _ = publish(tmp_path/'binding', unpack(binding))
    reference = partial.reference(path)
    runtime = envelope(dict(schema_version=entry.RESOURCE_VERSION+'-runtime',
        protocol_sha256=protocol['sha256'], matrix_sha256=matrix['sha256'],
        ledger_directory=str((tmp_path/'ABSENT-successor').resolve()), predecessor=binding,
        predecessor_reference=reference, input_paths=deepcopy(inputs)))
    contract = dict(runtime_manifest_sha256=runtime['sha256'], protocol_sha256=protocol['sha256'],
        matrix_sha256=matrix['sha256'], ledger_directory=unpack(runtime)['ledger_directory'],
        resource_policy=policy, execution_sha256=digest('new-execution'), approval_sha256=digest('new-approval'),
        phase_caps_ns={'input':7200}, total_cap_ns=7200, workloads=deepcopy(old_contract['workloads']),
        max_generated_forecasts=2, max_attempts_per_item=1)
    saved = {'synthetic': 'UNADMITTED-scientific-index'}
    calls = []

    def history(value):
        assert value == binding
        calls.append(value['sha256'])
        return unpack(value), cost, saved, original, old_contract

    monkeypatch.setattr(resource, '_history', history)  # Explicit synthetic history seam.
    return dict(protocol=protocol, matrix=matrix, inputs=inputs, original=original, old_contract=old_contract,
                cost=cost, policy=policy, binding=binding, path=path, runtime=runtime,
                contract=contract, saved=saved, calls=calls)


def validate(f):
    return launch.validate_partial_runtime(f['runtime'], f['contract'], protocol=f['protocol'], matrix=f['matrix'])


def test_resource_runtime_validates_history_once_and_rereads_physical_binding(sample, monkeypatch):
    f = sample
    original_load = partial._load
    loaded = []
    def load(ref):
        loaded.append(ref)
        return original_load(ref)
    monkeypatch.setattr(partial, '_load', load)
    # Calling the old redundant predecessor._history would reach this sentinel.
    monkeypatch.setattr(partial, '_history', lambda *args: pytest.fail('duplicate resource history'))
    assert validate(f) == unpack(f['runtime'])
    assert f['calls'] == [f['binding']['sha256']]
    assert loaded == [unpack(f['runtime'])['predecessor_reference']]


def test_public_floor_still_returns_three_values_and_same_meaning(sample):
    f = sample
    assert resource.resource_floor(f['runtime'], f['contract']) == (f['binding'], f['cost'], f['saved'])
    assert f['calls'] == [f['binding']['sha256']]


@pytest.mark.parametrize('key', ['protocol_sha256', 'matrix_sha256', 'ledger_directory', 'resource_policy',
    'execution_sha256', 'approval_sha256', 'phase_caps_ns', 'total_cap_ns', 'workloads',
    'max_generated_forecasts', 'max_attempts_per_item', 'runtime_manifest_sha256'])
def test_every_existing_floor_scope_predicate_remains_required(sample, key):
    f = sample
    contract = deepcopy(f['contract'])
    if key in {'execution_sha256', 'approval_sha256', 'ledger_directory'}:
        contract[key] = f['cost'][key]
    elif key == 'resource_policy': contract[key] = envelope({'synthetic': 'wrong-policy'})
    elif key == 'phase_caps_ns': contract[key]['input'] += 1
    elif key == 'workloads': contract[key] = []
    elif key in {'total_cap_ns', 'max_generated_forecasts', 'max_attempts_per_item'}: contract[key] += 1
    else: contract[key] = digest('wrong-scope')
    with pytest.raises(ValueError):
        resource._resource_floor_and_history(f['runtime'], contract)


@pytest.mark.parametrize('changed', ['protocol', 'matrix', 'input_paths', 'physical_bytes', 'reference_content'])
def test_reused_history_cannot_skip_runtime_or_physical_reference_checks(sample, changed):
    f = sample
    if changed in {'protocol', 'matrix'}:
        f[changed] = envelope({'synthetic': 'different-'+changed})
    elif changed == 'physical_bytes':
        # Equal JSON meaning with different bytes must still fail the pin.
        f['path'].write_bytes(b' '+f['path'].read_bytes())
    else:
        runtime = deepcopy(unpack(f['runtime']))
        if changed == 'input_paths': runtime['input_paths']['trajectory_path'] += '-changed'
        else: runtime['predecessor_reference']['content_sha256'] = digest('wrong-reference-content')
        f['runtime'] = envelope(runtime)
        f['contract']['runtime_manifest_sha256'] = f['runtime']['sha256']
    with pytest.raises(ValueError): validate(f)


def test_history_is_fresh_each_invocation_not_a_global_success_cache(sample):
    f = sample
    assert validate(f) == unpack(f['runtime'])
    f['original']['protocol'] = envelope({'synthetic': 'changed-source'})
    with pytest.raises(ValueError): validate(f)
    assert len(f['calls']) == 2


@pytest.mark.parametrize('version', [resource.RECOVERY_VERSION, resource.CONTINUATION_VERSION])
def test_latest_accounting_floor_does_not_relabel_original_scientific_history(sample, monkeypatch, version):
    f = sample
    runtime = deepcopy(unpack(f['runtime']))
    f['binding']['payload']['schema_version'] = version
    f['binding']['sha256'] = digest(f['binding']['payload'])
    runtime['predecessor'] = f['binding']
    record = envelope(runtime)
    f['contract']['runtime_manifest_sha256'] = record['sha256']
    latest = dict(f['cost'], latest_accounting=True)
    loaded = []
    def load(ref):
        loaded.append(ref)
        return envelope(latest), latest
    monkeypatch.setattr(resource.policy, '_load_reference', load)  # Explicit synthetic latest-cost seam.
    binding, floor_cost, saved, original, old_contract = resource._resource_floor_and_history(record, f['contract'])
    assert binding == f['binding'] and floor_cost == latest and saved == f['saved']
    assert original is f['original'] and old_contract is f['old_contract']
    assert loaded == [unpack(f['policy'])['cost_reference']] and len(f['calls']) == 1


def test_legacy_partial_runtime_keeps_original_floor_then_physical_then_history_route(sample, monkeypatch):
    f = sample
    runtime = deepcopy(unpack(f['runtime']))
    runtime['schema_version'] = entry.PARTIAL_VERSION+'-runtime'
    events = []
    def floor(record, contract):
        events.append('legacy-floor')
        return f['binding'], f['cost'], f['saved']
    load = partial._load
    def physical(ref):
        events.append('physical-binding')
        return load(ref)
    def history(binding):
        events.append('legacy-history')
        assert binding == f['binding']
        return unpack(binding), f['cost'], f['saved'], f['original'], f['old_contract']
    monkeypatch.setattr(launch, '_floor', floor)  # Explicit legacy routing-only seam.
    monkeypatch.setattr(partial, '_load', physical)
    monkeypatch.setattr(partial, '_history', history)
    monkeypatch.setattr(resource, '_resource_floor_and_history', lambda *args: pytest.fail('resource path on legacy'))
    assert launch.validate_partial_runtime(envelope(runtime), f['contract'], protocol=f['protocol'], matrix=f['matrix']) == runtime
    assert events == ['legacy-floor', 'physical-binding', 'legacy-history']
