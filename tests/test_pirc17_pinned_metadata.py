"""Bounded metadata/software seams only; not real native/domain admission."""
from copy import deepcopy
import os
from types import SimpleNamespace

import pytest

from experiments.pirc17 import formal_pinned_metadata as module
from experiments.pirc17 import formal_partial_imports as imports
from experiments.pirc17.protocol_core import canonical, digest, envelope, file_hash, publish


def saved(tmp_path):
    path, record = publish(tmp_path/'owned', dict(entries={str(i):dict(tags=['software', i]) for i in range(100)}))
    ref = dict(path=str(path), content_sha256=record['sha256'], file_sha256=file_hash(path))
    return path, record, ref


def test_parse_once_but_recheck_every_byte_on_every_read(tmp_path, monkeypatch):
    path, record, ref = saved(tmp_path)
    reader = module.PinnedMetadata(ref, root=path.parent)
    # Parse and content validation happened once from the same bounded bytes.
    monkeypatch.setattr(module, 'decode', lambda *a: pytest.fail('reparsed inventory'))
    monkeypatch.setattr(module, 'unpack', lambda *a, **kw: pytest.fail('rehashed whole payload'))
    for _ in range(4):
        actual, payload = reader.read(ref)
        assert actual == record and payload == record['payload']
    before = path.stat()
    raw = path.read_bytes()
    changed = raw.replace(b'"software"', b'"changed!"', 1)  # Same length, restored mtime.
    assert len(changed) == len(raw) and changed != raw
    path.write_bytes(changed)  # Deliberate corruption confined to pytest tempdir.
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match='bytes changed'):
        reader.read(ref)


def test_owned_envelope_fast_path_cannot_cache_a_mutable_header_or_alias(tmp_path, monkeypatch):
    _, original, _ = saved(tmp_path)
    owned = module.OwnedRecord(original)
    expected_bytes = canonical(original)
    assert module.record_bytes(owned) == expected_bytes
    assert module.record_payload(owned) == original['payload']
    original['payload']['entries']['0']['tags'].append('changed caller')
    assert module.record_bytes(owned) != canonical(original)
    one, two = module.mutable_record(owned), module.mutable_record(owned)
    one['payload']['entries']['0']['tags'].append('legacy deserializer mutation')
    assert two['payload']['entries']['0']['tags'] == ['software', 0]
    assert module.record_payload(owned)['entries']['0']['tags'] == ['software', 0]
    with pytest.raises(ValueError, match='content identity'):
        module.record_payload(original)
    with pytest.raises(TypeError): owned._encoded = b'changed'
    with pytest.raises(TypeError): owned.__init__(original)
    monkeypatch.setattr(module, 'canonical', lambda *a:pytest.fail('serialized owned envelope again'))
    monkeypatch.setattr(module, 'unpack', lambda *a,**kw:pytest.fail('unpacked owned envelope again'))
    assert module.record_bytes(owned) == expected_bytes
    assert module.record_payload(owned)['entries']['0']['tags'] == ['software', 0]


def test_forecast_scope_does_not_reparse_owned_fit_training_rows(monkeypatch):
    from experiments.pirc17.formal_forecast_records import forecast_scope
    fit = envelope(dict(parameter_identity=digest('software parameters'),
        training=[dict(software_only=True, row=i) for i in range(1000)]))
    own = module.OwnedRecord(fit)
    args = dict(protocol=dict(sha256=digest('protocol')), execution=dict(sha256=digest('execution')),
        matrix=dict(sha256=digest('matrix')), input_identity=envelope(dict(approval_sha256=digest('approval'),
        population_sha256=digest('population'))), case=None)
    work = dict(work_id=digest('work'), origin_mode='software', fit_identity='software', scientific=True)
    expected = forecast_scope(work, fit_receipt=fit, **args)
    actual_unpack = module.unpack
    def only_mutable_small(record):
        if record is own: pytest.fail('rehashed immutable full fit per forecast')
        return actual_unpack(record)
    monkeypatch.setattr(module, 'unpack', only_mutable_small)
    for _ in range(10): assert forecast_scope(work, fit_receipt=own, **args) == expected
    fit['payload']['training'][0]['row'] += 1
    with pytest.raises(ValueError, match='content identity'):
        forecast_scope(work, fit_receipt=fit, **args)


def test_bridge_fit_source_reuses_parse_but_rechecks_pinned_bytes(tmp_path, monkeypatch):
    from experiments.pirc17.formal_import_scope import ScopeBridge
    path, record = publish(tmp_path, dict(schema_version='SYNTHETIC metadata only', model={'p':1.0}))
    artifact = dict(path=path.name, content_sha256=record['sha256'], file_sha256=file_hash(path))
    work = dict(work_id=digest('software fit'), kind='method_fit')
    bridge = ScopeBridge.__new__(ScopeBridge)
    bridge.work = {work['work_id']:work}
    bridge.source = dict(ledger_directory=str(tmp_path))
    bridge.science = dict(completed_sources={work['work_id']:dict(artifact=artifact)})
    bridge._fit_sources = {}
    monkeypatch.setattr(bridge, '_nested_bridge', lambda *a:None)  # No typed/native admission asserted.
    ref, first, completed = bridge._source(work)
    monkeypatch.setattr(module, 'decode', lambda *a:pytest.fail('reparsed cached fit source'))
    assert bridge._source(work)[1] is first
    assert len(bridge._fit_sources) == 1
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='bytes changed'): bridge._source(work)


@pytest.mark.parametrize('change', ['path', 'file_sha256', 'content_sha256'])
def test_reference_changes_never_inherit_cached_trust(tmp_path, change):
    path, _, ref = saved(tmp_path)
    reader = module.PinnedMetadata(ref, root=path.parent)
    altered = dict(ref, **{change:str(path.parent/'other') if change == 'path' else digest(change)})
    with pytest.raises(ValueError, match='reference changed'):
        reader.read(altered)


def test_no_mutable_caller_or_returned_alias(tmp_path):
    path, record, ref = saved(tmp_path)
    reader = module.PinnedMetadata(ref, root=path.parent)
    current, payload = reader.read(ref)
    record['payload']['entries']['0']['tags'].append('external mutation')
    assert canonical(current) != canonical(record)
    with pytest.raises(TypeError): payload['entries']['0']['tags'].append('mutate cached')
    with pytest.raises(TypeError): current['payload'] = {}
    with pytest.raises(TypeError): reader.reference.update(file_sha256=digest('replace'))
    with pytest.raises(TypeError): reader.reference = dict(ref)
    with pytest.raises(TypeError): del reader._record
    with pytest.raises(TypeError): reader.__init__(ref, root=path.parent)
    assert deepcopy(payload) is payload


@pytest.mark.parametrize('bad', ['content', 'bytes', 'extra_field', 'outside', 'missing', 'oversize'])
def test_initial_pin_path_and_size_fail_closed(tmp_path, monkeypatch, bad):
    path, _, ref = saved(tmp_path)
    root = path.parent
    if bad == 'content': ref['content_sha256'] = digest('not content')
    elif bad == 'bytes': ref['file_sha256'] = digest('not bytes')
    elif bad == 'extra_field': ref['producer_pass'] = True
    elif bad == 'outside': root = tmp_path/'different-root'
    elif bad == 'missing': path.unlink()
    else: monkeypatch.setattr(module, 'MAX_ROOT_BYTES', 10)
    with pytest.raises(ValueError): module.PinnedMetadata(ref, root=root)


def test_duplicate_json_is_not_admitted_even_with_correct_byte_pin(tmp_path):
    path = tmp_path/'duplicate.json'
    path.write_bytes(b'{"payload":{},"payload":{},"sha256":"' + digest({}).encode() + b'"}')
    ref = dict(path=str(path), content_sha256=digest({}), file_sha256=file_hash(path))
    with pytest.raises(ValueError, match='duplicate'):
        module.PinnedMetadata(ref, root=tmp_path)


def test_per_item_reader_still_verifies_owned_bytes_and_original_completion(tmp_path, monkeypatch):
    source = dict(result_sha256=digest('result'), settlement_sha256=digest('settlement'),
        observation_sha256=digest('observation'), controller_elapsed_ns=42)
    path, record = publish(tmp_path, dict(imported_entries={'work':dict(directory=str(tmp_path/'item'),
        manifest=dict(artifact_sha256=digest('software artifact')), artifacts={'item':dict(bytes=1)},
        source_completion=source)}))
    ref = dict(path=str(path), content_sha256=record['sha256'], file_sha256=file_hash(path))
    reader = imports.ImportedOutputs.__new__(imports.ImportedOutputs)
    reader.reference = ref
    reader._manifest_reader = module.PinnedMetadata(ref, root=tmp_path)
    reader.record, reader.manifest = reader._manifest_reader.read(ref)
    reader.closed = SimpleNamespace(_state=SimpleNamespace(partial_imports=dict(software_only=True)))
    calls = []
    original = dict(source, directory=tmp_path/'source')
    reader.original = SimpleNamespace(read=lambda work:(calls.append('original bytes') or original),
        read_metadata=lambda work:(calls.append('original metadata') or original))
    monkeypatch.setattr(imports, '_files', lambda entry, work:calls.append('owned bytes'))
    work = dict(work_id='work')  # Explicit native/domain seam, not scientific PASS.
    assert reader.read(work)['imported'] is True
    assert calls == ['owned bytes', 'original bytes']
    calls.clear()
    result = reader.read_metadata(work)
    assert result['metadata_only'] is True and not result['owned_source_bytes_rechecked']
    assert calls == ['original metadata']
    original['result_sha256'] = digest('changed original completion')
    with pytest.raises(ValueError, match='completion differs'): reader.read(work)
    path.write_bytes(path.read_bytes() + b' ')
    calls.clear()
    with pytest.raises(ValueError, match='bytes changed'): reader.read(work)
    assert calls == []
