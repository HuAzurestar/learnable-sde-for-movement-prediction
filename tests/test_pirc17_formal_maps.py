"""Temporary metadata/raw-byte fixtures, never real map/data authorization."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.pirc17 import formal_maps as module
from experiments.pirc17.protocol_core import canonical, digest, envelope, file_hash, unpack
from tests.test_pirc17_formal_inputs import qualified_fixture


def map_fixture(root):
    snapshot, raw = root/'snapshot', root/'raw'
    snapshot.mkdir(); raw.mkdir()
    entries, parents = [], []
    for i, split in enumerate(('train', 'validation', 'final_eval')):
        relative = f'srtm/N{10+i}E010.hgt.gz'
        target = raw/relative
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(b'SOFTWARE RAW-BYTE IDENTITY ONLY '+bytes([i]))
        parent = f'registered-file:{relative}:sha256:{file_hash(target)}'
        parents.append(parent)
        table = pa.table({module.PARENT_COLUMNS[0]: [parent, None],
                          module.PARENT_COLUMNS[1]: [None, 'receipt-asset:already-admitted'],
                          'numeric_feature': ['MUST NOT DECODE', 'MUST NOT DECODE']})
        path = snapshot/f'{split}.parquet'
        pq.write_table(table, path)
        entries.append(dict(path=path.name, sha256=file_hash(path), row_count=2, split=split, file_id=split))
    inventory = hashlib.sha256()
    for e in sorted(entries, key=lambda r: r['path']):
        inventory.update(f"{e['path']}\0{e['sha256']}\0{e['row_count']}\n".encode())
    spec = {'feature_spec_id': 'synthetic-spec'}
    manifest = dict(status='valid', dataset_id='synthetic', snapshot_id='synthetic',
                    feature_spec_id='synthetic-spec', feature_spec_sha256=digest(spec),
                    files=entries, content_inventory_sha256=inventory.hexdigest())
    for name, value in [('manifest.json', manifest), ('feature_spec.json', spec)]:
        (snapshot/name).write_bytes(canonical(value))
    receipt = raw/'receipt.json'
    receipt.write_bytes(canonical({'assets': []}))
    backend, modules = module.resolve_map_backend('multicell')
    query = backend(raw, [receipt])
    identity = query.identity
    query.close()
    policy = dict(receipt_sha256={'receipt.json': file_hash(receipt)}, admitted_receipt_assets_sha256={},
                  **{k: identity[k] for k in ('query_version', 'raster_cache_limits', 'geometry_cache_limits')})
    binding = dict(dataset_id='synthetic', online_maps=policy, snapshot=dict(snapshot_id='synthetic',
        manifest_sha256=file_hash(snapshot/'manifest.json'), feature_spec_id='synthetic-spec',
        feature_spec_file_sha256=file_hash(snapshot/'feature_spec.json'), feature_spec_content_sha256=digest(spec),
        content_inventory_sha256=inventory.hexdigest(), files=3))
    protocol = envelope({'fixture': 'NOT FORMAL AUTHORIZATION', 'dataset_inputs': binding})
    execution = envelope({'source_sha256': {'DSDE-SDE/'+m.__name__.replace('.', '/')+'.py': file_hash(m.__file__) for m in modules}})
    access = SimpleNamespace(access_kind='final_eval_features', population_sha256=digest('SOFTWARE POPULATION'),
        protocol_sha256=protocol['sha256'], execution_sha256=execution['sha256'], legacy_cohort_ack='synthetic',
        approval_sha256=digest('NO HUMAN APPROVAL'), access_started_sha256=digest('SOFTWARE ACCESS'))
    return access, dict(protocol=protocol, execution=execution, snapshot=snapshot, data_root=raw), parents


def test_all_snapshot_parents_bound_without_target_or_numeric_decode(tmp_path, monkeypatch):
    access, args, parents = map_fixture(tmp_path)
    original = pq.ParquetFile.iter_batches
    columns = []
    def observe(self, *a, **kw):
        columns.append(kw['columns'])
        return original(self, *a, **kw)
    monkeypatch.setattr(pq.ParquetFile, 'iter_batches', observe)
    owner = module._prepare_maps(access, **args)
    try:
        catalog = unpack(owner.catalog)
        assert set(catalog['registered_parent_witnesses']) == set(parents)
        assert catalog['snapshot_files_inspected'] == 3 and catalog['snapshot_rows_inspected'] == 6
        assert columns == [list(module.PARENT_COLUMNS)]*3
        assert catalog['raw_asset_value_queries'] == 0 and owner.observation()['identity']['verified_assets'] == {}
        fresh = owner.fresh_provider()
        try:
            assert fresh.catalog == owner.catalog and fresh.query is not owner.query
            relative = next(iter(owner.query.assets))
            owner.query._path(owner.query.assets[relative])  # Actual byte SHA check, no raster decoding.
            assert list(owner.observation()['identity']['verified_assets']) == [relative]
            assert fresh.observation()['identity']['verified_assets'] == {}
            owner.query.assets[relative]['checksum']['value'] = '0'*64
            with pytest.raises(ValueError, match='admitted static catalog'): owner.observation()
        finally:
            fresh.close()
    finally:
        owner.close(); owner.close()
    with pytest.raises(ValueError, match='closed'): owner(np.zeros((1, 2)))


@pytest.mark.parametrize('change', ['feature-bytes', 'receipt-bytes', 'source', 'spec', 'access', 'population'])
def test_changed_binding_fails_before_provider_use(tmp_path, change):
    access, args, _ = map_fixture(tmp_path)
    if change == 'feature-bytes': (args['snapshot']/'train.parquet').write_bytes(b'changed')
    elif change == 'receipt-bytes': (args['data_root']/'receipt.json').write_bytes(b'changed')
    elif change == 'source':
        args['execution'] = envelope({'source_sha256': {}})
        access.execution_sha256 = args['execution']['sha256']
    elif change == 'spec':
        p = deepcopy(unpack(args['protocol']))
        p['dataset_inputs']['snapshot']['feature_spec_content_sha256'] = '0'*64
        args['protocol'] = envelope(p); access.protocol_sha256 = args['protocol']['sha256']
    elif change == 'access': access.access_kind = 'final_eval_eligibility'
    else: access.population_sha256 = None
    with pytest.raises(ValueError): module._prepare_maps(access, **args)


def test_attempted_rows_are_not_lost_when_provider_fails():
    class Provider:
        def __call__(self, xy): raise ValueError('synthetic query failure')
        def close(self): pass
    owner = module.RegisteredMaps(Provider(), {}, root=None, policy={}, parents=(), execution={})
    with pytest.raises(ValueError): owner(np.zeros((3, 2)))
    assert owner.attempted_query_rows == 3 and owner.completed_query_rows == 0
    with pytest.raises(ValueError): owner(np.zeros((2, 3)))
    assert owner.attempted_query_rows == 3


@pytest.mark.parametrize('failure', ['approval', 'population'])
def test_actual_public_guard_denies_before_any_map_preparation(tmp_path, monkeypatch, failure):
    args = qualified_fixture(tmp_path, monkeypatch)
    keys = ('protocol', 'execution', 'approval_path', 'approval_sha256', 'test_path', 'review_path',
            'journal_directory', 'population_path', 'population_sha256', 'eligibility_path', 'data_root')
    call = {k: args[k] for k in keys}
    call['snapshot'] = tmp_path/'MUST-NOT-OPEN'
    if failure == 'approval': call['approval_sha256'] = '0'*64
    else: call['population_path'] = call['population_sha256'] = None
    monkeypatch.setattr(module, '_prepare_maps', lambda *a, **kw: pytest.fail('unauthorized map initialization'))
    with pytest.raises(ValueError): module.prepare_formal_maps(**call)
