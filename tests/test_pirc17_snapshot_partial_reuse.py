"""Synthetic reconstruction tests; no empirical trajectory or feature inputs."""
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.pirc17.snapshot_partial_reuse import PartialReuse, factor_results
from experiments.pirc17.snapshot_recovery import PRODUCER, sha


@pytest.fixture
def producer(monkeypatch):
    monkeypatch.syspath_prepend(str(PRODUCER))
    module = importlib.import_module('trajectory.feature_materialization')
    assert Path(module.__file__).resolve() == (PRODUCER / 'trajectory/feature_materialization.py').resolve()
    return module


@pytest.fixture
def inputs(tmp_path, producer):
    location = PRODUCER / 'tests/test_feature_materialization.py'
    spec = importlib.util.spec_from_file_location('pirc21_public_recovery_fixture', location)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    release, conditions = fixture._pirc20_release(tmp_path)
    return release, conditions, PRODUCER / 'registry/pirc21_feature_spec.contract.json'


def build(producer, inputs, root, sidecars=None):
    return producer.build_feature_snapshot(*inputs, root, snapshot_id='resume-fixture',
                                           factor_sidecars=sidecars)


def test_full_public_rebuild_has_identical_bytes_and_coverage(tmp_path, producer, inputs):
    original = build(producer, inputs, tmp_path / 'first')
    reuse = PartialReuse(producer, [original], sha)
    with reuse.installed():
        restored = build(producer, inputs, tmp_path / 'second')
    assert reuse.stats['reused_files'] == 3
    assert reuse.stats['reused_rows'] == 42
    assert reuse.stats['computed_files'] == 0
    for name in ('manifest.json', 'coverage_report.json', 'feature_spec.json'):
        assert sha(original / name) == sha(restored / name)
    for first in original.glob('features/*/*.parquet'):
        assert sha(first) == sha(restored / first.relative_to(original))


def test_partial_bad_footer_is_not_reused_and_original_cache_is_preserved(tmp_path, producer, inputs):
    original = build(producer, inputs, tmp_path / 'first')
    partial = tmp_path / 'partial'
    for index, source in enumerate(original.glob('features/*/*.parquet')):
        target = partial / source.relative_to(original)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes() if index == 0 else b'incomplete')
    reuse = PartialReuse(producer, [partial], sha)
    with reuse.installed():
        restored = build(producer, inputs, tmp_path / 'second')
    assert reuse.stats['valid_cache_files'] == reuse.stats['reused_files'] == 1
    assert reuse.stats['incomplete_cache_files'] == reuse.stats['computed_files'] == 2
    assert sha(original / 'manifest.json') == sha(restored / 'manifest.json')
    assert sum(p.read_bytes() == b'incomplete' for p in partial.glob('features/*/*.parquet')) == 2


def test_cached_bytes_changed_after_inspection_are_rejected(tmp_path, producer, inputs):
    original = build(producer, inputs, tmp_path / 'first')
    reuse = PartialReuse(producer, [original], sha)
    path = next(original.glob('features/*/*.parquet'))
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='changed after inspection'), reuse.installed():
        build(producer, inputs, tmp_path / 'second')
    assert not (tmp_path / 'second/CURRENT').exists()


def test_wrong_cached_point_identity_is_rejected(tmp_path, producer, inputs):
    original = build(producer, inputs, tmp_path / 'first')
    path = next(original.glob('features/*/*.parquet'))
    table = pq.ParquetFile(path).read()
    index = table.column_names.index('point_id')
    table = table.set_column(index, 'point_id', pa.array(['wrong-point'] * table.num_rows))
    pq.write_table(table, path)
    reuse = PartialReuse(producer, [original], sha)
    with pytest.raises(ValueError, match='point identity'), reuse.installed():
        build(producer, inputs, tmp_path / 'second')


def test_sidecar_source_and_raw_validity_semantics_are_preserved(producer):
    factor = dict(factor_id='road', value_columns=[dict(name='value', dtype='float32')],
                  status_column='status', provenance=dict(parent_asset_column='parent'))
    array = producer._numeric_array(np.array([1e40, 3.0, np.nan, 4.0]), 'float32')
    table = pa.table(dict(value=array, status=['valid', 'source_missing', 'valid', 'not_materialized'],
                         parent=['asset', 'asset', 'asset', 'asset']))
    result = factor_results(producer, dict(factors=[factor]), table, {'road'})['road']
    assert result.source_mask.tolist() == [True, False, True, False]
    assert result.valid_mask.tolist() == [True, False, False, False]
    assert np.isinf(table['value'][0].as_py())  # Not a reason to alter raw-valid coverage.


def test_public_sidecar_rebuild_coverage_and_feature_bytes_match(tmp_path, producer, inputs):
    path = tmp_path / 'road.parquet'
    statuses = ['valid'] * 14
    statuses[2], statuses[3] = 'source_missing', 'not_materialized'
    distances = np.arange(14, dtype=float)
    distances[4] = np.nan
    pq.write_table(pa.table(dict(file_id=['file-a'] * 14, point_index=range(14),
        road_distance_m=distances, road_direction_east=np.ones(14), road_direction_north=np.zeros(14),
        overture_road_status=statuses, overture_road_parent_asset_id=['asset'] * 14)), path)
    sidecars = {'overture_road': [path]}
    original = build(producer, inputs, tmp_path / 'first', sidecars)
    reuse = PartialReuse(producer, [original], sha)
    with reuse.installed():
        restored = build(producer, inputs, tmp_path / 'second', sidecars)
    assert reuse.stats['reused_files'] == 3
    assert sha(original / 'manifest.json') == sha(restored / 'manifest.json')
    assert sha(original / 'coverage_report.json') == sha(restored / 'coverage_report.json')


def test_cache_metadata_drift_and_disagreeing_trees_are_rejected(tmp_path, producer, inputs):
    original = build(producer, inputs, tmp_path / 'first')
    other = build(producer, inputs, tmp_path / 'second')
    source = next(other.glob('features/*/*.parquet'))
    table = pq.ParquetFile(source).read()
    metadata = dict(table.schema.metadata)
    metadata[b'pirc21.dataset_id'] = b'wrong-dataset'
    pq.write_table(table.replace_schema_metadata(metadata), source)
    with pytest.raises(ValueError, match='caches disagree'):
        PartialReuse(producer, [original, other], sha)
    reuse = PartialReuse(producer, [other], sha)
    original_hook = producer._feature_table
    with pytest.raises(ValueError, match='metadata'), reuse.installed():
        build(producer, inputs, tmp_path / 'third')
    assert producer._feature_table is original_hook


def test_direct_resume_worker_without_owned_job_is_denied_before_inputs():
    child = subprocess.run([sys.executable, '-m', 'experiments.pirc17.snapshot_recovery', 'worker-resume'],
        cwd=Path(__file__).parents[1], env=dict(os.environ, PYTHONIOENCODING='utf-8'),
        capture_output=True, text=True, encoding='utf-8', timeout=30)
    assert child.returncode != 0
    assert 'worker must belong to its live reserved process job' in child.stderr
