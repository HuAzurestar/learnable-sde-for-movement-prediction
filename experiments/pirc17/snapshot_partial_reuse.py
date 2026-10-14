"""Reuse closed recovery files without treating a partial tree as a snapshot.

The original producer still validates releases, conditions and sidecars, walks
every alignment group, recomputes coverage and builds the complete manifest.
Only byte-identified completed feature files can bypass repeated computation.
Final original inventory/coverage/manifest verification remains mandatory.
"""
from __future__ import annotations

import contextlib
from pathlib import Path
import shutil

import numpy as np
import pyarrow.parquet as pq


def factor_results(producer, spec, table, sidecar_factors):
    """Reconstruct the producer's coverage masks from its exact output masks.

Use Arrow validity, not float32 finiteness: a finite raw float64 may overflow
when stored as float32. Sidecar source status and fallback parent semantics
are distinct; a parent can still be present for an unsourced sidecar row.
"""
    results = {}
    for factor in spec['factors']:
        name = factor['factor_id']
        values = {column['name']: table[column['name']].combine_chunks()
                  for column in factor['value_columns']}
        statuses = table[factor['status_column']].to_pylist()
        parents = table[factor['provenance']['parent_asset_column']].to_pylist()
        valid = np.logical_and.reduce([array.is_valid().to_numpy(zero_copy_only=False)
                                       for array in values.values()])
        if name in sidecar_factors:
            source = np.asarray([status not in {'source_missing', 'out_of_extent', 'not_materialized'}
                                 for status in statuses], dtype=bool)
            valid &= np.asarray([status == 'valid' for status in statuses], dtype=bool)
        else:
            source = np.asarray([parent is not None for parent in parents], dtype=bool)
        results[name] = producer._FactorResult(values, statuses, parents, source, valid)
    return results


class PartialReuse:
    """One process-local patch, restored on exit; never edits the producer tree."""

    def __init__(self, producer, directories, sha):
        self.producer, self.sha = producer, sha
        self.files = {}
        self.pending = None
        self.stats = dict(valid_cache_files=0, incomplete_cache_files=0,
                          reused_files=0, reused_rows=0, computed_files=0)
        for directory in directories:
            directory = Path(directory).resolve()
            for path in sorted(directory.glob('features/*/*.parquet')):
                if path.is_symlink() or not path.resolve().is_relative_to(directory):
                    raise ValueError('unsafe partial feature file')
                try:
                    count = pq.ParquetFile(path).metadata.num_rows
                except Exception:
                    self.stats['incomplete_cache_files'] += 1
                    continue  # An interrupted final write is not a cache entry.
                key = tuple(path.parts[-3:])
                entry = (path, sha(path), count)
                old = self.files.get(key)
                if old is not None and old[1:] != entry[1:]:
                    raise ValueError('closed recovery caches disagree')
                self.files[key] = entry
        self.stats['valid_cache_files'] = len(self.files)

    def feature_table(self, spec, dataset_id, rows, condition, **kwargs):
        producer = self.producer
        key = ('features', str(rows[0]['split']), producer._file_name(str(rows[0]['file_id'])))
        entry = self.files.get(key)
        if entry is None or kwargs['history_window_points'] != 3:
            return self.original_feature_table(spec, dataset_id, rows, condition, **kwargs)
        path, digest, count = entry
        if self.sha(path) != digest:
            raise ValueError('cached feature bytes changed after inspection')
        table = pq.ParquetFile(path).read()
        if table.num_rows != count or count != len(rows):
            raise ValueError('cached row count differs from current alignment')
        metadata = table.schema.metadata or {}
        expected = {
            b'pirc21.dataset_id': str(dataset_id).encode(),
            b'pirc21.feature_spec_id': str(spec['feature_spec_id']).encode(),
            b'pirc21.feature_spec_sha256': producer.feature_spec_fingerprint(spec).encode(),
            b'pirc21.schema_version': producer.FEATURE_SNAPSHOT_SCHEMA_VERSION.encode(),
        }
        if any(metadata.get(name) != value for name, value in expected.items()):
            raise ValueError('cached feature metadata differs from current recipe')
        ordered = sorted(rows, key=lambda row: int(row['source_point_index']))
        for name, array in producer._identity_arrays(spec, dataset_id, ordered).items():
            if not table[name].combine_chunks().equals(array):
                raise ValueError('cached point identity differs from current alignment')
        sidecars = kwargs['sidecars']
        sidecar_factors = set(sidecars._table_by_factor) if sidecars is not None else set()
        results = factor_results(producer, spec, table, sidecar_factors)
        self.pending = (key, path, digest, table)
        return table, results

    def write_table(self, table, where, *args, **kwargs):
        if self.pending is None:
            self.stats['computed_files'] += 1
            return self.original_write_table(table, where, *args, **kwargs)
        key, source, digest, cached_table = self.pending
        self.pending = None
        target = Path(where)
        if (tuple(target.parts[-3:]) != key or target.exists()
                or not table.equals(cached_table, check_metadata=True)
                or self.sha(source) != digest):
            raise ValueError('producer output differs from the validated cached feature file')
        shutil.copyfile(source, target)
        if self.sha(target) != digest:
            raise ValueError('cached byte copy failed verification')
        self.stats['reused_files'] += 1
        self.stats['reused_rows'] += table.num_rows

    @contextlib.contextmanager
    def installed(self):
        self.original_feature_table = self.producer._feature_table
        self.original_write_table = self.producer.pq.write_table
        self.producer._feature_table = self.feature_table
        self.producer.pq.write_table = self.write_table
        try:
            yield self
            if self.pending is not None:
                raise ValueError('cached feature was not written by the producer')
        finally:
            self.producer._feature_table = self.original_feature_table
            self.producer.pq.write_table = self.original_write_table
