"""Content-bound upstream metadata takeover, never trajectory authorization.

An external frozen acceptance catalog is an operator trust boundary, like an
imported exposure report. This verifies its exact bindings and actual files;
it does not authenticate off-platform decisions or create new acceptance.
"""
from contextlib import contextmanager, ExitStack
import hashlib
import json
import os
from pathlib import Path
import re

from infrastructure.research_files import opened_regular_file
from infrastructure.research_json import read_json, read_metadata_header
from infrastructure.research_store import ResearchError, digest, encode, identifier, utc_now


_SCHEMA = 'pirc25-upstream-snapshot-v1'
_ERRORS = {'MISSING_ARTIFACT', 'IDENTITY_MISMATCH', 'UNACCEPTED_VERSION'}
_HASH_FIELDS = ('data_hash', 'split_hash', 'fold_hash', 'feature_hash', 'selection_hash')
_REQUIRED = ('object_id', 'issue', 'acceptance_commit', 'code_sha', 'schema_version', 'artifact_id',
             'artifact_hash', 'artifact_size_bytes', 'path', 'format', 'role', 'kind', 'license',
             'metadata_checks') + _HASH_FIELDS
_HEADERS = {'schema_version', 'status', 'overall_status', 'scientific_role', 'final_eval_read_count',
            'source_selection_identity_sha256', 'immutability_policy', 'data_hash', 'split_hash',
            'fold_hash', 'feature_hash', 'selection_hash', 'code_sha'}


def _clone(value):
    return json.loads(encode(value))


def _hash(value, length=64):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{' + str(length) + '}', value) is not None


def _identity(record):
    return {key: value for key, value in record.items() if key != 'source_branch'}


def _version(record):
    return tuple(record.get(key) for key in ('issue', 'acceptance_commit', 'code_sha', 'object_id'))


def _validate_input(record):
    if any(key not in record for key in _REQUIRED):
        raise ResearchError('MISSING_ARTIFACT', 'snapshot input lacks required acceptance/identity metadata')
    for key in ('object_id', 'artifact_id'):
        identifier(record[key])
    if (not all(isinstance(record[key], str) and record[key].strip() for key in ('issue', 'schema_version', 'path'))
            or not _hash(record['acceptance_commit'], 40) or not _hash(record['code_sha'], 40)
            or not _hash(record['artifact_hash']) or type(record['artifact_size_bytes']) is not int
            or record['artifact_size_bytes'] < 0):
        raise ResearchError('IDENTITY_MISMATCH', 'invalid snapshot input identity/hash/size')
    for key in _HASH_FIELDS:
        value = record[key]
        if not (_hash(value) or isinstance(value, dict) and set(value) == {'not_applicable'}
                and isinstance(value['not_applicable'], str) and value['not_applicable'].strip()):
            raise ResearchError('IDENTITY_MISMATCH', 'explicit hash or non-applicability reason required')
    license = record['license']
    if (record['role'] != 'metadata-only' or not isinstance(license, dict)
            or not isinstance(license.get('id'), str) or not license['id'].strip()
            or license.get('scope') != 'metadata-only'):
        raise ResearchError('UNACCEPTED_VERSION', 'snapshot has no metadata-only license/scope')
    checks = record['metadata_checks']
    if (record['format'] != 'json-metadata' or not isinstance(checks, dict)
            or set(checks) - _HEADERS
            or checks.get('schema_version') != record['schema_version']
            or any(not isinstance(key, str) or not key or type(value) not in (str, int, bool, type(None))
                   for key, value in checks.items())):
        raise ResearchError('IDENTITY_MISMATCH', 'explicit scalar JSON metadata schema checks required')
    if record['kind'] not in {'metadata', 'terrain-selection'}:
        raise ResearchError('IDENTITY_MISMATCH', 'unknown upstream metadata kind')
    if record['kind'] == 'terrain-selection':
        if (not _hash(record['selection_hash']) or checks.get('status') != 'selected'
                or type(checks.get('final_eval_read_count')) is not int or checks['final_eval_read_count'] != 0
                or checks.get('immutability_policy') != 'new_selection_version_required'
                or checks.get('source_selection_identity_sha256') != record['selection_hash']):
            raise ResearchError('UNACCEPTED_VERSION', 'immutable zero-final-eval selection binding required')
        binding = record.get('benchmark_binding')
        if (not isinstance(binding, dict) or set(binding) != {'matrix_object_id', 'matrix_lock_object_id'}
                or any(not isinstance(value, str) or not value for value in binding.values())):
            raise ResearchError('MISSING_ARTIFACT', 'explicit frozen selection matrix/lock references required')


def _failure(object_id, exception):
    code = (exception.code if isinstance(exception, ResearchError) and exception.code in _ERRORS
            else 'MISSING_ARTIFACT' if isinstance(exception, OSError) else 'IDENTITY_MISMATCH')
    return {'object_id': object_id, 'code': code, 'detail': 'upstream metadata verification refused'}


@contextmanager
def _held_input(root, record, failures):
    entered = False
    try:
        path = Path(os.path.abspath(root / record['path']))
        if Path(record['path']).is_absolute() or not path.is_relative_to(root):
            raise ResearchError('IDENTITY_MISMATCH', 'snapshot metadata path escapes its lexical root')
        with opened_regular_file(root, path, expected_size=record['artifact_size_bytes']) as handle:
            entered = True
            yield handle
    except (OSError, ResearchError) as exc:
        if not entered:
            raise
        # Closing one changed input cannot erase other independently validated
        # inputs or silently keep this input ready. Preserve its first refusal.
        failures.setdefault(record['object_id'], _failure(record['object_id'], exc))


def _verify_input(stream, size, verify, record, *, semantic=False):
    hasher, remaining = hashlib.sha256(), size
    while remaining:
        chunk = stream.read(min(1024 * 1024, remaining))
        if not chunk:
            raise ResearchError('IDENTITY_MISMATCH', 'snapshot metadata shrank during hash verification')
        remaining -= len(chunk)
        hasher.update(chunk)
        del chunk
    if stream.read(1) or hasher.hexdigest() != record['artifact_hash']:
        raise ResearchError('IDENTITY_MISMATCH', 'snapshot metadata raw hash differs')
    verify()
    stream.seek(0)
    headers = read_metadata_header(stream, size, record['metadata_checks'])
    if any(key not in headers or type(headers[key]) is not type(expected) or headers[key] != expected
           for key, expected in record['metadata_checks'].items()):
        raise ResearchError('IDENTITY_MISMATCH', 'snapshot metadata schema/status/selection fields differ')
    verify()
    if semantic:
        if not _hash(record.get('canonical_hash')):
            raise ResearchError('MISSING_ARTIFACT', 'accepted semantic metadata content identity required')
        stream.seek(0)
        # Large encoded sources are validated first, never buffered wholesale.
        # Original decoded values retain their natural cost; no small-file cap.
        document = read_json(stream, size, encoding='utf-8', verify_identity=verify)
        verify()
        if digest(document) != record['canonical_hash']:
            raise ResearchError('IDENTITY_MISMATCH', 'semantic metadata canonical identity differs')
        return document


def _selection_contract(record, inputs, documents, dependencies):
    try:
        from experiments.pirc22.consumer import validate_benchmark_selection_binding
        from experiments.pirc22.representations import validate_representation_matrix
    except ImportError as exc:
        raise ResearchError('MISSING_ARTIFACT', 'original selection validator dependencies unavailable') from exc

    binding = record['benchmark_binding']
    references = [binding['matrix_object_id'], binding['matrix_lock_object_id']]
    if (len(set([record['object_id'], *references])) != 3
            or any(name not in dependencies or name not in inputs or name not in documents for name in references)):
        raise ResearchError('MISSING_ARTIFACT', 'complete declared selection matrix/lock inputs required')
    try:
        matrix = validate_representation_matrix(documents[references[0]], documents[references[1]])
        consumer = validate_benchmark_selection_binding(documents[record['object_id']], matrix=matrix)
    except (ValueError, KeyError, TypeError) as exc:
        raise ResearchError('IDENTITY_MISMATCH', 'original frozen BenchmarkSelection contract refused') from exc
    return {'selection_hash': consumer['source_selection_identity_sha256'],
            'consumer_identity_sha256': consumer['consumer_identity_sha256'],
            'matrix_identity_sha256': consumer['matrix_identity_sha256'],
            'configuration': _clone(consumer['selected_configuration'])}


class UpstreamSnapshot:
    """Detached versioned definition; display branches do not alter identity."""
    def __init__(self, manifest):
        if (not isinstance(manifest, dict) or manifest.get('schema_version') != _SCHEMA
                or any(not isinstance(manifest.get(key), list) for key in ('inputs', 'studies', 'cells'))):
            raise ResearchError('CONTRACT_MISMATCH', 'complete upstream snapshot definition required')
        self._manifest = _clone(manifest)
        for key, field in (('inputs', 'object_id'), ('studies', 'study_id'), ('cells', 'cell_id')):
            rows = self._manifest[key]
            if (any(not isinstance(row, dict) or not isinstance(row.get(field), str) or not row[field] for row in rows)
                    or len({row[field] for row in rows}) != len(rows)):
                raise ResearchError('CONTRACT_MISMATCH', 'ambiguous snapshot input/study/cell identity')
        self._definition = _clone(self._manifest)
        self._definition.pop('source_branch', None)
        for record in self._definition['inputs']:
            record.pop('source_branch', None)
        self._snapshot_hash = digest(self._definition)

    @property
    def snapshot_hash(self):
        return self._snapshot_hash

    @property
    def definition(self):
        return _clone(self._definition)

    def resolve(self, *, root, accepted_versions, cell_id=None):
        root = Path(os.path.abspath(root))
        selected_cells = self._manifest['cells']
        selected_inputs = self._manifest['inputs']
        if cell_id is not None:
            selected_cells = [cell for cell in selected_cells if cell['cell_id'] == cell_id]
            if len(selected_cells) != 1:
                raise ResearchError('MISSING_ARTIFACT', 'selected cell absent from frozen upstream snapshot')
            dependencies = selected_cells[0].get('upstream_ids', [])
            if not isinstance(dependencies, list) or any(not isinstance(name, str) for name in dependencies):
                raise ResearchError('IDENTITY_MISMATCH', 'invalid selected cell dependency list')
            required = set(dependencies)
            selected_inputs = [record for record in selected_inputs if record['object_id'] in required]
        if not isinstance(accepted_versions, (list, tuple)):
            raise ResearchError('CONTRACT_MISMATCH', 'explicit external frozen acceptance catalog required')
        catalog = {}
        for entry in accepted_versions:
            if not isinstance(entry, dict) or not isinstance(entry.get('input'), dict):
                raise ResearchError('CONTRACT_MISMATCH', 'invalid external acceptance catalog row')
            expected = entry['input']
            key = _version(expected)
            if any(not isinstance(value, str) for value in key):
                continue  # Cannot assert acceptance of any complete version.
            if key in catalog:
                previous = catalog[key]
                if (previous.get('status') != entry.get('status') or 'input' not in previous
                        or _identity(previous['input']) != _identity(expected)):
                    catalog[key] = {'status': 'ambiguous'}
                continue
            catalog[key] = _clone(entry)
        inputs = {record['object_id']: record for record in self._manifest['inputs']}
        semantic_ids = set()
        for record in selected_inputs:
            if record.get('kind') == 'terrain-selection':
                semantic_ids.add(record['object_id'])
                binding = record.get('benchmark_binding')
                if isinstance(binding, dict):
                    semantic_ids.update(value for value in binding.values() if isinstance(value, str))
        failures, checked, held, documents, selections = {}, {}, [], {}, {}
        with ExitStack() as stack:
            for record in selected_inputs:
                object_id = record['object_id']
                try:
                    _validate_input(record)
                    entry = catalog.get(_version(record))
                    if entry is None or entry.get('status') != 'accepted':
                        raise ResearchError('UNACCEPTED_VERSION', 'input version has no external acceptance')
                    _validate_input(entry['input'])
                    if _identity(record) != _identity(entry['input']):
                        raise ResearchError('IDENTITY_MISMATCH', 'input differs from its frozen accepted identity')
                    stream, size, verify = stack.enter_context(_held_input(root, record, failures))
                    document = _verify_input(stream, size, verify, record, semantic=object_id in semantic_ids)
                    if object_id in semantic_ids:
                        documents[object_id] = document
                    held.append((object_id, verify))
                    checked[object_id] = {**_clone(record), 'physical_size_bytes': size,
                        **({'semantic_document': document} if object_id in semantic_ids else {})}
                except (OSError, ResearchError, ValueError, UnicodeError) as exc:
                    failures.setdefault(object_id, _failure(object_id, exc))
            for record in selected_inputs:
                name = record['object_id']
                if record.get('kind') == 'terrain-selection' and name not in failures:
                    try:
                        selections[name] = _selection_contract(record, inputs, documents, {r['object_id'] for r in selected_inputs})
                    except (ResearchError, ValueError, TypeError, KeyError) as exc:
                        failures.setdefault(name, _failure(name, exc))
            # Pin every admitted descriptor through all other input checks;
            # input1 cannot be changed during inputN and retain a ready cell.
            for object_id, verify in held:
                try:
                    verify()
                except (OSError, ResearchError) as exc:
                    failures.setdefault(object_id, _failure(object_id, exc))
        finished = utc_now()
        resolved = [{**record, 'last_validated_at': finished} for object_id, record in checked.items()
                    if object_id not in failures]
        studies = {study['study_id']: study for study in self._manifest['studies']}
        inputs = {record['object_id']: record for record in self._manifest['inputs']}
        cell_results = []
        for cell in selected_cells:
            dependencies = cell.get('upstream_ids')
            errors = []
            if (not isinstance(dependencies, list) or any(not isinstance(name, str) for name in dependencies)
                    or len(set(dependencies)) != len(dependencies)):
                errors.append({'code': 'IDENTITY_MISMATCH', 'detail': 'invalid frozen cell dependency list'})
                dependencies = []
            for name in dependencies:
                if name not in inputs:
                    errors.append({'object_id': name, 'code': 'MISSING_ARTIFACT', 'detail': 'missing frozen input'})
                elif name in failures:
                    errors.append(failures[name])
            study = studies.get(cell.get('study_id'), {})
            cutover = study.get('pirc22_cutover', {})
            mode = cutover.get('mode') if isinstance(cutover, dict) else None
            terrain = [inputs[name] for name in dependencies if name in inputs and inputs[name].get('kind') == 'terrain-selection']
            if (mode not in {'adopted_primary', 'pending_addendum', 'not_applicable'}
                    or cell.get('role') not in {'primary', 'secondary'}
                    or mode in {'pending_addendum', 'not_applicable'} and terrain
                    or mode == 'adopted_primary' and cell.get('role') == 'primary' and
                    (len(terrain) != 1 or terrain[0].get('selection_hash') != cutover.get('selection_hash'))):
                errors.append({'code': 'IDENTITY_MISMATCH', 'detail': 'cell differs from explicit frozen pirc22_cutover'})
            if mode == 'adopted_primary' and cell.get('role') == 'primary':
                common = cutover.get('conditioner_binding')
                primary = [row for row in self._manifest['cells']
                           if row.get('study_id') == cell.get('study_id') and row.get('role') == 'primary']
                binding = terrain[0].get('benchmark_binding') if len(terrain) == 1 else None
                references = ({terrain[0]['object_id'], *binding.values()} if isinstance(binding, dict)
                              and all(isinstance(value, str) for value in binding.values()) else None)
                if (not isinstance(common, dict) or len(terrain) != 1 or references is None
                        or selections.get(terrain[0]['object_id']) != common
                        or any(row.get('conditioner_binding') != common
                               or not isinstance(row.get('upstream_ids'), list)
                               or any(not isinstance(name, str) for name in row['upstream_ids'])
                               or not references <= set(row['upstream_ids']) for row in primary)):
                    errors.append({'code': 'IDENTITY_MISMATCH', 'detail': 'primary cells lack common frozen selection conditioner'})
            cell_results.append({**_clone(cell), 'status': 'rejected' if errors else 'ready', 'rejected_inputs': _clone(errors)})
        return {'schema_version': 'pirc25-upstream-validation-v1', 'snapshot_hash': self.snapshot_hash,
                'data_authorization': 'none', 'resolved': resolved, 'rejected_inputs': list(failures.values()),
                'cells': cell_results, 'validation_finished_at': finished}

    def register(self, store, *, root, accepted_versions):
        receipt = self.resolve(root=root, accepted_versions=accepted_versions)
        store.publish('upstream-snapshot-' + self.snapshot_hash, self.definition)
        receipt_hash = digest(receipt)
        store.publish('upstream-validation-' + receipt_hash, receipt)
        return {**receipt, 'receipt_hash': receipt_hash}


def resolve_snapshot(manifest, *, root, accepted_versions):
    return UpstreamSnapshot(manifest).resolve(root=root, accepted_versions=accepted_versions)
