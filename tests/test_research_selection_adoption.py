"""Full original consumer semantics and common frozen primary conditioning."""
from copy import deepcopy
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tracemalloc

import pytest

from application.research_budget import BudgetSpec
from experiments.pirc22.consumer import BenchmarkConsumerError, validate_benchmark_selection_binding
from experiments.pirc22.representations import validate_representation_matrix
from experiments.pirc25.snapshot import UpstreamSnapshot
from infrastructure.research_store import ResearchError, digest, encode
from tests.research_audit_fixtures import payload_opens
from tests.research_selection_fixtures import public_fingerprints, selection_admission, selection_fixture
from application.research_evidence import export_evidence
from tests.test_research_upstream_paper import invoke_paper, reseal as reseal_receipt


def reseal(consumer):
    consumer["consumer_identity_sha256"] = digest({key: value for key, value in consumer.items()
                                                  if key != "consumer_identity_sha256"})


def damage(documents, change):
    consumer = documents["selection"]
    if change == "identity":
        consumer["consumer_identity_sha256"] = "0" * 64
    elif change == "dimension":
        consumer["selected_configuration"]["model_input_dim"] += 1
        reseal(consumer)
    elif change == "conditioner":
        consumer["selected_configuration"]["conditioner_id"] = "not-a-published-conditioner"
        reseal(consumer)
    elif change == "training":
        consumer["selected_configuration"]["training_config"]["maximum_epochs"] = 0
        reseal(consumer)
    elif change == "matrix":
        consumer["matrix_identity_sha256"] = "0" * 64
        reseal(consumer)


def resolved(root, manifest, accepted):
    value = UpstreamSnapshot(manifest).resolve(root=root, accepted_versions=accepted)
    (root / "selection-observed.json").write_bytes(encode(value))
    return {row["cell_id"]: row for row in value["cells"]}, value


def test_real_published_consumer_and_frozen_matrix_remain_valid_without_provider_reads(tmp_path):
    before = public_fingerprints()
    documents, manifest, accepted = selection_fixture(tmp_path)
    matrix = validate_representation_matrix(documents["matrix"], documents["matrix-lock"])
    assert validate_benchmark_selection_binding(documents["selection"], matrix=matrix) == documents["selection"]
    with payload_opens([tmp_path / "plugin-input.bin"]) as opened:
        rows, value = resolved(tmp_path, manifest, accepted)
    assert rows["dependent"]["status"] == rows["independent"]["status"] == "ready"
    assert value["data_authorization"] == "none" and not opened and public_fingerprints() == before


@pytest.mark.parametrize("change", ["identity", "dimension", "conditioner", "training", "matrix"])
def test_hash_bound_selected_headers_do_not_replace_original_consumer_validation(tmp_path, change):
    before = public_fingerprints()
    documents, manifest, accepted = selection_fixture(tmp_path, lambda d: damage(d, change))
    matrix = validate_representation_matrix(documents["matrix"], documents["matrix-lock"])
    with pytest.raises(BenchmarkConsumerError):
        validate_benchmark_selection_binding(documents["selection"], matrix=matrix)
    rows, value = resolved(tmp_path, manifest, accepted)
    assert rows["dependent"]["status"] == "rejected", value
    assert rows["independent"]["status"] == "ready" and public_fingerprints() == before


def test_full_selection_requires_explicit_frozen_matrix_reference(tmp_path):
    _, manifest, accepted = selection_fixture(tmp_path)
    manifest["inputs"][0].pop("benchmark_binding")
    accepted[0]["input"] = deepcopy(manifest["inputs"][0])
    rows, value = resolved(tmp_path, manifest, accepted)
    assert rows["dependent"]["status"] == "rejected", value
    assert rows["independent"]["status"] == "ready"


@pytest.mark.parametrize("change", ["missing", "different"])
def test_adopted_primary_requires_complete_common_conditioner_before_read(tmp_path, change):
    _, manifest, accepted = selection_fixture(tmp_path)
    if change == "missing":
        manifest["studies"][0]["pirc22_cutover"].pop("conditioner_binding")
    else:
        other = deepcopy(manifest["cells"][0])
        other.update(cell_id="other-primary", conditioner_binding={"unfrozen": True})
        manifest["cells"].append(other)
    rows, value = resolved(tmp_path, manifest, accepted)
    assert rows["dependent"]["status"] == "rejected", value
    assert rows["independent"]["status"] == "ready"


@pytest.mark.parametrize("change", ["dimension", "unshared-primary"])
def test_actual_formal_runner_refuses_bad_or_unshared_selection_before_provider_or_worker(tmp_path, change):
    before = public_fingerprints()
    store, spec, runner, _ = selection_admission(tmp_path,
        mutate=(lambda d: damage(d, change)) if change == "dimension" else None,
        cell_change=(lambda cells: cells[1].pop("conditioner_binding")) if change == "unshared-primary" else None)
    with payload_opens([tmp_path / "plugin-input.bin"]) as opened:
        try:
            outcome = runner.run_cell(spec["study_id"], digest(spec["cells"][0]), budget=BudgetSpec(10))
            observed = {"outcome": outcome, "error": None}
        except ResearchError as exc:
            observed = {"outcome": None, "error": exc.code}
    observed.update(opens=len(opened), worker_started=any(e["event_kind"] == "WORKER_STARTED" for e in store.events()))
    (tmp_path / "selection-runner-observed.json").write_bytes(encode(observed))
    assert observed["error"] in {"IDENTITY_MISMATCH", "UNACCEPTED_VERSION"}, observed
    assert not opened and not observed["worker_started"]
    assert public_fingerprints() == before


@pytest.fixture(scope='module')
def actual_selection_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp('actual-shared-selection')
    store, spec, runner, grant = selection_admission(root)
    for cell in spec['cells']:
        outcome = runner.run_cell(spec['study_id'], digest(cell), budget=BudgetSpec(10))
        assert outcome['state'] == 'SUCCEEDED', outcome
    return export_evidence(store, spec['study_id'], grant)


def test_actual_shared_primary_selection_is_preserved_in_independent_paper_evidence(actual_selection_bundle, tmp_path):
    result = invoke_paper(deepcopy(actual_selection_bundle), tmp_path)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('change', ['semantic-dimension', 'missing-document', 'unbound-extra'])
def test_independent_paper_refuses_resealed_actual_selection_content(actual_selection_bundle, tmp_path, change):
    bundle = deepcopy(actual_selection_bundle)
    row = bundle['cells'][0]
    resolved = row['admission']['documents']['upstream_snapshot']['validation']['resolved']
    selection = next(item for item in resolved if item['kind'] == 'terrain-selection')
    if change == 'semantic-dimension':
        selection['semantic_document']['selected_configuration']['model_input_dim'] += 1
        reseal(selection['semantic_document'])
    elif change == 'missing-document':
        selection.pop('semantic_document')
    else:
        selection['extra_unbound_semantics'] = {'conditioner_id': 'other'}
    # Only disposable validation/transport envelopes. Original registration,
    # catalog, grants, READ receipts and stores are neither resealed nor edited.
    reseal_receipt(row, record_events=True)
    result = invoke_paper(bundle, tmp_path)
    assert result.returncode != 0 and 'invalid admission evidence: upstream' in result.stderr, result.stderr


@pytest.mark.parametrize('change', ['valid', 'identity', 'dimension', 'conditioner', 'training', 'matrix'])
def test_stdlib_paper_checks_original_consumer_semantics_without_runtime_imports(tmp_path, change):
    # Pure contract controls. Modified catalog-bound documents below are NOT
    # admissions, accepted scientific outputs or real data authorization.
    documents, manifest, _ = selection_fixture(tmp_path, lambda d: damage(d, change))
    source = tmp_path / 'pure-contract.json'
    source.write_bytes(encode({'inputs': manifest['inputs'], 'documents': documents}))
    paper = Path(__file__).resolve().parents[2] / 'TSDE-SDE'
    code = """
import importlib.abc,json,sys
class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'application','infrastructure','experiments','torch','pyarrow','numpy'}:
            raise RuntimeError('offline contract attempted to import runtime/training')
sys.meta_path.insert(0,NoRuntime())
sys.path.insert(0,sys.argv[1])
from scripts.pirc25.selection import selection_documents
with open(sys.argv[2],encoding='utf-8') as stream:
    value=json.load(stream)
inputs={row['object_id']:row for row in value['inputs']}
resolved={name:{'semantic_document':doc} for name,doc in value['documents'].items()}
print(json.dumps(selection_documents(inputs['selection'],inputs,resolved,list(inputs))[0],sort_keys=True))
"""
    command = [sys.executable, '-I', '-B', '-c', code, str(paper), str(source)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    (tmp_path / 'pure-paper-observed.json').write_bytes(encode({'command': command,
        'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}))
    if change == 'valid':
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0 and 'invalid admission evidence: upstream' in result.stderr, result.stderr


@pytest.mark.parametrize('change', ['size-change', 'restored-mtime'])
def test_matrix_descriptor_remains_held_through_original_consumer_semantics(tmp_path, monkeypatch, change):
    import experiments.pirc22.consumer as original
    before = public_fingerprints()
    _, manifest, accepted = selection_fixture(tmp_path)
    validator = original.validate_benchmark_selection_binding
    def change_after_validation(payload, **kwargs):
        result = validator(payload, **kwargs)
        path = tmp_path / 'matrix.json'
        before = path.stat()
        content = path.read_bytes()
        changed = content.replace(b'FROZEN', b'BROKEN' if change == 'restored-mtime' else b'CHANGED', 1)
        assert changed != content
        path.write_bytes(changed)
        if change == 'restored-mtime':
            assert len(changed) == len(content)
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        after = path.stat()
        (tmp_path / 'source-stat-observed.json').write_bytes(encode({
            'platform': os.name, 'change': change,
            'before': {key: getattr(before, key) for key in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')},
            'after': {key: getattr(after, key) for key in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')},
            'original_hash': hashlib.sha256(content).hexdigest(),
            'changed_hash': hashlib.sha256(path.read_bytes()).hexdigest()}))
        return result
    monkeypatch.setattr(original, 'validate_benchmark_selection_binding', change_after_validation)
    rows, value = resolved(tmp_path, manifest, accepted)
    after_public = public_fingerprints()
    (tmp_path / 'publisher-observed.json').write_bytes(encode({'before': before, 'after': after_public}))
    assert after_public == before
    assert rows['dependent']['status'] == 'rejected', value
    assert rows['independent']['status'] == 'ready'


def test_legal_large_formatted_selection_keeps_full_semantics_without_encoded_file_buffer(tmp_path):
    _, manifest, accepted = selection_fixture(tmp_path)
    path = tmp_path / 'selection.json'
    body = path.read_bytes()
    with path.open('wb') as stream:
        stream.write(body)
        for _ in range(12):
            stream.write(b' ' * (1024 * 1024))
    record = manifest['inputs'][0]
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(64 * 1024), b''):
            hasher.update(chunk)
    record.update(artifact_hash=hasher.hexdigest(), artifact_size_bytes=path.stat().st_size)
    accepted[0]['input'] = deepcopy(record)
    tracemalloc.start()
    try:
        rows, _ = resolved(tmp_path, manifest, accepted)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert rows['dependent']['status'] == 'ready'
    assert peak < 4 * 1024 * 1024, peak
