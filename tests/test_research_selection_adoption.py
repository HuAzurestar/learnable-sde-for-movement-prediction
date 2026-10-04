"""Full original consumer semantics and common frozen primary conditioning."""
from copy import deepcopy
import json

import pytest

from application.research_budget import BudgetSpec
from experiments.pirc22.consumer import BenchmarkConsumerError, validate_benchmark_selection_binding
from experiments.pirc22.representations import validate_representation_matrix
from experiments.pirc25.snapshot import UpstreamSnapshot
from infrastructure.research_store import ResearchError, digest, encode
from tests.research_audit_fixtures import payload_opens
from tests.research_selection_fixtures import public_fingerprints, selection_admission, selection_fixture


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
