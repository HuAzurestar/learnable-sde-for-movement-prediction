"""A rehashed four-file input cannot become formal statistical evidence."""
from copy import deepcopy
import hashlib
import json

import pytest

from application.research_evidence import (
    accept_aggregate, accept_evidence_package, expected_metrics_csv, expected_paper_index,
)
from infrastructure.research_store import ResearchError, digest, encode
from tests.test_research_comparison import comparison_source


@pytest.fixture(scope="module")
def descriptive_source(tmp_path_factory):
    store, spec, _, _, package = comparison_source(tmp_path_factory.mktemp("descriptive-source"))
    metadata = store.manifest("artifact-" + package["aggregate_id"])
    aggregate = json.loads(store._verified_artifact_content(metadata))
    assert aggregate["qualification"] == "descriptive"
    assert all(row["qualification"] == "fixture" for row in aggregate["cell_dispositions"])
    return store, spec, aggregate


@pytest.mark.parametrize("entry", ["aggregate", "package"])
@pytest.mark.parametrize("tamper", ["label", "metric-and-label"])
def test_rehashed_fixture_cannot_be_imported_as_formal(descriptive_source, tmp_path, entry, tamper):
    store, spec, original = descriptive_source
    value = deepcopy(original)
    value["qualification"] = "formal"
    if tamper == "metric-and-label":
        for arm in value["arms"]:
            arm["metrics"]["error"] = 999999.0
    value["aggregate_hash"] = digest({key: item for key, item in value.items() if key != "aggregate_hash"})
    assert "adjudication" not in value and "computation_ref" not in value
    before = store.events()
    if entry == "aggregate":
        with pytest.raises(ResearchError, match="UNQUALIFIED.*managed computation proof"):
            accept_aggregate(store, value, spec["study_id"])
    else:
        table = expected_metrics_csv(value)
        files = {"aggregate.json": encode(value), "metrics.csv": table,
                 "PaperEvidenceIndex.json": encode(expected_paper_index(value, table))}
        files["manifest.json"] = encode({"schema_version": "pirc25-evidence-package-v1",
            "aggregate_hash": value["aggregate_hash"],
            "files": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}})
        for name, content in files.items():
            (tmp_path / name).write_bytes(content)
        with pytest.raises(ResearchError, match="UNQUALIFIED.*managed computation proof"):
            accept_evidence_package(store, tmp_path, value["aggregate_hash"])
    assert store.events() == before, "a refused package must not publish evidence artifacts/manifests"


def test_legacy_engineering_package_remains_descriptive(descriptive_source):
    store, spec, original = descriptive_source
    value = {**original, "qualification": "engineering-fixture"}
    value["aggregate_hash"] = digest({key: item for key, item in value.items() if key != "aggregate_hash"})
    artifact = accept_aggregate(store, value, spec["study_id"])
    assert json.loads(store._verified_artifact_content(artifact))["qualification"] == "engineering-fixture"


@pytest.mark.parametrize("qualification", [True, [], {}, "qualified"])
def test_malformed_qualification_is_a_typed_refusal(descriptive_source, qualification):
    store, spec, original = descriptive_source
    value = {**original, "qualification": qualification}
    value["aggregate_hash"] = digest({key: item for key, item in value.items() if key != "aggregate_hash"})
    with pytest.raises(ResearchError, match="UNQUALIFIED"):
        accept_aggregate(store, value, spec["study_id"])
