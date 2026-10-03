"""Real registered synthetic executions, independent Paper CLI, no stand-ins."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest

from application.research_budget import BudgetSpec
from application.research_evidence import export_evidence
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_store import digest, encode
from tests.test_research_upstream_admission import prepared
from tests.test_shared_upstream_snapshot import snapshot_fixture


@pytest.fixture(scope="module")
def actual_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("actual-upstream-paper-source")
    _path, manifest, accepted = snapshot_fixture(root)
    store, value, registry, grant = prepared(root, formal=True, two_arms=True,
        upstream_inputs=manifest["inputs"], accepted_versions=accepted)
    store.register(value, digest(value))
    for cell in value["cells"]:
        outcome = SharedRunner(store, registry).run_cell("synthetic", digest(cell), budget=BudgetSpec(10))
        assert outcome["state"] == "SUCCEEDED"
    bundle = export_evidence(store, "synthetic", grant)
    assert all(row["admission"]["documents"]["upstream_snapshot"]["validation"]["resolved"] for row in bundle["cells"])
    return bundle


def invoke_paper(bundle, root):
    paper = Path(__file__).resolve().parents[2] / "TSDE-SDE/scripts/pirc25/aggregate.py"
    assert paper.is_file(), "independent paired Paper checkout required, not skipped"
    bundle["bundle_hash"] = digest({key: value for key, value in bundle.items() if key != "bundle_hash"})
    source = root / "actual-paper-bundle.json"
    source.write_bytes(encode(bundle))
    command = [sys.executable, "-B", str(paper), str(source), "--expected-hash", bundle["bundle_hash"],
               "--formal", "--output", str(root / "paper-evidence")]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    observed = {"command": command, "bundle_hash": bundle["bundle_hash"], "exit_code": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    (root / "actual-paper-observed.json").write_bytes(encode(observed))
    return result


def reseal(row):
    receipt = row["admission"]
    if "upstream_snapshot" in receipt["documents"]:
        evidence = receipt["documents"]["upstream_snapshot"]
        evidence["validation_hash"] = digest(evidence["validation"])
    receipt["admission_hash"] = digest({key: value for key, value in receipt.items() if key != "admission_hash"})
    row["admission_hash"] = receipt["admission_hash"]


def test_actual_upstream_bundle_passes_independent_formal_paper_cli(actual_bundle, tmp_path):
    result = invoke_paper(deepcopy(actual_bundle), tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "paper-evidence/PaperEvidenceIndex.json").is_file()


@pytest.mark.parametrize("mutation", ["missing-snapshot", "rejected-cell", "other-cell", "other-catalog"])
def test_resealed_actual_upstream_counterexamples_are_refused_by_independent_paper(actual_bundle, tmp_path, mutation):
    bundle = deepcopy(actual_bundle)
    row = bundle["cells"][0]
    docs = row["admission"]["documents"]
    if mutation == "missing-snapshot":
        docs.pop("upstream_snapshot")
    else:
        validation = docs["upstream_snapshot"]["validation"]
        if mutation == "rejected-cell":
            validation["cells"][0].update(status="rejected", rejected_inputs=[{"code": "MISSING_ARTIFACT"}])
        elif mutation == "other-cell":
            validation["cells"][0]["cell_id"] = digest("another synthetic cell")
        else:
            validation["acceptance_catalog_hash"] = "0" * 64
    reseal(row)
    result = invoke_paper(bundle, tmp_path)
    assert result.returncode != 0, "Independent current Paper accepted " + mutation
    assert "upstream" in result.stderr.lower(), result.stderr
