"""The offline consumer must reject grants accepted by older unsafe runtimes."""

from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest

from application.research_budget import BudgetSpec
from application.research_evidence import export_evidence
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_store import digest, encode
from tests.research_admission_fixtures import attach_foreign_model
from tests.test_research_admission_chain import prepared


@pytest.mark.parametrize("mutation", ["missing-protocol", "wrong-protocol", "wrong-purpose"])
def test_paper_rejects_legacy_model_source_permission_bypass(tmp_path, mutation):
    aggregator = Path(__file__).resolve().parents[2] / "TSDE-SDE/scripts/pirc25/aggregate.py"
    assert aggregator.is_file(), "paired checkout is required for cross-repository acceptance"
    store, value, registry, grant = prepared(tmp_path, formal=True, two_arms=True)
    attach_foreign_model(store, value)
    store.register(value, digest(value))
    for cell in value["cells"]:
        assert SharedRunner(store, registry).run_cell("synthetic", digest(cell), budget=BudgetSpec(10))["state"] == "SUCCEEDED"
    bundle = deepcopy(export_evidence(store, "synthetic", grant))
    for row in bundle["cells"]:
        receipt = row["admission"]
        source_grant = receipt["documents"]["model_authorization"]
        if mutation == "missing-protocol":
            source_grant.pop("protocol_hash")
        elif mutation == "wrong-protocol":
            source_grant["protocol_hash"] = "0" * 64
        else:
            source_grant["purposes"] = ["preview"]
        receipt["admission_hash"] = digest({key: item for key, item in receipt.items() if key != "admission_hash"})
        row["admission_hash"] = receipt["admission_hash"]
    bundle["bundle_hash"] = digest({key: item for key, item in bundle.items() if key != "bundle_hash"})
    source = tmp_path / "legacy-model-permission.json"
    source.write_bytes(encode(bundle))
    completed = subprocess.run([sys.executable, "-B", str(aggregator), str(source),
        "--expected-hash", bundle["bundle_hash"], "--formal", "--output", str(tmp_path / "output")],
        capture_output=True, text=True, timeout=30)
    assert completed.returncode != 0, "offline consumer accepted incomplete source permission"
    assert "admission evidence" in completed.stderr
