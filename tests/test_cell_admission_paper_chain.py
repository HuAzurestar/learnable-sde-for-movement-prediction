"""Real shared synthetic receipts -> independent paper CLI; no research claim."""

from copy import deepcopy
import json
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


def paper_validate(tmp_path, bundle, *, name):
    # Exercise the independent stdlib-only consumer in a separate interpreter.
    paper = Path(__file__).resolve().parents[2] / "TSDE-SDE"
    script = paper / "scripts/pirc25/aggregate.py"
    assert script.is_file(), "paired paper checkout required"
    source = tmp_path / (name + ".json")
    source.write_bytes(encode(bundle))
    return subprocess.run([sys.executable, "-B", str(script), str(source), "--expected-hash", bundle["bundle_hash"],
        "--formal", "--output", str(tmp_path / (name + "-evidence"))], cwd=paper,
        capture_output=True, text=True, timeout=30)


@pytest.fixture(scope="module", params=[False, True], ids=["execution-only", "foreign-model"])
def source(request, tmp_path_factory):
    root = tmp_path_factory.mktemp("cell-paper-source")
    # Explicit synthetic operator attestations; never real model/data permission.
    store, spec, registry, grant = prepared(root, formal=True, two_arms=True)
    if request.param:
        attach_foreign_model(store, spec)
    settings = spec["admission"]
    package_hash = settings.pop("package_hash")
    model = {field: settings.pop(field) for field in ("model_authorization_id",
        "model_authorization_version", "model_protocol_id") if field in settings}
    settings["cell_packages"] = {"schema_version": "pirc25-cell-packages-v1", "bindings": [
        {"cell_hash": digest(cell), "package_hash": package_hash, **model} for cell in spec["cells"]]}
    store.register(spec, digest(spec))
    for cell in spec["cells"]:
        outcome = SharedRunner(store, registry).run_cell(spec["study_id"], digest(cell), budget=BudgetSpec(10))
        assert outcome["state"] == "SUCCEEDED", outcome
    return export_evidence(store, spec["study_id"], grant)


def test_actual_shared_mapped_receipts_are_accepted_by_independent_formal_reader(source, tmp_path):
    outcome = paper_validate(tmp_path, source, name="valid")
    assert outcome.returncode == 0, outcome.stderr
    result = json.loads((tmp_path / "valid-evidence/aggregate.json").read_bytes())
    assert result["successful_cell_count"] == result["expected_cell_count"] == 2
    # --formal verifies recorded admission, but the offline writer does not
    # manufacture the managed computation proof required for formal output.
    assert result["qualification"] == "descriptive"
    assert result["source_bundle_hash"] == source["bundle_hash"]


@pytest.mark.parametrize("fault", ["missing-selection", "wrong-table", "wrong-binding", "wrong-cell",
    "wrong-package", "wrong-mode", "wrong-package-body", "missing-binding", "duplicate-binding",
    "foreign-binding", "default", "mode-override", "grant-override", "upstream-override"])
def test_rehashed_admission_chain_cannot_bypass_independent_cell_selection(source, tmp_path, fault):
    # Forge ONLY a disposable transport claim to reach the independent gate.
    # No ledger/manifests, grants, raw inputs or qualification are rewritten.
    bundle = deepcopy(source)
    spec = bundle["registered_spec"]
    receipt = bundle["cells"][0]["admission"]
    table = spec["admission"]["cell_packages"]
    selection = receipt["admission_selection"]
    if fault == "missing-selection":
        receipt.pop("admission_selection")
    elif fault in {"wrong-table", "wrong-binding", "wrong-cell", "wrong-package"}:
        selection[fault.removeprefix("wrong-") + "_hash"] = digest("substitution")
    elif fault == "wrong-mode":
        receipt["mode"] = "fixture"
    elif fault == "wrong-package-body":
        receipt["documents"]["package"]["payload"] = {"substitution": True}
    elif fault == "missing-binding":
        table["bindings"].pop()
    elif fault == "duplicate-binding":
        table["bindings"].append(deepcopy(table["bindings"][0]))
    elif fault == "foreign-binding":
        table["bindings"][1]["cell_hash"] = digest("foreign")
    elif fault == "default":
        spec["admission"]["package_hash"] = table["bindings"][0]["package_hash"]
    else:
        field = {"mode-override": "mode", "grant-override": "authorization_id",
                 "upstream-override": "upstream_root"}[fault]
        table["bindings"][0][field] = "forged"
    bundle["spec_hash"] = digest(spec)
    for row in bundle["cells"]:
        admission = row["admission"]
        admission["spec"] = spec
        admission["spec_hash"] = bundle["spec_hash"]
        admission["admission_hash"] = digest({key: value for key, value in admission.items() if key != "admission_hash"})
        row["admission_hash"] = admission["admission_hash"]
    bundle["bundle_hash"] = digest({key: value for key, value in bundle.items() if key != "bundle_hash"})
    outcome = paper_validate(tmp_path, bundle, name="refused")
    assert outcome.returncode != 0
    assert "cell package selection" in outcome.stderr or (fault == "wrong-mode" and "formal admission" in outcome.stderr), outcome.stderr
    assert not (tmp_path / "refused-evidence").exists()
