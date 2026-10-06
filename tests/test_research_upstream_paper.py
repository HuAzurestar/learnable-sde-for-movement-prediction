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


def reseal(row, *, record_events=False):
    receipt = row["admission"]
    if "upstream_snapshot" in receipt["documents"]:
        evidence = receipt["documents"]["upstream_snapshot"]
        evidence["validation_hash"] = digest(evidence["validation"])
        if record_events:
            # Deliberately reseal disposable transport event envelopes too.
            # This reaches semantic checks instead of only a stale hash check;
            # original authoritative stores/events are never changed.
            publication = evidence.get("publication_events", {}).get("validation")
            if publication is not None:
                publication["payload"] = {"object_id": "upstream-validation-" + evidence["validation_hash"],
                                          "sha256": evidence["validation_hash"]}
                publication["hash"] = digest({key: value for key, value in publication.items() if key != "hash"})
            event = evidence.get("validation_event")
            if event is not None:
                event["payload"]["validation_hash"] = evidence["validation_hash"]
                if publication is not None:
                    event["previous_hash"] = publication["hash"]
                event["hash"] = digest({key: value for key, value in event.items() if key != "hash"})
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


@pytest.mark.parametrize("mutation", ["physical-size", "bad-record-hash", "missing-input", "extra-input",
    "other-consumer", "future-time", "naive-time", "metadata-grant", "missing-publications",
    "bad-publication-hash", "late-publication", "late-validation", "other-event-consumer"])
def test_resealed_actual_receipt_semantics_are_not_replaced_by_ready_flag(actual_bundle, tmp_path, mutation):
    bundle = deepcopy(actual_bundle)
    row = bundle["cells"][0]
    evidence = row["admission"]["documents"]["upstream_snapshot"]
    validation = evidence["validation"]
    if mutation == "physical-size":
        validation["resolved"][0]["physical_size_bytes"] += 1
    elif mutation == "bad-record-hash":
        validation["resolved"][0]["artifact_hash"] = "0" * 64
    elif mutation == "missing-input":
        validation["resolved"] = []
    elif mutation == "extra-input":
        validation["resolved"].append({**validation["resolved"][0], "object_id": "unrelated-input"})
    elif mutation == "other-consumer":
        validation["consumer"]["attempt_id"] = "another-consumer"
    elif mutation in {"future-time", "naive-time"}:
        validation["validation_finished_at"] = "2099-01-01T00:00:00+00:00" if mutation == "future-time" else "2026-10-04T00:00:00"
        validation["resolved"][0]["last_validated_at"] = validation["validation_finished_at"]
    elif mutation == "metadata-grant":
        validation["data_authorization"] = "granted"
    elif mutation == "missing-publications":
        evidence.pop("publication_events")
    elif mutation == "bad-publication-hash":
        evidence["publication_events"]["snapshot"]["hash"] = "0" * 64
    elif mutation == "late-publication":
        event = evidence["publication_events"]["snapshot"]
        event["sequence"] = row["admission"]["documents"]["preregistration_event"]["sequence"] + 1
        event["hash"] = digest({key: value for key, value in event.items() if key != "hash"})
    elif mutation == "late-validation":
        evidence["validation_event"]["sequence"] = row["admission"]["input_evidence"][0]["sequence"]
    else:
        evidence["validation_event"]["payload"]["attempt_id"] = "another-consumer"
    reseal(row, record_events=True)
    result = invoke_paper(bundle, tmp_path)
    assert result.returncode != 0, mutation
    assert "upstream" in result.stderr.lower(), result.stderr


def test_new_formal_snapshot_route_without_legacy_recipe_passes_paper(tmp_path):
    source_root = tmp_path / "actual-no-legacy-source"
    source_root.mkdir()
    _path, manifest, accepted = snapshot_fixture(source_root)
    store, value, registry, grant = prepared(source_root, formal=True, two_arms=True,
        upstream_inputs=manifest["inputs"], accepted_versions=accepted, legacy_upstream=False)
    store.register(value, digest(value))
    for cell in value["cells"]:
        assert SharedRunner(store, registry).run_cell("synthetic", digest(cell), budget=BudgetSpec(10))["state"] == "SUCCEEDED"
    bundle = export_evidence(store, "synthetic", grant)
    assert all("upstream" not in row["admission"]["documents"] for row in bundle["cells"])
    result = invoke_paper(bundle, tmp_path)
    assert result.returncode == 0, result.stderr


def test_independent_full_paper_cli_never_imports_runtime_or_opens_source_root(actual_bundle, tmp_path):
    source = tmp_path / "isolated-source-bundle.json"
    source.write_bytes(encode(actual_bundle))
    paper = Path(__file__).resolve().parents[2] / "TSDE-SDE/scripts/pirc25/aggregate.py"
    code = """
import importlib.abc,json,os,runpy,sys
bundle=json.load(open(sys.argv[2],encoding='utf-8'))
roots=[os.path.normcase(os.path.abspath(bundle['registered_spec']['admission']['upstream_root']))]
def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        path=os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
        if any(path==root or path.startswith(root+os.sep) for root in roots):
            raise RuntimeError('Paper attempted to open source root')
class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'application','infrastructure','experiments'}:
            raise RuntimeError('Paper attempted to import runtime/provider')
sys.addaudithook(guard)
sys.meta_path.insert(0,NoRuntime())
try:
    sys.audit('open',os.path.join(roots[0],'metadata.json'),'r',0)
except RuntimeError:
    pass
else:
    raise AssertionError('source-open audit guard not installed')
script=sys.argv[1]
sys.path.insert(0,os.path.dirname(script))
sys.argv=[script,sys.argv[2],'--expected-hash',bundle['bundle_hash'],'--formal','--output',sys.argv[3]]
runpy.run_path(script,run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(paper), str(source),
                             str(tmp_path / "isolated-evidence")], capture_output=True, text=True, timeout=30)
    (tmp_path / "isolated-observed.json").write_bytes(encode({"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}))
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "isolated-evidence/PaperEvidenceIndex.json").is_file()
