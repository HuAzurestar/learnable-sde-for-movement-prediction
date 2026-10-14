"""Independent stdlib reader/CLI against disposable authorized owner exports."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import subprocess
import sys

from infrastructure.research_store import digest, encode


def independent_reader():
    return _load_paper_module("path_qualification")


def independent_aggregate():
    return _load_paper_module("aggregate")


def _load_paper_module(name):
    path = Path(__file__).resolve().parents[2]/("TSDE-SDE/scripts/pirc25/"+name+".py")
    sys.path.insert(0, str(path.parent))
    try:
        entry = importlib.util.spec_from_file_location("independent_path_"+name, path)
        reader = importlib.util.module_from_spec(entry)
        entry.loader.exec_module(reader)
    finally:
        sys.path.pop(0)
    return reader


def paper_validate(tmp_path, bundle):
    paper = Path(__file__).resolve().parents[2]/"TSDE-SDE"
    path = tmp_path/"path-bundle.json"
    path.write_bytes(encode(bundle))
    return subprocess.run([sys.executable, "-B", str(paper/"scripts/pirc25/validate_paths.py"), str(path),
        "--expected-hash", bundle["bundle_hash"]], cwd=paper, capture_output=True, text=True, timeout=30)


def reseal_transport(bundle):
    """Rehash exported copies only; never change the journal or grants."""
    row = bundle["cells"][0]
    receipt = row["admission"]
    evidence = receipt["documents"].get("propagation_qualification")
    if evidence:
        source = evidence["source_result"]
        analysis = source["forecast"]["path_qualification_analysis"]
        for key in ("continuous", "target"):
            analysis[key+"_certificate_hash"] = digest(analysis[key+"_certificate"])
        analysis["completed_statistics_hash"] = digest(analysis["completed_statistics"])
        analysis["analysis_hash"] = digest({k: v for k, v in analysis.items() if k != "analysis_hash"})
        source["output_hash"] = digest({k: source[k] for k in ("metrics", "forecast", "fit", "source_schema")})
        metadata = evidence["source_artifact"]
        metadata.update(artifact_id=digest(source), sha256=digest(source), size_bytes=len(encode(source)))
        evidence["source_attempt"].update(artifact_id=metadata["artifact_id"], artifact_manifest_hash=digest(metadata))
        receipt["documents"]["package"]["payload"]["managed_path_qualification"]["source_artifact_id"] = metadata["artifact_id"]
        for key in ("reservation_event", "worker_event", "stop_event", "settlement_event", "admission_event", "completion_event"):
            if key == "completion_event":
                evidence[key]["payload"] = deepcopy(evidence["source_attempt"])
            evidence[key]["hash"] = digest({k: v for k, v in evidence[key].items() if k != "hash"})
        evidence["evidence_hash"] = digest({k: v for k, v in evidence.items() if k != "evidence_hash"})
    receipt["admission_hash"] = digest({k: v for k, v in receipt.items() if k != "admission_hash"})
    row["admission_hash"] = receipt["admission_hash"]
    result = row["result"]
    result["admission_hash"] = receipt["admission_hash"]
    current = result["forecast"]["path_output_analysis"]
    if evidence:
        current["qualification_evidence_hash"] = evidence["evidence_hash"]
    current["completed_statistics_hash"] = digest(current["completed_statistics"])
    current["analysis_hash"] = digest({k: v for k, v in current.items() if k != "analysis_hash"})
    result["output_hash"] = digest({k: result[k] for k in ("metrics", "forecast", "fit", "source_schema")})
    row["result_artifact"].update(artifact_id=digest(result), sha256=digest(result), size_bytes=len(encode(result)))
    row["artifact_id"] = digest(result)
    bundle["bundle_hash"] = digest({k: v for k, v in bundle.items() if k != "bundle_hash"})
