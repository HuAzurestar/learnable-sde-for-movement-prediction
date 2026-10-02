"""Actual managed graph provenance, tamper rejection and SVG safety."""

import hashlib
import json

import pytest

from application.research_budget import BudgetSpec
from application.research_evidence import accept_evidence_package
from application.research_query import ResearchQuery
from infrastructure.research_store import ResearchError, encode
from tests.test_research_comparison import source, compute


def test_actual_managed_figures_are_frozen_authorized_and_import_verified(source, tmp_path):
    result = compute(source, budget=BudgetSpec(20), output=tmp_path / "figures")
    assert result["state"] == "SUCCEEDED", result
    store = source[0]
    query = ResearchQuery(store, "viewer")
    viewed = query.comparison(result["comparison"]["aggregate_hash"])
    index = viewed["figure_index"]
    assert index["aggregate_hash"] == viewed["aggregate"]["aggregate_hash"]
    assert index["computation_ref"] == viewed["aggregate"]["computation_ref"]
    entry = viewed["package"]["figures"][0]
    content, media = query.artifact(entry["artifact_id"], export=True)
    assert media == "image/svg+xml" and hashlib.sha256(content).hexdigest() == entry["sha256"]
    filename = entry["filename"]
    assert (tmp_path / "figures" / filename).read_bytes() == content
    grant = {**source[2], "authorization_id": "preview-only", "purposes": ["preview"]}
    store.authorize(grant)
    assert ResearchQuery(store, "preview-only").artifact(entry["artifact_id"])[0] == content
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        ResearchQuery(store, "preview-only").artifact(entry["artifact_id"], export=True)
    # A locally rehashed manifest cannot replace the actual worker's graph.
    directory = tmp_path / "figures"
    (directory / filename).write_bytes(content.replace(b"Frozen comparison", b"Changed comparison"))
    manifest = json.loads((directory / "manifest.json").read_bytes())
    manifest["files"][filename] = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
    (directory / "manifest.json").write_bytes(encode(manifest))
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        accept_evidence_package(store, directory, result["comparison"]["aggregate_hash"])


@pytest.mark.parametrize("payload", [
    '<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>',
    '<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)" />',
    '<svg xmlns="http://www.w3.org/2000/svg"><image href="https://external.example" /></svg>',
    '<!DOCTYPE svg [<!ENTITY e SYSTEM "file:///private">]><svg xmlns="http://www.w3.org/2000/svg">&e;</svg>',
    '<svg xmlns="http://www.w3.org/2000/svg"><foreignObject /></svg>',
    '<svg xmlns="http://www.w3.org/2000/svg"><rect style="fill:url(https://external.example)" /></svg>',
])
def test_active_or_external_svg_is_not_an_allowed_read_artifact(payload):
    from application.research_figures import validate_figure_svg
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        validate_figure_svg(payload.encode())


def test_registered_graph_plan_is_bounded_before_worker_launch():
    from application.research_computation import comparison_plan
    rows = [{"block_id": "block", "comparison_dimensions": {"horizon": str(i)}, "metrics": {}} for i in range(257)]
    with pytest.raises(ResearchError, match="RESOURCE_PLAN_REJECTED"):
        comparison_plan({"cells": rows}, 20_000_000)
