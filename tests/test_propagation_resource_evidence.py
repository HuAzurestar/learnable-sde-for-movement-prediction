"""Immutable all-attempt resource evidence in isolated real owner stores."""
import json

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_evidence import export_evidence
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_store import digest, encode
from tests.test_propagation_shared_adapter import prepare


def run(root, *, fail=False):
    changes = {"samples": 2, "steps": 4}
    if fail:
        changes["initial_mean"] = (1.79e308, 0., 1.79e308, 0.)
    store, spec, registry = prepare(root, changes=changes, resource_observation=True)
    store.register(spec, digest(spec))
    result = SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]), budget=BudgetSpec(60))
    grant = store.authorization(spec["admission"]["authorization_id"])
    return store, spec, registry, grant, result


@pytest.mark.parametrize("fail", [False, True])
def test_actual_stopped_settled_worker_resource_artifact_exports_even_on_failure(tmp_path, fail):
    store, spec, _, grant, outcome = run(tmp_path, fail=fail)
    assert outcome["state"] == ("FAILED" if fail else "SUCCEEDED"), outcome
    balance = BudgetLedger(store).balance(spec["cells"][0]["arm_id"])
    assert balance["committed_ms"] == outcome["elapsed_ms"] > 0
    bundle = export_evidence(store, spec["study_id"], grant)
    row = bundle["cells"][0]
    usage = row["resource_usage"]
    assert usage["schema_version"] == "pirc27-resource-history-v1"
    assert usage["peak_aggregation"] == "not-additive;not-a-phase-delta;not-a-tree-peak"
    assert len(usage["attempts"]) == 1
    source = usage["attempts"][0]
    assert source["attempt_id"] == outcome["attempt_id"] and source["status"] == "RECORDED"
    assert source["packet"]["execution_state"] == ("FAILED" if fail else "SUCCEEDED")
    assert source["packet"]["memory"]["status"] == "MEASURED"
    assert source["artifact"]["role"] == "propagation-resource-observation"
    assert source["artifact"]["artifact_id"] == digest(source["packet"])
    assert source["worker_stop_event"]["sequence"] < source["settlement_event"]["sequence"] < source["publication_event"]["sequence"]
    event = source["publication_event"]
    assert event["event_kind"] == "PROPAGATION_RESOURCE_RECORDED"
    assert event["payload"]["result_artifact_id"] == outcome["artifact_id"]
    assert source["settlement_event"]["payload"]["charged_ms"] == outcome["elapsed_ms"]
    assert store.read_artifact(source["artifact"]["artifact_id"], purpose="evaluate", authorization=grant) == encode(source["packet"])
    assert BudgetLedger(store).balance(spec["cells"][0]["arm_id"]) == balance
    assert bundle == export_evidence(store, spec["study_id"], grant)
    if fail:
        assert row["metrics"] is None and row["artifact_id"] is None
