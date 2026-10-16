"""Immutable all-attempt resource evidence in isolated real owner stores."""
import json
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_evidence import export_evidence
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_store import digest, encode
from tests.test_propagation_shared_adapter import prepare
from application.propagation_resource_evidence import ResourceEventIndex, _publication, record_observation, resource_history
from infrastructure.research_store import ResearchError


def run(root, *, fail=False):
    changes = {"samples": 2, "steps": 4}
    if fail:
        changes["initial_mean"] = (1.79e308, 0., 1.79e308, 0.)
    store, spec, registry = prepare(root, changes=changes, resource_observation=True)
    store.register(spec, digest(spec))
    result = SharedRunner(store, registry).run_cell(spec["study_id"], digest(spec["cells"][0]), budget=BudgetSpec(60))
    grant = store.authorization(spec["admission"]["authorization_id"])
    return store, spec, registry, grant, result


def paper_check(bundle):
    paper = Path(__file__).resolve().parents[2]/"TSDE-SDE"
    if not paper.is_dir():
        pytest.skip("independent resource reader requires sibling Paper checkout")
    return subprocess.run([sys.executable, "-B", "-c",
        "import json,sys; from scripts.pirc25.aggregate import validate_bundle; "
        "validate_bundle(json.load(sys.stdin)); print('independent-resource-ok')"],
        cwd=paper, input=encode(bundle), capture_output=True, timeout=30)


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
    independent = paper_check(bundle)
    assert independent.returncode == 0, independent.stderr.decode("utf-8", errors="replace")
    if fail:
        assert row["metrics"] is None and row["artifact_id"] is None


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    return run(tmp_path_factory.mktemp("recorded-resource"))


def test_resource_replay_does_not_read_protected_prediction_or_charge_again(recorded, monkeypatch):
    store, spec, _, _, outcome = recorded
    before = store.events()
    def forbidden(*args, **kwargs):
        raise AssertionError("idempotent publication cannot reopen an artifact")
    monkeypatch.setattr(store, "read_artifact", forbidden)
    record_observation(store, outcome["attempt_id"], spec, spec["cells"][0], None)
    assert store.events() == before


@pytest.mark.parametrize("mutation", ["extra-field", "run", "code", "state", "result", "stop", "cost", "ordering", "reason", "status"])
def test_resealed_publication_source_substitution_is_refused(recorded, mutation):
    store, spec, _, _, outcome = recorded
    index = ResourceEventIndex(store.events())
    event = deepcopy(next(e for e in index.by_attempt[outcome["attempt_id"]]
                          if e["event_kind"] == "PROPAGATION_RESOURCE_RECORDED"))
    fields = {"run": "run_id", "code": "code_hash", "state": "attempt_state", "result": "result_artifact_id",
              "stop": "worker_stop_event_hash", "cost": "settlement_event_hash"}
    if mutation in fields:
        event["payload"][fields[mutation]] = "0"*64
    elif mutation == "ordering":
        event["sequence"] = 1
    elif mutation == "extra-field":
        event["payload"]["private_path"] = "not-allowed"
    elif mutation == "reason":
        event["payload"]["reason"] = "MISSING_OBSERVATION"
    else:
        event["payload"].update(status="UNAVAILABLE", reason="INVENTED", artifact_id=None, packet_hash=None)
    event["hash"] = digest({k: v for k, v in event.items() if k != "hash"})
    with pytest.raises(ResearchError):
        _publication(index, event, spec, spec["cells"][0], store.attempts()[outcome["attempt_id"]])


def test_unpublished_attempt_is_unknown_not_zero_and_not_backfilled(recorded):
    store, spec, _, grant, outcome = recorded
    attempt = store.attempts()[outcome["attempt_id"]]
    index = ResourceEventIndex([e for e in store.events() if e["event_kind"] != "PROPAGATION_RESOURCE_RECORDED"])
    usage = resource_history(store, spec, spec["cells"][0], [attempt], grant, index=index)
    source = usage["attempts"][0]
    assert source["status"] == "UNAVAILABLE" and source["reason"] == "NOT_RECORDED"
    assert all(source[k] is None for k in ("artifact", "packet", "publication_event", "worker_stop_event", "settlement_event"))


def test_resource_export_rechecks_original_grant(recorded):
    store, spec, _, grant, outcome = recorded
    narrow = deepcopy(grant)
    narrow["purposes"] = ["evaluate"]
    with pytest.raises(ResearchError):
        resource_history(store, spec, spec["cells"][0], [store.attempts()[outcome["attempt_id"]]], narrow)


def test_native_proof_does_not_accept_absent_stop_or_unsettled_cost(recorded):
    store, _, _, _, outcome = recorded
    events = store.events()
    for omitted, reason in (("WORKER_STARTED", "NOT_LAUNCHED"), ("WORKER_TREE_STOPPED", "UNCONFIRMED_STOP"),
                             ("SETTLE", "UNSETTLED_COST")):
        index = ResourceEventIndex([e for e in events if e["event_kind"] != omitted])
        assert index.proof(outcome["attempt_id"])[2] == reason


@pytest.fixture(scope="module")
def exported(recorded):
    store, spec, _, grant, _ = recorded
    return export_evidence(store, spec["study_id"], grant)


@pytest.mark.parametrize("mutation", ["unit", "scope", "counter", "negative", "boolean", "cpu-zero",
    "unknown-zero", "resumed", "source-code", "extra-packet", "history", "missing-usage", "peak-sum", "settlement", "publication",
    "strip-registration", "empty-registration", "strip-both"])
def test_independent_reader_refuses_resealed_invalid_resources(exported, mutation):
    bundle = deepcopy(exported)
    row = bundle["cells"][0]
    usage = row["resource_usage"]
    source = usage["attempts"][0]
    value = source["packet"]
    if mutation == "unit":
        value["memory"]["units"] = "KiB"
    elif mutation == "scope":
        value["memory"]["includes_children"] = True
    elif mutation == "counter":
        value["memory"]["resident_counter"] = "declared-memory-limit"
    elif mutation == "negative":
        value["duration"]["wall_ns"] = -1
    elif mutation == "boolean":
        value["includes_previous_attempts"] = 0
    elif mutation == "cpu-zero":
        value["duration"]["cpu_ns"] = False
    elif mutation == "unknown-zero":
        value["memory"].update(status="UNAVAILABLE", peak_resident_bytes=0, resident_counter=None,
                               reason="NATIVE_COUNTER_UNAVAILABLE")
    elif mutation == "resumed":
        value["resumed"] = True
    elif mutation == "source-code":
        value["code_hash"] = "0"*64
    elif mutation == "extra-packet":
        value["process_id"] = 123
    elif mutation == "history":
        usage["attempts"] = []
    elif mutation == "missing-usage":
        del row["resource_usage"]
    elif mutation == "peak-sum":
        usage["peak_aggregation"] = "sum"
    elif mutation == "settlement":
        # The real export shares the same Python event object with cost.sources.
        # Detach it to test a resource/cost disagreement, not a cost-schema error.
        source["settlement_event"] = deepcopy(source["settlement_event"])
        source["settlement_event"]["payload"]["charged_ms"] = 0
        source["settlement_event"]["hash"] = digest({k: v for k, v in source["settlement_event"].items() if k != "hash"})
        source["publication_event"]["payload"]["settlement_event_hash"] = source["settlement_event"]["hash"]
    elif mutation in {"strip-registration", "empty-registration", "strip-both"}:
        if mutation == "empty-registration":
            row["registered_cell"] = {}
        else:
            del row["registered_cell"]
        if mutation == "strip-both":
            del row["resource_usage"]
    else:
        source["publication_event"]["sequence"] = 1
    # Reseal every outer transport/content pointer. Refusal must come from the
    # independent scientific source/scope checks, not a stale packet hash.
    aid = digest(value)
    source["artifact"].update(artifact_id=aid, sha256=aid, size_bytes=len(encode(value)))
    source["publication_event"]["payload"].update(artifact_id=aid, packet_hash=aid)
    event = source["publication_event"]
    event["hash"] = digest({k: v for k, v in event.items() if k != "hash"})
    bundle["bundle_hash"] = digest({k: v for k, v in bundle.items() if k != "bundle_hash"})
    checked = paper_check(bundle)
    assert checked.returncode != 0, checked.stdout
    assert b"resource evidence" in checked.stderr, checked.stderr
