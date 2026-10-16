"""Closed-arm isolation controls; never use the canonical shared store."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
op = importlib.import_module("pirc27_remaining_path_after_closed_arm")
from infrastructure.research_store import ResearchError, digest

CANONICAL_REVIEWED = deepcopy(op.REVIEWED)


@pytest.fixture(scope="module")
def compiled():
    arms = {m: {"arm_id": "pirc27-affine-"+m+"-mean-x", "model_family_id": op.original.original.FAMILY,
        "method_family_id": m, "objective_id": "endpoint-x", "budget_seconds": 86400}
        for m in op.original.METHODS}
    cells = op.original.frozen(arms)[1]
    return arms, cells, op.original.code_hash(), op.original.source_binding()


@pytest.fixture
def fixture(monkeypatch, tmp_path, compiled):
    arms, cells, source, binding = deepcopy(compiled)
    # The synthetic fixture owns its locally constructed cell. Execution
    # bindings include callable bytecode, which legitimately differs by Python
    # version. Never pretend this fixture is the real Windows/3.10 receipt.
    monkeypatch.setattr(op, "REVIEWED", {**CANONICAL_REVIEWED, "cell_hash": digest(cells[4])})
    spec = {"arms": list(arms.values()), "cells": cells, "code_hash": source,
        "orchestrator_source_sha256": binding,
        "user_authority_hash": digest(op.original.DECISION)}
    attempts, runs = {}, {}
    for index in range(5):
        aid = op.REVIEWED["attempt_id"] if index == 4 else "fixture-"+str(index)
        state = "INTERRUPTED" if index == 4 else "FAILED" if index == 2 else "SUCCEEDED"
        a = {"attempt_id": aid, "run_id": "r"+str(index), "state": state,
            "error_code": "HEARTBEAT_LOST" if index == 4 else "WORKER_FAILED" if index == 2 else None,
            "artifact_id": "fixture-result" if state == "SUCCEEDED" else None}
        attempts[aid] = a
        runs[a["run_id"]] = {"study_id": op.original.STUDY, "spec_hash": digest(spec),
            "arm_id": cells[index]["arm_id"], "cell_hash": digest(cells[index]), "cell": cells[index]}
    balances = {a["arm_id"]: {"closed": m == "reversible-heun", "committed_ms": 340284}
        for m, a in arms.items()}
    stop = {"hash": op.REVIEWED["native_stop_hash"], "payload": {"observed_elapsed_ms": 139783}}
    settled = {"hash": op.REVIEWED["settlement_hash"], "payload": {"charged_ms": 139783}}
    resource = {"status": "RECORDED", "publication_event": {"hash": op.REVIEWED["publication_hash"]},
                "artifact": {"sha256": op.REVIEWED["packet_hash"]}}
    class Store:
        path = tmp_path
        @contextmanager
        def _read_transaction(self):
            yield
        def manifest(self, name):
            if name == "study-"+op.original.STUDY:
                return {"spec": spec}
            if name == op.original.original.ORIGINAL:
                return {"arms": list(arms.values())}
            return runs[name.removeprefix("run-")]
        def attempts(self):
            return attempts
        def events(self):
            return []
        def authorization(self, name):
            assert name == op.original.AUTHORIZATION
            return {"fixture": True}
    store = Store()
    calls = []
    class Index:
        def __init__(self, events):
            pass
        def proof(self, aid):
            fee = {**settled, "payload": {**settled["payload"], "outcome": attempts[aid]["state"]}}
            return stop, fee, None
    monkeypatch.setattr(op, "SPEC_HASH", digest(spec))
    monkeypatch.setattr(op, "ResourceEventIndex", Index)
    monkeypatch.setattr(op, "BudgetLedger", lambda s: SimpleNamespace(balance=lambda aid: balances[aid]))
    monkeypatch.setattr(op, "resource_history", lambda *a, **k: {"attempts": [resource]})
    monkeypatch.setattr(op.original, "accounting", lambda s: {"charged_ms": 3577213, "reserved_ms": 0})
    monkeypatch.setattr(op, "authorize_study", lambda *a: calls.append(("authorize", a[-1])))
    monkeypatch.setattr(op, "require_unexpired_disclosure", lambda *a: calls.append(("expiry", a[-1])))
    monkeypatch.setattr(op, "verify_log", lambda s, aid: calls.append(("log", aid)))
    return SimpleNamespace(store=store, arms=arms, cells=cells, spec=spec, attempts=attempts,
        runs=runs, balances=balances, stop=stop, settled=settled, resource=resource, calls=calls)


def test_production_review_receipt_keeps_exact_canonical_cell():
    assert op.REVIEWED == CANONICAL_REVIEWED
    assert op.REVIEWED["cell_hash"] == "98dc0413b42fab65683e26f249e93a1053d2425fee8918610997e187e761ef8f"
    assert op.REVIEWED["attempt_id"] == "f6b05e77497e48a6bad2c5f04455c0f7"


def test_unchanged_original_binding_and_exact_closed_inventory(fixture):
    f = fixture
    assert op.original.source_binding() == op.SOURCE_BINDING
    before = deepcopy((f.spec, f.attempts, f.runs, f.balances))
    report = op.preflight(f.store)
    assert report["attempted_cells"] == 5 and report["eligible_cells"] == 7
    assert len(report["closed_unattempted_cell_hashes"]) == 4
    assert report["next_cell"] == f.cells[6]
    assert report["next_cell"]["comparison_dimensions"]["method"] == "heun"
    assert report["next_cell"]["comparison_dimensions"]["model_configuration"] == "near-critical-v1"
    assert (f.spec, f.attempts, f.runs, f.balances) == before
    assert ("log", op.REVIEWED["attempt_id"]) in f.calls
    assert f.calls[-1] == ("expiry", "preview")


@pytest.mark.parametrize("fault", ["open", "unreviewed", "error", "artifact", "cell", "arm",
    "stop", "settle", "fee", "publication", "packet"])
def test_exact_diagnosed_receipt_is_not_generic_interruption_permission(fixture, fault):
    f = fixture
    a = deepcopy(f.attempts[op.REVIEWED["attempt_id"]])
    r = deepcopy(f.runs[a["run_id"]])
    b, stop, settled, resource = {"closed": True}, deepcopy(f.stop), deepcopy(f.settled), deepcopy(f.resource)
    if fault == "open": b["closed"] = False
    elif fault == "unreviewed": a["attempt_id"] = "other"
    elif fault == "error": a["error_code"] = "SUPERVISOR_ERROR"
    elif fault == "artifact": a["artifact_id"] = "salvaged"
    elif fault in {"cell", "arm"}: r[fault+"_hash" if fault == "cell" else "arm_id"] = "other"
    elif fault == "stop": stop["hash"] = "other"
    elif fault == "settle": settled["hash"] = "other"
    elif fault == "fee": settled["payload"]["charged_ms"] += 1
    elif fault == "publication": resource["publication_event"]["hash"] = "other"
    elif fault == "packet": resource["artifact"]["sha256"] = "other"
    with pytest.raises(ResearchError):
        op.verify_reviewed(a, r, b, stop, settled, resource)


@pytest.mark.parametrize("fault", ["running", "other-interruption", "control", "duplicate", "open-closed-arm",
    "unproved", "foreign-proof"])
def test_selector_never_dispatches_live_ambiguous_or_retried_attempts(fixture, fault):
    f = fixture
    reviewed = {op.REVIEWED["attempt_id"]}
    a = f.attempts["fixture-0"]
    if fault == "running": a["state"] = "RUNNING"
    elif fault == "other-interruption": a.update(state="INTERRUPTED", error_code="HEARTBEAT_LOST")
    elif fault == "control": a.update(state="PREFLIGHT_FAILED", error_code="UNAUTHORIZED_DATA")
    elif fault == "duplicate":
        f.attempts["duplicate"] = {**a, "attempt_id": "duplicate"}
    elif fault == "open-closed-arm": f.balances[op.REVIEWED["arm_id"]]["closed"] = False
    elif fault == "unproved": reviewed.clear()
    elif fault == "foreign-proof": reviewed.add("other")
    with pytest.raises(ResearchError):
        op.select_cells(f.spec, f.attempts, f.runs, f.balances, reviewed)


@pytest.mark.parametrize("fault", ["spec", "source", "authority", "arms", "stop", "outcome",
    "charge", "resource", "log", "grant", "over-cap", "reserved"])
def test_full_preflight_failures_never_reach_sdk(fixture, monkeypatch, fault):
    f = fixture
    if fault == "spec": monkeypatch.setattr(op, "SPEC_HASH", "0"*64)
    elif fault in {"source", "authority"}:
        f.spec["orchestrator_source_sha256" if fault == "source" else "user_authority_hash"] = "0"*64
        monkeypatch.setattr(op, "SPEC_HASH", digest(f.spec))
    elif fault == "arms": f.arms["heun"]["method_family_id"] = "foreign"
    elif fault == "stop":
        monkeypatch.setattr(op.ResourceEventIndex, "proof", lambda self, aid: (None, None, "UNCONFIRMED_STOP"))
    elif fault == "outcome":
        monkeypatch.setattr(op.ResourceEventIndex, "proof", lambda self, aid: (f.stop,
            {**f.settled, "payload": {**f.settled["payload"], "outcome": "RUNNING"}}, None))
    elif fault == "charge": f.stop["payload"]["observed_elapsed_ms"] += 1
    elif fault == "resource": f.resource["status"] = "UNAVAILABLE"
    elif fault in {"log", "grant"}:
        def reject(*args): raise ResearchError("UNAUTHORIZED_DATA", "fixture rejection")
        monkeypatch.setattr(op, "verify_log" if fault == "log" else "authorize_study", reject)
    elif fault in {"over-cap", "reserved"}:
        monkeypatch.setattr(op.original, "accounting", lambda s: {"charged_ms": 12300001,
            "reserved_ms": 1 if fault == "reserved" else 0})
    def no_runner(*args): pytest.fail("SDK must not be reached after failed preflight")
    monkeypatch.setattr(op, "SharedRunner", no_runner)
    with pytest.raises(ResearchError): op.run_next(f.store)


def test_sdk_receives_only_original_next_cell_and_unchanged_budget(fixture, monkeypatch):
    calls = []
    class Runner:
        def __init__(self, store, registry): assert store is fixture.store
        def run_cell(self, study, cell, *, budget):
            calls.append((study, cell, budget))
            return {"attempt_id": "new-distinct", "state": "SUCCEEDED", "artifact_id": "fixture"}
    monkeypatch.setattr(op, "SharedRunner", Runner)
    report = op.run_next(fixture.store)
    assert len(calls) == 1 and calls[0][:2] == (op.original.STUDY, digest(fixture.cells[6]))
    assert calls[0][2].job_seconds == 300 and calls[0][2].category == "pilot"
    assert len(report["closed_unattempted_cell_hashes"]) == 4


def test_all_closed_cells_remain_visible_and_no_worker_starts(fixture, monkeypatch):
    for balance in fixture.balances.values(): balance["closed"] = True
    monkeypatch.setattr(op, "SharedRunner", lambda *a: pytest.fail("no eligible cell"))
    report = op.run_next(fixture.store)
    assert report["state"] == "ALL_ELIGIBLE_CELLS_ATTEMPTED"
    assert report["attempted_cells"] == 5 and len(report["closed_unattempted_cell_hashes"]) == 11


@pytest.mark.parametrize("fault", [None, "changed", "symlink", "oversized"])
def test_owned_bounded_log_uses_exact_bytes_and_regular_handle(tmp_path, monkeypatch, fault):
    aid = op.REVIEWED["attempt_id"]
    directory = tmp_path/"artifacts"/(".attempt-"+aid)
    directory.mkdir(parents=True)
    raw = b"fixture\ndomain.errors.DataValidationError: discrete covariance is not symmetric\n"
    monkeypatch.setattr(op, "KNOWN_LOG", hashlib.sha256(raw).hexdigest())
    target = directory/"worker.log"
    if fault == "symlink":
        other = directory/"other.log"
        other.write_bytes(raw)
        try: target.symlink_to(other)
        except OSError: pytest.skip("symlink unavailable on this platform")
    else: target.write_bytes(b"x"*262145 if fault == "oversized" else raw+b"changed" if fault == "changed" else raw)
    if fault is None: op.verify_log(SimpleNamespace(path=tmp_path), aid)
    else:
        with pytest.raises((ResearchError, OSError, ValueError)):
            op.verify_log(SimpleNamespace(path=tmp_path), aid)
