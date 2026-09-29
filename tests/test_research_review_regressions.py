"""Independent-review counterexamples: regressions must fail before fixes."""

import json

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_evidence import export_evidence
from application.research_supervisor import ResearchSupervisor
from experiments.pirc25.__main__ import main
from infrastructure.research_store import ResearchError, ResearchStore, digest, encode
from tests.test_research_budget import registered
from tests.test_research_evidence import setup


def test_r2_generic_cli_show_cannot_disclose_restricted_bundle(tmp_path, capsys):
    store, value, grant = setup(tmp_path)
    cell = value["cells"][0]
    # An authorized export already exists; this does not authorize a later show.
    restricted = {**grant, "authorization_id": "restricted-export",
                  "visibilities": ["restricted", "synthetic"]}
    store.authorize(restricted)
    run_id = store.register_run("synthetic", cell)
    attempt = store.new_attempt(run_id)
    store.transition(attempt, "RUNNING")
    result = {"spec_hash": digest(value), "cell_hash": digest(cell),
              "protocol_hash": value["protocol_hash"], "metrics": {"error": 1},
              "metric_units": {"error": "m"}, "qualification": "fixture",
              "state_order": ["x", "y", "vx", "vy"], "units": ["m", "m", "m/s", "m/s"]}
    artifact = store.artifact(encode(result), role="result", visibility="restricted",
                              block_ids=[cell["block_id"]], study_id="synthetic")
    store.transition(attempt, "SUCCEEDED", artifact_id=artifact["artifact_id"])
    bundle = export_evidence(store, "synthetic", restricted)
    before = len(store.events())
    status = main(["--root", str(tmp_path), "--store-id", "evidence",
                   "show", "bundle-" + bundle["bundle_hash"]])
    response = json.loads(capsys.readouterr().out)
    assert status != 0, "generic show bypassed export authorization"
    assert response["error"]["code"] == "UNAUTHORIZED_DATA"
    assert "metrics" not in json.dumps(response)
    assert len(store.events()) == before  # Denial never discloses a result.


@pytest.mark.parametrize("kind", ["bundle", "comparison", "authorization", "future-result"])
def test_r2_generic_show_fails_closed_for_non_metadata_kinds(tmp_path, capsys, kind):
    store, _, _ = setup(tmp_path)
    store.publish(kind + "-private", {"metrics": {"private": 42}})
    assert main(["--root", str(tmp_path), "--store-id", "evidence", "show", kind + "-private"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["error"]["code"] == "UNAUTHORIZED_DATA"
    assert set(result) == {"error"}
    assert "metrics" not in json.dumps(result)


def test_r2_show_still_supports_registration_metadata(tmp_path, capsys):
    _, value, _ = setup(tmp_path)
    assert main(["--root", str(tmp_path), "--store-id", "evidence", "show", "study-synthetic"]) == 0
    assert json.loads(capsys.readouterr().out)["spec_hash"] == digest(value)


@pytest.mark.parametrize("settled", [False, True])
def test_r4_budget_rejection_persists_actionable_terminal_attempt(tmp_path, settled):
    store, attempts = registered(tmp_path, 13)
    ledger = BudgetLedger(store)
    reservations = [ledger.reserve(attempt, BudgetSpec()) for attempt in attempts[:12]]
    if settled:
        for reservation in reservations:
            ledger.settle(reservation["reservation_id"], 7200000, outcome="FAILED")
    launched = []
    with pytest.raises(ResearchError) as caught:
        ResearchSupervisor(store).run(attempts[-1], lambda _: launched.append(True), BudgetSpec())
    rejected = store.attempts()[attempts[-1]]
    assert not launched
    assert rejected["state"] in store.TERMINAL
    assert rejected["error_code"] == ("BUDGET_EXHAUSTED" if settled else "BUDGET_BUSY")
    assert rejected["ended_at"]
    assert caught.value.envelope()["retryable"] is (not settled)
    reopened = ResearchStore(tmp_path, "budget-fixture")
    assert reopened.attempts()[attempts[-1]] == rejected
    assert len(ledger._state()[0]) == 12
    # Rejection must not release or settle another attempt's reservation.
    assert all(r["settled"] == settled for r in ledger._state()[0].values())
    replacement = store.new_attempt(rejected["run_id"], parent_attempt_id=attempts[-1],
                                     reason="retry after recorded preflight rejection")
    assert replacement != attempts[-1]
    if not settled:
        ledger.settle(reservations[0]["reservation_id"], 0, outcome="FAILED")
        accepted = ledger.reserve(replacement, BudgetSpec())
        assert accepted["attempt_id"] == replacement
        assert not ledger.balance("affine")["closed"]
