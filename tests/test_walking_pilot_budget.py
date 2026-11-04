"""Shared journal reservations and real foreground synthetic worker timeouts."""
from concurrent.futures import ThreadPoolExecutor
import json
import sys

import pytest

from application.research_budget import BudgetLedger,BudgetSpec
from application.research_supervisor import ResearchSupervisor
from application.walking_pilot_budget import remaining,charge_owner,usage
from application.walking_pilot import ARMS
from infrastructure.research_store import ResearchStore,ResearchError,digest
from tests.test_research_store import spec


def fixture(tmp_path):
    store=ResearchStore(tmp_path,"walking-fixture",initialize=True)
    arms={a:"fixture-"+a.lower() for a in ARMS}
    arms.update({"M0-UO2":"pirc26-m0-o1","M2-UO2":"pirc26-m2-o1"})
    value=spec()
    value["arms"]=[{"arm_id":v,"model_family_id":a,"method_family_id":"synthetic","objective_id":"uo2","budget_seconds":86400} for a,v in arms.items()]
    value["cells"]=[{"arm_id":v,"pilot_arm":a,"pilot_id":"synthetic-pilot","pilot_stage":"predict" if i==2 else "train",
                     "seed":i,"block_id":"synthetic","visibility":"synthetic"} for a,v in arms.items() for i in range(3)]
    store.register(value,digest(value))
    approval={"decider":"azurestarhu","feature":"PIRC-26","decision":"DECISION-01@v1","allowed_pilot":"synthetic-pilot",
              "source_wording_sha256":"ed9e71a39ca450e244851f1ab20ac24e6c1207132d8ee1b93ee9daedcc9ff00e"}
    store.publish("fixture-decision",approval)
    policy={"schema_version":"pirc26-walking-budget-v1","pilot_id":"synthetic-pilot","study_id":"synthetic",
            "global_ms":1800000,"training_ms":120000,"expires_at":"2026-11-03T23:59:00+08:00",
            "motion":"walking","speed_limit":2.,"final_test":False,"arms":arms,
            "approval":{"manifest_id":"fixture-decision","sha256":digest(approval)}}
    store.publish("walking-pilot-synthetic-pilot",policy)
    attempts={arm:[store.new_attempt(store.register_run("synthetic",cell)) for cell in value["cells"] if cell["arm_id"]==arm] for arm in arms.values()}
    return store,arms,attempts


def budget(seconds,phase="train"):
    return BudgetSpec(seconds,category="pilot",pilot_id="synthetic-pilot",pilot_phase=phase)


def test_training_limit_cumulative_but_prediction_remains_funded(tmp_path):
    store,arms,attempts=fixture(tmp_path); ledger=BudgetLedger(store); arm=arms["M0-UO2"]
    first=ledger.reserve(attempts[arm][0],budget(120))
    ledger.settle(first["reservation_id"],119000,outcome="TIMEOUT")
    assert not ledger.balance(arm)["closed"]
    assert remaining(ledger,"synthetic-pilot",arm,"train")==1.
    with pytest.raises(ResearchError,match="training limit"):
        ledger.reserve(attempts[arm][1],budget(1.001))
    prediction=ledger.reserve(attempts[arm][2],budget(10,"predict"))
    ledger.settle(prediction["reservation_id"],2000,outcome="SUCCEEDED")
    reopened=BudgetLedger(ResearchStore(tmp_path,"walking-fixture"))
    assert reopened.balance(arm)["committed_ms"]==121000
    assert remaining(reopened,"synthetic-pilot",arm,"train")==1.
    assert usage(reopened,"synthetic-pilot")["global_ms"]==121000


def test_concurrent_pilot_reservations_serialized_and_new_arms_cannot_bypass(tmp_path):
    store,arms,attempts=fixture(tmp_path); ledger=BudgetLedger(store)
    def reserve(arm):
        try:
            return ledger.reserve(attempts[arm][0],budget(1))
        except ResearchError as exc:
            assert exc.code=="BUDGET_BUSY"
    with ThreadPoolExecutor(max_workers=2) as pool:
        values=list(pool.map(reserve,[arms["A-S"],arms["B-S"]]))
    assert sum(v is not None for v in values)==1
    for v in values:
        if v:
            ledger.settle(v["reservation_id"],100,outcome="FAILED")
    with pytest.raises(ResearchError,match="restricted"):
        ledger.reserve(attempts[arms["C-T-S"]][0],BudgetSpec(1))


def test_actual_slow_worker_train_timeout_partial_checkpoint_and_next_arm(tmp_path):
    store,arms,attempts=fixture(tmp_path); arm=arms["M0-UO2"]
    # Stdlib worker writes atomic usable progress before deliberately hanging.
    code="import json,pathlib,sys,time,os; p=pathlib.Path(sys.argv[1]); stage=p.with_name('progress.staging'); stage.write_text(json.dumps({'checkpoint':{'value':2},'updates':3})); os.replace(stage,p.with_name('progress.json')); time.sleep(30)"
    result=ResearchSupervisor(store).run(attempts[arm][0],lambda output:[sys.executable,"-c",code,str(output)],budget(2),
        resource_plan={"maximum_result_bytes":4*1024**2,"tree_memory_bytes":2*1024**3},
        partial_result_validator=lambda v:assert_partial(v))
    assert result["state"]=="TIMEOUT" and result["partial_checkpoint_artifact_id"]
    assert store.manifest("artifact-"+result["partial_checkpoint_artifact_id"])["role"]=="checkpoint"
    assert BudgetLedger(store).balance(arm)["committed_ms"]>0
    assert not BudgetLedger(store).balance(arm)["closed"]
    other=arms["A-S"]
    r=BudgetLedger(store).reserve(attempts[other][0],budget(1))
    BudgetLedger(store).settle(r["reservation_id"],10,outcome="FAILED")
    assert any(e["event_kind"]=="WORKER_TREE_STOPPED" for e in store.events())


def assert_partial(v):
    assert v["checkpoint"]["value"]==2 and v["updates"]==3


def test_global_timeout_charges_owner_and_failed_worker_without_time_refund(tmp_path):
    store,arms,attempts=fixture(tmp_path); ledger=BudgetLedger(store)
    charge_owner(ledger,"synthetic-pilot",1799500,"owner-fixture")
    assert remaining(ledger,"synthetic-pilot",arms["A-N"],"predict")==.5
    result=ResearchSupervisor(store).run(attempts[arms["A-N"]][2],lambda output:[sys.executable,"-c","import time; time.sleep(30)"],budget(.5,"predict"))
    assert result["state"]=="TIMEOUT"
    with store._read_transaction():
        assert usage(ledger,"synthetic-pilot")["global_ms"]>=1800000
    assert remaining(ledger,"synthetic-pilot",arms["B-N"],"train")==0


def test_unknown_worker_cost_retained_and_arm_closed(tmp_path):
    store,arms,attempts=fixture(tmp_path); ledger=BudgetLedger(store)
    arm=arms["C-R-S"]
    r=ledger.reserve(attempts[arm][0],budget(2))
    ledger.settle(r["reservation_id"],None,outcome="INTERRUPTED")
    assert ledger.balance(arm)["closed"]
    assert usage(ledger,"synthetic-pilot")["global_ms"]==2000
