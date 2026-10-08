import copy

import pytest

from experiments.pirc17.numerical_check import audit


def fixture():
    header={"type":"header","schema_version":"pirc17-development-rollout-v1",
        "purpose":"bounded_validation_engineering_pilot","final_eval_label_prediction_metric_reads":0,
        "sample_ids":["a"],"configurations":["base"],"seeds":[1],"particle_counts":[8,16],
        "max_steps_seconds":[1,2],"expected_run_count":4,"brownian_driver":"pirc17-coupled-brownian-v2"}
    runs=[]
    for particles in (8,16):
        for step in (1,2):
            runs.append({"type":"run","sample_id":"a","independent_block_id":"b","configuration":"base",
                "seed":1,"particles":particles,"max_step_seconds":step,"actual_horizons_seconds":[1,2],
                "brownian_identity":{"version":header["brownian_driver"],"seed":1,"stream_id":"a","max_particles":16,"path_sha256":"fixture"},
                "status":"success","scores":{"time_weighted_energy_score_m":10+step,
                    "by_time":[{"elapsed_seconds":t,"energy_score_m":10+step} for t in (1,2)]}})
    return [header,*runs,{"type":"completion","attempted_run_count":4,"success_count":4,"failure_count":0}]


def test_complete_audit_is_not_certification():
    result=audit(fixture(),tolerance_m=.5)
    assert result["certified"] is False
    assert result["independent_block_count"]==1
    assert len(result["sensitivities"])==4
    assert [r["all_scoring_times_within_tolerance"] for r in result["sensitivities"]]==[False,False,True,True]


@pytest.mark.parametrize("change",["missing","duplicate","path","stream","score_time","completion","final"])
def test_inconsistent_ledger_fails_closed(change):
    rows=fixture()
    if change=="missing":
        rows.pop(1)
    elif change=="duplicate":
        rows[2]=copy.deepcopy(rows[1])
    elif change=="path":
        rows[1]["brownian_identity"]["path_sha256"]="changed"
    elif change=="stream":
        rows[1]["brownian_identity"]["stream_id"]="other"
    elif change=="score_time":
        rows[1]["scores"]["by_time"][0]["elapsed_seconds"]=.5
    elif change=="completion":
        rows[-1]["success_count"]=3
    else:
        rows[0]["final_eval_label_prediction_metric_reads"]=1
    with pytest.raises(ValueError):
        audit(rows,tolerance_m=.5)


def test_failure_is_retained_and_blocks_sensitivity_claim():
    rows=fixture()
    rows[1]["status"]="failure"
    rows[1].pop("scores")
    rows[-1].update(success_count=3,failure_count=1)
    result=audit(rows,tolerance_m=.5)
    assert len(result["failures"])==1
    assert result["sensitivities"]==[]
    assert not result["certified"]
