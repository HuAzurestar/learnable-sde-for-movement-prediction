import numpy as np
import pytest

from experiments.pirc17.inference import InferenceConfig, SEEDS, factor_verdict, holm_adjust, infer, paired_blocks, planning_power

FAMILY={"effect":("candidate","control")}


def records(deltas, *, seed_changes=None, extra_origins=0):
    rows=[]
    for b,delta in enumerate(deltas):
        for origin in range(1+(extra_origins if b==0 else 0)):
            for s,seed in enumerate(SEEDS):
                for configuration,score in (("control",100.),("candidate",100.+delta+(seed_changes[s] if seed_changes else 0))):
                    rows.append({"configuration":configuration,"origin_id":f"o{b}-{origin}","independent_block_id":f"b{b}",
                        "seed":seed,"score_m":score,"status":"success"})
    return rows


def run(deltas, **kwargs):
    return infer(records(deltas,**kwargs),config=InferenceConfig(delta_m=1),mechanism_passed={"effect":True},family=FAMILY)["results"]["effect"]


@pytest.mark.parametrize("delta,expected",[(-3,"beneficial"),(3,"harmful"),(0,"equivalent"),(-1,"inconclusive"),(1,"inconclusive")])
def test_known_effects_and_exact_practical_boundaries(delta,expected):
    assert run([delta]*30)["verdict"]==expected


def test_holm_stepdown_and_order_restoration():
    np.testing.assert_allclose(holm_adjust([.03,.001,.02]),[.04,.003,.04])


def test_windows_and_seeds_do_not_inflate_block_count_or_weight():
    data=records([-3]*30,extra_origins=40)
    report=infer(data,config=InferenceConfig(delta_m=1),mechanism_passed={"effect":True},family=FAMILY)
    assert report["independent_block_count"]==30
    assert sum(report["origin_count_by_block"])==70
    assert report["results"]["effect"]["delta_estimate_m"]==-3
    assert run([-3]*29)["reason"]=="insufficient_independent_blocks"


def test_mechanism_and_seed_stability_override_strong_mean():
    result=infer(records([-3]*30),config=InferenceConfig(delta_m=1),mechanism_passed={"effect":False},family=FAMILY)
    assert result["results"]["effect"]["reason"]=="mechanism_gate_failed"
    assert run([-10]*30,seed_changes=[0,0,0,11,11])["reason"]=="training_seed_direction_unstable"


def test_missing_failed_duplicate_rows_are_not_intersected_away():
    data=records([-3]*30)
    with pytest.raises(ValueError,match="complete matched"):
        paired_blocks(data[:-1],FAMILY)
    with pytest.raises(ValueError,match="duplicate"):
        paired_blocks(data+[data[0]],FAMILY)
    data[0]["status"]="failure"
    with pytest.raises(ValueError,match="failed"):
        paired_blocks(data,FAMILY)


def test_tail_check_reproducibility_and_underresolved_family_rejection():
    data=records(np.linspace(-4,2,35))
    config=InferenceConfig(delta_m=1,tail_tolerance_fraction_of_delta=1e-12)
    a=infer(data,config=config,mechanism_passed={"effect":True},family=FAMILY)
    b=infer(data,config=config,mechanism_passed={"effect":True},family=FAMILY)
    assert a==b and a["results"]["effect"]["reason"]=="bootstrap_tail_monte_carlo_unstable"
    with pytest.raises(ValueError,match="ten expected"):
        InferenceConfig(delta_m=1,bootstrap_iterations=100).validate(5)


def test_planning_power_is_forward_assumption_not_observed_p_value():
    modest=planning_power(paired_sd_m=10,blocks=30,true_improvement_m=2,delta_m=1)
    larger=planning_power(paired_sd_m=10,blocks=300,true_improvement_m=2,delta_m=1)
    assert larger["approximate_power"]>modest["approximate_power"]
    assert modest["required_blocks_normal_approximation"]>300


def test_degenerate_bootstrap_variance_is_not_false_precision():
    result=run([0]*29+[100])
    assert result["verdict"]=="inconclusive"
    assert not result["interval_finite"]
    assert result["reason"]=="degenerate_bootstrap_uncertainty"


def test_all_five_factor_dispositions_and_conflict_precedence():
    good=run([-3]*30)
    harmful=run([3]*30)
    equivalent=run([0]*30)
    gates=dict(conflict_unresolved=False,correlated_group_rule_passed=True,power_qualified=True)
    assert factor_verdict(good,good,**gates)["verdict"]=="retain"
    assert factor_verdict(harmful,harmful,**gates)["verdict"]=="harmful"
    redundant=factor_verdict(equivalent,good,**gates)
    assert redundant["verdict"]=="redundant" and "not absence" in redundant["scope"]
    assert factor_verdict(good,harmful,**gates)["verdict"]=="inconclusive"
    assert factor_verdict(None,None,unavailable_reason="no valid paired runs",**gates)["verdict"]=="unavailable"
    gates["power_qualified"]=False
    assert factor_verdict(good,good,**gates)["reason"]=="power_qualification_not_met"
