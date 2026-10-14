"""Paired independent-block inference conditional on the five registered seeds.

Studentized bootstrap bands have nominal, not exact finite-sample coverage.
Holm adjusted p-values and Bonferroni bootstrap-t bands are distinct outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
import math

import numpy as np

INFERENCE_VERSION = "pirc17-paired-block-inference-v2"
SEEDS = (20260814,20260815,20260816,20260817,20260818)
PRIMARY_FAMILY = {
    "all-vs-base": ("all-terrain","base"),
    "road": ("all-terrain","loo-road"),
    "river": ("all-terrain","loo-river"),
    "worldcover": ("all-terrain","loo-worldcover"),
    "surface": ("all-terrain","loo-surface"),
}


@dataclass(frozen=True)
class InferenceConfig:
    delta_m: float
    alpha: float = .05
    bootstrap_iterations: int = 2000
    bootstrap_seed: int = 20260926
    mc_check_seed: int = 20260927
    minimum_blocks: int = 30
    required_agreeing_seeds: int = 4
    tail_tolerance_fraction_of_delta: float = .25

    def validate(self, family_size):
        if not np.isfinite(self.delta_m) or self.delta_m<=0 or not 0<self.alpha<1:
            raise ValueError("positive practical margin and valid family alpha required")
        for name in ("bootstrap_iterations","minimum_blocks","required_agreeing_seeds"):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError(f"positive integer {name} required")
        if self.required_agreeing_seeds>len(SEEDS):
            raise ValueError("agreement threshold exceeds registered seeds")
        if self.bootstrap_seed==self.mc_check_seed:
            raise ValueError("Monte Carlo tail check requires a distinct fixed seed")
        if self.bootstrap_iterations*self.alpha/(2*family_size)<10:
            raise ValueError("fewer than ten expected bootstrap draws per simultaneous-interval tail")
        if not np.isfinite(self.tail_tolerance_fraction_of_delta) or self.tail_tolerance_fraction_of_delta<=0:
            raise ValueError("positive tail stability tolerance required")


def holm_adjust(p_values):
    values=np.asarray(p_values,dtype=float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or np.any((values<0)|(values>1)):
        raise ValueError("finite p-values in [0,1] required")
    order=np.argsort(values,kind="stable")
    adjusted=np.minimum(1,np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values)))))
    result=np.empty_like(values)
    result[order]=adjusted
    return result


def paired_blocks(rows, family=PRIMARY_FAMILY):
    """Require a complete Cartesian config x origin x seed ledger, never intersect away failures."""
    if not family or any(a==b for a,b in family.values()):
        raise ValueError("nonempty directed contrasts with distinct configurations required")
    configurations={c for pair in family.values() for c in pair}
    ledger={c:{} for c in configurations}
    origin_blocks={}
    for row in rows:
        configuration=row["configuration"]
        if configuration not in configurations:
            raise ValueError("submit primary/supporting families separately; unknown configuration")
        if row.get("status")!="success" or not np.isfinite(row.get("score_m",np.nan)):
            raise ValueError("failed or nonfinite score in required paired ledger")
        origin,block,seed=row["origin_id"],row["independent_block_id"],row["seed"]
        if not origin or not block or seed not in SEEDS:
            raise ValueError("explicit origin, block and registered seed required")
        if origin in origin_blocks and origin_blocks[origin]!=block:
            raise ValueError("origin changes independent block")
        origin_blocks[origin]=block
        key=(origin,seed)
        if key in ledger[configuration]:
            raise ValueError("duplicate origin/seed/configuration row")
        ledger[configuration][key]=float(row["score_m"])
    expected={(origin,seed) for origin in origin_blocks for seed in SEEDS}
    if not expected or any(set(table)!=expected for table in ledger.values()):
        raise ValueError("paired ledger lacks a complete matched origin/seed grid")
    blocks=sorted(set(origin_blocks.values()))
    contrasts=list(family)
    values=np.empty((len(blocks),len(SEEDS),len(contrasts)))
    origin_counts=[]
    for b,block in enumerate(blocks):
        origins=sorted(o for o,v in origin_blocks.items() if v==block)
        origin_counts.append(len(origins))
        for s,seed in enumerate(SEEDS):
            for c,contrast in enumerate(contrasts):
                candidate,control=family[contrast]
                values[b,s,c]=np.mean([ledger[candidate][o,seed]-ledger[control][o,seed] for o in origins])
    return blocks,contrasts,values,origin_counts


def _resample_means(block_means, iterations, seed):
    rng=np.random.default_rng(seed)
    output=np.empty((iterations,block_means.shape[1]))
    standard_errors=np.empty_like(output)
    for start in range(0,iterations,128):
        stop=min(start+128,iterations)
        indexes=rng.integers(len(block_means),size=(stop-start,len(block_means)))
        selected=block_means[indexes]
        output[start:stop]=selected.mean(axis=1)
        standard_errors[start:stop]=selected.std(axis=1,ddof=1)/np.sqrt(len(block_means))
    return output,standard_errors


def _studentized(bootstrap_means, bootstrap_errors, estimates):
    residual=bootstrap_means-estimates
    tolerance=32*np.finfo(float).eps*np.maximum(1,np.abs(estimates))
    valid=bootstrap_errors>tolerance
    ratios=np.zeros_like(residual)
    np.divide(residual,bootstrap_errors,out=ratios,where=valid)
    # A heterogeneous bootstrap draw with zero estimated variance contributes
    # an unbounded pivot, never a spuriously precise finite interval.
    ratios[(~valid)&(residual>tolerance)]=np.inf
    ratios[(~valid)&(residual < -tolerance)]=-np.inf
    return ratios


def _t_interval(pivots, estimates, errors, tail):
    # Order statistics avoid interpolating infinities from degenerate draws.
    quantiles=np.quantile(pivots,[tail,1-tail],axis=0,method="inverted_cdf")
    return np.stack((estimates-quantiles[1]*errors,estimates-quantiles[0]*errors))


def infer(rows, *, config: InferenceConfig, mechanism_passed, family=PRIMARY_FAMILY):
    blocks,names,values,origin_counts=paired_blocks(rows,family)
    if len(blocks)<2:
        raise ValueError("at least two independent blocks needed to estimate uncertainty")
    config.validate(len(names))
    if set(mechanism_passed)!=set(names) or any(type(v) is not bool for v in mechanism_passed.values()):
        raise ValueError("explicit boolean mechanism status for every comparison required")
    block_means=values.mean(axis=1)  # Seeds are NOT additional independent blocks.
    estimates=block_means.mean(axis=0)
    errors=block_means.std(axis=0,ddof=1)/np.sqrt(len(blocks))
    primary,primary_errors=_resample_means(block_means,config.bootstrap_iterations,config.bootstrap_seed)
    repeat,repeat_errors=_resample_means(block_means,config.bootstrap_iterations,config.mc_check_seed)
    primary_pivots=_studentized(primary,primary_errors,estimates)
    repeat_pivots=_studentized(repeat,repeat_errors,estimates)
    tail=config.alpha/(2*len(names))
    simultaneous=_t_interval(primary_pivots,estimates,errors,tail)
    repeat_simultaneous=_t_interval(repeat_pivots,estimates,errors,tail)
    marginal=_t_interval(primary_pivots,estimates,errors,config.alpha/2)
    # Centred bootstrap-t empirical null, H0 mean paired delta=0 (two-sided).
    observed=_studentized(estimates[None,:],errors[None,:],np.zeros(len(names)))[0]
    p=(1+np.sum(np.abs(primary_pivots)>=np.abs(observed),axis=0))/(config.bootstrap_iterations+1)
    adjusted=holm_adjust(p)
    seed_means=values.mean(axis=0)
    results={}
    for c,name in enumerate(names):
        lower,upper=simultaneous[:,c]
        finite=np.isfinite(simultaneous[:,c]).all() and np.isfinite(repeat_simultaneous[:,c]).all()
        tail_shift=float(np.max(np.abs(simultaneous[:,c]-repeat_simultaneous[:,c]))) if finite else None
        stable=bool(finite and tail_shift<=config.delta_m*config.tail_tolerance_fraction_of_delta)
        beneficial=int(np.count_nonzero(seed_means[:,c]<0))
        harmful=int(np.count_nonzero(seed_means[:,c]>0))
        equivalent=int(np.count_nonzero(np.abs(seed_means[:,c])<config.delta_m))
        verdict,reason="inconclusive","interval_overlaps_practical_boundary_or_not_significant"
        if not mechanism_passed[name]:
            reason="mechanism_gate_failed"
        elif len(blocks)<config.minimum_blocks:
            reason="insufficient_independent_blocks"
        elif not finite:
            reason="degenerate_bootstrap_uncertainty"
        elif not stable:
            reason="bootstrap_tail_monte_carlo_unstable"
        elif upper < -config.delta_m and adjusted[c]<=config.alpha:
            if beneficial>=config.required_agreeing_seeds:
                verdict,reason="beneficial","simultaneous_interval_below_negative_margin_and_holm_zero_test"
            else:
                reason="training_seed_direction_unstable"
        elif lower > config.delta_m and adjusted[c]<=config.alpha:
            if harmful>=config.required_agreeing_seeds:
                verdict,reason="harmful","simultaneous_interval_above_positive_margin_and_holm_zero_test"
            else:
                reason="training_seed_direction_unstable"
        elif lower > -config.delta_m and upper < config.delta_m:
            if equivalent>=config.required_agreeing_seeds:
                verdict,reason="equivalent","simultaneous_interval_strictly_inside_practical_band"
            else:
                reason="training_seed_equivalence_unstable"
        results[name]={"candidate":family[name][0],"control":family[name][1],"delta_estimate_m":float(estimates[c]),
            "marginal_interval_m":[float(v) if np.isfinite(v) else None for v in marginal[:,c]],
            "simultaneous_interval_m":[float(v) if np.isfinite(v) else None for v in (lower,upper)],
            "interval_finite":bool(finite),"improvement_m":float(-estimates[c]),
            "improvement_simultaneous_interval_m":[float(-v) if np.isfinite(v) else None for v in (upper,lower)],
            "p_zero_two_sided":float(p[c]),"holm_adjusted_p_zero":float(adjusted[c]),
            "seed_delta_m":seed_means[:,c].tolist(),"tail_check_max_shift_m":tail_shift,"tail_check_passed":stable,
            "verdict":verdict,"reason":reason}
    return {"schema_version":INFERENCE_VERSION,"independent_block_count":len(blocks),"block_ids":blocks,
        "origin_count_by_block":origin_counts,"registered_seeds":list(SEEDS),
        "estimand":"equal blocks; equal registered seeds within block; equal matched origins within block/seed",
        "sign_convention":"candidate_ES_minus_control_ES; negative favours candidate",
        "uncertainty_scope":"independent-block sampling conditional on these five fitted seeds",
        "interval_method":"Bonferroni studentized bootstrap-t, nominal family coverage only; null bounds indicate unbounded/degenerate tails",
        "simultaneous_confidence_per_comparison":1-config.alpha/len(names),
        "marginal_confidence":1-config.alpha,"holm_null":"two-sided zero mean paired delta, not equivalence",
        "equivalence_rule":"strict simultaneous-interval containment in [-delta,+delta]; not non-rejection of zero",
        "config":vars(config),"results":results}


def factor_verdict(loo, lio, *, conflict_unresolved, correlated_group_rule_passed,
                   power_qualified, unavailable_reason=None):
    """HC-06 disposition; a supporting LIO result cannot replace the primary LOO.

    Evidence-card generation additionally binds raw files, protocol identity and
    final-eval authorization. This function alone does not certify final evidence.
    """
    if any(type(v) is not bool for v in (conflict_unresolved,correlated_group_rule_passed,power_qualified)):
        raise ValueError("explicit conflict/group/power gate status required")
    if loo is None or lio is None:
        if not unavailable_reason:
            raise ValueError("missing comparison requires a concrete unavailable reason")
        return {"verdict":"unavailable","reason":unavailable_reason,"scope":"required LOO/LIO comparison cannot be established"}
    if conflict_unresolved:
        return {"verdict":"inconclusive","reason":"unresolved_LOO_LIO_or_group_conflict"}
    if not power_qualified:
        return {"verdict":"inconclusive","reason":"power_qualification_not_met"}
    if not correlated_group_rule_passed:
        return {"verdict":"inconclusive","reason":"correlated_group_rule_not_met"}
    mapping={"beneficial":"retain","harmful":"harmful","equivalent":"redundant","inconclusive":"inconclusive"}
    if loo.get("verdict") not in mapping or lio.get("verdict") not in mapping:
        raise ValueError("unknown comparison verdict")
    # Opposite significant practical directions are a conflict even if a caller
    # mistakenly says there is none. Resolving it requires new explicit evidence.
    if {loo["verdict"],lio["verdict"]}=={"beneficial","harmful"}:
        return {"verdict":"inconclusive","reason":"opposite_LOO_LIO_practical_directions"}
    verdict=mapping[loo["verdict"]]
    result={"verdict":verdict,"reason":loo["reason"],"primary_basis":"LOO","supporting_LIO":lio["verdict"],
        "improvement_sign":"positive favours adding the factor"}
    if verdict=="redundant":
        result["scope"]="incremental equivalence in this full model only; not absence of standalone information"
    return result


def planning_power(*, paired_sd_m, blocks, true_improvement_m, delta_m, alpha=.05, family_size=5, target=.8):
    """Normal planning approximation for crossing the practical margin, NOT observed post-hoc power.

    Caller must identify the allowed-data source and uncertainty of paired_sd_m.
    This does not guarantee bootstrap power, nor waive the 30-block requirement.
    """
    numbers=[paired_sd_m,true_improvement_m,delta_m,alpha,target]
    if not np.isfinite(numbers).all() or paired_sd_m<0 or delta_m<=0 or true_improvement_m<=delta_m:
        raise ValueError("finite SD and an assumed improvement beyond the margin required")
    if not 0<alpha<1 or not 0<target<1 or blocks<1 or family_size<1:
        raise ValueError("invalid planning counts/probabilities")
    normal=NormalDist()
    z=normal.inv_cdf(1-alpha/(2*family_size))
    excess=true_improvement_m-delta_m
    if paired_sd_m==0:
        power,required=1.,1
    else:
        power=normal.cdf(excess*np.sqrt(blocks)/paired_sd_m-z)
        required=math.ceil(((z+normal.inv_cdf(target))*paired_sd_m/excess)**2)
    return {"approximate_power":power,"required_blocks_normal_approximation":required,
        "assumed_true_improvement_m":true_improvement_m,"practical_margin_m":delta_m,
        "paired_sd_m":paired_sd_m,"target_power":target,"method":"normal_margin_crossing_planning_not_bootstrap_guarantee"}
