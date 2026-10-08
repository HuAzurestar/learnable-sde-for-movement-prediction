"""Independent saved-output arithmetic for the fixed development method batch.

Only stdlib/NumPy are used for scientific recomputation. No producer, fitter,
forecast engine, score implementation or trajectory/map loader is imported.
One exclusive audit receipt; no predictions, fit retries or final-eval access.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT/"artifacts/pirc17/dev10/method-qualification-p512-h5-v1"
PLAN = ROOT/"experiments/pirc17/plans/method-qualification-p512-h5-v1.json"
PLAN_SHA = "1aa3732f7da584295eaa54d76c5d417d3977ac3e7ae4d20e798fd1485f768cbb"
MECHANISM_VERSION = "pirc17-causal-method-mechanisms-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(value):
    a = np.asarray(value, dtype="<f8")
    return digest({"shape": list(a.shape), "bytes": hashlib.sha256(a.tobytes()).hexdigest()})


def close(observed, expected):
    np.testing.assert_allclose(observed, expected, rtol=1e-11, atol=1e-8)


def energy(samples, target):
    # Independent unchunked pairwise-distance implementation, including 8D paths.
    n = len(samples)
    pairs = np.sqrt(np.sum((samples[:, None]-samples[None, :])**2, axis=-1))
    return float(np.sqrt(np.sum((samples-target)**2, axis=-1)).mean()-pairs.sum()/(2*n*(n-1)))


def common_scores(x, y, times, saved):
    assert x.shape == (512, 4, 2) and y.shape == (4, 2) and np.isfinite(x).all() and np.isfinite(y).all()
    assert saved["particle_count"] == 512 and saved["time_weights"] == [.25]*4
    assert len(saved["by_time"]) == 4
    es, entropies = [], []
    for t, row in enumerate(saved["by_time"]):
        assert row["elapsed_seconds"] == times[t]
        samples = x[:, t]
        es.append(energy(samples, y[t]))
        close(row["energy_score_m"], es[-1])
        absolute_pairs = np.abs(samples[:, None]-samples[None, :]).sum(axis=(0, 1))
        close(row["marginal_crps_m"], np.abs(samples-y[t]).mean(axis=0)-absolute_pairs/(2*512*511))
        mean = samples.mean(axis=0)
        radii = np.sort(np.linalg.norm(samples-mean, axis=1))
        close(row["region"]["center_m"], mean)
        assert len(row["region"]["levels"]) == 4
        for level, detail in zip((.5, .8, .9, .95), row["region"]["levels"]):
            radius = radii[math.ceil(level*512)-1]
            close([detail["level"], detail["radius_m"], detail["area_m2"], detail["empirical_mass"]],
                  [level, radius, math.pi*radius**2, np.mean(radii <= radius)])
            assert detail["covered"] == bool(np.linalg.norm(y[t]-mean) <= radius)
        counts = np.histogram2d(samples[:, 0], samples[:, 1], bins=(np.arange(-10000, 10001, 250),)*2)[0]
        overflow = 512-counts.sum()
        masses = np.append(counts.ravel(), overflow)/512
        p = masses[masses > 0]
        entropies.append(float(-np.sum(p*np.log(p))))
        close([row["position_entropy"]["entropy_nats"], row["position_entropy"]["overflow_mass"],
               row["position_entropy"]["total_mass"]], [entropies[-1], overflow/512, 1.])
    errors = np.linalg.norm(x.mean(axis=0)-y, axis=1)
    close([saved["time_weighted_energy_score_m"], saved["ade_grid_mean_m"],
           saved["time_weighted_displacement_error_m"], saved["fde_m"]], [np.mean(es), errors.mean(), errors.mean(), errors[-1]])
    for level in (.5, .9, .95):
        close(saved["particle_endpoint_error_quantiles_m"][str(level)], np.quantile(np.linalg.norm(x[:, -1]-y[-1], axis=1), level))
    close(saved["path_energy_score_m"], energy((.5*x).reshape(512, 8), (.5*y).ravel()))
    close(saved["entropy_change_from_first_scoring_time_nats"], np.array(entropies)-entropies[0])
    assert saved["point_estimator"] == "ensemble_mean_not_best_of_N"
    assert all(saved[k]["status"] == "unavailable" for k in ("nll", "mode_entropy", "joint_path_entropy"))
    return float(np.mean(es))


def model_statistic(model, statistic):
    p = np.asarray(model["mode_probabilities"])
    w = [np.asarray(a) for a in model["weights"]]
    if statistic == "mode_entropy": return float(-np.sum(p*np.log(p+1e-15)))
    if statistic == "mode_switch_rate": return float(1-np.sum(p*p))
    if statistic == "covariance_min_eigenvalue": return float(min(np.linalg.eigvalsh(c).min() for c in model["covariances"]))
    if statistic == "component_separation": return float(max(np.linalg.norm(a[0]-b[0]) for a in w for b in w))
    if statistic == "decomposition_coefficient_norm": return float(sum(np.linalg.norm(a[3:6]) for a in w))
    if statistic == "effective_mode_count": return float(np.sum(p >= .05))
    if statistic in {"validation_energy_improvement", "objective_improvement"}:
        return model["validation_objective_before"]-model["validation_objective_after"]
    if statistic == "mean_log_likelihood": return -model["validation_objective_after"]
    if statistic == "adaptation_parameter_delta": return model["adaptation_parameter_delta"]
    if statistic == "adaptation_sample_count": return model["training_sample_count"]
    if statistic == "meta_task_count": return model["meta_task_count"]
    if statistic == "condition_coefficient_norm": return float(sum(np.linalg.norm(a[6 if model["model_kind"] == "explicit_decomp" else 3:]) for a in w))
    if statistic == "density_mass_error": return float(abs(p.sum()-1))
    raise ValueError("unknown raw-model statistic")


def check_gate(gate, value, definitions):
    definition = definitions[gate["slot_id"]]
    assert (gate["statistic"], gate["operator"], gate["threshold"]) == (
        definition["statistic"], definition.get("operator", "ge"), definition["threshold"])
    assert digest({k: v for k, v in gate.items() if k != "sha256"}) == gate["sha256"]
    if value is None:
        assert gate["value"] is None and gate["passed"] is None and gate["unavailable_reason"]
    else:
        close(gate["value"], value)
        assert gate["passed"] == bool(value >= gate["threshold"] if gate["operator"] == "ge" else value <= gate["threshold"])
    assert not gate["scientific_claim_authorized"]


def scoring_diagnostic(x, y, gate, definitions):
    e = gate["evidence"]
    assert len(e["per_time"]) == 4 and e["diagnostic_draws_per_time"] == 256 and e["quadrature_order"] == 12
    assert e["positions_sha256"] == array_digest(x) and e["targets_sha256"] == array_digest(y)
    roots, weights = np.polynomial.hermite.hermgauss(12)
    grid = np.array(list(itertools.product(roots, repeat=2)))
    mass = np.array([a*b/math.pi for a, b in itertools.product(weights, repeat=2)])
    errors = []
    for t, row in enumerate(e["per_time"]):
        mean = x[:, t].mean(axis=0)
        chol = np.linalg.cholesky(np.cov(x[:, t], rowvar=False)+1e-9*np.eye(2))
        closed = np.dot(mass, np.linalg.norm(mean+math.sqrt(2)*grid@chol.T-y[t], axis=1))
        closed -= .5*np.dot(mass, np.linalg.norm(2*grid@chol.T, axis=1))
        entropy = bytes.fromhex(digest([MECHANISM_VERSION, "scoring-Gaussian-not-forecast", e["origin_id"], t]))
        words = [int.from_bytes(entropy[i:i+4], "little") for i in range(0, 32, 4)]
        draws = mean+np.random.default_rng(np.random.SeedSequence([e["seed"], *words])).normal(size=(256, 2))@chol.T
        cyclic = lambda a: np.linalg.norm(a-y[t], axis=1).mean()-.5*np.linalg.norm(a-np.roll(a, 1, axis=0), axis=1).mean()
        mc = cyclic(draws)
        error = abs(closed-mc)/max(abs(closed), 1e-9)
        close([row["moment_gaussian_quadrature_es_m"], row["gaussian_mc_es_m"],
               row["saved_forecast_cyclic_es_256_m"], row["relative_error"]], [closed, mc, cyclic(x[:256, t]), error])
        assert row["diagnostic_draws_sha256"] == array_digest(draws)
        errors.append(error)
    check_gate(gate, max(errors), definitions)


def integration(a, r, gate, definitions):
    am, rm = a["conditional_means_method_frame_m"], r["conditional_means_method_frame_m"]
    ac, rc = a["conditional_covariances_method_frame_m2"], r["conditional_covariances_method_frame_m2"]
    roots = []
    for covariance in (ac, rc):
        eigenvalues, vectors = np.linalg.eigh((covariance+covariance.swapaxes(2, 3))/2)
        roots.append((vectors*np.sqrt(np.maximum(0., eigenvalues))[..., None, :])@vectors.swapaxes(2, 3))
    means_squared = np.sum((am-rm)**2, axis=2)
    values = np.sqrt(np.mean(means_squared+np.sum((roots[0]-roots[1])**2, axis=(2, 3)), axis=0))
    e = gate["evidence"]
    close(e["per_time_gaussian_coupling_rms_m"], values)
    close(e["per_time_rms_conditional_mean_error_m"], np.sqrt(means_squared.mean(axis=0)))
    close(e["per_time_rms_covariance_difference_m2"], np.sqrt(np.sum((ac-rc)**2, axis=(2, 3)).mean(axis=0)))
    assert e["conditional_moment_sha256"] == [array_digest(v) for v in (am, rm, ac, rc)]
    check_gate(gate, float(values.max()), definitions)


def planning(rows, families, definitions, analysis, plan):
    lookup = {(r["origin_id"], r["seed"], r["slot_id"]): r for r in rows if r["status"] == "success"}
    origins = [origins[0] for block, origins in sorted(plan["expected_origins_by_block"].items())]
    seeds = plan["seeds"]
    normal, delta, verified = NormalDist(), 41.100259, 0
    for family_id, contrasts in families.items():
        saved = analysis["families"][family_id]
        needed = {(o, s, c) for o in origins for s in seeds for c in set(contrasts)|set(contrasts.values())}
        if not needed <= lookup.keys():
            assert saved["status"] == "unavailable"
            continue
        assert saved["status"] == "computed" and saved["independent_block_ids"] == sorted(plan["expected_origins_by_block"])
        for candidate, control in contrasts.items():
            differences = np.array([[lookup[o, s, candidate]["score_m"]-lookup[o, s, control]["score_m"]
                for s in seeds] for o in origins])
            block_means, seed_means = differences.mean(axis=1), differences.mean(axis=0)
            sd = float(block_means.std(ddof=1))
            result = saved["results"][candidate]
            close(result["per_block_per_seed_difference_m"], differences)
            close(result["seed_mean_difference_m"], seed_means)
            close([result["paired_block_sd_m"], result["observed_development_improvement_m"],
                   result["simulation_mean_standard_error_m"]], [sd, -block_means.mean(), seed_means.std(ddof=1)/np.sqrt(5)])
            assert len(result["planning_scenarios"]) == 2
            for multiplier, scenario in zip((1., 2.), result["planning_scenarios"]):
                sigma = sd*multiplier
                z = normal.inv_cdf(1-.05/(2*len(contrasts)))
                probability = 1. if sigma == 0 else normal.cdf(delta*math.sqrt(46)/sigma-z)
                required = 1 if sigma == 0 else math.ceil(((z+normal.inv_cdf(.8))*sigma/delta)**2)
                close(scenario["approximate_power"], probability)
                assert scenario["required_blocks_normal_approximation"] == required
                assert scenario["planning_blocks"] == 46 and scenario["sd_multiplier"] == multiplier
                assert scenario["assumed_true_improvement_m"] == 2*delta
            verified += 1
    slots = ("arm-20/full", "arm-21/mc", "arm-21/crn")
    needed = {(o, s, c) for o in origins for s in seeds for c in slots}
    if needed <= lookup.keys():
        values = []
        for o in origins:
            differences = np.array([[lookup[o, s, c]["score_m"]-lookup[o, s, slots[0]]["score_m"]
                for c in slots[1:]] for s in seeds])
            values.append(differences.var(axis=0, ddof=1))
        values = np.array(values)
        mean = values.mean(axis=0)
        ratio = None if mean[1] == 0 else float(mean[0]/mean[1])
        draws = values[np.random.default_rng(20260926).integers(3, size=(2000, 3))].mean(axis=1)
        valid = draws[:, 1] > 0
        for gate in analysis["variance"]["gates"].values():
            check_gate(gate, ratio, definitions)
            e = gate["evidence"]
            close([e["numerator_m2"], e["denominator_m2"]], mean)
            assert e["independent_block_count"] == 3 and e["seed_count"] == 5 and e["variance_ddof"] == 1
            assert e["undefined_bootstrap_replicates"] == int(np.count_nonzero(~valid))
            if valid.all(): close(e["descriptive_block_bootstrap_interval"], np.quantile(draws[:, 0]/draws[:, 1], [.025, .975], method="inverted_cdf"))
            else: assert e["descriptive_block_bootstrap_interval"] is None
            actual = sorted([r for r in rows if (r["origin_id"], r["seed"], r["slot_id"]) in needed],
                            key=lambda r: (r["block_id"], r["origin_id"], r["seed"], r["slot_id"]))
            assert e["complete_input_sha256"] == digest(actual)
    else:
        assert analysis["variance"]["status"] == "unavailable"
    return verified


def run():
    started = time.perf_counter()
    def guard():
        if time.perf_counter()-started > 200:
            raise TimeoutError("fixed saved-output audit cap exhausted; no new experiments")
    def read(path, sha=None):
        assert Path(path).stat().st_size <= 2*1024**2
        if sha is not None: assert file_sha(path) == sha, str(path)
        return json.loads(Path(path).read_text(encoding="utf-8"))
    assert not (BATCH/"audit.json").exists()
    plan = read(PLAN, PLAN_SHA)
    sources = plan["source_sha256"]
    for name, sha in sources.items(): assert file_sha(ROOT/name) == sha, name
    supervision = read(BATCH/"supervision.json")
    assert supervision["plan_sha256"] == PLAN_SHA and supervision["process_tree_closed"]
    assert supervision["accounting"]["active_processes"] == 0 and supervision["worker_returncode"] == 0
    assert not supervision["timed_out"] and not supervision["interrupted"] and supervision["supervisor_error"] is None
    ledger = [json.loads(line) for line in (BATCH/"ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert ledger[-1]["event"] == "complete"
    report = read(BATCH/"result.json", ledger[-1]["result_sha256"])
    assert report["plan_sha256"] == PLAN_SHA and report["input_sha256"] == plan["expected_input_sha256"]
    assert report["final_eval_reads"] == 0
    assert all(report[k] is False for k in ("formal_training_accepted", "numerically_qualified", "power_qualified", "scientific_claim_authorized", "automatic_retry"))
    spec = read(ROOT/"experiments/nex326/experiment.json")
    definitions = {f"arm-{a['arm_id']:02d}/{s['subconfig_id']}": {**a["mechanism_gate"], **s.get("mechanism_gate", {})}
        for a in spec["arms"] for s in a["subconfigs"]}
    fits = {}
    assert len(report["fits"]) == 16
    assert [r["slot_id"] for r in report["fits"]] == plan["cached_fit_slots"]+plan["new_fit_slots"]
    for row in report["fits"]:
        if row["status"] != "success":
            assert row["reason"]
            continue
        value = read(ROOT/row["path"], row["sha256"])
        d, t = value["dynamics"], value["training"]
        assert digest(value["model"]) == d["model_sha256"]
        assert digest({k: v for k, v in t.items() if k != "training_identity_sha256"}) == t["training_identity_sha256"]
        assert digest({"training_identity_sha256": t["training_identity_sha256"], "model": value["model"]}) == d["fit_identity"]
        assert t["input_sha256"] == plan["expected_input_sha256"] and not value["formal_training_accepted"]
        assert d["fit_identity"] not in fits
        fits[d["fit_identity"]] = value
        guard()
    models_checked = 0
    special = {"arm-10/d2_mc", "arm-10/d2_closed", "arm-18/full", "arm-19/em", "arm-19/euler", "arm-21/mc", "arm-21/crn"}
    assert set(report["model_gates"]) == set(plan["required_slots"])-special
    for slot, item in report["model_gates"].items():
        if item["status"] == "computed":
            gate = item["gate"]
            model = fits[gate["evidence"]["dynamics"]["fit_identity"]]["model"]
            check_gate(gate, model_statistic(model, gate["statistic"]), definitions)
            models_checked += 1
    expected = {(o, s, c) for origins in plan["expected_origins_by_block"].values() for o in origins
                for s in plan["seeds"] for c in plan["forecast_order_per_origin_seed"]}
    arrays, details, rows = {}, {}, []
    for index, row in enumerate(report["forecasts"]):
        guard()
        key = row["origin_id"], row["seed"], row["slot_id"]
        assert key in expected and key not in details
        details[key] = None
        if row["status"] != "success":
            assert row["reason"]
            rows.append(row)
            continue
        assert row["path"] == f"forecast-{index:04d}.json"
        detail = read(BATCH/row["path"], row["sha256"])
        assert all(detail[k] == row[k] for k in ("origin_id", "block_id", "seed", "slot_id", "split", "context_sha256", "elapsed_seconds"))
        assert row["split"] == "validation" and row["origin_id"] in plan["expected_origins_by_block"][row["block_id"]]
        d = detail["diagnostics"]
        assert all(d[k] == v for k, v in {"origin_id": row["origin_id"], "seed": row["seed"],
            "run_id": row["slot_id"], "origin_mode": "causal_prefix", "particles": 512, "max_step_seconds": 5.}.items())
        stream_key = ([row["origin_id"], "paired", d["crn_pair_id"]] if d["crn_pair_id"]
                      else [row["origin_id"], "independent", d["propagation"], row["slot_id"]])
        assert d["random_stream_sha256"] == digest(stream_key)
        array_path = (BATCH/row["path"]).with_suffix(".npz")
        assert file_sha(array_path) == detail["artifact_sha256"]
        with np.load(array_path, allow_pickle=False) as z: a = {k: z[k] for k in z.files}
        assert digest(detail["context"]) == row["context_sha256"] == detail["context_sha256"]
        close(a["target_positions_m"], detail["context"]["target_positions_scoring_frame_m"])
        assert a["elapsed_seconds"].tolist() == row["elapsed_seconds"] == detail["context"]["elapsed_seconds"]
        assert detail["diagnostics"]["dynamics"] == fits[detail["diagnostics"]["dynamics"]["fit_identity"]]["dynamics"]
        value = common_scores(a["positions_m"], a["target_positions_m"], a["elapsed_seconds"], detail["scores"])
        close(row["score_m"], value)
        assert not detail["scientific_claim_authorized"] and detail["development_calibration_exposed"]
        if detail["score_gate"] is not None:
            scoring_diagnostic(a["positions_m"], a["target_positions_m"], detail["score_gate"], definitions)
        arrays[key], details[key] = a, detail
        rows.append({**row, "diagnostics": detail["diagnostics"], "score_gate": detail["score_gate"]})
    assert set(details) == expected and len(report["forecasts"]) == 435
    assert sum(r["status"] == "success" for r in rows) == report["analysis"]["successful_forecasts"]
    numerical_keys = {(o, s, c) for o, s, _ in expected for c in ("arm-18/full", "arm-19/em", "arm-19/euler")}
    assert len(report["integration_gates"]) == 45
    assert {(r["origin_id"], r["seed"], r["slot_id"]) for r in report["integration_gates"]} == numerical_keys
    assert len(report["score_gates"]) == 30
    for row in report["score_gates"]:
        key = row["origin_id"], row["seed"], row["slot_id"]
        assert row["slot_id"] in {"arm-10/d2_mc", "arm-10/d2_closed"}
        if row["status"] == "success": assert row["gate"] == details[key]["score_gate"]
    for row in report["integration_gates"]:
        if row["status"] != "computed": continue
        key = row["origin_id"], row["seed"], row["slot_id"]
        reference = row["origin_id"], row["seed"], "diagnostic/full-exact-kernel"
        gate = row["gate"]
        assert gate["evidence"]["approximate_diagnostics"] == details[key]["diagnostics"]
        assert gate["evidence"]["reference_diagnostics"] == details[reference]["diagnostics"]
        assert details[key]["context_sha256"] == details[reference]["context_sha256"] == gate["evidence"]["context_sha256"]
        assert details[key]["diagnostics"]["random_stream_sha256"] == details[reference]["diagnostics"]["random_stream_sha256"]
        integration(arrays[key], arrays[reference], gate, definitions)
    # Read the pre-run, hash-bound comparison mapping as data, without importing
    # either the producer or its statistical implementation.
    tree = ast.parse((ROOT/"experiments/pirc17/method_comparisons.py").read_text(encoding="utf-8"))
    families = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "FAMILY_DEFINITIONS" for t in n.targets))
    comparisons = planning(rows, families, definitions, report["analysis"], plan)
    guard()
    result = {"schema_version": "pirc17-independent-method-qualification-audit-v1", "status": "audited-saved-evidence",
        "plan_sha256": PLAN_SHA, "result_sha256": file_sha(BATCH/"result.json"),
        "ledger_sha256": file_sha(BATCH/"ledger.jsonl"), "supervision_sha256": file_sha(BATCH/"supervision.json"),
        "auditor_source_sha256": file_sha(Path(__file__)), "source_bindings": len(sources),
        "fitted_models_checked": len(fits), "model_gates_checked": models_checked,
        "forecast_arrays_checked": len(arrays), "observed_time_score_records_checked": 4*len(arrays),
        "method_planning_comparisons_checked": comparisons,
        "new_fits": 0, "new_forecasts": 0, "final_eval_reads": 0, "scientific_claim_authorized": False,
        "elapsed_seconds": time.perf_counter()-started}
    result["sha256"] = digest(result)
    with (BATCH/"audit.json").open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    # Only lifecycle containment is shared with the production path.
    from .seed_resume_session import require_owned_job, run_owned
    job = "PIRC17-METHOD-AUDIT-"+PLAN_SHA[:24]
    if args.worker:
        require_owned_job(job)
        print(json.dumps(run(), sort_keys=True), flush=True)
        return 0
    # Exclusive start survives a failed audit; no automatic restart.
    with (BATCH/"audit-start.json").open("x", encoding="utf-8") as out:
        json.dump({"plan_sha256": PLAN_SHA, "auditor_source_sha256": file_sha(Path(__file__)),
            "supervisor_pid": os.getpid(), "job_name": job, "outer_seconds_including_cleanup": 240}, out)
    started = time.perf_counter()
    result = run_owned([sys.executable, "-u", "-m", "experiments.pirc17.method_qualification_audit", "--worker"],
        timeout=200, job_name=job)
    result["elapsed_seconds_including_supervision"] = time.perf_counter()-started
    with (BATCH/"audit-supervision.json").open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
    print(json.dumps({"event": "owned_independent_method_audit_closed", **result}), flush=True)
    return 0 if (result["worker_returncode"] == 0 and result["process_tree_closed"] and not result["timed_out"]
                 and not result["interrupted"] and result["supervisor_error"] is None) else 1


if __name__ == "__main__":
    raise SystemExit(main())
