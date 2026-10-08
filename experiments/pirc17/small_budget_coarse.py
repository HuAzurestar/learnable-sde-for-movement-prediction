"""Bounded coarse forecasts against already audited, saved fine paths.

Reuse the unchanged causal kernel, models, maps and full-horizon Brownian
identity. A short screen is descriptive only; it never qualifies 30 minutes.
No checkpoint continuation, new fitting, or final-evaluation loading is added.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import time

import numpy as np

from . import direct_linear_rollout as engine
from . import small_budget_review as offline
from .brownian import BrownianPath, integration_grid
from .contrast_refinement import joint_precision
from .direct_linear_evidence import read_bound
from .inference import PRIMARY_FAMILY, SEEDS
from .precision import paired_energy_precision
from .precision_check import KEYS, aggregate_precision, load_particle_evidence
from .qualification import _hash
from .resource_replay import MemoryObserver

VERSION = "pirc17-small-budget-coarse-v1"
CONFIGURATIONS = ["base", "all-terrain", "loo-road", "loo-river", "loo-worldcover", "loo-surface"]


def source_hashes():
    return {**offline.source_hashes(), **{name: _hash(Path(__file__).with_name(name))
        for name in ("small_budget_coarse.py", "contrast_refinement.py")}}


def validate_plan(spec):
    short = spec.get("scoring_slots") == 2
    if (spec.get("schema_version") != VERSION or spec.get("final_eval_authorized") is not False
            or type(spec.get("scoring_slots")) is not int or spec["scoring_slots"] not in (2, 4)
            or spec.get("step_seconds") not in (2.5, 5.) or type(spec.get("particles")) is not int
            or spec["particles"] < 3 or spec.get("seeds") != list(SEEDS)
            or spec.get("time_weights") != ([0., 1.] if short else [.25]*4)
            or spec.get("map_backend") != "multicell"
            or spec.get("new_empirical_total_cap_seconds") != 7200
            or not isinstance(spec.get("prior_new_empirical_batches"), list)
            or type(spec.get("wall_seconds")) is not int or not 0 < spec["wall_seconds"] <= (900 if short else 3600)
            or spec.get("outer_wall_seconds") != spec["wall_seconds"]+60):
        raise ValueError("fixed bounded coarse plan, five seeds and original history semantics required")
    for field, allowed in (("sample_ids", None), ("configurations", CONFIGURATIONS)):
        values = spec.get(field)
        if (not isinstance(values, list) or not values or len(set(values)) != len(values)
                or any(not isinstance(v, str) or not v for v in values)
                or (allowed is not None and not set(values) <= set(allowed))):
            raise ValueError("unique registered origin/configuration axes required")
    expected = len(spec["sample_ids"])*len(spec["configurations"])*len(SEEDS)
    if (spec.get("expected_run_count") != expected or "all-terrain" not in spec["configurations"]
            or (short and expected > 40) or (not short and spec["configurations"] != CONFIGURATIONS)):
        raise ValueError("coarse workload denominator or complete-primary-family scope differs")


def prior_budget(spec):
    """Bind the whole ordered history before admitting another bounded batch."""
    spent, seen = 0., set()
    for index, item in enumerate(spec["prior_new_empirical_batches"]):
        path = Path(item["ledger"]).resolve()
        if path in seen:
            raise ValueError("duplicate prior empirical batch")
        seen.add(path)
        rows = read_bound(path, item["ledger_sha256"], jsonl=True)
        first, last = rows[0], rows[-1]
        elapsed = last.get("elapsed_seconds")
        if (first.get("schema_version") != VERSION or last.get("type") != "completion"
                or first["plan"]["offline_result_sha256"] != spec["offline_result_sha256"]
                or first["plan"]["prior_new_empirical_batches"] != spec["prior_new_empirical_batches"][:index]
                or isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                or not np.isfinite(elapsed) or elapsed < 0):
            raise ValueError("complete ordered empirical cost history required; unresolved runs cannot be restarted")
        spent += elapsed
    if spent+spec["wall_seconds"] > spec["new_empirical_total_cap_seconds"]:
        raise ValueError("remaining total empirical budget cannot admit this batch cap")
    return spent


def reference_driver(row):
    """Generate the old FULL grid even if dynamics will stop at five minutes."""
    driver = BrownianPath(row["actual_horizons_seconds"], [row["max_step_seconds"]],
        history_step_seconds=5., particles=row["brownian_identity"]["max_particles"],
        seed=row["seed"], stream_id=row["sample_id"])
    if driver.identity != row["brownian_identity"]:
        raise ValueError("original full-horizon Brownian identity does not reproduce")
    return driver


def validate_pair(row, window, driver, particles, step, slots, model):
    horizons = window.horizon_seconds[:slots]
    if (row["brownian_identity"] != driver.identity or row["seed"] != driver.identity["seed"]
            or row["sample_id"] != window.sample_id or row["independent_block_id"] != window.block_id
            or row["model_identity_sha256"] != model.identity["sha256"] or window.role != "validation"
            or row["actual_horizons_seconds"] != window.horizon_seconds.tolist()
            or not 3 <= particles <= row["particles"]
            or not set(integration_grid(horizons, step, 5.)) <= set(driver.index)):
        raise ValueError("coarse forecast changes model, block, full stream or integration grid")
    return horizons


def recheck(bound, guard=lambda: None):
    for path, digest in bound.items():
        guard()
        if _hash(Path(path)) != digest:
            raise ValueError("bound evidence or map asset changed: "+str(path))


def asset_bindings(identity, data_root):
    root = Path(data_root).resolve()
    result = {Path(p): sha for p, sha in identity["receipt_sha256"].items()}
    for relative, digest in identity["verified_assets"].items():
        path = (root/relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(root):
            raise ValueError("map asset must remain inside the registered data root")
        result[path] = digest
    return result


def context(plan_path, plan_sha256):
    """Read immutable audited sources; do not repeat forecasts or fit models."""
    spec = read_bound(plan_path, plan_sha256)
    validate_plan(spec)
    spent = prior_budget(spec)
    previous = read_bound(spec["offline_result"], spec["offline_result_sha256"])
    registered = read_bound(spec["offline_plan"], spec["offline_plan_sha256"])
    if (previous.get("schema_version") != offline.VERSION or previous.get("status") != "complete"
            or previous.get("plan_sha256") != spec["offline_plan_sha256"]
            or previous.get("source_sha256") != offline.source_hashes()
            or previous.get("new_rollouts") != 0 or previous.get("final_eval_label_prediction_metric_reads") != 0
            or spec["particles"] not in registered["particle_budgets"]):
        raise ValueError("complete unchanged offline particle review required")
    bound = {Path(plan_path): plan_sha256, Path(spec["offline_result"]): spec["offline_result_sha256"],
             Path(spec["offline_plan"]): spec["offline_plan_sha256"]}
    bound.update({Path(p["ledger"]): p["ledger_sha256"] for p in spec["prior_new_empirical_batches"]})
    references, header, map_identities = {}, None, []
    for item in registered["inputs"]:
        rows = read_bound(item["ledger"], item["ledger_sha256"], jsonl=True)
        read_bound(item["audit"], item["audit_sha256"])
        bound.update({Path(item["ledger"]): item["ledger_sha256"], Path(item["audit"]): item["audit_sha256"]})
        header = header or rows[1]
        map_identities.append(rows[-1]["maps"])
        for row in rows[2:-1]:
            # Preserve verification of every original parent array, not just a favorable subset.
            arrays = load_particle_evidence(row, Path(item["ledger"]).parent)
            bound[Path(item["ledger"]).parent/row["particle_artifact"]["path"]] = row["particle_artifact"]["sha256"]
            if row["particles"] == registered["particle_budgets"][-1] and row["max_step_seconds"] == registered["step_seconds"]:
                key = (row["sample_id"], row["configuration"], row["seed"])
                if key in references:
                    raise ValueError("duplicate fine reference")
                references[key] = (row, arrays)
    expected = set(itertools.product(header["sample_ids"], CONFIGURATIONS, SEEDS))
    if (set(references) != expected or len(references) != previous["selected_runs"]
            or not set(spec["sample_ids"]) <= set(header["sample_ids"])
            or (spec["scoring_slots"] == 4 and spec["sample_ids"] != header["sample_ids"])):
        raise ValueError("incomplete reference grid or changed selected origins")
    evidence = registered["candidate_evidence"]
    for field in ("fit", "fit_ledger"):
        bound[Path(evidence[field])] = evidence[field+"_sha256"]
    return spec, previous, evidence, references, header, map_identities, bound


def analyze(rows, directory, spec, previous, references):
    """Independent saved-array consumer; a partial grid never produces a pass."""
    expected = set(itertools.product(spec["sample_ids"], spec["configurations"], SEEDS))
    if len(rows) != len(expected) or any(row.get("status") != "success" for row in rows):
        raise ValueError("all registered successful coarse forecasts required")
    n, slots, step = spec["particles"], spec["scoring_slots"], spec["step_seconds"]
    loaded = {}
    for row in rows:
        key = row["sample_id"], row["configuration"], row["seed"]
        if key not in expected or key in loaded or row["particles"] != n or row["max_step_seconds"] != step:
            raise ValueError("unexpected, duplicated or differently resolved coarse forecast")
        ref, (fine, target, times) = references[key]
        coarse, actual_target, actual_times = load_particle_evidence(row, directory)
        if (any(row[k] != ref[k] for k in ("brownian_identity", "model_identity_sha256", "independent_block_id"))
                or not np.array_equal(actual_target, target[:slots]) or not np.array_equal(actual_times, times[:slots])):
            raise ValueError("coarse/fine pairing identity, targets or scoring times differ")
        precision = engine.energy_precision(coarse, actual_target, spec["time_weights"])
        if precision != row["particle_precision"]:
            raise ValueError("saved coarse precision does not reproduce")
        loaded[key] = (row, coarse, fine[:n, :slots], actual_target)
    records = []
    family = {k: pair for k, pair in PRIMARY_FAMILY.items() if set(pair) <= set(spec["configurations"])}
    for sample, seed in itertools.product(spec["sample_ids"], SEEDS):
        for comparison, (candidate, control) in family.items():
            a, b = (loaded[(sample, name, seed)] for name in (candidate, control))
            if (a[0]["brownian_identity"] != b[0]["brownian_identity"]
                    or not np.array_equal(a[3], b[3]) or a[0]["independent_block_id"] != b[0]["independent_block_id"]):
                raise ValueError("paired configurations do not share a stream, block or target")
            ordinary = paired_energy_precision(a[1], b[1], a[3], spec["time_weights"])
            change = joint_precision(a[1], b[1], a[2], b[2], a[3], spec["time_weights"], kind="integration")
            for kind, precision in (("coarse_contrast", ordinary), ("coarse_minus_fine", change)):
                first = dict(zip(KEYS, (sample, candidate, seed, n, step)))
                second = dict(zip(KEYS, (sample, control if kind == "coarse_contrast" else candidate,
                    seed, n, step if kind == "coarse_contrast" else references[(sample, candidate, seed)][0]["max_step_seconds"])))
                records.append(dict(kind=kind, comparison=comparison, candidate_workload=first,
                    control_workload=second, independent_block_id=a[0]["independent_block_id"], precision=precision))
    header = dict(sample_ids=spec["sample_ids"], seeds=list(SEEDS), time_weights=spec["time_weights"],
                  brownian_driver=engine.BROWNIAN_VERSION)
    aggregates = aggregate_precision(records, header)
    for row in aggregates:
        row["normal_mc_margin_m"] = offline.Q*row["time_weighted_standard_error_m"]
        if row["kind"] == "coarse_minus_fine":
            row["absolute_change_plus_margin_m"] = abs(row["time_weighted_difference_m"])+row["normal_mc_margin_m"]
    return dict(paired_diagnostics=records, aggregate_precision=aggregates,
        missing_primary_comparisons=sorted(set(PRIMARY_FAMILY)-set(family)),
        scoring_slots=slots, epsilon_m=previous["parameters"]["epsilon_m"],
        normal_quantile=offline.Q, diagnostic_family_size=offline.FAMILY_SIZE,
        short_screen_only=slots == 2, numerically_qualified=False, certified=False,
        scope="conditional paired numerical diagnostics; short horizons never qualify the full horizon",
        interval_caveat="asymptotic path-jackknife normal margins, not exact coverage or scientific block intervals")


def run(*, plan_path, plan_sha256, output, eligibility, release, snapshot, data_root, progress=None):
    output = Path(output)
    audit_path = output.with_suffix(".audit.json")
    if output.suffix != ".jsonl" or any(p.exists() for p in (output, audit_path, output.with_suffix(".particles"))):
        raise FileExistsError("new exclusive .jsonl ledger, particle directory and audit required")
    spec = read_bound(plan_path, plan_sha256)
    validate_plan(spec)
    spent = prior_budget(spec)
    started = time.perf_counter()
    observer = MemoryObserver()
    sources, maps, attempts, successes, errors, map_modules = source_hashes(), None, [], 0, [], []
    map_identity, bindings, result = None, {}, None

    def guard():
        if time.perf_counter()-started >= spec["wall_seconds"]:
            raise TimeoutError("fixed coarse batch wall-time cap reached; no automatic extension")
        observer.guard()

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        def emit(row):
            target.write(json.dumps(row, allow_nan=False)+"\n")
            target.flush()

        def fail(exc):
            errors.append(dict(error_type=type(exc).__name__, error_message=str(exc)[:400]))
            emit(dict(type="failure", **errors[-1]))

        emit(dict(type="initialization", schema_version=VERSION, plan=spec, plan_sha256=plan_sha256,
            source_sha256=sources, started_at=datetime.now(timezone.utc).isoformat(),
            final_eval_label_prediction_metric_reads=0, certified=False))
        try:
            guard()
            _, previous, evidence, references, header, old_maps, bindings = context(plan_path, plan_sha256)
            for identity in old_maps:
                bindings.update(asset_bindings(identity, data_root))
            recheck(bindings, guard)
            bundle, models, eligibility_sha256 = engine.load_candidate(**evidence)
            encoder = engine.CanonicalEncoder.frozen_pirc22(Path(snapshot))
            windows, parents, identity = engine.load_development(Path(eligibility), eligibility_sha256,
                Path(release), Path(data_root), encoder)
            engine.validate_population(bundle, windows, parents, identity)
            selected = {w.sample_id: w for w in windows["validation"]}
            query_type, map_modules = engine.resolve_map_backend(spec["map_backend"])
            map_sources = {m.__name__: _hash(Path(m.__file__)) for m in map_modules}
            if map_sources != header["map_source_sha256"]:
                raise ValueError("map implementation differs from the audited fine reference")
            maps = query_type(Path(data_root), [Path(data_root)/p for p in engine.RECEIPTS])
            for parent in parents:
                maps.register_snapshot_parent(parent)
            static = {k: v for k, v in maps.identity.items() if k != "verified_assets"}
            if any(static != {k: v for k, v in old.items() if k != "verified_assets"} for old in old_maps):
                raise ValueError("map backend or receipts changed from fine references")
            catalog = {p: v["checksum"]["value"] for p, v in maps.assets.items()}
            if any(catalog.get(p) != sha for old in old_maps for p, sha in old["verified_assets"].items()):
                raise ValueError("registered map catalog changes an original asset")
            emit(dict(type="header", input_validation_seconds=time.perf_counter()-started,
                development_identity=identity["sha256"], map_source_sha256=map_sources,
                physical_history_step_seconds=5., full_reference_grid_preserved=True,
                sample_ids=spec["sample_ids"], expected_run_count=spec["expected_run_count"]))
            if progress:
                progress(dict(phase="inputs_verified", expected=spec["expected_run_count"]))
            for sample, seed in itertools.product(spec["sample_ids"], SEEDS):
                guard()
                window = selected[sample]
                driver = reference_driver(references[(sample, "all-terrain", seed)][0])
                for name in spec["configurations"]:
                    guard()
                    ref, (_, target_positions, _) = references[(sample, name, seed)]
                    if not np.array_equal(window.target_positions_m, target_positions):
                        raise ValueError("loaded development targets differ from saved reference")
                    model = models[name][seed]
                    horizons = validate_pair(ref, window, driver, spec["particles"], spec["step_seconds"], spec["scoring_slots"], model)
                    provider = engine.PredictedPositionFeatures(engine.configuration_encoder(encoder, name),
                        (lambda p: [{} for _ in p]) if name == "base" else maps, window.frame)

                    def terrain(state):
                        guard()
                        return provider(state)

                    row = dict(type="run", sample_id=sample, configuration=name, seed=seed,
                        particles=spec["particles"], max_step_seconds=spec["step_seconds"], status="failure",
                        independent_block_id=window.block_id, brownian_identity=driver.identity,
                        model_identity_sha256=model.identity["sha256"], actual_horizons_seconds=horizons.tolist())
                    begin = time.perf_counter()
                    try:
                        prediction = engine.rollout(window.origin, horizons, particles=spec["particles"], seed=seed,
                            max_step_seconds=spec["step_seconds"], history_step_seconds=5.,
                            base_drift=model.base_drift, diffusion=model.diffusion, terrain=terrain,
                            conditioner=model.correction, brownian_increments=driver)
                        row["rollout_wall_seconds_including_lazy_map_initialization"] = time.perf_counter()-begin
                        guard()
                        row.update(invalid_feature_rows=prediction.invalid_feature_rows, feature_query_rows=prediction.feature_query_rows,
                            particle_precision=engine.energy_precision(prediction.positions_m, target_positions[:len(horizons)], spec["time_weights"]))
                        row["particle_artifact"] = engine.save_particle_artifact(output, row, prediction.positions_m,
                            target_positions[:len(horizons)], horizons)
                        row["status"] = "success"
                        successes += 1
                    except Exception as exc:
                        row.update(error_type=type(exc).__name__, error_message=str(exc)[:300],
                            elapsed_seconds_before_failure=time.perf_counter()-begin)
                        raise
                    finally:
                        attempts.append(row)
                        emit(row)
                        if progress:
                            progress(dict(phase="run", completed=successes, attempted=len(attempts),
                                expected=spec["expected_run_count"], configuration=name, seed=seed, status=row["status"]))
                del driver
            guard()
            result = analyze(attempts, output.parent, spec, previous, references)
            map_identity = maps.identity
            if ({k: v for k, v in map_identity.items() if k != "verified_assets"} != static
                    or any(catalog.get(p) != sha for p, sha in map_identity["verified_assets"].items())):
                raise ValueError("coarse paths used changed or unregistered maps")
            bindings.update(asset_bindings(map_identity, data_root))
            recheck(bindings, guard)
            for row in attempts:
                load_particle_evidence(row, output.parent)
            if sources != source_hashes() or map_sources != {m.__name__: _hash(Path(m.__file__)) for m in map_modules}:
                raise ValueError("coarse runtime sources changed during execution")
            guard()
        except Exception as exc:
            fail(exc)
        finally:
            if maps is not None:
                try:
                    maps.close()
                except Exception as exc:
                    fail(exc)
        completion = dict(type="completion", status="complete" if result is not None and not errors else "failed",
            expected_run_count=spec["expected_run_count"], attempted_run_count=len(attempts), success_count=successes,
            failure_count=len(attempts)-successes, unattempted_run_count=spec["expected_run_count"]-len(attempts),
            terminal_error_count=len(errors), resource_stopped=any(e["error_type"] in ("TimeoutError", "MemoryError") for e in errors),
            elapsed_seconds=time.perf_counter()-started, maps=map_identity, memory_observer=observer.summary(),
            prior_empirical_wall_seconds=spent, cumulative_empirical_wall_seconds=spent+time.perf_counter()-started,
            final_eval_label_prediction_metric_reads=0, numerically_qualified=False, certified=False)
        # Closing and internal analysis are included in the fixed batch cap.
        if completion["elapsed_seconds"] >= spec["wall_seconds"]:
            completion.update(status="failed", resource_stopped=True)
        emit(completion)
    if completion["status"] == "complete":
        result.update(schema_version=VERSION+"-audit", status="complete", ledger_sha256=_hash(output),
            plan_sha256=plan_sha256, source_sha256=sources, counts={k: completion[k] for k in
                ("expected_run_count", "attempted_run_count", "success_count", "failure_count", "unattempted_run_count")},
            original_numerical_failures=previous["original_numerical_failures"], final_eval_label_prediction_metric_reads=0)
        with audit_path.open("x", encoding="utf-8") as target:
            json.dump(result, target, indent=2, allow_nan=False)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan-path", "output", "eligibility", "release", "snapshot", "data-root"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    result = run(**vars(parser.parse_args()), progress=lambda row: print(json.dumps(row), flush=True))
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
