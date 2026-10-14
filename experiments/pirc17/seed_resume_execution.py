"""Checkpoint-aware seed execution core using the unchanged prediction math.

Not a production CLI: a separately tested owned-process supervisor and closed
attempt ancestry reader must wrap this core before empirical resumed launches.
Results distinguish inherited work from new predictions and never impersonate
the old engine's ledger or claim the missing old resource summaries exist.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import time

from . import resume_journal as journal
from . import seed_resume as admission
from . import seed_resume_lineage as lineage
from .numerical_check import audit as numerical_audit
from .precision_check import KEYS, load_particle_evidence, replay

native, engine = admission.native, admission.engine
VERSION = "pirc17-seed-resume-execution-core-v1"
SOURCE_ARGUMENTS = {"plan", "plan_sha256", "reference_ledger", "reference_audit", "fit", "fit_ledger",
                    "envelope", "envelope_sha256", "forecast_sha256", "supervisor_sha256"}


def source_hashes():
    return lineage.source_hashes()


def runtime_identity():
    return {"python": engine.platform.python_version(), "numpy": engine.np.__version__,
        "torch": engine.torch.__version__, "torch_intraop_threads": engine.torch.get_num_threads(),
        "torch_interop_threads": engine.torch.get_num_interop_threads()}


def _copy_particles(row, source_directory, destination, relative):
    # Materialization is a byte copy, not a new prediction or inherited success
    # claim. Original row/path/hash provenance remains in the attempt manifest.
    load_particle_evidence(row, source_directory)
    source = Path(source_directory)/row["particle_artifact"]["path"]
    path = destination/relative
    def copy(target):
        with source.open("rb") as original:
            shutil.copyfileobj(original, target)
    journal._publish(path, copy)
    if native._hash(path) != row["particle_artifact"]["sha256"]:
        raise ValueError("inherited particle bytes changed during materialization")
    return dict(deepcopy(row), particle_artifact={"path": relative, "sha256": native._hash(path)})


def _whole_math(prefix, rows, directory, reference, reference_directory, guard):
    """Validate every scientific row, then adapt only to generic numerical math.

    No old-engine completion or ledger is emitted. The in-memory completion
    below is solely the full Cartesian denominator required by generic math.
    It says nothing about resource conformance or a single uninterrupted run.
    """
    expected = prefix.spec["expected_run_count"]
    if len(rows) != expected or any(row["status"] != "success" for row in rows):
        raise ValueError("whole resumed math requires every registered successful workload")
    admission.validate_prefix_rows(prefix.spec, reference,
        [prefix.initialization, prefix.header, *rows], directory, reference_directory, guard=guard)
    generic = [dict(prefix.header, schema_version="pirc17-development-rollout-v1",
        purpose="bounded_validation_engineering_pilot"), *rows,
        {"type": "completion", "attempted_run_count": expected, "success_count": expected, "failure_count": 0}]
    return {"schema_version": VERSION+"-whole-math", "expected_run_count": expected,
        "numerical_audit": numerical_audit(generic, tolerance_m=prefix.spec["tolerance_m"]),
        "particle_precision": replay(generic, directory, tolerance_m=prefix.spec["tolerance_m"]),
        "reference_numerical": prefix.report["reference_numerical"],
        "adapter": "validated resumed scientific contract -> in-memory generic score/particle math only",
        "numerically_qualified": False, "certified": False, "formal_training_accepted": False,
        "final_eval_label_prediction_metric_reads": 0,
        "historical_resource_gaps": {"inner_completion": None, "maps": None, "memory_summary": None}}


def run(*, source, eligibility, release, snapshot, data_root, output, progress=None,
        resume_directory=None, root_sha256=None, attempt_index=None):
    """Compute ONLY the missing suffix of a freshly validated closed legacy seed batch.

    This library core enforces a conservative cooperative residual budget:
    charge the entire closed supervisor elapsed time against BOTH original
    budgets (including its preflight overhead). It never renews an expired cap.
    It deliberately reports outer_cap_enforced=False; do not use directly as a
    production launch. A successful core result still needs supervisory closure.
    """
    if set(source) != SOURCE_ARGUMENTS:
        raise ValueError("exact closed-source arguments required; no prevalidated object or guard override")
    source = {key: value if key.endswith("_sha256") else str(Path(value).resolve()) for key, value in source.items()}
    locations = {key: str(Path(value).resolve()) for key, value in
                 (("eligibility", eligibility), ("release", release), ("snapshot", snapshot), ("data_root", data_root))}
    eligibility, release, snapshot, data_root = (Path(locations[key]) for key in
                                                ("eligibility", "release", "snapshot", "data_root"))
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("new exclusive resume-attempt directory required")
    started = time.perf_counter()
    spec = native.load_plan(source["plan"], source["plan_sha256"])
    legacy = native.output_paths(source["envelope"])
    prior = admission.read_closed(legacy["supervisor"], source["supervisor_sha256"])
    spent = prior.get("elapsed_seconds")
    if isinstance(spent, bool) or not isinstance(spent, (int, float)) or not 0 <= spent < float("inf"):
        raise ValueError("finite closed-attempt resource charge required")
    continuation, inherited_directories = None, None
    reserved = any(value is not None for value in (resume_directory, root_sha256, attempt_index))
    if reserved:
        if any(value is None for value in (resume_directory, root_sha256, attempt_index)):
            raise ValueError("complete ordered continuation reservation required")
        root, _, reservation, _ = lineage.history(resume_directory, root_sha256, current_index=attempt_index)
        expected_output = Path(resume_directory).resolve()/"attempts"/f"{attempt_index:06d}"/"work"
        if root["source"] != source or root["data_locations"] != locations or output != expected_output:
            raise ValueError("worker source, locations or output differ from its exclusive reservation")
        spent = reservation["prior_elapsed_seconds"]
        # Worker import/preflight time is part of this attempt too. The outer
        # supervisor independently caps the entire owned process tree.
        elapsed_before_core = max(0., (lineage.datetime.now(lineage.timezone.utc)-lineage._date(reservation["started_at"])).total_seconds())
    else:
        elapsed_before_core = 0.
    residual = spec["wall_seconds"]-spent
    observer = native.MemoryObserver()
    def guard():
        if time.perf_counter()-started+elapsed_before_core >= residual:
            raise TimeoutError("original cumulative cooperative budget exhausted; no renewal on resume")
        observer.guard()
    guard()
    sources = source_hashes()
    prefix = admission.inspect(**source, guard=guard)
    evidence = {"fit": source["fit"], "fit_ledger": source["fit_ledger"], **{key: spec[key] for key in
                ("fit_sha256", "fit_ledger_sha256", "training_policy_sha256")}}
    bundle, models, fitted_eligibility = engine.load_candidate(**evidence)
    if fitted_eligibility != spec["eligibility_sha256"]:
        raise ValueError("resumed fit changes original eligibility")
    encoder = engine.CanonicalEncoder.frozen_pirc22(Path(snapshot))
    windows, parents, identity = engine.load_development(Path(eligibility), spec["eligibility_sha256"],
        Path(release), Path(data_root), encoder)
    engine.validate_population(bundle, windows, parents, identity)
    selected = engine.select_windows(windows["validation"], count=spec["limit_origins"], policy=spec["selection_policy"])
    if [window.sample_id for window in selected] != prefix.header["sample_ids"]:
        raise ValueError("resumed selection changes original cohort or order")
    encoders = {name: engine.configuration_encoder(encoder, name) for name in spec["configurations"]}
    if any(len(value.columns)*2 != engine.configuration_width(name) for name, value in encoders.items()):
        raise ValueError("resumed encoder changes frozen configuration")
    query_type, modules = engine.resolve_map_backend(spec["map_backend"])
    map_sources = {module.__name__: native._hash(Path(module.__file__)) for module in modules}
    if map_sources != prefix.header["map_source_sha256"]:
        raise ValueError("resumed map sources changed")
    reference = admission.read_closed(source["reference_ledger"], spec["reference_ledger_sha256"], jsonl=True)
    if reserved:
        inherited_directories, continuation, _ = lineage.inherit(prefix, source, locations,
            resume_directory, root_sha256, attempt_index, reference, guard=guard)
    else:
        inherited_directories = [str(legacy["forecast"].parent)]*len(prefix.rows)
    originals = native.validate_seed_result(spec, reference, [prefix.initialization, prefix.header])
    for window in selected:
        for name in spec["configurations"]:
            original = originals[(window.sample_id, name)]
            arrays = load_particle_evidence(original, Path(source["reference_ledger"]).resolve().parent)
            if (window.block_id != original["independent_block_id"]
                    or not engine.np.array_equal(window.target_positions_m, arrays[1])
                    or not engine.np.array_equal(window.horizon_seconds, arrays[2])):
                raise ValueError("resumed development changes observed targets, times or block")
    guard()
    bound = {Path(source[key]).resolve(): spec[key+"_sha256"] for key in
             ("fit", "fit_ledger", "reference_ledger", "reference_audit")}
    bound.update({Path(source["plan"]).resolve(): source["plan_sha256"],
        legacy["envelope"]: source["envelope_sha256"], legacy["forecast"]: source["forecast_sha256"],
        legacy["supervisor"]: source["supervisor_sha256"], Path(eligibility).resolve(): spec["eligibility_sha256"]})
    for row, directory in zip(prefix.rows, inherited_directories):
        bound[(Path(directory)/row["particle_artifact"]["path"]).resolve()] = row["particle_artifact"]["sha256"]
    if reserved:
        # Bind the root, every previous sealed attempt and this reservation;
        # changes anywhere in this lineage forbid a whole-result publication.
        for path in Path(resume_directory).resolve().rglob("*"):
            if path.is_file():
                bound[path] = native._hash(path)
    keys = admission.workload_keys(spec, prefix.header["sample_ids"])
    inherited_count = len(prefix.rows)
    output.mkdir()  # Reserve only after strict read-only admission and data checks.
    (output/"inherited").mkdir()
    resources = lineage.resources(spec, spent)
    writer = journal.Writer(output/"journal", workloads=[dict(zip(KEYS, key)) for key in keys],
        inherited_count=inherited_count,
        ancestry={"legacy_arguments": {key: str(value) for key, value in source.items()},
                  "validated_prefix": prefix.report, "inherited_rows": prefix.rows,
                  "inherited_directories": inherited_directories, "continuation": continuation},
        contract={"schema_version": VERSION, "source_sha256": sources, "plan": spec,
            "scientific_initialization": prefix.initialization, "scientific_header": prefix.header,
            "data_locations": locations, "resources": resources, "observer": native.physical_memory.identity()})
    rows, errors, maps, maps_identity, audit = [], [], None, None, None
    try:
        for index, (row, directory) in enumerate(zip(prefix.rows, inherited_directories)):
            guard()
            rows.append(_copy_particles(row, directory, output, f"inherited/{index:06d}.npz"))
        journal._publish_json(output/"inherited.json", {"source_forecast_sha256": source["forecast_sha256"], "rows": rows})
        if progress:
            progress({"phase": "prefix_inherited", "inherited_count": inherited_count, "expected": len(keys)})
        maps = query_type(Path(data_root), [Path(data_root)/relative for relative in engine.RECEIPTS])
        for parent in parents:
            maps.register_snapshot_parent(parent)
        grid = admission.entropy_grid()
        cursor = 0
        for window in selected:
            for seed in spec["seeds"]:
                group_size = len(spec["configurations"])*len(spec["particles"])*len(spec["steps"])
                if cursor+group_size <= inherited_count:
                    cursor += group_size
                    continue
                guard()
                driver = engine.BrownianPath(window.horizon_seconds, spec["steps"], history_step_seconds=5.,
                    particles=max(spec["particles"]), seed=seed, stream_id=window.sample_id)
                for name in spec["configurations"]:
                    model = models[name][seed]
                    query = (lambda positions: [{} for _ in positions]) if name == "base" else maps
                    provider = engine.PredictedPositionFeatures(encoders[name], query, window.frame)
                    def guarded_features(state):
                        guard()
                        return provider(state)
                    for count in spec["particles"]:
                        for step in spec["steps"]:
                            index = cursor
                            cursor += 1
                            if index < inherited_count:
                                continue
                            guard()
                            row = {"type": "run", **dict(zip(KEYS, keys[index])), "independent_block_id": window.block_id,
                                "model_identity_sha256": model.identity["sha256"], "brownian_identity": driver.identity,
                                "actual_horizons_seconds": window.horizon_seconds.tolist(), "status": "failure"}
                            begin = time.perf_counter()
                            try:
                                prediction = engine.rollout(window.origin, window.horizon_seconds, particles=count, seed=seed,
                                    max_step_seconds=step, history_step_seconds=5., base_drift=model.base_drift,
                                    diffusion=model.diffusion, terrain=guarded_features, conditioner=model.correction,
                                    brownian_increments=driver)
                                row["rollout_wall_seconds_including_lazy_map_initialization"] = time.perf_counter()-begin
                                row.update(invalid_feature_rows=prediction.invalid_feature_rows, feature_query_rows=prediction.feature_query_rows)
                                guard()
                                row["scores"] = engine.score_path(prediction.positions_m, window.target_positions_m,
                                    window.horizon_seconds, time_weights=engine.TIME_WEIGHTS, entropy_grid=grid)
                                row["particle_precision"] = engine.energy_precision(prediction.positions_m, window.target_positions_m, engine.TIME_WEIGHTS)
                                guard()
                            except Exception as exc:
                                row.update(error_type=type(exc).__name__, error_message=str(exc)[:300],
                                           elapsed_seconds_before_failure=time.perf_counter()-begin)
                                writer.append(row)
                                if isinstance(exc, (MemoryError, TimeoutError)):
                                    raise
                            else:
                                row["status"] = "success"
                                writer.append(row, (prediction.positions_m, window.target_positions_m, window.horizon_seconds))
                            guard()
                            if progress:
                                progress({"phase": "run_committed", "index": index, "expected": len(keys),
                                          "new_committed": writer.tip["record_count"], "status": row["status"]})
                del driver
    except Exception as exc:
        errors.append({"error_type": type(exc).__name__, "error_message": str(exc)[:300]})
    finally:
        if maps is not None:
            try:
                maps_identity = maps.identity
            except Exception as exc:
                errors.append({"error_type": type(exc).__name__, "error_message": str(exc)[:300]})
            try:
                maps.close()
            except Exception as exc:
                errors.append({"error_type": type(exc).__name__, "error_message": str(exc)[:300]})
    # Inventory, not the writer's possibly unacknowledged in-memory cursor,
    # determines committed progress. KeyboardInterrupt intentionally propagates
    # above: an outer supervisor must record actual closure for that case.
    saved = journal.read(writer.directory, writer.manifest_sha256)
    for record in saved.records:
        row = deepcopy(record["row"])
        if "particle_artifact" in row:
            row["particle_artifact"]["path"] = "journal/"+row["particle_artifact"]["path"]
        rows.append(row)
    try:
        guard()
        if (source_hashes() != sources or runtime_identity() != prefix.initialization["runtime"]
                or any(native._hash(path) != digest for path, digest in bound.items())
                or {module.__name__: native._hash(Path(module.__file__)) for module in modules} != map_sources):
            raise ValueError("resume sources, plan, inputs or inherited particles changed")
        if not errors and len(rows) == len(keys) and not saved.failure_count:
            checked_math = _whole_math(prefix, rows, output, reference, Path(source["reference_ledger"]).resolve().parent, guard)
            guard()
            if (source_hashes() != sources or runtime_identity() != prefix.initialization["runtime"]
                    or any(native._hash(path) != digest for path, digest in bound.items())
                    or {module.__name__: native._hash(Path(module.__file__)) for module in modules} != map_sources):
                raise ValueError("resume evidence changed during whole-workload math")
            journal._publish_json(output/"whole-math.json", checked_math)
            audit = checked_math
    except Exception as exc:
        errors.append({"error_type": type(exc).__name__, "error_message": str(exc)[:300]})
    result = {"schema_version": VERSION, "status": "computed_pending_supervisory_closure" if audit and not errors else "failed",
        "expected_run_count": len(keys), "inherited_success_count": inherited_count,
        "new_committed_count": len(saved.records), "new_failure_count": saved.failure_count,
        "missing_completed_record_count": len(keys)-inherited_count-len(saved.records),
        "journal_tip": saved.tip, "uncommitted_files": saved.uncommitted_files, "errors": errors,
        "resources": resources, "observer": observer.summary(), "new_attempt_maps": maps_identity,
        "elapsed_seconds": time.perf_counter()-started, "whole_math_sha256": native._hash(output/"whole-math.json") if audit else None,
        "production_resume_ready": False, "certified": False, "formal_training_accepted": False,
        "final_eval_label_prediction_metric_reads": 0}
    journal._publish_json(output/"core-result.json", result)
    return result
