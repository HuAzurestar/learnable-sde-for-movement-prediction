"""Compute only new particle paths and commit each complete child forecast.

The owned session enforces process closure and the outer deadline. This core
enforces fresh memory/cooperative checks and never reopens an attempt. A
complete child forecast may be inherited from a prior closed interruption;
its original per-particle prefix always comes from the independently admitted
complete parent. Neither form of inheritance is charged as new integration.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

from . import particle_suffix_contract as science
from . import particle_suffix_lineage as lineage
from . import particle_suffix_maps as maps
from .particle_suffix import rollout_suffix
from .particle_extension_execution import assembled
from .seed_resume_execution import _copy_particles
from .precision_check import KEYS, load_particle_evidence

native, engine, journal = science.native, science.engine, lineage.journal
VERSION = science.VERSION+"-execution"
RESULT_FIELDS = {"schema_version", "status", "expected_run_count", "inherited_success_count",
    "new_committed_count", "new_failure_count", "missing_completed_record_count", "journal_tip",
    "uncommitted_files", "errors", "resources", "reference_resource_charges", "observer",
    "new_attempt_maps", "map_catalog_sha256", "map_observations", "elapsed_seconds",
    "whole_science_sha256", *science.UNQUALIFIED}


def load_inputs(prepared, *, guard=lambda: None):
    """Bind actual causal windows and all fitted models, not just their labels."""
    guard()
    source, spec = prepared.parent_root["source"], prepared.spec
    locations = prepared.parent_root["data_locations"]
    evidence = {"fit": source["fit"], "fit_ledger": source["fit_ledger"],
        **{k: spec[k] for k in ("fit_sha256", "fit_ledger_sha256", "training_policy_sha256")}}
    bundle, models, eligibility = engine.load_candidate(**evidence)
    if eligibility != spec["eligibility_sha256"]:
        raise ValueError("suffix candidate eligibility changed")
    encoder = engine.CanonicalEncoder.frozen_pirc22(Path(locations["snapshot"]))
    windows, parents, identity = engine.load_development(Path(locations["eligibility"]), spec["eligibility_sha256"],
        Path(locations["release"]), Path(locations["data_root"]), encoder)
    engine.validate_population(bundle, windows, parents, identity)
    selected = engine.select_windows(windows["validation"], count=spec["limit_origins"], policy=spec["selection_policy"])
    header = prepared.scientific_header
    if ([w.sample_id for w in selected] != header["sample_ids"] or identity["sha256"] != header["development_identity"]
            or {name: {str(seed): m.identity["sha256"] for seed, m in group.items()} for name, group in models.items()}
                != header["model_identities"]):
        raise ValueError("suffix selection, development identity or fitted models changed")
    encoders = {name: engine.configuration_encoder(encoder, name) for name in spec["configurations"]}
    if any(len(e.columns)*2 != engine.configuration_width(name) for name, e in encoders.items()):
        raise ValueError("suffix encoder differs from frozen configuration")
    query_type, modules = engine.resolve_map_backend(spec["map_backend"])
    if {m.__name__: native._hash(Path(m.__file__)) for m in modules} != header["map_source_sha256"]:
        raise ValueError("suffix map code changed")
    for window in selected:
        guard()
        original = next(r for r in prepared.parent_rows if r["sample_id"] == window.sample_id)
        _, targets, times = load_particle_evidence(original, prepared.parent_work)
        if (window.block_id != original["independent_block_id"] or not engine.np.array_equal(window.target_positions_m, targets)
                or not engine.np.array_equal(window.horizon_seconds, times)):
            raise ValueError("suffix actual targets, times or independent block changed")
    prepared.recheck(guard)
    return selected, models, encoders, parents, query_type


def run(*, directory, root_sha256, index, start_sha256, progress=None):
    began, entered_at = time.perf_counter(), lineage.datetime.now(lineage.timezone.utc)
    directory = Path(directory).resolve()
    root, closed, start, actual = lineage.history(directory, root_sha256, current_index=index)
    if actual != start_sha256:
        raise ValueError("suffix worker must bind the exact latest reservation")
    work = directory/"attempts"/f"{index:06d}"/"work"
    if work.exists():
        raise FileExistsError("suffix worker never reopens an existing attempt")
    spec = root["plan"]
    elapsed_before_core = max(0., (entered_at-lineage.storage._date(start["started_at"])).total_seconds())
    observer = native.MemoryObserver()

    def guard():
        if time.perf_counter()-began+elapsed_before_core >= start["cooperative_remaining_seconds"]:
            raise TimeoutError("original cumulative suffix budget exhausted; no renewal")
        observer.guard()

    guard()
    prepared = science.prepare(source=root["source"], guard=guard)
    registered = maps.catalog(prepared, guard=guard)
    rows, folders, ancestry, _, _ = lineage.inherit(prepared, directory, root_sha256,
        current_index=index, registered=registered, guard=guard)
    inherited_count = len(rows)
    selected, models, encoders, parents, query_type = load_inputs(prepared, guard=guard)
    locations = prepared.parent_root["data_locations"]
    bound = {directory/item["path"]: item["sha256"] for item in lineage.storage.inventory(directory)}

    def recheck():
        prepared.recheck(guard)
        if (lineage.source_hashes() != root["source_sha256"]
                or science.science.runtime_identity() != prepared.original_reference[0]["runtime"]
                or any(native._hash(path) != digest for path, digest in bound.items())):
            raise ValueError("suffix sources, runtime or closed attempt ancestry changed")
        # A prior child can visit an asset outside the original parent's used
        # set. It remains a dependency even when this attempt never queries it
        # (including audit-only continuation), also after whole-science replay.
        for item in closed:
            if item["saved"]:
                maps.verify_attempt(item["directory"]/"work", item["saved"], root_sha256=root_sha256,
                    start_sha256=item["start_sha256"], registered=registered,
                    data_root=locations["data_root"], guard=guard)

    recheck()
    work.mkdir()
    (work/"inherited").mkdir()
    (work/"map-observations").mkdir()
    journal._publish_json(work/"map-catalog.json", registered)
    writer = journal.Writer(work/"journal", workloads=[dict(zip(KEYS, k)) for k in prepared.keys],
        inherited_count=inherited_count, ancestry=ancestry,
        contract=lineage.journal_contract(root, root_sha256, index, start_sha256, prepared, registered))
    errors, query_maps, last_maps, audit, map_report = [], None, None, None, None
    try:
        inherited_rows = []
        for position, (row, folder) in enumerate(zip(rows, folders)):
            guard()
            inherited_rows.append(_copy_particles(row, folder, work, f"inherited/{position:06d}.npz"))
        journal._publish_json(work/"inherited.json", dict(root_sha256=root_sha256, start_sha256=start_sha256, rows=inherited_rows))
        if progress:
            progress({"phase": "prefix_inherited", "inherited_count": inherited_count, "expected": len(prepared.keys)})
        grid, cursor = science.admission.entropy_grid(), 0
        for window in selected:
            for seed in spec["seeds"]:
                group_size = len(spec["configurations"])*len(spec["steps"])
                if cursor+group_size <= inherited_count:
                    cursor += group_size
                    continue
                guard()
                total, prefix = spec["particles"][0], spec["prefix_particles"]
                driver = engine.BrownianPath(window.horizon_seconds, spec["steps"], history_step_seconds=5.,
                    particles=total, seed=seed, stream_id=window.sample_id)
                for name in spec["configurations"]:
                    model = models[name][seed]
                    for step in spec["steps"]:
                        position, cursor = cursor, cursor+1
                        if position < inherited_count:
                            continue
                        guard()
                        original = prepared.parent_rows[position]
                        old_positions, _, _ = load_particle_evidence(original, prepared.parent_work)
                        row = {"type": "run", "schema_version": science.VERSION+"-run",
                            **dict(zip(KEYS, prepared.keys[position])), "status": "failure",
                            "independent_block_id": window.block_id, "model_identity_sha256": model.identity["sha256"],
                            "brownian_identity": deepcopy(driver.identity), "actual_horizons_seconds": window.horizon_seconds.tolist(),
                            "prefix_binding": science.prefix_binding(prepared, position)}
                        raw_rows = [0]
                        begin = time.perf_counter()
                        try:
                            if name != "base" and query_maps is None:
                                query_maps = query_type(Path(locations["data_root"]), [Path(locations["data_root"])/p for p in engine.RECEIPTS])
                                for parent in parents:
                                    query_maps.register_snapshot_parent(parent)

                            def query(positions):
                                raw_rows[0] += len(positions)
                                return query_maps(positions)

                            provider = engine.PredictedPositionFeatures(encoders[name],
                                (lambda positions: [{} for _ in positions]) if name == "base" else query, window.frame)

                            def features(state):
                                guard()
                                return provider(state)

                            suffix = rollout_suffix(window.origin, window.horizon_seconds, prefix_particles=prefix, driver=driver,
                                seed=seed, stream_id=window.sample_id, max_step_seconds=step, history_step_seconds=5.,
                                base_drift=model.base_drift, diffusion=model.diffusion, terrain=features, conditioner=model.correction)
                            elapsed = time.perf_counter()-begin
                            prediction = suffix.forecast
                            positions = engine.np.concatenate((old_positions, prediction.positions_m), axis=0)
                            row["suffix_accounting"] = {"particle_start": prefix, "particle_stop": total, "computed_particles": total-prefix,
                                "new_feature_query_rows": prediction.feature_query_rows, "new_invalid_feature_rows": prediction.invalid_feature_rows,
                                "new_raw_map_query_rows": raw_rows[0], "inherited_feature_query_rows": original["feature_query_rows"],
                                "inherited_invalid_feature_rows": original["invalid_feature_rows"],
                                "ensemble_feature_rows": original["feature_query_rows"]+prediction.feature_query_rows,
                                "ensemble_invalid_rows": original["invalid_feature_rows"]+prediction.invalid_feature_rows,
                                "new_rollout_wall_seconds_including_lazy_map_initialization": elapsed}
                            guard()
                            row["scores"] = engine.score_path(positions, window.target_positions_m, window.horizon_seconds,
                                time_weights=engine.TIME_WEIGHTS, entropy_grid=grid)
                            row["particle_precision"] = engine.energy_precision(positions, window.target_positions_m, engine.TIME_WEIGHTS)
                            row["status"] = "success"
                            observation = maps.observation(row, index=position, root_sha256=root_sha256,
                                start_sha256=start_sha256, registered=registered, identity=None if name == "base" else query_maps.identity)
                            maps.check_observation(observation, row, index=position, root_sha256=root_sha256,
                                start_sha256=start_sha256, registered=registered, data_root=locations["data_root"])
                            guard()
                        except Exception as exc:
                            row.update(status="failure", error_type=type(exc).__name__, error_message=str(exc)[:300],
                                elapsed_seconds_before_failure=time.perf_counter()-begin)
                            writer.append(row)
                            if isinstance(exc, (MemoryError, TimeoutError)):
                                raise
                        else:
                            # A stop between these two publications leaves an observation
                            # orphan, not an invented committed successful forecast.
                            journal._publish_json(work/"map-observations"/f"{position:06d}.json", observation)
                            writer.append(row, (positions, window.target_positions_m, window.horizon_seconds))
                        guard()
                        if progress:
                            progress({"phase": "run_committed", "index": position, "expected": len(prepared.keys),
                                "new_committed": writer.tip["record_count"], "status": row["status"]})
                del driver
    except Exception as exc:
        errors.append({"error_type": type(exc).__name__, "error_message": str(exc)[:300]})
    finally:
        if query_maps is not None:
            try:
                last_maps = deepcopy(query_maps.identity)
            except Exception as exc:
                errors.append({"error_type": type(exc).__name__, "error_message": str(exc)[:300]})
            try:
                query_maps.close()
            except Exception as exc:
                errors.append({"error_type": type(exc).__name__, "error_message": str(exc)[:300]})
    # KeyboardInterrupt deliberately propagates without a completion result.
    saved = journal.read(writer.directory, writer.manifest_sha256)
    try:
        recheck()
        if not errors and not saved.failure_count and inherited_count+len(saved.records) == len(prepared.keys):
            map_report = maps.verify_attempt(work, saved, root_sha256=root_sha256, start_sha256=start_sha256,
                registered=registered, data_root=locations["data_root"], guard=guard)
            checked = science.whole_audit(prepared, assembled(work, saved), work, guard=guard)
            recheck()
            maps.verify_attempt(work, saved, root_sha256=root_sha256, start_sha256=start_sha256,
                registered=registered, data_root=locations["data_root"], guard=guard)
            journal._publish_json(work/"whole-science.json", checked)
            audit = checked
    except Exception as exc:
        errors.append({"error_type": type(exc).__name__, "error_message": str(exc)[:300]})
    result = {"schema_version": VERSION, "status": "computed_pending_supervisory_closure" if audit and not errors else "failed",
        "expected_run_count": len(prepared.keys), "inherited_success_count": inherited_count,
        "new_committed_count": len(saved.records), "new_failure_count": saved.failure_count,
        "missing_completed_record_count": len(prepared.keys)-inherited_count-len(saved.records),
        "journal_tip": saved.tip, "uncommitted_files": saved.uncommitted_files, "errors": errors,
        "resources": lineage.resources(spec, start["prior_elapsed_seconds"]),
        "reference_resource_charges": prepared.reference_resource_charges, "observer": observer.summary(),
        "new_attempt_maps": last_maps, "map_catalog_sha256": science._digest(registered), "map_observations": map_report,
        "elapsed_seconds": time.perf_counter()-began, "whole_science_sha256": native._hash(work/"whole-science.json") if audit else None,
        **science.UNQUALIFIED}
    journal._publish_json(work/"core-result.json", result)
    return result
