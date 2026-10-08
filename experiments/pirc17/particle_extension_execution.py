"""Missing-particle execution core; a tested owned supervisor is still required.

Only an immutable latest reservation selects data, work and output paths.
Reference budgets are never rerun. Successful extension ancestors are replayed
and byte-copied, then only their missing suffix is integrated and committed.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

from . import particle_extension as science
from . import particle_extension_lineage as lineage
from . import resume_journal as journal
from .seed_resume_execution import _copy_particles
from .precision_check import KEYS, load_particle_evidence

native, engine = science.native, science.engine
VERSION = science.VERSION+"-execution"


def assembled(work, saved):
    value = science.admission.read_closed(work/"inherited.json", native._hash(work/"inherited.json"))
    if set(value) != {"root_sha256", "start_sha256", "rows"}:
        raise ValueError("exact inherited materialization manifest required")
    rows = deepcopy(value["rows"])
    for record in saved.records:
        row = deepcopy(record["row"])
        if "particle_artifact" in row:
            row["particle_artifact"]["path"] = "journal/"+row["particle_artifact"]["path"]
        rows.append(row)
    return rows


def run(*, directory, root_sha256, index, start_sha256, progress=None):
    """Library core, not a production launch or process-closure certificate."""
    began = time.perf_counter()
    entered_at = lineage.datetime.now(lineage.timezone.utc)
    directory = Path(directory).resolve()
    root, _, start, actual_digest = lineage.history(directory, root_sha256, current_index=index)
    if actual_digest != start_sha256:
        raise ValueError("worker must bind the exact latest extension reservation")
    work = directory/"attempts"/f"{index:06d}"/"work"
    if work.exists():
        raise FileExistsError("extension worker never reopens an existing attempt")
    source, spec, locations = root["source"], root["plan"], root["data_locations"]
    elapsed_before_core = max(0., (entered_at-lineage.storage._date(start["started_at"])).total_seconds())
    observer = native.MemoryObserver()
    def guard():
        if time.perf_counter()-began+elapsed_before_core >= start["cooperative_remaining_seconds"]:
            raise TimeoutError("original cumulative extension budget exhausted; no renewal")
        observer.guard()
    guard()
    if root["source_sha256"].get("particle_extension_execution.py") != native._hash(Path(__file__)):
        raise ValueError("extension root must bind this execution core before any forecast")
    prepared = science.prepare(source=source,guard=guard)
    rows, directories, ancestry, _, _ = lineage.inherit(prepared,directory,root_sha256,current_index=index,guard=guard)
    inherited_count = len(rows)
    evidence = {"fit":source["fit"],"fit_ledger":source["fit_ledger"], **{k:spec[k] for k in
                ("fit_sha256","fit_ledger_sha256","training_policy_sha256")}}
    bundle, models, fitted_eligibility = engine.load_candidate(**evidence)
    if fitted_eligibility != spec["eligibility_sha256"]:
        raise ValueError("extension candidate changes original eligibility")
    encoder = engine.CanonicalEncoder.frozen_pirc22(Path(locations["snapshot"]))
    windows, parents, identity = engine.load_development(Path(locations["eligibility"]),spec["eligibility_sha256"],
        Path(locations["release"]),Path(locations["data_root"]),encoder)
    engine.validate_population(bundle,windows,parents,identity)
    selected = engine.select_windows(windows["validation"],count=spec["limit_origins"],policy=spec["selection_policy"])
    header = prepared.scientific_header
    if ([w.sample_id for w in selected] != header["sample_ids"] or identity["sha256"] != header["development_identity"]
            or {name:{str(seed):m.identity["sha256"] for seed,m in group.items()} for name,group in models.items()}
                != header["model_identities"]):
        raise ValueError("extension selection, development identity or complete fitted models changed")
    encoders = {name:engine.configuration_encoder(encoder,name) for name in spec["configurations"]}
    if any(len(e.columns)*2 != engine.configuration_width(name) for name,e in encoders.items()):
        raise ValueError("extension encoder differs from frozen configuration")
    query_type, modules = engine.resolve_map_backend(spec["map_backend"])
    if {m.__name__:native._hash(Path(m.__file__)) for m in modules} != header["map_source_sha256"]:
        raise ValueError("extension map code changed")
    for window in selected:
        original = next(r for r in prepared.reference[2:-1] if r["sample_id"] == window.sample_id)
        _, targets, times = load_particle_evidence(original,Path(source["reference_ledger"]).parent)
        if (window.block_id != original["independent_block_id"] or not engine.np.array_equal(window.target_positions_m,targets)
                or not engine.np.array_equal(window.horizon_seconds,times)):
            raise ValueError("extension changes original targets, times or independent block")
    bound = {directory/item["path"]:item["sha256"] for item in lineage.storage.inventory(directory)}
    bound[Path(locations["eligibility"])] = spec["eligibility_sha256"]
    def recheck():
        prepared.recheck(guard)
        if (lineage.source_hashes() != root["source_sha256"]
                or science.runtime_identity() != prepared.reference[0]["runtime"]
                or any(native._hash(path) != sha for path,sha in bound.items())):
            raise ValueError("extension sources, runtime, data or closed ancestry changed")
    recheck()
    work.mkdir()
    (work/"inherited").mkdir()
    writer = journal.Writer(work/"journal",workloads=[dict(zip(KEYS,k)) for k in prepared.keys],
        inherited_count=inherited_count,ancestry=ancestry,
        contract=lineage.journal_contract(root,root_sha256,index,start_sha256,prepared))
    errors, maps, maps_identity, audit = [], None, None, None
    inherited_rows = []
    try:
        for position,(row,folder) in enumerate(zip(rows,directories)):
            guard()
            inherited_rows.append(_copy_particles(row,folder,work,f"inherited/{position:06d}.npz"))
        journal._publish_json(work/"inherited.json",dict(root_sha256=root_sha256,start_sha256=start_sha256,rows=inherited_rows))
        if progress:
            progress({"phase":"prefix_inherited","inherited_count":inherited_count,"expected":len(prepared.keys)})
        maps = query_type(Path(locations["data_root"]),[Path(locations["data_root"])/p for p in engine.RECEIPTS])
        for parent in parents:
            maps.register_snapshot_parent(parent)
        grid, cursor = science.admission.entropy_grid(), 0
        for window in selected:
            for seed in spec["seeds"]:
                group_size = len(spec["configurations"])*len(spec["steps"])
                if cursor+group_size <= inherited_count:
                    cursor += group_size
                    continue
                guard()
                count = spec["particles"][0]
                driver = engine.BrownianPath(window.horizon_seconds,spec["steps"],history_step_seconds=5.,
                    particles=count,seed=seed,stream_id=window.sample_id)
                for name in spec["configurations"]:
                    model = models[name][seed]
                    query = (lambda positions:[{} for _ in positions]) if name=="base" else maps
                    provider = engine.PredictedPositionFeatures(encoders[name],query,window.frame)
                    def features(state):
                        guard()
                        return provider(state)
                    for step in spec["steps"]:
                        position = cursor
                        cursor += 1
                        if position < inherited_count:
                            continue
                        guard()
                        row = {"type":"run",**dict(zip(KEYS,prepared.keys[position])),"status":"failure",
                            "independent_block_id":window.block_id,"model_identity_sha256":model.identity["sha256"],
                            "brownian_identity":driver.identity,"actual_horizons_seconds":window.horizon_seconds.tolist()}
                        begin = time.perf_counter()
                        try:
                            prediction = engine.rollout(window.origin,window.horizon_seconds,particles=count,seed=seed,
                                max_step_seconds=step,history_step_seconds=5.,base_drift=model.base_drift,diffusion=model.diffusion,
                                terrain=features,conditioner=model.correction,brownian_increments=driver)
                            row["rollout_wall_seconds_including_lazy_map_initialization"] = time.perf_counter()-begin
                            row.update(invalid_feature_rows=prediction.invalid_feature_rows,feature_query_rows=prediction.feature_query_rows)
                            guard()
                            row["scores"] = engine.score_path(prediction.positions_m,window.target_positions_m,window.horizon_seconds,
                                time_weights=engine.TIME_WEIGHTS,entropy_grid=grid)
                            row["particle_precision"] = engine.energy_precision(prediction.positions_m,window.target_positions_m,engine.TIME_WEIGHTS)
                            guard()
                        except Exception as exc:
                            row.update(error_type=type(exc).__name__,error_message=str(exc)[:300],
                                elapsed_seconds_before_failure=time.perf_counter()-begin)
                            writer.append(row)
                            if isinstance(exc,(MemoryError,TimeoutError)):
                                raise
                        else:
                            row["status"] = "success"
                            writer.append(row,(prediction.positions_m,window.target_positions_m,window.horizon_seconds))
                        guard()
                        if progress:
                            progress({"phase":"run_committed","index":position,"expected":len(prepared.keys),
                                "new_committed":writer.tip["record_count"],"status":row["status"]})
                del driver
    except Exception as exc:
        errors.append({"error_type":type(exc).__name__,"error_message":str(exc)[:300]})
    finally:
        if maps is not None:
            try:
                maps_identity = maps.identity
            except Exception as exc:
                errors.append({"error_type":type(exc).__name__,"error_message":str(exc)[:300]})
            try:
                maps.close()
            except Exception as exc:
                errors.append({"error_type":type(exc).__name__,"error_message":str(exc)[:300]})
    # KeyboardInterrupt propagates; only the owner can attest real termination.
    # Journal publication failures retain orphans and the last durable cursor.
    saved = journal.read(writer.directory,writer.manifest_sha256)
    try:
        recheck()
        if not errors and inherited_count+len(saved.records)==len(prepared.keys) and not saved.failure_count:
            rows = assembled(work,saved)
            checked = science.whole_audit(prepared,rows,work,guard=guard)
            recheck()
            journal._publish_json(work/"whole-science.json",checked)
            audit = checked
    except Exception as exc:
        errors.append({"error_type":type(exc).__name__,"error_message":str(exc)[:300]})
    result = {"schema_version":VERSION,"status":"computed_pending_supervisory_closure" if audit and not errors else "failed",
        "expected_run_count":len(prepared.keys),"inherited_success_count":inherited_count,
        "new_committed_count":len(saved.records),"new_failure_count":saved.failure_count,
        "missing_completed_record_count":len(prepared.keys)-inherited_count-len(saved.records),
        "journal_tip":saved.tip,"uncommitted_files":saved.uncommitted_files,"errors":errors,
        "resources":lineage.resources(spec,start["prior_elapsed_seconds"]),"observer":observer.summary(),
        "new_attempt_maps":maps_identity,"elapsed_seconds":time.perf_counter()-began,
        "whole_science_sha256":native._hash(work/"whole-science.json") if audit else None,
        "production_resume_ready":False,**science.UNQUALIFIED}
    journal._publish_json(work/"core-result.json",result)
    return result
