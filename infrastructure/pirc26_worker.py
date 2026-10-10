"""Internal numerical worker; only the shared supervisor dispatches this path."""

from pathlib import Path
import sys
import time

from infrastructure.research_control import WorkerControl, write_frame
from infrastructure.pirc26_process_resources import process_resources
from infrastructure.research_store import ResearchError, atomic_write, digest, encode


def run(output, handoff_hash):
    # Reject absent ownership before importing torch or allocating any model.
    from application.pirc26_runtime import load_handoff
    control = WorkerControl.from_environment()
    receipt, plugin, job, transport, restored, ownership = load_handoff(output, handoff_hash, control)
    import torch
    torch.set_num_threads(1)
    from application.pirc26_components import construct_components
    from application.pirc26_data import decode_block
    from application.pirc26_training_control import ManagedTrainingControl
    from application.pirc26_forecast_control import (is_forecast_state, validate_saved_job, materialize_fit,
        pack_fit, ManagedForecastControl)
    from inference.phase_space_resume import array, materialize, moments
    from infrastructure.pirc26_forecast_codec import inspect_array, require
    from models.phase_space import ModelContractError
    from evaluation.phase_space import evaluate_forecast
    from application.pirc26_metrics import metric_binding, aggregate_metric
    from estimation.phase_space_o2 import HorizonTrainingExample
    spec, cell = receipt["spec"], receipt["cell"]
    metric = metric_binding(job, receipt)
    parts, plan = construct_components(plugin.registry_entry, cell["execution"]["components"],
        matrix_cells=len(spec["cells"]), seed=cell["seed"], registries=plugin.component_registries,
        initial_checkpoint=job["initial_checkpoint"])
    model = parts["model"]
    profile = cell["execution"]["inputs"]
    block = decode_block(transport, model, max_observations=profile["observations"])
    population_evidence = None
    selection_evidence = None
    if plugin.plugin_id.startswith("pirc26-population-"):
        from application.pirc26_population_runtime import source
        population, population_evidence = source(receipt)
        require([s["segment_id"] for s in block.segments] == [s["pooled_segment_id"] for s in population["provenance"]["segments"]],
                "decoded training population segment identity differs")
    elif plugin.plugin_id.startswith("pirc26-selection-"):
        from application.pirc26_selection_runtime import source
        selected, selection_evidence = source(receipt)
        require([s["segment_id"] for s in block.segments] == selection_evidence["segment_ids"],
                "decoded selection segment membership differs")
    requests = [block.forecast_request(recipe["segment_id"], recipe["origin_index"], recipe["time_grid"],
        sample_count=recipe["sample_count"], brownian_root_id=recipe["brownian_root_id"], chunk_size=recipe["chunk_size"])
        for recipe in job["origins"]]
    from dataclasses import asdict
    if [digest(asdict(req)) for req in requests] != cell["execution"]["config"]["forecast_request_hashes"]:
        raise ResearchError("CONTRACT_MISMATCH", "causal requests differ from the registered origin recipes")
    fit = {"status": "FROZEN", "checkpoint": model.checkpoint()}
    completed_predictions, origin_index, active = [], 0, None
    fit_frame, batch_count = None, 0
    forecast_recovery = is_forecast_state(restored)
    if forecast_recovery:
        _, completed_predictions, origin_index, active = validate_saved_job(restored, job, receipt)
        fit_frame = restored["method_state"]["fit"]
        fit = materialize_fit(fit_frame)
        from estimation.phase_space_checkpoint import restore_model, decode_state, restore_rng
        restore_model(model, fit["checkpoint"])
        restore_rng(decode_state(restored["rng_state"]))
    elif job["operation"] == "fit-and-forecast":
        trainer = parts["trainer"]
        data = block.transitions(batch_size=job["batch_size"]) if trainer.document["objective"] == "O1" else [
            HorizonTrainingExample(req, block.truth(recipe["segment_id"], req), model.spec.train_binding_hash)
            for recipe, req in zip(job["origins"], requests)]
        batch_count = len(data)
        if population_evidence is not None:
            require(batch_count == population_evidence["batch_count"], "complete training batch population differs")
        if plugin.resume_level == "restart-only":
            if restored is not None:
                raise ResearchError("CHECKPOINT_INCOMPATIBLE", "basis QR cannot resume an optimizer state")
            fit = trainer.fit(model, data, o1_result=job["o1_result"], cancellation=ownership.requested)
        else:
            managed = ManagedTrainingControl(trainer.plan.max_steps)
            fit = trainer.fit(model, data, o1_result=job["o1_result"], **managed.arguments(restored))
        if fit["status"] == "CHECKPOINTED":
            return 85  # save() has received the actual owner ACK.
    elif restored is not None:
        raise ResearchError("CHECKPOINT_INCOMPATIBLE", "training recovery cannot substitute a frozen forecast job")
    if population_evidence is not None:
        if forecast_recovery:
            require(fit.get("training_population") == population_evidence, "recovered training population differs")
        else:
            fit["training_population"] = population_evidence
    if control is not None and fit_frame is None:
        if selection_evidence is not None:
            fit["selection_input"] = selection_evidence
        fit["producer_attempt_id"] = ownership.attempt_id
        fit_frame = pack_fit(fit, job, cell["execution"]["config"], batch_count)
    elif selection_evidence is not None:
        require(fit.get("selection_input") == selection_evidence, "recovered selection identity differs")
    forecasts, forecast_phase_times = [], []
    for index, (recipe, req) in enumerate(zip(job["origins"], requests)):
        if index < origin_index:
            prediction = dict(completed_predictions[index])
            shape = [len(prediction["sample_ids"]),len(req.time_grid),4]
            raw = inspect_array(prediction["samples"], shape, str(model.velocity_factor.dtype).split(".")[-1])
            prediction["samples"] = materialize(raw, shape, model.velocity_factor.dtype)
            mean, m2 = moments(prediction["samples"])
            expected_moments = {"estimator_id": "ordered-sample-welford-full-state-v1", "sample_count": shape[0],
                "mean": mean.tolist(), "covariance": (m2/(shape[0]-1)).tolist()}
            require(prediction["moments"] == expected_moments, "completed origin moment population differs")
        else:
            managed_forecast = None if control is None else ManagedForecastControl(
                control, job, fit_frame, completed_predictions, index, ownership.attempt_id,
                active if index == origin_index else None)
            phase_started = time.monotonic()
            prediction = parts["predictor"].predict(model, req,
                cancellation=ownership.requested if control is None else None,
                resume_state=active if index == origin_index else None,
                checkpoint_requested=None if managed_forecast is None else managed_forecast.requested,
                checkpoint_handler=None if managed_forecast is None else managed_forecast.save)
            if prediction.get("status") == "CHECKPOINTED":
                return 85  # Actual control.save has received the owner's ACK.
            forecast_phase_times.append({"origin_index": index,
                "started_monotonic_seconds": phase_started,
                "finished_monotonic_seconds": time.monotonic()})
        completion = prediction.pop("completion_state", None)
        try:
            evaluation = evaluate_forecast(prediction, block.truth(recipe["segment_id"], req),
                cancellation=None if control is None or index < origin_index else managed_forecast.requested)
        except ModelContractError as exc:
            if (completion is None or control is None or not str(exc).startswith("INTERRUPTED:")
                    or not managed_forecast.requested()):
                raise
            managed_forecast.save(completion,{"completed_steps": req.sample_count*(len(req.time_grid)-1),
                                              "total_steps": req.sample_count*(len(req.time_grid)-1)})
            return 85
        if evaluation["status"] != "SUCCEEDED":
            raise ResearchError("NONFINITE", "incomplete paths cannot become a successful comparison result")
        if index >= origin_index:
            completed_predictions.append({**prediction, "samples": array(prediction["samples"])})
        forecasts.append({"request_hash": digest(asdict(req)), "evaluation": evaluation,
                          "sample_ids": prediction["sample_ids"],
                          "moments": prediction["moments"],
                          "samples": prediction["samples"].detach().cpu().tolist()})
    # Hash-covered observation through fitting/forecast/evaluation. Keep
    # telemetry out of exact scientific forecast/history equality comparisons.
    fit["worker_resource_observation"] = {**process_resources("job-payload-before-encoding"),
                                          "attempt_id": ownership.attempt_id}
    # Honest current-attempt solver intervals, separate from scientific output
    # and lifetime memory peaks. Reused completed origins contribute no copied
    # interval; startup/admission, scoring and publication are not solver work.
    fit["worker_phase_observation"] = {"schema_version": "pirc26-worker-phase-observation-v1",
        "attempt_id": ownership.attempt_id, "units": "seconds", "clock": "time.monotonic",
        "scope": "current-attempt-forecast-solver-only", "includes_previous_attempts": False,
        "forecast": forecast_phase_times}
    payload = {"metrics": aggregate_metric(forecasts, metric), "forecast": {"origins": forecasts, "metric_binding": metric},
        "fit": fit, "source_schema": "pirc26-observed-phase-space-result-v1"}
    result = {"schema_version": "pirc25-result-v1", "status": "SUCCEEDED", "spec_hash": digest(spec),
        "cell_hash": digest(cell), "protocol_hash": spec["protocol_hash"], "input_hash": spec["data_hash"],
        "output_hash": digest(payload), "state_order": list(plugin.state_order), "units": list(plugin.units),
        "resume_level": plugin.resume_level, "qualification": receipt["qualification"],
        "admission_hash": receipt["admission_hash"], "component_plan_hash": plan["component_plan_hash"],
        "metric_units": {"energy_score": "m"}, "metric_definitions": {"energy_score": metric["definition"]},
        "source_identity": transport["source_identity"], **payload}
    content = encode(result)
    if population_evidence is not None:
        from application.pirc26_population_runtime import result_validate
        result_validate(receipt, result)
    elif selection_evidence is not None:
        from application.pirc26_selection_runtime import result_validate
        result_validate(receipt, result)
    if len(content) > receipt["resource_plan"]["limits"]["result_bytes"]:
        raise ResearchError("RESOURCE_PLAN_REJECTED", "actual numerical result exceeds the declared publication quota")
    atomic_write(Path(output), content)
    return 0


def main():
    if len(sys.argv) != 3:
        raise ResearchError("MISSING_INPUT", "internal worker requires owner output and handoff identity")
    from infrastructure.pirc26_worker_control import require_owned_worker
    ownership = require_owned_worker(sys.argv[1])
    # No data or exception message in logs; stable codes only. The owner keeps
    # its existing FAILED/WORKER_FAILED semantics for a nonzero worker exit.
    try:
        code = run(sys.argv[1], sys.argv[2])
    except Exception as exc:
        code = 1
        error_code = getattr(exc, "code", None)
        if error_code is None:
            # Numerical components have a closed typed-prefix vocabulary;
            # retain only that code, never a raw exception message/data value.
            prefix = str(exc).partition(":")[0]
            error_code = prefix if prefix in {"MODEL_CONTRACT_ERROR", "OBJECTIVE_INCOMPATIBLE", "RESOURCE_PLAN_REJECTED",
                "CHECKPOINT_INCOMPATIBLE", "UNAUTHORIZED_DATA", "NONFINITE", "ILL_CONDITIONED", "INTERRUPTED"} else "WORKER_FAILED"
        diagnostic = {"schema_version": "pirc26-worker-failure-v1", "error_code": error_code}
        atomic_write(ownership.directory / "pirc26-failure.json", encode(diagnostic))
    try:
        write_frame(ownership.directory / "pirc26-resources.json",
            {**process_resources("worker-exit-after-output-or-ack"), "attempt_id": ownership.attempt_id},
            16384, deadline=ownership.deadline)
    except (OSError, ValueError):
        # Hard stop/IO failure may prevent this optional terminal observation.
        # Never replace a prior pre-ACK report, actual outcome or native exit.
        pass
    from application.pirc26_training_control import exit_managed_worker
    exit_managed_worker(code, output=sys.argv[1])


if __name__ == "__main__":
    main()
