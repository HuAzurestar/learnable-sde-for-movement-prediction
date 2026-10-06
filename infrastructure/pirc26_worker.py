"""Internal numerical worker; only the shared supervisor dispatches this path."""

from pathlib import Path
import sys

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
    requests = [block.forecast_request(recipe["segment_id"], recipe["origin_index"], recipe["time_grid"],
        sample_count=recipe["sample_count"], brownian_root_id=recipe["brownian_root_id"], chunk_size=recipe["chunk_size"])
        for recipe in job["origins"]]
    from dataclasses import asdict
    if [digest(asdict(req)) for req in requests] != cell["execution"]["config"]["forecast_request_hashes"]:
        raise ResearchError("CONTRACT_MISMATCH", "causal requests differ from the registered origin recipes")
    fit = {"status": "FROZEN", "checkpoint": model.checkpoint()}
    if job["operation"] == "fit-and-forecast":
        trainer = parts["trainer"]
        data = block.transitions(batch_size=job["batch_size"]) if trainer.document["objective"] == "O1" else [
            HorizonTrainingExample(req, block.truth(recipe["segment_id"], req), model.spec.train_binding_hash)
            for recipe, req in zip(job["origins"], requests)]
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
    forecasts = []
    for recipe, req in zip(job["origins"], requests):
        prediction = parts["predictor"].predict(model, req,
            cancellation=lambda: ownership.requested() if control is None else control.poll() is not None)
        evaluation = evaluate_forecast(prediction, block.truth(recipe["segment_id"], req))
        if evaluation["status"] != "SUCCEEDED":
            raise ResearchError("NONFINITE", "incomplete paths cannot become a successful comparison result")
        forecasts.append({"request_hash": digest(asdict(req)), "evaluation": evaluation,
                          "sample_ids": prediction["sample_ids"],
                          "samples": prediction["samples"].detach().cpu().tolist()})
    # Hash-covered observation through fitting/forecast/evaluation. Keep
    # telemetry out of exact scientific forecast/history equality comparisons.
    fit["worker_resource_observation"] = {**process_resources("job-payload-before-encoding"),
                                          "attempt_id": ownership.attempt_id}
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
