"""Owned inference phase checkpoints; fresh admission/authorization stays outside."""

import time
import math

from infrastructure.pirc26_forecast_codec import pack_json, unpack_json, require, LIMIT, inspect_array
from infrastructure.research_control import canonical, write_frame
from infrastructure.pirc26_process_resources import process_resources
from infrastructure.research_store import digest, ResearchError

SCHEMA = "pirc26-owned-forecast-state-v1"


def is_forecast_state(state):
    return type(state) is dict and type(state.get("method_state")) is dict and state["method_state"].get("schema_version") == SCHEMA


def preflight_job_capacity(job, config, inputs):
    # Original route stays bounded; no sampling/training-plan shortening. Bound
    # full scientific history, worst-width model numbers, samples and moments.
    cp = job["initial_checkpoint"]
    numbers = 0
    stack = [cp]
    while stack:
        value = stack.pop()
        if type(value) is dict:
            stack.extend(value.values())
        elif type(value) is list:
            stack.extend(value)
        elif type(value) in (int, float):
            numbers += 1
    model_bytes = len(canonical(cp, 2 * 1024 * 1024)) + numbers * 32
    sample_values = sum(r["sample_count"] * len(r["time_grid"]) * 4 for r in job["origins"])
    moment_values = sum(len(r["time_grid"]) * 24 for r in job["origins"])
    history_bytes = 0
    if job["operation"] == "fit-and-forecast":
        # Each row has two full-precision floats, exact counters and O2's fixed
        # Brownian/estimator/horizon metadata. No lossy/truncated representation.
        history_bytes = config["plan"]["max_steps"] * 76
    common = model_bytes + (sample_values + moment_values) * 26 + 65536
    full_history_bytes = 0
    if job["operation"] == "fit-and-forecast":
        width = 150 if config["objective"] == "O1" else 300 + 4 * len(config["plan"]["horizon_indices"])
        full_history_bytes = config["plan"]["max_steps"] * width
    publication = common + full_history_bytes
    # Saved completed origin bodies are losslessly compressed canonical JSON;
    # a compression ratio is NEVER assumed for admission/capacity.
    checkpoint = ((common + history_bytes) * 4 + 2) // 3 + 65536
    if publication > LIMIT or checkpoint > LIMIT:
        raise ResearchError("RESOURCE_PLAN_REJECTED", "complete forecast/history publication or continuation frame exceeds registered capacity")


def pack_fit(fit, job, config, batch_count):
    if "history" not in fit:
        return pack_json({"result": fit, "history_recipe": None})
    from estimation.phase_space_checkpoint import pack_history
    identity = ([None] * batch_count if config["objective"] == "O1" else {"examples": [
        {"request": {"brownian_root_id": r["brownian_root_id"]}} for r in job["origins"]]})
    recipe = {"objective": config["objective"], "plan": config["plan"], "data_identity": identity}
    result = {**fit, "history": pack_history(fit["history"], recipe)}
    return pack_json({"result": result, "history_recipe": recipe})


def materialize_fit(frame):
    value = unpack_json(frame)
    fit, recipe = value["result"], value["history_recipe"]
    if recipe is not None:
        from estimation.phase_space_checkpoint import unpack_history
        fit["history"] = unpack_history(fit["history"], recipe, fit["steps"])
    return fit


def validate_saved_job(state, job, receipt):
    canonical(state, LIMIT)
    require(type(state) is dict and set(state) == {"step", "data_position", "method_state", "rng_state"}
            and type(state["step"]) is int and state["step"] >= 0, "closed owned forecast envelope")
    method = state["method_state"]
    require(type(method) is dict and set(method) == {"schema_version", "job_hash", "origin_index", "fit", "finished", "active", "sha256"}
            and method["schema_version"] == SCHEMA and method["job_hash"] == digest(job)
            and method["sha256"] == digest({k:v for k,v in method.items() if k != "sha256"}), "owned forecast job binding differs")
    index, active = method["origin_index"], method["active"]
    require(type(index) is int and 0 <= index < len(job["origins"]), "owned forecast origin cursor")
    packed, finished = unpack_json(method["fit"]), unpack_json(method["finished"])
    require(type(packed) is dict and set(packed) == {"result", "history_recipe"}, "closed fitted-state representation")
    fit, history_recipe = packed["result"], packed["history_recipe"]
    require(type(fit) is dict and type(finished) is list and len(finished) == index, "completed origin population differs")
    from application.pirc26_components import preflight_model
    config, inputs = receipt["cell"]["execution"]["config"], receipt["cell"]["execution"]["inputs"]
    cp = fit.get("checkpoint")
    preflight_model(cp, {**config, "initial_model_hash": cp.get("sha256") if type(cp) is dict else None}, inputs)
    if job["operation"] == "forecast":
        require(fit.get("status") == "FROZEN" and cp == job["initial_checkpoint"] and history_recipe is None, "frozen inference checkpoint differs")
    else:
        columns = fit.get("history")
        require(fit.get("status") in {"CONVERGED", "MAX_STEPS"} and fit.get("objective") == config["objective"]
                and type(fit.get("steps")) is int and 0 < fit["steps"] <= config["plan"]["max_steps"]
                and type(columns) is dict and set(columns) == {"schema_version", "objective", "gradient_norm"}
                and columns["schema_version"] == "pirc26-history-columns-v1" and all(type(columns[k]) is list
                and len(columns[k]) == fit["steps"] and all(type(v) is float and math.isfinite(v) for v in columns[k])
                for k in ("objective", "gradient_norm")), "finished training state differs")
        require(type(history_recipe) is dict and set(history_recipe) == {"objective", "plan", "data_identity"}
                and history_recipe["objective"] == config["objective"] and history_recipe["plan"] == config["plan"], "fitted history recipe differs")
        identity = history_recipe["data_identity"]
        if config["objective"] == "O1":
            require(type(identity) is list and 1 <= len(identity) <= inputs["batches"]
                    and all(v is None for v in identity), "fitted history batch population differs")
        else:
            expected = {"examples": [{"request": {"brownian_root_id": r["brownian_root_id"]}} for r in job["origins"]]}
            require(identity == expected, "fitted history origin population differs")
    work = sum(r["sample_count"] * (len(r["time_grid"]) - 1) for r in job["origins"][:index])
    require(type(active) is dict and set(active) == {"schema_version", "scope", "first", "step", "completed", "sample_ids",
        "failed_sample_ids", "pending", "alive", "mean", "m2", "sha256"} and active.get("schema_version") == "pirc26-forecast-state-v1"
            and active.get("sha256") == digest({k:v for k,v in active.items() if k != "sha256"})
            and active.get("scope", {}).get("request_hash") == config["forecast_request_hashes"][index]
            and active["scope"].get("model_hash") == cp["sha256"], "active forecast/request/model differs")
    recipe = job["origins"][index]
    first, step = active.get("first"), active.get("step")
    require(type(first) is int and 0 <= first <= recipe["sample_count"] and type(step) is int
            and 0 <= step < len(recipe["time_grid"]) and (first == recipe["sample_count"] or first % recipe["chunk_size"] == 0),
            "active forecast work cursor")
    batch = min(recipe["chunk_size"], recipe["sample_count"] - first)
    work += first * (len(recipe["time_grid"]) - 1) + (0 if active.get("pending") is None else batch * step)
    require(state["step"] == work and state["data_position"] == {"origin_index": index, "first": first, "step": step},
            "owned forecast position/work differs")
    # No tensor allocations: inspect every binary field before numerical import.
    t, kind = len(recipe["time_grid"]), inputs["dtype"]
    ids, failed = active.get("sample_ids"), active.get("failed_sample_ids")
    require(type(ids) is list and type(failed) is list and all(type(i) is int for i in ids + failed)
            and ids == sorted(set(ids)) and failed == sorted(set(failed))
            and sorted(ids + failed) == list(range(first)), "active prefix IDs differ")
    inspect_array(active["completed"], [len(ids), t, 4], kind)
    inspect_array(active["mean"], [t,4], "float64")
    inspect_array(active["m2"], [t,4,4], "float64")
    if active.get("pending") is not None:
        inspect_array(active["pending"], [step + 1,batch,4], kind)
        require(type(active.get("alive")) is list and len(active["alive"]) == batch
                and all(type(a) is bool for a in active["alive"]), "active failure mask differs")
    else:
        require(step == 0 and active.get("alive") is None, "empty active chunk cursor differs")
    for prediction, recipe in zip(finished, job["origins"]):
        require(type(prediction) is dict and prediction.get("schema_version") == "pirc26-forecast-result-v1"
                and prediction.get("sample_ids") == list(range(recipe["sample_count"]))
                and prediction.get("failed_sample_ids") == [] and prediction.get("requested_paths") == recipe["sample_count"]
                and prediction.get("valid_paths") == recipe["sample_count"] and prediction.get("time_grid") == recipe["time_grid"]
                and prediction.get("brownian_root_id") == recipe["brownian_root_id"], "completed origin binding/population differs")
        inspect_array(prediction["samples"], [recipe["sample_count"],len(recipe["time_grid"]),4], kind)
    return packed, finished, index, active


class ManagedForecastControl:
    def __init__(self, control, job, fit_frame, finished, index, attempt_id, active=None):
        self.control, self.job, self.fit, self.finished, self.index = control, job, fit_frame, finished, index
        self.attempt_id, self.started = attempt_id, time.perf_counter()
        recipe = job["origins"][index]
        self.first_work = 0 if active is None else active["first"] * (len(recipe["time_grid"])-1) + (
            0 if active["pending"] is None else min(recipe["chunk_size"],recipe["sample_count"]-active["first"]) * active["step"])

    def requested(self):
        return self.control.poll() is not None

    def save(self, active, row):
        from estimation.phase_space_checkpoint import encode_state, rng_state
        offset = sum(r["sample_count"] * (len(r["time_grid"])-1) for r in self.job["origins"][:self.index])
        completed = offset + row["completed_steps"]
        total = sum(r["sample_count"] * (len(r["time_grid"])-1) for r in self.job["origins"])
        body = {"schema_version": SCHEMA, "job_hash": digest(self.job), "origin_index": self.index,
                "fit": self.fit, "finished": pack_json(self.finished), "active": active}
        envelope = {"step": completed, "data_position": {"origin_index": self.index, "first": active["first"], "step": active["step"]},
                    "method_state": {**body, "sha256": digest(body)}, "rng_state": encode_state(rng_state())}
        # Capture and save actual RNG, never a balance/deadline or old telemetry.
        elapsed = max(1e-9, time.perf_counter() - self.started)
        local_work = row["completed_steps"]
        rate = max(0, local_work - self.first_work) / elapsed
        progress = {"completed_steps": completed, "total_steps": total,
                    "throughput_per_second": rate, "eta_seconds": (total-completed) / max(rate,1e-9)}
        canonical(envelope, min(LIMIT, self.control.descriptor["byte_limit"]) - 16384)
        write_frame(self.control.directory / "pirc26-resources.json",
            {**process_resources("checkpoint-save-before-ack"), "attempt_id": self.attempt_id}, 16384,
            deadline=self.control.descriptor["deadline"])
        self.control.save(envelope, progress)
