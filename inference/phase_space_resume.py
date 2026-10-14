"""Exact CPU inference continuation, with ordered Welford full-state moments."""

from dataclasses import asdict
import hashlib
from pathlib import Path
import platform
import sys

import torch

from infrastructure.pirc26_forecast_codec import chunks, inspect_array, require, LIMIT
from infrastructure.research_control import canonical
from infrastructure.research_files import source_file_hash
from infrastructure.research_store import digest

SCHEMA = "pirc26-forecast-state-v1"


def scope(model, request):
    root = Path(__file__).resolve().parents[1]
    files = {name: source_file_hash(root, root / name) for name in (
        "inference/phase_space.py", "inference/phase_space_resume.py", "models/phase_space.py",
        "infrastructure/pirc26_forecast_codec.py")}
    return {"request_hash": digest(asdict(request)), "model_hash": model.checkpoint()["sha256"],
        "code_hash": digest(files), "environment": {"python": sys.version.split()[0], "torch": str(torch.__version__),
        "platform": platform.platform(), "machine": platform.machine(), "threads": torch.get_num_threads(),
        "torch_build_hash": hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
        "deterministic": torch.are_deterministic_algorithms_enabled(), "precision": torch.get_float32_matmul_precision(),
        "dtype": str(model.velocity_factor.dtype)}}


def array(tensor):
    value = tensor.detach().cpu().contiguous()
    return {"shape": list(value.shape), "dtype": str(value.dtype).split(".")[-1], "data": chunks(value.numpy().tobytes())}


def materialize(raw, shape, dtype):
    if not raw:
        return torch.empty(shape, dtype=dtype)
    return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape).clone()


def moments(samples):
    mean = torch.zeros(samples.shape[1:], dtype=torch.float64)
    m2 = torch.zeros((*samples.shape[1:-1], 4, 4), dtype=torch.float64)
    for count, sample in enumerate(samples, 1):
        delta = sample.double() - mean
        mean += delta / count
        m2 += delta[:, :, None] * (sample.double() - mean)[:, None, :]
    return mean, m2


def snapshot(identity, first, step, completed, ids, failed, states, alive, mean, m2):
    body = {"schema_version": SCHEMA, "scope": identity, "first": first, "step": step,
        "completed": array(completed), "sample_ids": ids, "failed_sample_ids": failed,
        "pending": None if states is None else array(torch.stack(states)),
        "alive": None if alive is None else alive.tolist(), "mean": array(mean), "m2": array(m2)}
    result = {**body, "sha256": digest(body)}
    canonical(result, LIMIT)
    return result


def restore(value, identity, request, dtype):
    canonical(value, LIMIT)  # Before any numerical allocation/materialization.
    require(type(value) is dict and set(value) == {"schema_version", "scope", "first", "step", "completed", "sample_ids",
        "failed_sample_ids", "pending", "alive", "mean", "m2", "sha256"}
        and value["schema_version"] == SCHEMA and value["scope"] == identity
        and value["sha256"] == digest({k:v for k,v in value.items() if k != "sha256"}), "forecast source/request/model identity differs")
    first, step = value["first"], value["step"]
    require(type(first) is int and 0 <= first <= request.sample_count and
            (first == request.sample_count or first % request.chunk_size == 0)
            and type(step) is int and 0 <= step < len(request.time_grid), "forecast cursor differs")
    ids, failed = value["sample_ids"], value["failed_sample_ids"]
    require(type(ids) is list and type(failed) is list and all(type(i) is int for i in ids + failed)
            and ids == sorted(set(ids)) and failed == sorted(set(failed))
            and sorted(ids + failed) == list(range(first)), "prefix IDs must be a disjoint complete partition")
    t = len(request.time_grid)
    kind = str(dtype).split(".")[-1]
    raw = inspect_array(value["completed"], [len(ids), t, 4], kind)
    raw_mean = inspect_array(value["mean"], [t, 4], "float64")
    raw_m2 = inspect_array(value["m2"], [t, 4, 4], "float64")
    pending, alive = value["pending"], value["alive"]
    if pending is None:
        require(step == 0 and alive is None, "empty chunk must have no pending cursor")
        raw_pending = None
    else:
        size = min(request.chunk_size, request.sample_count - first)
        require(size > 0 and type(alive) is list and len(alive) == size and all(type(a) is bool for a in alive), "pending failure mask")
        raw_pending = inspect_array(pending, [step + 1, size, 4], kind)
    # All fields, even the last one, have been semantically inspected now.
    completed = materialize(raw, [len(ids), t, 4], dtype)
    mean, m2 = materialize(raw_mean, [t,4], torch.float64), materialize(raw_m2, [t,4,4], torch.float64)
    expected_mean, expected_m2 = moments(completed)
    require(torch.equal(mean, expected_mean) and torch.equal(m2, expected_m2), "saved moments differ from ordered prefix population")
    states = None if raw_pending is None else list(materialize(raw_pending, pending["shape"], dtype).unbind(0))
    mask = None if alive is None else torch.tensor(alive, dtype=torch.bool)
    if states is not None:
        require(torch.equal(states[0], torch.tensor(request.initial_state, dtype=dtype).expand(len(mask),4))
                and (step == 0 and bool(mask.all()) or step > 0)
                and bool((states[-1][~mask] == 0).all()), "pending initial/quarantined state differs")
    return first, step, completed, list(ids), list(failed), states, mask, mean, m2
