"""Bounded JSON training state and exact CPU continuation, without pickle."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import platform
import sys

import numpy as np
import torch

from infrastructure.research_files import source_file_hash
from infrastructure.research_store import digest
from models.phase_space import ModelContractError, PhaseSpaceSDE


LIMIT = 4 * 1024 * 1024
SOURCE_FILES = ("estimation/phase_space_checkpoint.py", "estimation/phase_space.py",
                "estimation/phase_space_basis.py",
                "estimation/phase_space_o2.py", "models/phase_space.py",
                "inference/phase_space.py", "evaluation/phase_space.py")


def _encode(value, depth=0):
    if depth > 32:
        raise ModelContractError("MODEL_CONTRACT_ERROR: checkpoint nesting quota")
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.numel() > 100000 or not torch.isfinite(value).all():
            raise ModelContractError("MODEL_CONTRACT_ERROR: bounded finite CPU checkpoint tensor required")
        return {"tensor": {"dtype": str(value.dtype).split(".")[-1], "shape": list(value.shape),
                           "data": value.detach().tolist()}}
    if isinstance(value, np.ndarray):
        if value.dtype.name not in ("uint32", "int64", "float32", "float64") or value.size > 100000 or not np.isfinite(value).all():
            raise ModelContractError("MODEL_CONTRACT_ERROR: bounded checkpoint RNG array required")
        return {"array": {"dtype": value.dtype.name, "shape": list(value.shape), "data": value.tolist()}}
    if isinstance(value, dict):
        return {"map": [[_encode(key, depth + 1), _encode(item, depth + 1)] for key, item in value.items()]}
    if isinstance(value, tuple):
        return {"tuple": [_encode(item, depth + 1) for item in value]}
    if isinstance(value, list):
        return [_encode(item, depth + 1) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ModelContractError("MODEL_CONTRACT_ERROR: unsupported checkpoint object")


def _decode(value, depth=0):
    if depth > 32:
        raise ModelContractError("MODEL_CONTRACT_ERROR: checkpoint nesting quota")
    if isinstance(value, dict):
        if len(value) != 1:
            raise ModelContractError("MODEL_CONTRACT_ERROR: checkpoint tag mismatch")
        kind, item = next(iter(value.items()))
        if kind in ("tensor", "array"):
            if (type(item) is not dict or set(item) != {"dtype", "shape", "data"}
                    or type(item["shape"]) is not list or len(item["shape"]) > 8
                    or any(type(i) is not int or not 0 <= i <= 100000 for i in item["shape"])):
                raise ModelContractError("MODEL_CONTRACT_ERROR: checkpoint array shape")
            count = 1
            for size in item["shape"]:
                count *= size
            if count > 100000:
                raise ModelContractError("MODEL_CONTRACT_ERROR: checkpoint array element quota")
            if kind == "tensor":
                if item["dtype"] not in ("float32", "float64", "int64", "int32", "uint8", "bool"):
                    raise ModelContractError("MODEL_CONTRACT_ERROR: checkpoint tensor dtype")
                array = torch.tensor(item["data"], dtype=getattr(torch, item["dtype"]))
                if array.numel() != count or not torch.isfinite(array).all():
                    raise ModelContractError("MODEL_CONTRACT_ERROR: checkpoint tensor content")
            else:
                if item["dtype"] not in ("uint32", "int64", "float32", "float64"):
                    raise ModelContractError("MODEL_CONTRACT_ERROR: checkpoint RNG dtype")
                array = np.asarray(item["data"], dtype=item["dtype"])
                if array.size != count or not np.isfinite(array).all():
                    raise ModelContractError("MODEL_CONTRACT_ERROR: checkpoint RNG content")
            return array.reshape(item["shape"])
        if kind == "tuple":
            if type(item) is not list:
                raise ModelContractError("MODEL_CONTRACT_ERROR: checkpoint tuple content")
            return tuple(_decode(child, depth + 1) for child in item)
        if kind == "map":
            if type(item) is not list or any(type(pair) is not list or len(pair) != 2 for pair in item):
                raise ModelContractError("MODEL_CONTRACT_ERROR: checkpoint map content")
            decoded = {}
            for key, child in item:
                key = _decode(key, depth + 1)
                if type(key) not in (str, int) or key in decoded:
                    raise ModelContractError("MODEL_CONTRACT_ERROR: checkpoint map key")
                decoded[key] = _decode(child, depth + 1)
            return decoded
        raise ModelContractError("MODEL_CONTRACT_ERROR: unknown checkpoint tag")
    if isinstance(value, list):
        return [_decode(item, depth + 1) for item in value]
    return value


def encode_state(value):
    encoded = _encode(value)
    try:
        # Leave envelope/RNG duplication margin below the shared 65,536 nodes.
        stack, remaining = [(encoded, 0)], 45000
        while stack:
            item, depth = stack.pop()
            remaining -= 1
            if remaining < 0 or depth > 28:
                raise ValueError("checkpoint structural quota")
            if isinstance(item, dict):
                stack.extend((child, depth + 1) for child in item.values())
            elif isinstance(item, list):
                stack.extend((child, depth + 1) for child in item)
            elif type(item) is str and len(item) > 65536 or type(item) is int and item.bit_length() > 256:
                raise ValueError("checkpoint scalar quota")
        if len(json.dumps(encoded, allow_nan=False).encode()) > LIMIT:
            raise ValueError("checkpoint byte quota")
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ModelContractError("RESOURCE_PLAN_REJECTED: checkpoint structure/byte quota") from exc
    return encoded


def decode_state(value):
    try:
        raw = json.dumps(value, allow_nan=False)
        if len(raw.encode()) > LIMIT:
            raise ModelContractError("RESOURCE_PLAN_REJECTED: checkpoint byte quota")
        return _decode(json.loads(raw))
    except (ValueError, TypeError, KeyError, OverflowError, RuntimeError, RecursionError) as exc:
        if isinstance(exc, ModelContractError):
            raise
        raise ModelContractError("MODEL_CONTRACT_ERROR: malformed training state") from exc


def tensor_identity(value):
    if value is None:
        return None
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}


def training_scope(model, plan, data_identity, objective):
    root = Path(__file__).resolve().parents[1]
    files = {name: source_file_hash(root, root / name, maximum_bytes=2 * 1024 * 1024) for name in SOURCE_FILES}
    return {"objective": objective, "plan": asdict(plan), "data_identity": data_identity,
        "train_binding_hash": model.spec.train_binding_hash, "model_spec": model.model_card()["spec"],
        "model_configuration": model.acceleration_model.configuration(), "code_hash": digest(files),
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
                        "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
                        "torch_build_hash": hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                        "matmul_precision": torch.get_float32_matmul_precision(),
                        "cpu_threads": torch.get_num_threads(), "dtype": str(model.velocity_factor.dtype)}}


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": None, "cuda_reason": "CPU-only registered training component"}


def restore_rng(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])


def restore_model(model, checkpoint):
    restored = PhaseSpaceSDE.from_checkpoint(checkpoint)
    model.load_state_dict(restored.state_dict())


def train_loop(model, parameters, plan, scope, objective, monitor, *, auxiliary=None,
               resume_state=None, cancellation=None, progress=None,
               checkpoint_requested=None, checkpoint_handler=None):
    """Continue the actual Adam/batch-counter state, with immutable input scope."""
    import time
    if torch.get_num_threads() != 1:
        raise ModelContractError("OBJECTIVE_INCOMPATIBLE: exact CPU training requires one declared thread")
    if (checkpoint_requested is None) != (checkpoint_handler is None):
        raise ModelContractError("MODEL_CONTRACT_ERROR: checkpoint request and handler must be paired")
    scope_hash = digest(scope)
    optimizer = torch.optim.Adam(parameters, lr=plan.learning_rate)
    initial_model_hash = model.checkpoint()["sha256"]
    first_step, stale, best, history = 0, 0, None, []
    best_checkpoint = None
    if resume_state is not None:
        if (not isinstance(resume_state, dict) or resume_state.get("schema_version") != "pirc26-training-state-v1"
                or resume_state.get("scope_hash") != scope_hash
                or resume_state.get("initial_model_hash") != initial_model_hash):
            raise ModelContractError("CHECKPOINT_INCOMPATIBLE: code/data/model/objective/environment differs")
        detached = dict(resume_state)
        identity = detached.pop("sha256", None)
        if identity != digest(detached):
            raise ModelContractError("CHECKPOINT_INCOMPATIBLE: training state identity differs")
        state = decode_state(resume_state["state"])
        first_step, stale, best, history = state["step"], state["stale"], state["best"], state["history"]
        if type(first_step) is not int or not 0 <= first_step <= plan.max_steps or len(history) != first_step:
            raise ModelContractError("CHECKPOINT_INCOMPATIBLE: training position/history differs")
        restore_model(model, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if auxiliary is not None:
            auxiliary.load_state_dict(state["auxiliary"])
        elif state["auxiliary"] is not None:
            raise ModelContractError("CHECKPOINT_INCOMPATIBLE: diffusion optimizer state differs")
        best_checkpoint = state["best_checkpoint"]
        restore_rng(state["rng"])
    started, status, last_state = time.perf_counter(), "MAX_STEPS", None
    for step in range(first_step, plan.max_steps):
        if stale >= plan.patience:
            status = "CONVERGED"
            break
        if cancellation is not None and cancellation():
            status = "INTERRUPTED"
            break
        optimizer.zero_grad()
        loss, metadata = objective(step)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ModelContractError("NONFINITE: training objective")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            raise ModelContractError("NONFINITE: training gradient")
        gradient = torch.nn.utils.clip_grad_norm_(parameters, plan.gradient_norm_limit)
        optimizer.step()
        with torch.no_grad():
            value = monitor()
            if not math_isfinite(value):
                raise ModelContractError("NONFINITE: training monitor")
            if best is None or value < best - plan.tolerance:
                best, stale, best_checkpoint = value, 0, model.checkpoint()
            else:
                stale += 1
        row = {"step": step + 1, "objective": value, "gradient_norm": float(gradient), **metadata}
        history.append(row)
        if progress is not None:
            progress({**row, "elapsed_seconds": time.perf_counter() - started})
        if checkpoint_requested is not None and checkpoint_requested():
            state = {"step": step + 1, "stale": stale, "best": best, "history": history,
                "model": model.checkpoint(), "best_checkpoint": best_checkpoint,
                "optimizer": optimizer.state_dict(), "auxiliary": None if auxiliary is None else auxiliary.state_dict(),
                "rng": rng_state()}
            saved = {"schema_version": "pirc26-training-state-v1", "scope_hash": scope_hash,
                     "initial_model_hash": initial_model_hash, "state": encode_state(state)}
            last_state = {**saved, "sha256": digest(saved)}
            checkpoint_handler(last_state, {**row, "elapsed_seconds": time.perf_counter() - started})
            status = "CHECKPOINTED"
            break
        if stale >= plan.patience:
            status = "CONVERGED"
            break
    if best_checkpoint is not None:
        restore_model(model, best_checkpoint)
    return {"schema_version": "pirc26-fit-result-v1", "objective": scope["objective"],
            "gradient_route": "G0" if scope["objective"] == "O1" else "G1", "status": status,
            "steps": len(history), "best_train_objective": best, "history": history,
            "checkpoint": model.checkpoint(), "scope_hash": scope_hash, "code_hash": scope["code_hash"],
            "training_state": last_state, "resume_level": "exact",
            "wall_seconds": time.perf_counter() - started, "peak_memory_bytes": None}


def math_isfinite(value):
    import math
    return type(value) in (float, int) and math.isfinite(value)
