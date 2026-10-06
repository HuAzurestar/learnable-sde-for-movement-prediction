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
from infrastructure.research_control import canonical, ControlError, SCHEMA as CONTROL_SCHEMA
from infrastructure.pirc26_checkpoint_contract import array as validate_array, CheckpointContractError
from infrastructure.pirc26_process_resources import process_resources
from infrastructure.research_store import digest
from models.phase_space import ModelContractError, PhaseSpaceSDE


LIMIT = 4 * 1024 * 1024
TRAINING_SCHEMA = "pirc26-training-state-v2"
HISTORY_SCHEMA = "pirc26-history-columns-v1"
SOURCE_FILES = ("estimation/phase_space_checkpoint.py", "estimation/phase_space.py",
                "infrastructure/pirc26_checkpoint_contract.py",
                "infrastructure/research_control.py",
                "infrastructure/pirc26_process_resources.py",
                "estimation/phase_space_basis.py",
                "estimation/phase_space_o2.py", "models/phase_space.py",
                "inference/phase_space.py", "inference/phase_space_resume.py",
                "infrastructure/pirc26_forecast_codec.py", "evaluation/phase_space.py")


def _encode(value, depth=0):
    if depth > 32:
        raise ModelContractError("MODEL_CONTRACT_ERROR: checkpoint nesting quota")
    if isinstance(value, torch.Tensor):
        if (value.device.type != "cpu" or value.dtype not in (torch.float32, torch.float64, torch.int64,
                torch.int32, torch.uint8, torch.bool) or value.numel() > 100000
                or len(value.shape) > 8 or any(i > 100000 for i in value.shape)
                or not torch.isfinite(value).all()):
            raise ModelContractError("MODEL_CONTRACT_ERROR: bounded finite CPU checkpoint tensor required")
        return {"tensor": {"dtype": str(value.dtype).split(".")[-1], "shape": list(value.shape),
                           "data": value.detach().tolist()}}
    if isinstance(value, np.ndarray):
        if (value.dtype.name not in ("uint32", "int64", "float32", "float64") or value.size > 100000
                or len(value.shape) > 8 or any(i > 100000 for i in value.shape) or not np.isfinite(value).all()):
            raise ModelContractError("MODEL_CONTRACT_ERROR: bounded checkpoint RNG array required")
        return {"array": {"dtype": value.dtype.name, "shape": list(value.shape), "data": value.tolist()}}
    if isinstance(value, dict):
        if any(type(key) not in (str, int) for key in value):
            raise ModelContractError("MODEL_CONTRACT_ERROR: checkpoint map key")
        return {"map": [[_encode(key, depth + 1), _encode(item, depth + 1)] for key, item in value.items()]}
    if isinstance(value, tuple):
        return {"tuple": [_encode(item, depth + 1) for item in value]}
    if isinstance(value, list):
        return [_encode(item, depth + 1) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ModelContractError("MODEL_CONTRACT_ERROR: unsupported checkpoint object")


def _decode(value, depth=0, *, validate_only=False):
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
            allowed = ("float32", "float64", "int64", "int32", "uint8", "bool") if kind == "tensor" else (
                "uint32", "int64", "float32", "float64")
            if type(item["dtype"]) is not str or item["dtype"] not in allowed:
                raise ModelContractError("MODEL_CONTRACT_ERROR: checkpoint array dtype")
            try:
                validate_array(item["data"], item["shape"], dtype=item["dtype"], exact=True)
            except CheckpointContractError as exc:
                raise ModelContractError(str(exc)) from exc
            if validate_only:
                return None  # No array engine until the entire state has passed.
            if kind == "tensor":
                array = torch.tensor(item["data"], dtype=getattr(torch, item["dtype"]))
                if array.numel() != count or not torch.isfinite(array).all():
                    raise ModelContractError("MODEL_CONTRACT_ERROR: checkpoint tensor content")
            else:
                array = np.asarray(item["data"], dtype=item["dtype"])
                if array.size != count or not np.isfinite(array).all():
                    raise ModelContractError("MODEL_CONTRACT_ERROR: checkpoint RNG content")
            return array.reshape(item["shape"])
        if kind == "tuple":
            if type(item) is not list:
                raise ModelContractError("MODEL_CONTRACT_ERROR: checkpoint tuple content")
            return tuple(_decode(child, depth + 1, validate_only=validate_only) for child in item)
        if kind == "map":
            if type(item) is not list or any(type(pair) is not list or len(pair) != 2 for pair in item):
                raise ModelContractError("MODEL_CONTRACT_ERROR: checkpoint map content")
            decoded = {}
            for key, child in item:
                key = _decode(key, depth + 1, validate_only=validate_only)
                if type(key) not in (str, int) or key in decoded:
                    raise ModelContractError("MODEL_CONTRACT_ERROR: checkpoint map key")
                decoded[key] = _decode(child, depth + 1, validate_only=validate_only)
            return decoded
        raise ModelContractError("MODEL_CONTRACT_ERROR: unknown checkpoint tag")
    if isinstance(value, list):
        return [_decode(item, depth + 1, validate_only=validate_only) for item in value]
    return value


def _encoded_limits(encoded):
    try:
        # Shared preflight bounds key bytes/container lengths before either
        # extending a traversal stack or materializing the detached JSON.
        canonical(encoded, LIMIT)
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
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise ModelContractError("RESOURCE_PLAN_REJECTED: checkpoint structure/byte quota") from exc
    return encoded


def encode_state(value):
    return _encoded_limits(_encode(value))


def decode_state(value):
    try:
        _encoded_limits(value)
        raw = json.dumps(value, allow_nan=False)
        if len(raw.encode()) > LIMIT:
            raise ModelContractError("RESOURCE_PLAN_REJECTED: checkpoint byte quota")
        detached = json.loads(raw)
        _decode(detached, validate_only=True)
        return _decode(detached)
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


def history_metadata(scope, step):
    """Only immutable, source-bound counters; no loss/gradient recomputation."""
    if scope["objective"] == "O1":
        return {"batch_index": step % len(scope["data_identity"])}
    if scope["objective"] != "O2":
        raise ModelContractError("OBJECTIVE_INCOMPATIBLE: history codec requires O1/O2")
    plan, examples = scope["plan"], scope["data_identity"]["examples"]
    index = step % len(examples)
    horizons = plan["horizon_indices"][:min(len(plan["horizon_indices"]), 1 + step // plan["curriculum_steps"])]
    return {"batch_index": index, "horizon_indices": list(horizons),
            "brownian_root_id": digest({"root": examples[index]["request"]["brownian_root_id"],
                                        "phase": "o2-train", "step": step}),
            "estimator_id": "energy-u-exact-v1"}


def pack_history(history, scope):
    """Lossless JSON columns, preserving every actual monitor/gradient value."""
    if type(history) is not list or len(history) > scope["plan"]["max_steps"]:
        raise ModelContractError("CHECKPOINT_INCOMPATIBLE: bounded history required")
    objectives, gradients = [], []
    for index, row in enumerate(history):
        expected = history_metadata(scope, index)
        if (type(row) is not dict or set(row) != {"step", "objective", "gradient_norm", *expected}
                or type(row["step"]) is not int or row["step"] != index + 1
                or any(type(row[k]) is not float or not math_isfinite(row[k]) for k in ("objective", "gradient_norm"))
                or digest({key: row[key] for key in expected}) != digest(expected)):
            raise ModelContractError("CHECKPOINT_INCOMPATIBLE: history differs from immutable scope")
        objectives.append(row["objective"])
        gradients.append(row["gradient_norm"])
    return {"schema_version": HISTORY_SCHEMA, "objective": objectives, "gradient_norm": gradients}


def unpack_history(columns, scope, step):
    if (type(step) is not int or not 0 <= step <= scope["plan"]["max_steps"]
            or type(columns) is not dict or set(columns) != {"schema_version", "objective", "gradient_norm"}
            or columns["schema_version"] != HISTORY_SCHEMA
            or any(type(columns[k]) is not list or len(columns[k]) != step
                   or any(type(v) is not float or not math_isfinite(v) for v in columns[k])
                   for k in ("objective", "gradient_norm"))):
        raise ModelContractError("CHECKPOINT_INCOMPATIBLE: training position/history columns differ")
    return [{"step": i + 1, "objective": columns["objective"][i], "gradient_norm": columns["gradient_norm"][i],
             **history_metadata(scope, i)} for i in range(step)]


def managed_envelope(method_state, rng, step):
    return {"step": step, "data_position": step, "method_state": method_state, "rng_state": encode_state(rng)}


def _checkpoint_capacity(encoded, encoded_rng, step, byte_limit=LIMIT):
    """Exercise real owner codecs with conservative identifier/progress bounds.

    This is an unpublished capacity probe, not a model/checkpoint qualification,
    grant, reservation, budget restore or simulated owner checkpoint ACK.
    """
    if type(byte_limit) is not int or not 0 < byte_limit <= LIMIT:
        raise ModelContractError("RESOURCE_PLAN_REJECTED: managed checkpoint byte quota")
    _encoded_limits(encoded)
    method = {"schema_version": TRAINING_SCHEMA, "scope_hash": "f"*64,
              "initial_model_hash": "f"*64, "state": encoded, "sha256": "f"*64}
    state = {"step": step, "data_position": step, "method_state": method, "rng_state": encoded_rng}
    progress = {"completed_steps": step, "total_steps": step,
                "throughput_per_second": -sys.float_info.max, "eta_seconds": sys.float_info.max}
    frame = {"schema_version": CONTROL_SCHEMA, "attempt_id": "a"*128, "token": "f"*64,
             "request_id": "f"*32, "state": state, "progress": progress}
    publication = {"schema_version": "pirc25-checkpoint-v1", "parent_attempt_id": "a"*128,
        "run_id": "a"*128, "plugin_id": "a"*128, "plugin_version": "a"*128,
        "recovery_command_hash": "f"*64, "resume_level": "exact",
        "bindings": {**{k: "f"*64 for k in ("code_hash", "data_hash", "protocol_hash", "feature_hash",
                    "selection_hash", "model_hash", "objective_hash", "cell_hash", "execution_binding_hash")},
                     "schema_version": "a"*128},
        "payload_hash": "f"*64, "state": state, "admission_hash": "f"*64, "progress": progress}
    try:
        frame_bytes = len(canonical(frame, byte_limit))
        publication_bytes = len(canonical(publication, LIMIT))
        # The owner's _bounded_json additionally charges eight bytes per node.
        # Checking this stricter allowance avoids importing the application layer.
        stack, nodes = [publication], 0
        while stack:
            item = stack.pop()
            nodes += 1
            if type(item) is dict:
                stack.extend(item.values())
            elif type(item) is list:
                stack.extend(item)
        if publication_bytes + 8 * nodes > LIMIT:
            raise ControlError("owner conservative JSON byte allowance")
    except ControlError as exc:
        raise ModelContractError("RESOURCE_PLAN_REJECTED: complete managed checkpoint capacity") from exc
    return {"max_steps": step, "response_bytes": frame_bytes, "publication_bytes": publication_bytes,
            "publication_nodes": nodes, "byte_limit": byte_limit, "node_limit": 65536}


def preflight_training_checkpoint(model, optimizer, plan, scope, auxiliary=None, byte_limit=LIMIT):
    """Full-plan history + fully populated default CPU Adam, before any update.

    Shapes/groups come from the actual model/optimizer; Adam's source-bound
    default recipe has step/exp_avg/exp_avg_sq. No optimizer step, RNG draw,
    tensor mutation, state publication or data read is needed for this probe.
    """
    future = optimizer.state_dict()
    future["state"] = {}
    for group, actual in zip(future["param_groups"], optimizer.param_groups):
        if any(actual[k] for k in ("amsgrad", "capturable", "differentiable", "fused")):
            raise ModelContractError("RESOURCE_PLAN_REJECTED: unregistered Adam checkpoint recipe")
        for identity, parameter in zip(group["params"], actual["params"]):
            future["state"][identity] = {"step": torch.tensor(float(plan.max_steps)),
                                       "exp_avg": parameter.detach(), "exp_avg_sq": parameter.detach()}
    cp, rng = model.checkpoint(), rng_state()
    # Python's optional cached Gaussian may become a float without changing shape.
    rng["python"] = (*rng["python"][:2], -sys.float_info.max)
    columns = {"schema_version": HISTORY_SCHEMA, "objective": [-sys.float_info.max]*plan.max_steps,
               "gradient_norm": [-sys.float_info.max]*plan.max_steps}
    probe = _encode({"step": plan.max_steps, "stale": plan.max_steps, "best": -sys.float_info.max,
        "history": columns, "model": cp, "best_checkpoint": cp, "optimizer": future,
        "auxiliary": None if auxiliary is None else auxiliary.state_dict(), "rng": rng})
    def widest(item):
        if type(item) is dict:
            return {k: widest(v) for k, v in item.items()}
        if type(item) is list:
            return [widest(v) for v in item]
        if type(item) is float:
            return -sys.float_info.max
        if type(item) is int:
            return (1 << 255) - 1
        return item
    return _checkpoint_capacity(widest(probe), widest(_encode(rng)), plan.max_steps, byte_limit)


def train_loop(model, parameters, plan, scope, objective, monitor, *, auxiliary=None,
               resume_state=None, cancellation=None, progress=None,
               checkpoint_requested=None, checkpoint_handler=None, checkpoint_byte_limit=LIMIT):
    """Continue the actual Adam/batch-counter state, with immutable input scope."""
    import time
    if torch.get_num_threads() != 1:
        raise ModelContractError("OBJECTIVE_INCOMPATIBLE: exact CPU training requires one declared thread")
    if (checkpoint_requested is None) != (checkpoint_handler is None):
        raise ModelContractError("MODEL_CONTRACT_ERROR: checkpoint request and handler must be paired")
    scope_hash = digest(scope)
    optimizer = torch.optim.Adam(parameters, lr=plan.learning_rate)
    capacity = preflight_training_checkpoint(model, optimizer, plan, scope, auxiliary, checkpoint_byte_limit)
    initial_model_hash = model.checkpoint()["sha256"]
    first_step, stale, best, history = 0, 0, None, []
    best_checkpoint = None
    if resume_state is not None:
        if (not isinstance(resume_state, dict) or resume_state.get("schema_version") != TRAINING_SCHEMA
                or resume_state.get("scope_hash") != scope_hash
                or resume_state.get("initial_model_hash") != initial_model_hash):
            raise ModelContractError("CHECKPOINT_INCOMPATIBLE: code/data/model/objective/environment differs")
        detached = dict(resume_state)
        identity = detached.pop("sha256", None)
        if identity != digest(detached):
            raise ModelContractError("CHECKPOINT_INCOMPATIBLE: training state identity differs")
        state = decode_state(resume_state["state"])
        first_step, stale, best = state["step"], state["stale"], state["best"]
        history = unpack_history(state["history"], scope, first_step)
        if (type(stale) is not int or not 0 <= stale <= first_step
                or best is not None and not math_isfinite(best)):
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
        if type(metadata) is not dict or digest(metadata) != digest(history_metadata(scope, step)):
            raise ModelContractError("MODEL_CONTRACT_ERROR: objective history metadata differs from scope")
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
            state = {"step": step + 1, "stale": stale, "best": best, "history": pack_history(history, scope),
                "model": model.checkpoint(), "best_checkpoint": best_checkpoint,
                "optimizer": optimizer.state_dict(), "auxiliary": None if auxiliary is None else auxiliary.state_dict(),
                "rng": rng_state()}
            encoded = encode_state(state)
            _checkpoint_capacity(encoded, encode_state(state["rng"]), step + 1, checkpoint_byte_limit)
            saved = {"schema_version": TRAINING_SCHEMA, "scope_hash": scope_hash,
                     "initial_model_hash": initial_model_hash, "state": encoded}
            last_state = {**saved, "sha256": digest(saved)}
            checkpoint_handler(last_state, {**row, "elapsed_seconds": time.perf_counter() - started})
            status = "CHECKPOINTED"
            break
        if stale >= plan.patience:
            status = "CONVERGED"
            break
    if best_checkpoint is not None:
        restore_model(model, best_checkpoint)
    result = {"schema_version": "pirc26-fit-result-v1", "objective": scope["objective"],
            "gradient_route": "G0" if scope["objective"] == "O1" else "G1", "status": status,
            "steps": len(history), "best_train_objective": best, "history": history,
            "checkpoint": model.checkpoint(), "scope_hash": scope_hash, "code_hash": scope["code_hash"],
            "training_state": last_state, "resume_level": "exact",
            "checkpoint_capacity": capacity,
            "wall_seconds": time.perf_counter() - started}
    observation = process_resources("fit-return-before-encoding")
    return {**result, "peak_memory_bytes": observation["peak_resident_bytes"], "resource_observation": observation}


def math_isfinite(value):
    import math
    return type(value) in (float, int) and math.isfinite(value)
