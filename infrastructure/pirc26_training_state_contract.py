"""Pure typed JSON inspection for owned CPU training/inference recovery.

No engines, files, stores, grants or ambient RNG changes. Array descriptors are
inspected, never materialized; the admitted numerical consumer still validates
the actual source/data/environment scope and restores its real state.
"""

from dataclasses import dataclass
import json
import math
import re

from infrastructure.pirc26_checkpoint_contract import array, require, CheckpointContractError
from infrastructure.research_control import canonical

LIMIT = 4 * 1024 * 1024


@dataclass(frozen=True)
class TypedArray:
    kind: str
    dtype: str
    shape: tuple
    data: object


def encoded_limits(encoded):
    try:
        canonical(encoded, LIMIT)
        stack, remaining = [(encoded, 0)], 45000
        while stack:
            value, depth = stack.pop()
            remaining -= 1
            require(remaining >= 0 and depth <= 28, "checkpoint structural quota", "RESOURCE_PLAN_REJECTED")
            if type(value) is dict:
                stack.extend((child, depth + 1) for child in value.values())
            elif type(value) is list:
                stack.extend((child, depth + 1) for child in value)
            elif type(value) is str:
                require(len(value) <= 65536, "checkpoint scalar quota", "RESOURCE_PLAN_REJECTED")
            elif type(value) is int:
                require(value.bit_length() <= 256, "checkpoint scalar quota", "RESOURCE_PLAN_REJECTED")
        require(len(json.dumps(encoded, allow_nan=False).encode()) <= LIMIT,
                "checkpoint byte quota", "RESOURCE_PLAN_REJECTED")
    except CheckpointContractError:
        raise
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise CheckpointContractError("RESOURCE_PLAN_REJECTED", "checkpoint structure/byte quota") from exc
    return encoded


def _inspect(value, depth=0):
    require(depth <= 32, "checkpoint nesting quota")
    if type(value) is dict:
        require(len(value) == 1, "checkpoint tag mismatch")
        kind, item = next(iter(value.items()))
        if kind in ("tensor", "array"):
            require(type(item) is dict and set(item) == {"dtype", "shape", "data"}
                    and type(item["shape"]) is list and len(item["shape"]) <= 8
                    and all(type(i) is int and 0 <= i <= 100000 for i in item["shape"]), "checkpoint array shape")
            require(math.prod(item["shape"]) <= 100000, "checkpoint array element quota")
            allowed = ("float32", "float64", "int64", "int32", "uint8", "bool") if kind == "tensor" else (
                "uint32", "int64", "float32", "float64")
            require(type(item["dtype"]) is str and item["dtype"] in allowed, "checkpoint array dtype")
            array(item["data"], item["shape"], dtype=item["dtype"], exact=True)
            return TypedArray(kind, item["dtype"], tuple(item["shape"]), item["data"])
        if kind == "tuple":
            require(type(item) is list, "checkpoint tuple content")
            return tuple(_inspect(child, depth+1) for child in item)
        if kind == "map":
            require(type(item) is list and all(type(pair) is list and len(pair) == 2 for pair in item), "checkpoint map content")
            result = {}
            for key, child in item:
                key = _inspect(key, depth+1)
                require(type(key) in (str,int) and key not in result, "checkpoint map key")
                result[key] = _inspect(child, depth+1)
            return result
        require(False, "unknown checkpoint tag")
    if type(value) is list:
        return [_inspect(child, depth+1) for child in value]
    require(value is None or type(value) in (str,int,float,bool), "checkpoint primitive type")
    return value


def inspect_encoded(value):
    """Same closed codec, with all late fields validated before any engine."""
    encoded_limits(value)
    return _inspect(json.loads(json.dumps(value, allow_nan=False)))


def _tensor(value, shape, dtype):
    require(type(value) is TypedArray and value.kind == "tensor" and value.shape == tuple(shape)
            and value.dtype == dtype, "architecture-derived optimizer/auxiliary shape/dtype")


def _scalars(value):
    if type(value) is list:
        for child in value:
            yield from _scalars(child)
    else:
        yield value


def inspect_rng(encoded):
    value = inspect_encoded(encoded)
    _rng(value)
    return value


def _rng(value):
    require(type(value) is dict and set(value) == {"python","numpy","torch_cpu","torch_cuda","cuda_reason"}, "closed CPU RNG state")
    py, npstate, cpu = value["python"], value["numpy"], value["torch_cpu"]
    require(type(py) is tuple and len(py) == 3 and type(py[0]) is int and py[0] == 3
            and type(py[1]) is tuple and len(py[1]) == 625
            and all(type(i) is int and 0 <= i < 2**32 for i in py[1][:-1])
            and type(py[1][-1]) is int and 0 <= py[1][-1] <= 624
            and (py[2] is None or type(py[2]) is float and math.isfinite(py[2])), "Python MT RNG state")
    require(type(npstate) is tuple and len(npstate) == 5 and npstate[0] == "MT19937"
            and type(npstate[1]) is TypedArray and npstate[1].kind == "array"
            and npstate[1].dtype == "uint32" and npstate[1].shape == (624,)
            and type(npstate[2]) is int and 0 <= npstate[2] <= 624
            and type(npstate[3]) is int and npstate[3] in (0,1)
            and type(npstate[4]) is float and math.isfinite(npstate[4]), "NumPy MT RNG state")
    # Engine version/build belongs to numerical scope, not a guessed ABI length.
    require(type(cpu) is TypedArray and cpu.kind == "tensor" and cpu.dtype == "uint8"
            and len(cpu.shape) == 1 and 0 < cpu.shape[0] <= 100000
            and value["torch_cuda"] is None and value["cuda_reason"] == "CPU-only registered training component", "registered CPU Torch RNG state")


def inspect_training_envelope(envelope, job, config, inputs, *, model_preflight):
    """Inspect full state, model, history, Adam and duplicated RNG pre-read."""
    from infrastructure.research_store import digest
    try:
        canonical(envelope, LIMIT)
    except ValueError as exc:
        raise CheckpointContractError("RESOURCE_PLAN_REJECTED","training recovery envelope quota") from exc
    method, step = envelope["method_state"], envelope["step"]
    require(type(method.get("scope_hash")) is str and re.fullmatch(r"[0-9a-f]{64}",method["scope_hash"]), "training scope identity")
    state = inspect_encoded(method["state"])
    require(type(state) is dict and set(state) == {"step","stale","best","history","model","best_checkpoint","optimizer","auxiliary","rng"}
            and type(state["step"]) is int and state["step"] == step
            and type(state["stale"]) is int and 0 <= state["stale"] <= step
            and type(state["best"]) is float and math.isfinite(state["best"]), "closed training counters/best state")
    _rng(state["rng"])
    require(state["rng"] == inspect_rng(envelope["rng_state"]), "training duplicated RNG differs")
    history = state["history"]
    require(type(history) is dict and set(history) == {"schema_version","objective","gradient_norm"}
            and history["schema_version"] == "pirc26-history-columns-v1"
            and all(type(history[k]) is list and len(history[k]) == step
                    and all(type(v) is float and math.isfinite(v) for v in history[k]) for k in ("objective","gradient_norm")), "full training history columns")
    require(all(v >= 0 for v in history["gradient_norm"]), "nonnegative gradient norms")
    checked = None
    for name in ("model","best_checkpoint"):
        cp = state[name]
        require(type(cp) is dict, "training model checkpoint")
        candidate = model_preflight(cp,{**config,"initial_model_hash":cp.get("sha256")},inputs)
        if name == "model":
            checked = candidate
    # Shapes follow actual immutable architecture/parameter registration order,
    # not shapes or parameter IDs invented by the checkpoint.
    family = config["family"]
    require(family in ("M0","M2"), "registered Adam family")
    prefix = "acceleration_model." + ("affine." if family == "M2" else "")
    names = [prefix+n for n in ("A","B","a","context_weight")]
    if family == "M2":
        for index in range(len(checked["document"]["configuration"]["hidden"])+1):
            names.extend("acceleration_model.layers."+str(index)+"."+n for n in ("weight","bias"))
    shapes = [checked["shapes"][name] for name in names]
    diffusion = config["objective"] == "O1" and config["plan"]["fit_diffusion"]
    if diffusion:
        shapes.extend(([2],[]))
    auxiliary = state["auxiliary"]
    if diffusion:
        require(type(auxiliary) is dict and set(auxiliary) == {"raw_diagonal","off_diagonal"}, "diffusion auxiliary state")
        _tensor(auxiliary["raw_diagonal"],[2],inputs["dtype"])
        _tensor(auxiliary["off_diagonal"],[],inputs["dtype"])
    else:
        require(auxiliary is None, "frozen diffusion auxiliary state")
        require(state["model"]["state"]["velocity_factor"] == job["initial_checkpoint"]["state"]["velocity_factor"]
                and state["best_checkpoint"]["state"]["velocity_factor"] == job["initial_checkpoint"]["state"]["velocity_factor"], "frozen diffusion checkpoint differs")
    optimizer = state["optimizer"]
    require(type(optimizer) is dict and set(optimizer) == {"state","param_groups"}
            and type(optimizer["state"]) is dict and type(optimizer["param_groups"]) is list
            and len(optimizer["param_groups"]) == 1, "registered Adam state/group")
    group = optimizer["param_groups"][0]
    expected = {"lr":config["plan"]["learning_rate"],"betas":(.9,.999),"eps":1e-8,"weight_decay":0,
        "amsgrad":False,"maximize":False,"foreach":None,"capturable":False,"differentiable":False,"fused":None,
        "params":list(range(len(shapes)))}
    # Newer Torch exposes this default; numerical scope binds its actual build.
    if type(group) is dict and "decoupled_weight_decay" in group:
        expected["decoupled_weight_decay"] = False
    require(type(group) is dict and group == expected and digest(group) == digest(expected), "frozen Adam parameter recipe")
    for identity, values in optimizer["state"].items():
        require(type(identity) is int and 0 <= identity < len(shapes) and type(values) is dict
                and set(values) == {"step","exp_avg","exp_avg_sq"}, "Adam parameter identity/state")
        counter = values["step"]
        require(type(counter) is TypedArray and counter.kind == "tensor" and counter.shape == ()
                and counter.dtype in ("float32","float64") and 0 < counter.data <= step
                and counter.data.is_integer(), "Adam completed update counter")
        for name in ("exp_avg","exp_avg_sq"):
            _tensor(values[name],shapes[identity],inputs["dtype"])
        require(all(v >= 0 for v in _scalars(values["exp_avg_sq"].data)), "nonnegative Adam second moment")
    return state
