"""Allocation-free validation of the closed physical model checkpoint recipe.

No tensor engine, factory, file access or store is needed to reject an oversized
or malformed recipe. This validates transport/capacity, not data provenance or
scientific qualification. The numerical loader still checks actual weights.
"""

import hashlib
import json
import math
import re
import struct

from infrastructure.research_control import canonical, ControlError


class CheckpointContractError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(code + ": " + message)


def require(condition, message, code="MODEL_CONTRACT_ERROR"):
    if not condition:
        raise CheckpointContractError(code, message)


def identity(value):
    # Preserve the original model codec's ensure_ascii=True identity exactly.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def keys(value, expected):
    require(type(value) is dict and set(value) == set(expected), "closed checkpoint field set")


INTEGER_BOUNDS = {"int64": (-(1 << 63), (1 << 63)-1), "int32": (-(1 << 31), (1 << 31)-1),
                  "uint8": (0, 255), "uint32": (0, (1 << 32)-1)}


def array(value, shape, *, dtype="torch.float64", exact=False):
    """Validate exact JSON nesting and typed scalars without allocating arrays.

    Recipe floats may intentionally round to their frozen float32 buffers;
    serialized tensor data must already represent its declared dtype exactly.
    """
    kind = dtype.removeprefix("torch.") if type(dtype) is str else None
    require(kind in ("float32", "float64", "bool", *INTEGER_BOUNDS), "checkpoint array dtype")
    if shape:
        require(type(value) is list and len(value) == shape[0], "checkpoint tensor nesting/shape")
        for child in value:
            array(child, shape[1:], dtype=dtype, exact=exact)
    elif kind == "bool":
        require(type(value) is bool, "checkpoint boolean scalar")
    elif kind in INTEGER_BOUNDS:
        lo, hi = INTEGER_BOUNDS[kind]
        require(type(value) is int and lo <= value <= hi, "checkpoint integer scalar/range")
    else:
        require(type(value) in (int, float) and math.isfinite(value), "finite checkpoint scalar")
        if kind == "float32":
            require(abs(value) <= 3.4028234663852886e38, "checkpoint scalar exceeds dtype range")
        if exact:
            # Both numerical engines' tolist() emits Python floats for these
            # dtypes. Accepting an integer here changes the re-encoded identity
            # even when its numerical value happens to be representable.
            require(type(value) is float, "checkpoint floating scalar differs from dtype representation")
            represented = struct.unpack("f", struct.pack("f", value))[0] if kind == "float32" else float(value)
            require(represented == value, "checkpoint scalar differs from dtype representation")


def normalizer_buffers(means, scales):
    """The actual residual recipe freezes float32 buffers, even in double models."""
    require(type(means) in (list, tuple) and type(scales) in (list, tuple)
            and 4 <= len(means) == len(scales) <= 132, "bounded aligned normalizer buffers")
    for values in (means, scales):
        array(list(values), [len(values)], dtype="torch.float32")
    result = {name: [struct.unpack("f", struct.pack("f", v))[0] for v in values]
              for name, values in (("means", means), ("scales", scales))}
    require(all(v > 0 for v in result["scales"]), "positive normalization scales after float32 representation")
    return result


def inspect_checkpoint(checkpoint, *, component_limit=None):
    """Detach bounded canonical JSON and derive shapes before constructors.

    Shapes come from the validated architecture, not tensor metadata supplied
    by the caller. Empty [2,0] arrays retain their explicit empty row nesting.
    """
    try:
        content = canonical(checkpoint, 2 * 1024 * 1024)
        document = json.loads(content)
        keys(document, ("schema_version", "model_card", "configuration", "state", "sha256"))
        payload = {k: v for k, v in document.items() if k != "sha256"}
        require(document["sha256"] == identity(payload) and document["schema_version"] == "pirc26-dynamics-checkpoint-v1",
            "checkpoint identity/schema mismatch")
        card, config, state = document["model_card"], document["configuration"], document["state"]
        keys(card, ("schema_version", "state_names", "units", "time_unit", "noise_dim", "diffusion_support",
            "spec", "family", "dtype", "device", "qualification", "velocity_covariance"))
        require(card["schema_version"] == "pirc26-model-card-v1" and card["state_names"] == ["x", "y", "vx", "vy"]
            and card["units"] == ["m", "m", "m/s", "m/s"] and card["time_unit"] == "s"
            and type(card["noise_dim"]) is int and card["noise_dim"] == 2 and card["diffusion_support"] == ["vx", "vy"]
            and card["dtype"] in ("torch.float32", "torch.float64") and card["device"] == "cpu"
            and card["qualification"] == "unqualified", "physical model card contract")
        array(card["velocity_covariance"], [2, 2])
        spec = card["spec"]
        keys(spec, ("coordinate_frame", "train_binding_hash", "normalizer_hash", "context_hash", "context_dim", "means", "scales"))
        c = spec["context_dim"]
        require(type(c) is int and 0 <= c <= 128 and type(spec["coordinate_frame"]) is str and bool(spec["coordinate_frame"].strip()),
            "context/frame contract")
        for name in ("train_binding_hash", "normalizer_hash", "context_hash"):
            require(type(spec[name]) is str and re.fullmatch(r"[0-9a-f]{64}", spec[name]), "frozen provenance hash")
        for name in ("means", "scales"):
            array(spec[name], [4 + c], dtype="torch.float32")
        require(all(v > 0 for v in spec["scales"]), "positive normalization scales")
        require(type(config) is dict and config.get("family") == card["family"], "checkpoint model family")
        family = card["family"]
        require(family in ("M0", "M1-R", "M1-S", "M2"), "known checkpoint model family")
        residual = family != "M0"
        prefix = "acceleration_model." + ("affine." if residual else "")
        shapes = {"velocity_factor": [2, 2], prefix + "A": [2, 2], prefix + "B": [2, 2],
            prefix + "a": [2], prefix + "context_weight": [2, c]}
        widths, q = [], 0
        if not residual:
            keys(config, ("family",))
        else:
            common = {"family", "features", "residual_enabled"}
            extra = {"M1-R": {"centers", "bandwidth"}, "M1-S": {"knots", "degree", "extrapolation"},
                "M2": {"hidden", "seed", "amplitude", "spectral_bound", "alpha"}}[family]
            keys(config, common | extra)
            features = config["features"]
            require(type(features) is list and 1 <= len(features) <= 8
                and all(type(i) is int and 0 <= i < 4 + c for i in features) and len(set(features)) == len(features)
                and type(config["residual_enabled"]) is bool, "bounded unique feature selector/gate")
            d = len(features)
            shapes.update({"acceleration_model.means": [4+c], "acceleration_model.scales": [4+c]})
            if family == "M1-R":
                require(type(config["centers"]) is list and 1 <= len(config["centers"]) <= 128,
                    "registered RBF basis quota", "RESOURCE_PLAN_REJECTED")
                q = len(config["centers"])
                array(config["centers"], [q, d], dtype="torch.float32")
                array(config["bandwidth"], [q], dtype="torch.float32")
                require(all(v > 0 for v in config["bandwidth"]), "positive RBF widths")
                shapes.update({"acceleration_model.centers": [q, d], "acceleration_model.bandwidth": [q],
                    "acceleration_model.coefficients": [q, 2]})
            elif family == "M1-S":
                degree, knots = config["degree"], config["knots"]
                require(type(degree) is int and 1 <= degree <= 3 and config["extrapolation"] == "clamp", "spline degree/extrapolation")
                require(type(knots) is list and len(knots) == d and type(knots[0]) is list, "spline knot rows")
                k = len(knots[0])
                q = d * (k - degree - 1)
                require(k >= 2 * (degree+1) and q <= 128, "registered spline basis quota", "RESOURCE_PLAN_REJECTED")
                array(knots, [d, k], dtype="torch.float32")
                for row in knots:
                    require(all(a <= b for a, b in zip(row, row[1:])) and row[0] < row[-1]
                        and row[:degree+1] == [row[0]]*(degree+1) and row[-degree-1:] == [row[-1]]*(degree+1), "clamped ordered knots")
                shapes.update({"acceleration_model.knots": [d, k], "acceleration_model.coefficients": [q, 2]})
            else:
                widths = config["hidden"]
                require(type(widths) is list and 1 <= len(widths) <= 2 and all(type(w) is int and w in (16, 32) for w in widths),
                    "registered MLP capacity", "RESOURCE_PLAN_REJECTED")
                require(type(config["seed"]) is int and 0 <= config["seed"] < 2**63, "initializer seed")
                require(all(type(config[n]) in (int, float) and math.isfinite(config[n]) and config[n] > 0
                    for n in ("amplitude", "spectral_bound", "alpha")), "physical neural residual bounds")
                layers = [d, *widths, 2]
                for i, (a, b) in enumerate(zip(layers, layers[1:])):
                    shapes["acceleration_model.layers." + str(i) + ".weight"] = [b, a]
                    shapes["acceleration_model.layers." + str(i) + ".bias"] = [b]
        count = sum(math.prod(shape) for shape in shapes.values())
        require(count <= 100000, "model tensor element quota", "RESOURCE_PLAN_REJECTED")
        capacity = max(4, math.ceil(math.sqrt(count)), c, q, *widths)
        if component_limit is not None:
            require(type(component_limit) is int and 1 <= component_limit <= 128 and capacity <= component_limit,
                "actual model exceeds declared component workspace", "RESOURCE_PLAN_REJECTED")
        keys(state, shapes)
        for name, shape in shapes.items():
            item = state[name]
            keys(item, ("shape", "dtype", "data", "sha256"))
            require(type(item["shape"]) is list and all(type(i) is int for i in item["shape"])
                and item["shape"] == shape and item["dtype"] == card["dtype"], "architecture-derived tensor shape/dtype")
            array(item["data"], shape, dtype=card["dtype"], exact=True)
            require(item["sha256"] == identity({k: v for k, v in item.items() if k != "sha256"}), "checkpoint tensor identity")
        # These frozen buffers must describe the recipe before constructor work.
        if residual:
            buffers = normalizer_buffers(spec["means"], spec["scales"])
            for name in ("means", "scales"):
                require(state["acceleration_model."+name]["data"] == buffers[name], "normalizer buffer differs from recipe")
            for name in ("centers", "bandwidth") if family == "M1-R" else ("knots",) if family == "M1-S" else ():
                require(state["acceleration_model."+name]["data"] == config[name], "basis buffer differs from recipe")
        return {"document": document, "shapes": shapes, "tensor_elements": count, "component_capacity": capacity}
    except CheckpointContractError:
        raise
    except ControlError as exc:
        raise CheckpointContractError("RESOURCE_PLAN_REJECTED", "checkpoint JSON structure/byte/finite quota") from exc
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise CheckpointContractError("MODEL_CONTRACT_ERROR", "malformed bounded model checkpoint") from exc
