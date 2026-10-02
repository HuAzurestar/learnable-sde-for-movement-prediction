"""Versioned shared component declarations and allocation-free resource plans.

The typed JSON vocabulary below is deliberately bounded, not full JSON Schema:
no remote references, regexes, callbacks or unbounded schema evaluation. Code
identity covers the inspectable callable and its defining module; study and
admission identities must additionally bind dependencies/inputs/environment.
Declarations are not a sandbox or proof of scientific qualification. Production
gates must actually bind/revalidate these plans before invoking a worker.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
from types import CodeType, MappingProxyType, ModuleType

from infrastructure.research_store import ResearchError, digest, encode, identifier

GLOBAL_LIMITS = MappingProxyType({"matrix_cells": 100_000, "paths": 1_000_000,
    "steps": 1_000_000, "mixtures": 1024, "components": 4096, "observations": 10_000_000,
    "tensor_elements": 20_000_000, "tensor_bytes": 512 * 1024 * 1024,
    "result_bytes": 64 * 1024 * 1024})
COUNT_NAMES = frozenset({"paths", "steps", "mixtures", "components", "observations", "state_dim"})
JSON_TYPES = frozenset({"object", "array", "string", "integer", "number", "boolean", "null"})
RESUME_LEVELS = frozenset({"exact", "numerical-tolerance", "chunk", "restart-only"})


def reject(detail, code="CONTRACT_MISMATCH"):
    raise ResearchError(code, detail)


def _bounded_json(value, *, nodes=4096, depth=16, string_length=65536, byte_limit=4 * 1024 * 1024):
    remaining = nodes
    remaining_bytes = byte_limit
    def charge_string(value):
        nonlocal remaining_bytes
        try:
            remaining_bytes -= len(encode(value))
        except UnicodeError as exc:
            raise ResearchError("CONTRACT_MISMATCH", "JSON strings must be valid UTF-8") from exc
        if remaining_bytes < 0:
            reject("JSON value exceeds total byte quota")
    def visit(item, level):
        nonlocal remaining, remaining_bytes
        remaining -= 1
        # Conservative separator/container allowance, independent of nesting.
        remaining_bytes -= 8
        if remaining < 0 or level > depth or remaining_bytes < 0:
            reject("JSON declaration or value exceeds structural quota")
        if type(item) is dict:
            if len(item) > remaining:
                reject("JSON object exceeds structural quota")
            for key, child in item.items():
                if type(key) is not str or len(key) > 128:
                    reject("JSON keys must be bounded strings")
                charge_string(key)
                visit(child, level + 1)
        elif type(item) is list:
            if len(item) > remaining:
                reject("JSON array exceeds structural quota")
            for child in item:
                visit(child, level + 1)
        elif type(item) is str:
            if len(item) > string_length:
                reject("JSON string exceeds structural quota")
            charge_string(item)
        elif type(item) is int:
            # PCG RNG checkpoint state contains genuine unsigned 128-bit values.
            # This does not relax independently checked count/allocation quotas.
            if item.bit_length() > 256:
                reject("JSON integer exceeds bounded checkpoint range")
            remaining_bytes -= len(str(item))
        elif type(item) is float:
            if not math.isfinite(item):
                reject("JSON number must be finite")
            remaining_bytes -= len(encode(item))
        elif item is not None and type(item) is not bool:
            reject("declarations and values must contain only finite JSON types")
    visit(value, 0)
    if remaining_bytes < 0:
        reject("JSON value exceeds total byte quota")


def validate_schema(schema):
    """Validate this module's explicit typed-JSON-v1 vocabulary, fail closed."""
    _bounded_json(schema)
    remaining = 1024
    def visit(node, depth):
        nonlocal remaining
        remaining -= 1
        if (remaining < 0 or depth > 8 or type(node) is not dict or
                type(node.get("type")) is not str or node["type"] not in JSON_TYPES):
            reject("schema type or structural quota is invalid")
        kind = node["type"]
        allowed = {"type", "enum"}
        allowed.update({"object": {"properties", "required", "additionalProperties"},
            "array": {"items", "minItems", "maxItems"}, "string": {"minLength", "maxLength"},
            "integer": {"minimum", "maximum"}, "number": {"minimum", "maximum"}}.get(kind, set()))
        if set(node) - allowed:
            reject("schema keyword is unsupported; references and callbacks are forbidden")
        if "enum" in node and (type(node["enum"]) is not list or not 0 < len(node["enum"]) <= 256):
            reject("schema enum must be a bounded nonempty JSON list")
        if kind == "object":
            properties = node.get("properties", {})
            required = node.get("required", [])
            additional = node.get("additionalProperties", False)
            if (type(properties) is not dict or type(required) is not list or
                    any(type(key) is not str for key in required) or
                    len(set(required)) != len(required) or not set(required) <= set(properties)):
                reject("object schema required fields are invalid")
            for child in properties.values():
                visit(child, depth + 1)
            if type(additional) is dict:
                visit(additional, depth + 1)
            elif type(additional) is not bool:
                reject("additionalProperties must be boolean or a typed schema")
        if kind == "array":
            if "items" not in node:
                reject("array schema needs an explicit item schema")
            visit(node["items"], depth + 1)
        for lower, upper in (("minimum", "maximum"), ("minItems", "maxItems"), ("minLength", "maxLength")):
            for key in (lower, upper):
                if key in node:
                    bound = node[key]
                    if type(bound) not in {int, float} or not math.isfinite(bound):
                        reject("schema bounds must be finite numbers, never booleans")
                    if key not in {"minimum", "maximum"} and (type(bound) is not int or not 0 <= bound <= 2_000_000):
                        reject("schema length bounds must be bounded nonnegative integers")
            if lower in node and upper in node and node[lower] > node[upper]:
                reject("schema lower bound exceeds upper bound")
    visit(schema, 0)


def validate_value(schema, value):
    validate_schema(schema)
    _bounded_json(value, nodes=2_000_000, depth=32, string_length=1024 * 1024, byte_limit=64 * 1024 * 1024)
    def visit(node, item):
        kind = node["type"]
        valid = {"object": type(item) is dict, "array": type(item) is list,
            "string": type(item) is str, "integer": type(item) is int,
            "number": type(item) in {int, float}, "boolean": type(item) is bool, "null": item is None}[kind]
        if not valid:
            reject("value type differs from registered schema; coercion is forbidden")
        if "enum" in node and encode(item) not in {encode(option) for option in node["enum"]}:
            reject("value is not in the registered enum")
        if kind == "object":
            properties = node.get("properties", {})
            if not set(node.get("required", [])) <= set(item):
                reject("required configuration/input/output field is absent")
            for key, child in item.items():
                selected = properties.get(key, node.get("additionalProperties", False))
                if selected is False:
                    reject("undeclared configuration/input/output field")
                if selected is not True:
                    visit(selected, child)
        elif kind == "array":
            for child in item:
                visit(node["items"], child)
        for lower, upper in (("minimum", "maximum"), ("minItems", "maxItems"), ("minLength", "maxLength")):
            measured = item if lower == "minimum" else len(item) if kind in {"array", "string"} else None
            if measured is not None and ((lower in node and measured < node[lower]) or (upper in node and measured > node[upper])):
                reject("value exceeds registered schema bounds")
    visit(schema, value)


def _code_identity(code):
    remaining = 4096
    def constant(value, level):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or level > 16:
            reject("implementation constants exceed structural quota")
        if isinstance(value, CodeType):
            return {"code": visit(value, level + 1)}
        if type(value) in {tuple, frozenset}:
            if len(value) > remaining:
                reject("implementation constants exceed structural quota")
            result = [constant(item, level + 1) for item in value]
            return {"tuple" if type(value) is tuple else "frozenset": result if type(value) is tuple else sorted(result, key=encode)}
        if type(value) is bytes:
            if len(value) > 65536:
                reject("implementation bytes exceed structural quota")
            return {"bytes": value.hex()}
        if value is Ellipsis:
            return {"ellipsis": True}
        if value is None or type(value) in {str, int, float, bool}:
            _bounded_json(value)
            return value
        reject("implementation contains an unsupported code constant")
    def visit(value, level):
        if level > 16 or len(value.co_code) > 65536 or len(value.co_consts) > remaining:
            reject("implementation code exceeds structural quota")
        result = {"bytecode": value.co_code.hex(), "constants": [constant(item, level + 1) for item in value.co_consts],
            "names": list(value.co_names), "variables": list(value.co_varnames), "freevars": list(value.co_freevars),
            "cellvars": list(value.co_cellvars), "argcount": value.co_argcount, "posonly": value.co_posonlyargcount,
            "kwonly": value.co_kwonlyargcount, "flags": value.co_flags}
        _bounded_json(result, nodes=16384, depth=48, string_length=131072)
        return result
    return visit(code, 0)


def _function_identity(function, owner=None):
    # Captured configuration is implementation content, not merely source text.
    # Runtime stores/generators must instead be explicit execution context;
    # opaque closure objects cannot silently escape an immutable registration.
    code = _code_identity(function.__code__)
    remaining = 4096
    def capture(value, level=0):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or level > 16:
            reject("implementation bindings exceed structural quota")
        if owner is not None and value is owner:
            return {"declaring_class": owner.__module__ + "." + owner.__qualname__}
        if type(value) in {tuple, list}:
            if len(value) > remaining:
                reject("implementation bindings exceed structural quota")
            return {"tuple" if type(value) is tuple else "list": [capture(item, level + 1) for item in value]}
        if type(value) is dict:
            if len(value) > remaining or any(type(key) is not str or len(key) > 128 for key in value):
                reject("implementation bindings need bounded string keys")
            return {"dict": {key: capture(item, level + 1) for key, item in value.items()}}
        _bounded_json(value)
        return value
    bindings = {"defaults": capture(function.__defaults__), "kwdefaults": capture(function.__kwdefaults__),
        "closure": [capture(cell.cell_contents) for cell in (function.__closure__ or ())]}
    global_configuration = {}
    for name in function.__code__.co_names:
        if name not in function.__globals__:
            continue
        value = function.__globals__[name]
        if isinstance(value, ModuleType):
            # Imported modules and dependencies are additionally bound by the
            # StudySpec/environment; this is not a Python sandbox.
            global_configuration[name] = {"module": value.__name__}
        elif inspect.isfunction(value):
            global_configuration[name] = {"function": value.__module__ + "." + value.__qualname__,
                "code": _code_identity(value.__code__), "defaults": capture(value.__defaults__),
                "kwdefaults": capture(value.__kwdefaults__)}
        elif inspect.isclass(value):
            global_configuration[name] = {"class": value.__module__ + "." + value.__qualname__}
        else:
            global_configuration[name] = capture(value)
    _bounded_json(bindings)
    _bounded_json(global_configuration)
    result = {"code": code, "bindings": bindings, "globals": global_configuration}
    _bounded_json(result, nodes=32768, depth=48, string_length=131072)
    return result


def _class_identity(builder):
    bases = builder.__mro__[:-1]  # object has native, rather than declared code.
    if len(bases) > 16:
        reject("factory inheritance exceeds structural quota")
    result = []
    remaining_bytes = 16 * 1024 * 1024
    for base in bases:
        attributes = vars(base)
        if len(attributes) > 256:
            reject("factory attributes exceed structural quota")
        methods, constants = {}, {}
        for key, value in attributes.items():
            if key in {"__dict__", "__weakref__", "__module__", "__qualname__", "__doc__"}:
                continue
            value = value.__func__ if isinstance(value, (staticmethod, classmethod)) else value
            if inspect.isfunction(value):
                methods[key] = _function_identity(value, base)
            elif isinstance(value, property):
                methods[key] = {name: _function_identity(function, base) if function else None
                    for name, function in (("get", value.fget), ("set", value.fset), ("delete", value.fdel))}
            else:
                normalized = list(value) if type(value) is tuple else value
                _bounded_json(normalized)
                constants[key] = normalized
            remaining_bytes -= len(encode(methods.get(key, constants.get(key)))) + len(encode(key))
            if remaining_bytes < 0:
                reject("factory identity exceeds aggregate byte quota")
        result.append({"class": base.__module__ + "." + base.__qualname__, "methods": methods, "constants": constants})
    return result


def implementation_hash(builder):
    """Never invokes a factory; excludes local absolute filenames from identity."""
    try:
        path = Path(inspect.getsourcefile(builder))
        if path.stat().st_size > 4 * 1024 * 1024:
            reject("inspectable implementation exceeds source quota")
        with path.open("rb") as stream:
            defining_module = stream.read(4 * 1024 * 1024 + 1)
        if len(defining_module) > 4 * 1024 * 1024:
            reject("inspectable implementation exceeds source quota")
        source = inspect.getsource(builder)
        if len(source.encode("utf-8")) > 65536:
            reject("inspectable implementation exceeds source quota")
        if inspect.isclass(builder):
            codes = {"classes": _class_identity(builder)}
        else:
            if not inspect.isfunction(builder):
                reject("registered factories must be inspectable functions or classes, not opaque bound instances")
            codes = {"callable": _function_identity(builder)}
        return digest({"source": source, "module": builder.__module__, "qualname": builder.__qualname__,
            "defining_module_hash": hashlib.sha256(defining_module.decode("utf-8").replace("\r\n", "\n").encode()).hexdigest(), "codes": codes})
    except (OSError, TypeError, AttributeError, ValueError) as exc:
        if isinstance(exc, ResearchError):
            raise
        raise ResearchError("CONTRACT_MISMATCH", "implementation needs bounded inspectable source") from exc


@dataclass(frozen=True)
class RegistryEntry:
    component_id: str
    component_kind: str
    version: str
    code_hash: str
    config_schema: dict
    input_schema: dict
    output_schema: dict
    state_order: tuple[str, ...]
    units: tuple[str, ...]
    capabilities: frozenset[str]
    resource_class: str
    resume_level: str
    resource_contract: dict

    def manifest(self):
        return {"schema_version": "pirc25-registry-entry-v1", "component_id": self.component_id,
            "component_kind": self.component_kind, "version": self.version, "code_hash": self.code_hash,
            "config_schema": self.config_schema, "input_schema": self.input_schema, "output_schema": self.output_schema,
            "state_order": list(self.state_order), "units": list(self.units), "capabilities": sorted(self.capabilities),
            "resource_class": self.resource_class, "resume_level": self.resume_level, "resource_contract": self.resource_contract}


def validate_entry(entry):
    if not isinstance(entry, RegistryEntry):
        reject("versioned RegistryEntry is required")
    for value in (entry.component_id, entry.version):
        identifier(value)
    if (type(entry.component_kind) is not str or entry.component_kind not in {"model", "trainer", "predictor", "execution-adapter"} or
            type(entry.code_hash) is not str or len(entry.code_hash) != 64 or
            any(character not in "0123456789abcdef" for character in entry.code_hash) or
            type(entry.state_order) is not tuple or not 0 < len(entry.state_order) <= 32 or
            any(type(name) is not str or not name or len(name) > 64 for name in entry.state_order) or
            len(set(entry.state_order)) != len(entry.state_order) or type(entry.units) is not tuple or
            len(entry.units) != len(entry.state_order) or any(type(unit) is not str or not unit or len(unit) > 32 for unit in entry.units) or
            type(entry.capabilities) is not frozenset or not 0 < len(entry.capabilities) <= 64 or
            type(entry.resource_class) is not str or entry.resource_class not in {"cpu", "gpu"} or
            type(entry.resume_level) is not str or entry.resume_level not in RESUME_LEVELS):
        reject("component identity, state, units, capability or resource/recovery declaration is invalid")
    for capability in entry.capabilities:
        identifier(capability)
    for schema in (entry.config_schema, entry.input_schema, entry.output_schema):
        validate_schema(schema)
    validate_resource_contract(entry.resource_contract)
    if entry.resource_contract["counts"]["state_dim"] != {"constant": len(entry.state_order)}:
        reject("declared tensor state dimension differs from registered state order")
    _bounded_json(entry.manifest())
    if len(encode(entry.manifest())) > 65536:
        reject("registry metadata exceeds 64 KiB quota")


def validate_resource_contract(contract):
    _bounded_json(contract)
    if (type(contract) is not dict or set(contract) != {"schema_version", "counts", "tensors", "limits"} or
            contract.get("schema_version") != "pirc25-resource-contract-v1"):
        reject("resource contract schema is invalid")
    counts, tensors, limits = contract["counts"], contract["tensors"], contract["limits"]
    if type(counts) is not dict or set(counts) != COUNT_NAMES or type(limits) is not dict or set(limits) != set(GLOBAL_LIMITS):
        reject("all resource dimensions and quotas must be explicitly declared")
    for name, limit in limits.items():
        if type(limit) is not int or not 0 < limit <= GLOBAL_LIMITS[name]:
            reject("component resource quota exceeds shared hard limits", "RESOURCE_PLAN_REJECTED")
    for source in counts.values():
        if type(source) is not dict or len(source) != 1 or next(iter(source)) not in {"constant", "config", "input", "config_length", "input_length"}:
            reject("resource count needs a literal or a declared config/input path")
        kind, value = next(iter(source.items()))
        if kind == "constant":
            if type(value) is not int or not 0 <= value <= 10_000_000:
                reject("literal resource count must be a bounded nonnegative integer")
        elif (type(value) is not list or not 0 < len(value) <= 8 or
                any(type(key) is not str or not key or len(key) > 128 for key in value)):
            reject("resource count path must be explicit and bounded")
    if type(tensors) is not list or not 0 < len(tensors) <= 128:
        reject("resource contract needs bounded explicit tensor declarations")
    names = set()
    for tensor in tensors:
        if (type(tensor) is not dict or set(tensor) != {"name", "axes", "item_bytes"} or
                type(tensor["name"]) is not str or not tensor["name"] or len(tensor["name"]) > 128 or
                tensor["name"] in names or type(tensor["axes"]) is not list or not 0 < len(tensor["axes"]) <= 8 or
                any(type(axis) is not str or axis not in COUNT_NAMES | {"matrix_cells"} for axis in tensor["axes"]) or
                type(tensor["item_bytes"]) is not int or tensor["item_bytes"] not in {1, 2, 4, 8, 16}):
            reject("tensor axes, identity or element width is invalid")
        names.add(tensor["name"])


@dataclass(frozen=True)
class RegisteredComponent:
    _document: bytes
    builder: object

    @property
    def entry_hash(self):
        return digest(json.loads(self._document))

    @property
    def entry(self):
        value = json.loads(self._document)
        value.pop("schema_version")
        value["state_order"], value["units"] = tuple(value["state_order"]), tuple(value["units"])
        value["capabilities"] = frozenset(value["capabilities"])
        return RegistryEntry(**value)


class VersionedRegistry:
    def __init__(self):
        self._entries = {}

    def register(self, entry, builder):
        validate_entry(entry)
        if not callable(builder) or implementation_hash(builder) != entry.code_hash:
            reject("registered implementation code differs from its declared hash")
        registration = RegisteredComponent(encode(entry.manifest()), builder)
        key = (entry.component_id, entry.version)
        previous = self._entries.get(key)
        if previous is not None and previous._document != registration._document:
            reject("same component version has different immutable content", "IDENTITY_CONFLICT")
        self._entries[key] = registration
        return registration.entry_hash

    def resolve(self, component_id, version, *, required_capabilities=(), entry_hash=None,
                state_order=None, units=None, resource_class=None, resume_level=None):
        identifier(component_id)
        identifier(version)
        registration = self._entries.get((component_id, version))
        if registration is None:
            reject("requested exact component version is unavailable")
        entry = registration.entry
        if implementation_hash(registration.builder) != entry.code_hash:
            reject("registered live implementation changed", "IDENTITY_CONFLICT")
        if (type(required_capabilities) not in {tuple, list, set, frozenset} or
                any(type(value) is not str for value in required_capabilities) or
                not set(required_capabilities) <= entry.capabilities or
                (entry_hash is not None and entry_hash != registration.entry_hash) or
                (state_order is not None and (type(state_order) not in {tuple, list} or tuple(state_order) != entry.state_order)) or
                (units is not None and (type(units) not in {tuple, list} or tuple(units) != entry.units)) or
                (resource_class is not None and resource_class != entry.resource_class) or
                (resume_level is not None and resume_level != entry.resume_level)):
            reject("requested component capability, identity or compatibility differs")
        return registration


def plan_resources(entry, config, inputs, *, matrix_cells):
    """Constant-size declared arithmetic, never numerical tensors or factories."""
    validate_entry(entry)
    # Detach the policy before deriving the plan; the caller cannot invalidate
    # the already-hashed result by subsequently editing the declaration.
    entry = RegisteredComponent(encode(entry.manifest()), None).entry
    validate_value(entry.config_schema, config)
    validate_value(entry.input_schema, inputs)
    config, inputs = json.loads(encode(config)), json.loads(encode(inputs))
    contract = entry.resource_contract
    limits = contract["limits"]
    if type(matrix_cells) is not int or not 0 < matrix_cells <= limits["matrix_cells"]:
        reject("study matrix exceeds registered quota", "RESOURCE_PLAN_REJECTED")
    counts = {}
    for name, source in contract["counts"].items():
        kind, value = next(iter(source.items()))
        if kind != "constant":
            target = config if kind in {"config", "config_length"} else inputs
            for key in value:
                if type(target) is not dict or key not in target:
                    reject("declared resource count input is absent")
                target = target[key]
            value = target
            if kind.endswith("_length"):
                if type(value) is not list:
                    reject("resource length source must be an actual registered array")
                value = len(value)
        if type(value) is not int or value < 0 or (name != "state_dim" and value > limits[name]):
            reject("path/step/mixture/component/input count exceeds quota", "RESOURCE_PLAN_REJECTED")
        counts[name] = value
    tensors, total_elements, total_bytes = [], 0, 0
    axes = {**counts, "matrix_cells": matrix_cells}
    for declaration in contract["tensors"]:
        shape = [axes[axis] for axis in declaration["axes"]]
        elements = 1
        for size in shape:
            elements *= size
            if elements > limits["tensor_elements"]:
                reject("tensor shape exceeds element quota", "RESOURCE_PLAN_REJECTED")
        size_bytes = elements * declaration["item_bytes"]
        total_elements += elements
        total_bytes += size_bytes
        if total_elements > limits["tensor_elements"] or total_bytes > limits["tensor_bytes"]:
            reject("combined tensors exceed allocation quota", "RESOURCE_PLAN_REJECTED")
        tensors.append({"name": declaration["name"], "shape": shape, "elements": elements, "bytes": size_bytes})
    result = {"schema_version": "pirc25-resource-plan-v1", "registry_entry_hash": digest(entry.manifest()),
        "config_hash": digest(config), "input_hash": digest(inputs), "matrix_cells": matrix_cells,
        "counts": counts, "tensors": tensors, "tensor_elements": total_elements, "tensor_bytes": total_bytes,
        "maximum_result_bytes": limits["result_bytes"], "resource_class": entry.resource_class,
        "limits": dict(limits), "global_limits_hash": digest(dict(GLOBAL_LIMITS))}
    result["resource_plan_hash"] = digest(result)
    return result
