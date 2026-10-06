"""The composition root's model, estimator, and inference registries."""

from __future__ import annotations

from application.registry import ComponentRegistry
from config import Config
from estimation.base import Estimator
from estimation.em import SegmentEM
from inference.base import (
    CommonRandomNumberEngine,
    EulerMaruyamaEngine,
    ExactGaussianEngine,
    InferenceEngine,
    SplitStepEngine,
)
from models.base import SDEModel
from models.segment_constant import SegmentConstantSDE
from dataclasses import asdict
from application.research_execution import execution_binding
from application.research_registry import GLOBAL_LIMITS, RegistryEntry, implementation_hash
from infrastructure.research_store import ResearchError, digest


def _torch_dtype(name: str):
    import torch

    return {"float32": torch.float32, "float64": torch.float64}[name]


def _build_segment_constant(cfg: Config) -> SDEModel:
    model = cfg.model.get("I1", {}) or {}
    return SegmentConstantSDE(
        n_modes=model.get("n_modes", 3),
        kappa=model.get("kappa", 0.0),
        dt_ref=model.get("dt_ref", 60.0),
        dtype=_torch_dtype(cfg.dtype),
        device=cfg.device,
    )


def _build_em(cfg: Config) -> Estimator:
    options = cfg.protocol.get("em", {}) or {}
    return SegmentEM(
        max_iter=int(options.get("max_iter", 50)),
        tol=float(options.get("tol", 1e-5)),
    )


def _build_exact(cfg: Config) -> InferenceEngine:
    return ExactGaussianEngine()


def _build_euler(cfg: Config) -> InferenceEngine:
    options = cfg.protocol.get("euler", {}) or {}
    return EulerMaruyamaEngine(max_step=float(options.get("max_step", 1.0)))


def _build_split(cfg: Config) -> InferenceEngine:
    options = cfg.protocol.get("split", {}) or {}
    return SplitStepEngine(max_step=float(options.get("max_step", 1.0)))


def _build_crn(cfg: Config) -> InferenceEngine:
    return CommonRandomNumberEngine()


MODEL_REGISTRY: ComponentRegistry[Config, SDEModel] = ComponentRegistry()
MODEL_REGISTRY.register("I1", _build_segment_constant)

ESTIMATOR_REGISTRY: ComponentRegistry[Config, Estimator] = ComponentRegistry()
ESTIMATOR_REGISTRY.register("EM", _build_em)

INFERENCE_REGISTRY: ComponentRegistry[Config, InferenceEngine] = ComponentRegistry()
INFERENCE_REGISTRY.register("exact", _build_exact)
INFERENCE_REGISTRY.register("euler", _build_euler)
INFERENCE_REGISTRY.register("J1_split", _build_split)
INFERENCE_REGISTRY.register("J3_CRN", _build_crn)


def build_model(cfg: Config) -> SDEModel:
    cfg.validate()
    return MODEL_REGISTRY.create(cfg.components.model, cfg)


def build_estimator(cfg: Config) -> Estimator:
    cfg.validate()
    return ESTIMATOR_REGISTRY.create(cfg.components.estimator, cfg)


def build_inference_engine(cfg: Config) -> InferenceEngine:
    cfg.validate()
    return INFERENCE_REGISTRY.create(cfg.components.inference, cfg)


def _versioned_model(document, inputs, context):
    return _build_segment_constant(Config.from_dict(document))


def _versioned_trainer(document, inputs, context):
    return _build_em(Config.from_dict(document))


def _versioned_exact(document, inputs, context):
    return _build_exact(Config.from_dict(document))


def _versioned_euler(document, inputs, context):
    return _build_euler(Config.from_dict(document))


def _versioned_split(document, inputs, context):
    return _build_split(Config.from_dict(document))


def _versioned_crn(document, inputs, context):
    return _build_crn(Config.from_dict(document))


def _object_schema(properties):
    return {"type": "object", "properties": properties, "required": sorted(properties), "additionalProperties": False}


def _configuration_schema():
    # Explicit defaults are frozen by Config serialization, never guessed on
    # resolution. Existing ordinary CLI configurations still use legacy APIs.
    return _object_schema({
        "seed": {"type": "integer", "minimum": 0, "maximum": 2**63 - 1},
        "dtype": {"type": "string", "enum": ["float32", "float64"]},
        "device": {"type": "string", "enum": ["cpu"]},
        "components": _object_schema({
            "model": {"type": "string", "enum": ["I1"]}, "estimator": {"type": "string", "enum": ["EM"]},
            "inference": {"type": "string", "enum": ["exact", "euler", "J1_split", "J3_CRN"]},
            "transfer": {"type": "string", "enum": ["none"]}, "condition": {"type": "string", "enum": ["none"]}}),
        "model": _object_schema({"I1": _object_schema({
            "n_modes": {"type": "integer", "minimum": 1, "maximum": 1024},
            "kappa": {"type": "number", "minimum": 0}, "dt_ref": {"type": "number", "minimum": 1e-12}})}),
        "protocol": {"type": "object", "properties": {
            "em": {"type": "object", "properties": {
                "max_iter": {"type": "integer", "minimum": 1, "maximum": 1000000},
                "tol": {"type": "number", "minimum": 1e-15}}, "required": ["max_iter"], "additionalProperties": False},
            "euler": _object_schema({"max_step": {"type": "number", "minimum": 1e-12}}),
            "split": _object_schema({"max_step": {"type": "number", "minimum": 1e-12}})},
            "required": ["em"], "additionalProperties": False},
        **{name: {"type": "object", "additionalProperties": True}
            for name in ("paths", "ablation_matrix", "equivalence_regression", "config_key_map")}})


def _profile_schema():
    return _object_schema({
        "state_dim": {"type": "integer", "enum": [2]}, "noise_dim": {"type": "integer", "enum": [1]},
        "diffusion_support": {"type": "array", "items": {"type": "string"}, "enum": [["vx"]]},
        "dtype": {"type": "string", "enum": ["float32", "float64"]}, "device": {"type": "string", "enum": ["cpu"]},
        "observation_profile": {"type": "string", "enum": ["segment-trajectories"]},
        **{name: {"type": "integer", "minimum": 1, "maximum": GLOBAL_LIMITS[name]}
            for name in ("observations", "paths", "steps", "components")}})


def _entry(component_id, kind, builder, capabilities):
    tensors = {
        "model": [{"name": "model_covariances", "axes": ["mixtures", "state_dim", "state_dim"], "item_bytes": 8},
            {"name": "model_parameters", "axes": ["mixtures", "state_dim", "state_dim", "state_dim"], "item_bytes": 8}],
        "trainer": [{"name": "fit_workspace", "axes": ["components", "observations", "state_dim", "state_dim"], "item_bytes": 8}],
        "predictor": [{"name": "rollout", "axes": ["paths", "steps", "state_dim"], "item_bytes": 8}]}
    return RegistryEntry(component_id=component_id, component_kind=kind, version="1.0.0",
        code_hash=implementation_hash(builder), config_schema=_configuration_schema(), input_schema=_profile_schema(),
        output_schema={"type": "object", "additionalProperties": True}, state_order=("x", "vx"), units=("m", "m/s"),
        capabilities=frozenset(capabilities), resource_class="cpu", resume_level="restart-only",
        resource_contract={"schema_version": "pirc25-resource-contract-v1", "counts": {
            **{name: {"input": [name]} for name in ("paths", "steps", "components", "observations")},
            "mixtures": {"config": ["model", "I1", "n_modes"]}, "state_dim": {"constant": 2}},
            "tensors": tensors[kind], "limits": dict(GLOBAL_LIMITS)})


def _register_versions():
    declarations = [
        (MODEL_REGISTRY, "I1", "model", _versioned_model, {"exact-transition", "latent-regime", "segment-affine", "sde-dynamics"}),
        (ESTIMATOR_REGISTRY, "EM", "trainer", _versioned_trainer, {"segment-em", "requires-segment-affine"}),
        (INFERENCE_REGISTRY, "exact", "predictor", _versioned_exact, {"exact-transition", "requires-exact-transition"}),
        (INFERENCE_REGISTRY, "euler", "predictor", _versioned_euler, {"generic-rollout", "requires-sde-dynamics"}),
        (INFERENCE_REGISTRY, "J1_split", "predictor", _versioned_split, {"generic-rollout", "requires-segment-affine"}),
        (INFERENCE_REGISTRY, "J3_CRN", "predictor", _versioned_crn, {"coupled-level", "exact-transition", "requires-exact-transition"})]
    for registry, name, kind, builder, capabilities in declarations:
        registry.register_version(_entry(name, kind, builder, capabilities), builder)


def component_bindings(cfg, inputs, *, matrix_cells):
    """Freeze explicit versions for existing single-axis composition factories."""
    _register_versions()
    document = asdict(cfg)
    result = {}
    for role, registry, name in (("model", MODEL_REGISTRY, cfg.components.model),
            ("trainer", ESTIMATOR_REGISTRY, cfg.components.estimator), ("predictor", INFERENCE_REGISTRY, cfg.components.inference)):
        registration = registry._versions.resolve(name, "1.0.0")
        result[role] = execution_binding(registration.entry, document, inputs, matrix_cells=matrix_cells)
    plan_components(cfg, result, matrix_cells=matrix_cells)
    return result


def plan_components(cfg, bindings, *, matrix_cells):
    """Validate all roles before creating any model, optimizer or runtime."""
    _register_versions()
    if type(bindings) is not dict or set(bindings) != {"model", "trainer", "predictor"}:
        raise ResearchError("CONTRACT_MISMATCH", "complete model/trainer/predictor bindings required")
    document = asdict(cfg)
    plans, entries = {}, {}
    inputs = None
    for role, registry, name in (("model", MODEL_REGISTRY, cfg.components.model),
            ("trainer", ESTIMATOR_REGISTRY, cfg.components.estimator), ("predictor", INFERENCE_REGISTRY, cfg.components.inference)):
        binding = bindings[role]
        plan, registration = registry.plan_bound(binding, matrix_cells=matrix_cells)
        entry = registration.entry
        if (entry.component_kind != role or binding["component_id"] != name or binding["config"] != document
                or binding["inputs"]["dtype"] != cfg.dtype or binding["inputs"]["device"] != cfg.device
                or inputs is not None and binding["inputs"] != inputs):
            raise ResearchError("CONTRACT_MISMATCH", "component configuration or input profile differs")
        inputs = binding["inputs"]
        plans[role], entries[role] = plan, entry.manifest()
    # requires-* is an explicit component-contract vocabulary, not a component
    # name heuristic. Neither type names nor a fallback select an algorithm.
    for role in ("trainer", "predictor"):
        required = {capability.removeprefix("requires-") for capability in entries[role]["capabilities"]
            if capability.startswith("requires-")}
        if (entries["model"]["state_order"] != entries[role]["state_order"]
                or entries["model"]["units"] != entries[role]["units"]
                or not required <= set(entries["model"]["capabilities"])):
            raise ResearchError("CONTRACT_MISMATCH", "component requires unavailable model capability or state contract")
    if cfg.protocol["em"]["max_iter"] > inputs["steps"]:
        raise ResearchError("RESOURCE_PLAN_REJECTED", "trainer iteration bound exceeds frozen step bound")
    totals = {key: sum(plan[key] for plan in plans.values()) for key in ("tensor_elements", "tensor_bytes")}
    if any(totals[key] > GLOBAL_LIMITS[key] for key in totals):
        raise ResearchError("RESOURCE_PLAN_REJECTED", "combined component allocation exceeds shared quota")
    result = {"schema_version": "pirc25-component-plan-v1", "binding_hash": digest(bindings),
        "entries": entries, "plans": plans, **totals}
    result["component_plan_hash"] = digest(result)
    return result


__all__ = [
    "ESTIMATOR_REGISTRY",
    "INFERENCE_REGISTRY",
    "MODEL_REGISTRY",
    "build_estimator",
    "build_inference_engine",
    "build_model",
    "component_bindings",
    "plan_components",
]
