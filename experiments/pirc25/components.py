"""Versioned factories for the existing affine pipelines, not new algorithms."""

from application.registry import ComponentRegistry
from application.research_execution import execution_binding
from application.research_registry import GLOBAL_LIMITS, RegistryEntry, implementation_hash
from infrastructure.research_store import ResearchError


def single_axis_config(seed):
    from config import Components, Config
    return Config(seed=seed, components=Components(model="I1", estimator="EM", inference="exact"),
        model={"I1": {"n_modes": 2, "kappa": 0.0, "dt_ref": 1.0}}, protocol={"em": {"max_iter": 2}})


def input_profile(dimensions):
    return {"state_dim": 2 if dimensions == 1 else 4, "noise_dim": 1 if dimensions == 1 else 2,
        "diffusion_support": ["vx"] if dimensions == 1 else ["vx", "vy"], "dtype": "float64", "device": "cpu",
        "observation_profile": "segment-trajectories" if dimensions == 1 else "phase-space-trajectories",
        "observations": 40 if dimensions == 1 else 32, "paths": 8, "steps": 8, "components": 4}


def phase_model(document, inputs, context):
    from .affine import phase_space_api
    import numpy as np
    api = phase_space_api()
    def construct(**values):
        shapes = {"feature_mean": (2,), "feature_scale": (2,), "weights": (3, 2), "diffusion_covariance": (2, 2)}
        if (set(values) != {"condition_names", "feature_basis", "feature_names", "transition_count", *shapes}
                or values["condition_names"] != () or values["feature_basis"] != "direct"
                or values["feature_names"] != ("vx", "vy")
                or type(values["transition_count"]) is not int or not 0 < values["transition_count"] <= inputs["observations"]
                or any(not isinstance(values[name], np.ndarray) or values[name].shape != shape
                    or str(values[name].dtype) != inputs["dtype"] or not np.isfinite(values[name]).all()
                    for name, shape in shapes.items())):
            raise ResearchError("CONTRACT_MISMATCH", "constructed four-state model differs from frozen profile")
        return api.AffineVelocityModel(**values)
    return construct


def phase_trainer(document, inputs, context):
    from .affine import phase_space_api
    import numpy as np
    api = phase_space_api()
    expected = document["benchmark"]["velocity_model"]
    def fit(segments, *, condition_names, condition_resolver=None, feature_basis="direct", ridge=1e-6, model_factory=None):
        if (type(segments) not in {list, tuple} or not segments
                or len(segments) > inputs["components"]):
            raise ResearchError("RESOURCE_PLAN_REJECTED", "actual four-state segment count exceeds frozen bound")
        observations = 0
        for segment in segments:
            if (not isinstance(segment, api.Segment) or not isinstance(segment.time, np.ndarray)
                    or not isinstance(segment.state, np.ndarray) or segment.time.ndim != 1 or segment.state.ndim != 2
                    or segment.state.shape != (len(segment.time), 2)
                    or str(segment.state.dtype) != inputs["dtype"] or str(segment.time.dtype) != inputs["dtype"]):
                raise ResearchError("CONTRACT_MISMATCH", "actual four-state observation shape/dtype differs")
            observations += len(segment.time)
            if observations > inputs["observations"]:
                raise ResearchError("RESOURCE_PLAN_REJECTED", "actual four-state observations exceed frozen bound")
        if (tuple(condition_names) != () or condition_resolver is not None
                or feature_basis != expected.get("feature_basis", "direct") or ridge != expected["ridge"]):
            raise ResearchError("CONTRACT_MISMATCH", "actual four-state trainer differs from frozen configuration")
        return api.fit_affine_velocity_model(segments, condition_names=(), feature_basis=feature_basis,
            ridge=ridge, model_factory=model_factory)
    return fit


def phase_predictor(document, inputs, context):
    from .affine import phase_space_api
    import numpy as np
    api = phase_space_api()
    def predict(model, *, initial_state, time_grid, n_samples, rng, condition_field=None):
        if type(n_samples) is not int or not 2 <= n_samples <= inputs["paths"]:
            raise ResearchError("RESOURCE_PLAN_REJECTED", "actual four-state path count exceeds frozen bound")
        if (not isinstance(time_grid, np.ndarray) or time_grid.ndim != 1 or
                not isinstance(initial_state, np.ndarray) or initial_state.shape != (inputs["state_dim"],)
                or str(initial_state.dtype) != inputs["dtype"] or str(time_grid.dtype) != inputs["dtype"]):
            raise ResearchError("CONTRACT_MISMATCH", "actual four-state forecast shape/dtype differs")
        if not 2 <= len(time_grid) <= inputs["steps"]:
            raise ResearchError("RESOURCE_PLAN_REJECTED", "actual four-state time grid exceeds frozen bound")
        if (not isinstance(model, api.AffineVelocityModel) or model.diffusion_covariance.shape != (2, 2)
                or str(model.diffusion_covariance.dtype) != inputs["dtype"] or model.condition_names != ()
                or condition_field is not None or not isinstance(rng, np.random.Generator)):
            raise ResearchError("CONTRACT_MISMATCH", "actual four-state model/random/context profile differs")
        return api.rollout_phase_space_from_state(model, initial_state=initial_state, time_grid=time_grid,
            n_samples=n_samples, rng=rng, condition_field=None)
    return predict


def _fixed_schema(values):
    kinds = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}
    def schema(value):
        kind = kinds[type(value)]
        result = {"type": kind, "enum": [value]}
        if kind == "array":
            # These fixed public recipes contain homogeneous string lists.
            result["items"] = {"type": "string"}
        if kind == "object":
            result["additionalProperties"] = True
        return result
    return {"type": "object", "properties": {key: schema(value) for key, value in values.items()},
        "required": sorted(values), "additionalProperties": False}


def phase_configuration(seed):
    from .affine import phase_space_api
    return {"seed": seed, "dtype": "float64", "device": "cpu", "benchmark": phase_space_api().load_phase_space_spec()}


def _phase_entry(role, builder, capabilities):
    config = _fixed_schema(phase_configuration(0))
    config["properties"]["seed"] = {"type": "integer", "minimum": 0, "maximum": 2**63 - 1}
    return RegistryEntry(component_id="phase-affine-" + role, component_kind=role, version="1.0.0",
        code_hash=implementation_hash(builder), config_schema=config, input_schema=_fixed_schema(input_profile(4)),
        output_schema={"type": "object", "additionalProperties": True},
        state_order=("x", "y", "vx", "vy"), units=("m", "m", "m/s", "m/s"),
        capabilities=frozenset(capabilities), resource_class="cpu", resume_level="restart-only",
        resource_contract={"schema_version": "pirc25-resource-contract-v1", "counts": {
            **{name: {"input": [name]} for name in ("paths", "steps", "components", "observations")},
            "mixtures": {"constant": 1}, "state_dim": {"constant": 4}},
            "tensors": {"model": [
                {"name": "model_parameters", "axes": ["state_dim", "state_dim"], "item_bytes": 8},
                {"name": "model_covariance", "axes": ["state_dim", "state_dim"], "item_bytes": 8}],
                "trainer": [{"name": "fit_workspace", "axes": ["components", "observations", "state_dim", "state_dim"], "item_bytes": 8}],
                "predictor": [{"name": "rollout", "axes": ["paths", "steps", "state_dim"], "item_bytes": 8}]}[role],
            "limits": dict(GLOBAL_LIMITS)})


def component_registries(dimensions):
    if dimensions == 1:
        from registry import MODEL_REGISTRY, ESTIMATOR_REGISTRY, INFERENCE_REGISTRY, _register_versions
        _register_versions()
        return {"model": MODEL_REGISTRY, "trainer": ESTIMATOR_REGISTRY, "predictor": INFERENCE_REGISTRY}
    if dimensions != 4:
        raise ResearchError("CONTRACT_MISMATCH", "unsupported affine component dimension")
    result = {}
    for role, builder, capabilities in (("model", phase_model, {"sde-dynamics", "affine-velocity-dynamics", "velocity-noise"}),
            ("trainer", phase_trainer, {"affine-velocity-fit"}), ("predictor", phase_predictor, {"generic-rollout"})):
        registry = ComponentRegistry()
        registry.register_version(_phase_entry(role, builder, capabilities), builder)
        result[role] = registry
    return result


def composition_contract(dimensions):
    required = {"model": ["segment-affine"], "trainer": ["segment-em"], "predictor": ["exact-transition"]} if dimensions == 1 else {
        "model": ["affine-velocity-dynamics"], "trainer": ["affine-velocity-fit"], "predictor": ["generic-rollout"]}
    dependencies = {"model": [], "trainer": ["segment-affine"], "predictor": ["exact-transition"]} if dimensions == 1 else {
        "model": [], "trainer": ["affine-velocity-dynamics"], "predictor": ["sde-dynamics"]}
    return {"schema_version": "pirc25-composition-contract-v1", "shared_configuration": True,
        "roles": {role: {"required_capabilities": required[role], "required_model_capabilities": dependencies[role],
            "seed_path": ["seed"]} for role in ("model", "trainer", "predictor")}}


def fixture_component_bindings(dimensions, seed, *, matrix_cells, registries):
    if dimensions == 1:
        from registry import component_bindings
        return component_bindings(single_axis_config(seed), input_profile(1), matrix_cells=matrix_cells)
    config, profile = phase_configuration(seed), input_profile(4)
    result = {}
    for role, registry in registries.items():
        registration = registry.resolve_version("phase-affine-" + role, "1.0.0")
        result[role] = execution_binding(registration.entry, config, profile, matrix_cells=matrix_cells)
    return result
