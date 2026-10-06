"""Versioned actual phase-space factories over the existing shared registry.

Registration and composition planning are allocation-free. Numerical engines
are imported only by factories/fit/predict inside the admitted worker. These
factories confer neither data authority nor scientific qualification.
"""

from dataclasses import asdict
import json
import math

from application.registry import ComponentRegistry
from application.research_composition import component_plan
from application.research_execution import execution_binding
from application.research_registry import GLOBAL_LIMITS, RegistryEntry, implementation_hash, validate_value
from infrastructure.research_store import ResearchError, digest, encode


VERSION = "1.3.0"
STATE = ("x", "y", "vx", "vy")
UNITS = ("m", "m", "m/s", "m/s")
FAMILIES = ("M0", "M1-S", "M1-R", "M2")
HASH = {"type": "string", "minLength": 64, "maxLength": 64}


def object_schema(fields):
    return {"type": "object", "properties": fields, "required": sorted(fields), "additionalProperties": False}


def integer(low, high):
    return {"type": "integer", "minimum": low, "maximum": high}


def basis_family(family):
    return family in ("M1-S", "M1-R")


def resume_level(family):
    return "restart-only" if basis_family(family) else "exact"


def latent_applicability(objective):
    """Negative evidence for this registered observed contract, not all data.

    No caller-supplied latent flags can promote this report to qualification.
    A future latent contract needs its own registration and feasibility scope.
    Degeneracy alone is not a proof that every constrained latent model fails.
    """
    if objective not in ("L1", "L2"):
        raise ResearchError("CONTRACT_MISMATCH", "latent applicability requires L1 or L2")
    reasons = ["OBSERVATION_LIKELIHOOD_UNREGISTERED", "INITIAL_LATENT_LAWS_UNREGISTERED",
        "RECOGNITION_PROCESS_UNREGISTERED", "LATENT_NEED_AND_IDENTIFIABILITY_UNESTABLISHED",
        "PATH_LAW_CONDITIONS_UNQUALIFIED"]
    if objective == "L2":
        reasons += ["FULL_STATE_DIFFUSION_INVERSE_UNAVAILABLE",
            "CONSTRAINED_POSTERIOR_FLOW_AND_SCORE_UNREGISTERED"]
    report = {"schema_version": "pirc26-latent-applicability-v1", "objective_id": objective,
        "status": "INAPPLICABLE", "scope": "current-registered-observed-contract-only",
        "component_version": VERSION, "observation_profile": "causal-observed-phase-space-v1",
        "state_order": list(STATE), "noise_dim": 2, "diffusion_support": ["vx", "vy"],
        "reason_codes": reasons, "production_latent_need": "NOT_ASSESSED",
        "noise_support_requirement": "G r = prior_drift - posterior_drift; position drift must remain velocity",
        "qualification": "unqualified", "metric_value": None}
    return {**report, "report_hash": digest(report)}


def _refuse_latent(objective):
    if objective in ("L1", "L2"):
        report = latent_applicability(objective)
        raise ResearchError("INAPPLICABLE", "current observed contract: " + ", ".join(report["reason_codes"]))


def plan_schema(objective, family=None):
    _refuse_latent(objective)
    if objective == "O1" and basis_family(family):
        return object_schema({"solver_id": {"type": "string", "enum": ["streaming-qr-v1"]},
            "ridge": {"type": "number", "minimum": 0},
            "curvature_penalty": {"type": "number", "minimum": 0, **({"maximum": 0} if family == "M1-R" else {})},
            "condition_number_max": {"type": "number", "minimum": 1}, "max_batch_rows": integer(1, 4096),
            "identifiability": {"type": "string", "enum": ["full-rbf-v1" if family == "M1-R" else "reference-coded-additive-v1"]}})
    fields = {"max_steps": integer(1, 10000), "patience": integer(1, 10000),
              "learning_rate": {"type": "number", "minimum": 1e-12, "maximum": 1},
              "tolerance": {"type": "number", "minimum": 0},
              "gradient_norm_limit": {"type": "number", "minimum": 1e-12}}
    if objective == "O1":
        fields.update(fit_diffusion={"type": "boolean"},
                      diffusion_diagonal_floor={"type": "number", "minimum": 1e-12})
    else:
        fields.update(horizon_indices={"type": "array", "items": integer(1, 255), "minItems": 1, "maxItems": 32},
                      curriculum_steps=integer(1, 10000), max_grid_steps=integer(1, 256),
                      max_gradient_state_elements=integer(8, 262144))
    return object_schema(fields)


def config_schema(objective, family):
    _refuse_latent(objective)
    fields = {"seed": integer(0, 2**63 - 1), "family": {"type": "string", "enum": [family]},
              "objective": {"type": "string", "enum": [objective]}, "plan": plan_schema(objective, family),
              "initial_model_hash": HASH, "dynamics_spec_hash": HASH, "configuration_hash": HASH,
              "forecast_request_hashes": {"type": "array", "items": HASH, "minItems": 1, "maxItems": 32}}
    if objective == "O2":
        fields["o1_lineage_hash"] = HASH
    return object_schema(fields)


def profile_schema():
    return object_schema({"state_dim": {"type": "integer", "enum": [4]},
        "noise_dim": {"type": "integer", "enum": [2]},
        "diffusion_support": {"type": "array", "items": {"type": "string"}, "enum": [["vx", "vy"]]},
        "dtype": {"type": "string", "enum": ["float32", "float64"]},
        "device": {"type": "string", "enum": ["cpu"]},
        "observation_profile": {"type": "string", "enum": ["causal-observed-phase-space-v1"]},
        "observations": integer(1, 1048576), "batches": integer(1, 256), "origins": integer(1, 32),
        "paths": integer(2, 256), "steps": integer(2, 10000), "components": integer(1, 128)})


def _settings(document, inputs):
    _refuse_latent(document.get("objective"))
    validate_value(config_schema(document.get("objective"), document.get("family")), document)
    validate_value(profile_schema(), inputs)
    if document["family"] not in FAMILIES or document["objective"] not in ("O1", "O2"):
        raise ResearchError("CONTRACT_MISMATCH", "unknown registered phase-space combination")
    if document["objective"] == "O2" and document["family"] != "M2":
        raise ResearchError("OBJECTIVE_INCOMPATIBLE", "O2 requires the same observed-state M2")
    from estimation.phase_space import O1Plan
    from estimation.phase_space_o2 import O2Plan
    from estimation.phase_space_basis import BasisPlan
    constructor = BasisPlan if basis_family(document["family"]) else (O1Plan if document["objective"] == "O1" else O2Plan)
    plan = constructor(**document["plan"])
    plan.validate()
    if (inputs["batches"] if isinstance(plan, BasisPlan) else plan.max_steps) > inputs["steps"]:
        raise ResearchError("RESOURCE_PLAN_REJECTED", "actual training steps exceed frozen resource plan")
    hashes = document["forecast_request_hashes"]
    if len(set(hashes)) != len(hashes):
        raise ResearchError("CONTRACT_MISMATCH", "forecast request identities must be unique")
    return plan


def _model_profile(model, document, inputs):
    from models.phase_space import PhaseSpaceSDE
    if not isinstance(model, PhaseSpaceSDE):
        raise ResearchError("CONTRACT_MISMATCH", "actual phase-space model required")
    card, configuration = model.model_card(), model.acceleration_model.configuration()
    if (model.state_dim != inputs["state_dim"] or model.noise_dim != inputs["noise_dim"]
            or card["state_names"] != list(STATE) or card["units"] != list(UNITS)
            or card["diffusion_support"] != inputs["diffusion_support"] or card["time_unit"] != "s"
            or card["dtype"] != "torch." + inputs["dtype"] or card["device"] != inputs["device"]
            or card["family"] != document["family"] or digest(card["spec"]) != document["dynamics_spec_hash"]
            or digest(configuration) != document["configuration_hash"]):
        raise ResearchError("CONTRACT_MISMATCH", "actual dynamics differ from frozen state/units/spec/configuration")
    capacity = max((4, math.ceil(math.sqrt(sum(t.numel() for t in model.state_dict().values()))),
                    model.spec.context_dim, *configuration.get("hidden", []),
                    len(configuration.get("centers", [])),
                    sum(len(knots) - configuration.get("degree", 0) - 1 for knots in configuration.get("knots", []))))
    if capacity > inputs["components"]:
        raise ResearchError("RESOURCE_PLAN_REJECTED", "actual model capacity exceeds declared workspace")
    if card["family"] == "M2" and configuration["seed"] != document["seed"]:
        raise ResearchError("CONTRACT_MISMATCH", "actual M2 initializer differs from registered seed")


def model_factory(document, inputs, context):
    validate_value(config_schema(document.get("objective"), document.get("family")), document)
    validate_value(profile_schema(), inputs)
    if type(context) is not dict or set(context) != {"initial_checkpoint"}:
        raise ResearchError("MISSING_INPUT", "admitted initial checkpoint is required")
    checkpoint = context["initial_checkpoint"]
    preflight_model(checkpoint, document, inputs)
    _settings(document, inputs)
    from models.phase_space import PhaseSpaceSDE
    model = PhaseSpaceSDE.from_checkpoint(checkpoint)
    _model_profile(model, document, inputs)
    return model


def preflight_model(checkpoint, document, inputs):
    """Owner/factory transport and workspace checks before any constructor."""
    from infrastructure.pirc26_checkpoint_contract import inspect_checkpoint, CheckpointContractError
    if type(checkpoint) is not dict or checkpoint.get("sha256") != document["initial_model_hash"]:
        raise ResearchError("CONTRACT_MISMATCH", "actual initial checkpoint differs from frozen identity")
    try:
        checked = inspect_checkpoint(checkpoint, component_limit=inputs["components"])
    except CheckpointContractError as exc:
        raise ResearchError(exc.code, "bounded model checkpoint preflight refused") from exc
    card, config = checked["document"]["model_card"], checked["document"]["configuration"]
    if (card["family"] != document["family"] or card["dtype"] != "torch." + inputs["dtype"]
            or card["device"] != inputs["device"] or digest(card["spec"]) != document["dynamics_spec_hash"]
            or digest(config) != document["configuration_hash"] or card["state_names"] != list(STATE)
            or inputs["state_dim"] != 4 or inputs["noise_dim"] != 2 or inputs["diffusion_support"] != ["vx", "vy"]
            or card["family"] == "M2" and config["seed"] != document["seed"]):
        raise ResearchError("CONTRACT_MISMATCH", "frozen checkpoint differs from declared component recipe")
    return checked


class BoundTrainer:
    def __init__(self, document, inputs):
        self.document, self.inputs = json.loads(encode(document)), json.loads(encode(inputs))
        self.plan = _settings(self.document, self.inputs)

    def fit(self, model, data, *, o1_result=None, **control):
        _model_profile(model, self.document, self.inputs)
        if model.checkpoint()["sha256"] != self.document["initial_model_hash"]:
            raise ResearchError("CONTRACT_MISMATCH", "fitting must start from the registered initial model")
        if type(data) not in (list, tuple) or not data:
            raise ResearchError("CONTRACT_MISMATCH", "explicit bounded train data required")
        if self.document["objective"] == "O1":
            from estimation.phase_space import TransitionBatch, fit_o1
            if any(not isinstance(part, TransitionBatch) for part in data):
                raise ResearchError("CONTRACT_MISMATCH", "actual O1 transition batches required")
            if len(data) > self.inputs["batches"] or sum(len(part.time) for part in data) > self.inputs["observations"]:
                raise ResearchError("RESOURCE_PLAN_REJECTED", "actual O1 train data exceeds frozen plan")
            if o1_result is not None:
                raise ResearchError("CONTRACT_MISMATCH", "O1 cannot substitute an O2 lineage")
            if basis_family(self.document["family"]):
                if set(control) - {"cancellation", "progress"}:
                    raise ResearchError("CHECKPOINT_INCOMPATIBLE", "streaming QR is restart-only, not an optimizer continuation")
                from estimation.phase_space_basis import fit_basis
                result = fit_basis(model, data, self.plan, **control)
            else:
                result = fit_o1(model, data, self.plan, **control)
        else:
            from estimation.phase_space_o2 import HorizonTrainingExample, fit_o2
            if any(not isinstance(part, HorizonTrainingExample) for part in data):
                raise ResearchError("CONTRACT_MISMATCH", "actual O2 train examples required")
            if len(data) > self.inputs["origins"] or sum(len(part.target) for part in data) > self.inputs["observations"]:
                raise ResearchError("RESOURCE_PLAN_REJECTED", "actual O2 train data exceeds frozen plan")
            if type(o1_result) is not dict or digest(o1_result) != self.document["o1_lineage_hash"]:
                raise ResearchError("CONTRACT_MISMATCH", "actual O1 lineage differs from registered O2 input")
            for part in data:
                _request_profile(part.request, self.document, self.inputs)
            result = fit_o2(model, data, self.plan, o1_result, **control)
        return result


def trainer_factory(document, inputs, context):
    return BoundTrainer(document, inputs)


def _request_profile(request, document, inputs):
    from inference.phase_space import ForecastRequest
    if not isinstance(request, ForecastRequest) or digest(asdict(request)) not in document["forecast_request_hashes"]:
        raise ResearchError("CONTRACT_MISMATCH", "forecast origin/grid/context/noise recipe differs from frozen request")
    if request.sample_count > inputs["paths"] or len(request.time_grid) > inputs["steps"]:
        raise ResearchError("RESOURCE_PLAN_REJECTED", "actual forecast exceeds declared path/grid workspace")


class BoundPredictor:
    def __init__(self, document, inputs):
        self.document, self.inputs = json.loads(encode(document)), json.loads(encode(inputs))
        _settings(self.document, self.inputs)

    def predict(self, model, request, *, cancellation=None):
        _model_profile(model, self.document, self.inputs)
        _request_profile(request, self.document, self.inputs)
        from inference.phase_space import forecast
        return forecast(model, request, cancellation=cancellation)


def predictor_factory(document, inputs, context):
    return BoundPredictor(document, inputs)


def composition_contract(objective, family=None):
    _refuse_latent(objective)
    return {"schema_version": "pirc25-composition-contract-v1", "shared_configuration": True,
        "roles": {"model": {"required_capabilities": ["phase-space-dynamics", "velocity-noise"],
                             "required_model_capabilities": [], "seed_path": ["seed"]},
                  "trainer": {"required_capabilities": ["observed-" + objective.lower()] + (["streaming-qr"] if basis_family(family) else []),
                      "required_model_capabilities": ["velocity-noise"] + (["direct-rollout-gradient"] if objective == "O2" else []),
                      "seed_path": ["seed"]},
                  "predictor": {"required_capabilities": ["generic-rollout"],
                      "required_model_capabilities": ["phase-space-dynamics"], "seed_path": ["seed"]}}}


def entry(role, objective, family):
    _refuse_latent(objective)
    if role not in ("model", "trainer", "predictor") or objective not in ("O1", "O2") or family not in FAMILIES:
        raise ResearchError("CONTRACT_MISMATCH", "unknown exact component combination")
    if objective == "O2" and family != "M2":
        raise ResearchError("OBJECTIVE_INCOMPATIBLE", "O2 component declares only M2")
    builder = {"model": model_factory, "trainer": trainer_factory, "predictor": predictor_factory}[role]
    capabilities = {"model": {"phase-space-dynamics", "velocity-noise", "local-velocity-objective"},
                    "trainer": {"observed-" + objective.lower()}, "predictor": {"generic-rollout"}}[role]
    if role == "model" and family == "M2":
        capabilities.add("direct-rollout-gradient")
    tensors = {"model": [{"name": "model_workspace", "axes": ["components", "components"], "item_bytes": 16}],
               "trainer": [{"name": "train_states", "axes": ["observations", "state_dim"], "item_bytes": 16},
                           {"name": "optimizer", "axes": ["components", "components"], "item_bytes": 16},
                           {"name": "gradients", "axes": ["components", "components"], "item_bytes": 8},
                           *[{"name": "activation_" + str(i), "axes": ["observations", "components"], "item_bytes": 8}
                             for i in range(4)]],
               "predictor": [{"name": "rollout", "axes": ["paths", "steps", "state_dim"], "item_bytes": 8},
                             {"name": "score_pairs", "axes": ["paths", "paths"], "item_bytes": 8}]}[role]
    if role == "trainer" and objective == "O2":
        tensors.extend({"name": "path_activation_" + str(i), "axes": ["paths", "steps", "components"], "item_bytes": 8}
                       for i in range(4))
    if role == "trainer" and basis_family(family):
        # B is bounded by the already admitted observation count. Include
        # streamed basis/QR Q, triangular state and spline penalty workspace.
        tensors = [{"name": "train_states", "axes": ["observations", "state_dim"], "item_bytes": 16},
            *[{"name": "stream_workspace_" + str(i), "axes": ["observations", "components"], "item_bytes": 8}
              for i in range(32)],
            *[{"name": "basis_workspace_" + str(i), "axes": ["components", "components"], "item_bytes": 8}
              for i in range(16)]]
        capabilities.add("streaming-qr")
    return RegistryEntry(component_id="pirc26-" + family.lower() + "-" + objective.lower() + "-" + role,
        component_kind=role, version=VERSION, code_hash=implementation_hash(builder),
        config_schema=config_schema(objective, family), input_schema=profile_schema(),
        output_schema={"type": "object", "additionalProperties": True}, state_order=STATE, units=UNITS,
        capabilities=frozenset(capabilities), resource_class="cpu", resume_level=resume_level(family) if role == "trainer" else "restart-only",
        resource_contract={"schema_version": "pirc25-resource-contract-v1", "counts": {
            **{name: {"input": [name]} for name in ("observations", "paths", "steps", "components")},
            "mixtures": {"constant": 1}, "state_dim": {"constant": 4}}, "tensors": tensors,
            "limits": {**dict(GLOBAL_LIMITS), "matrix_cells": 10000, "result_bytes": 4 * 1024 * 1024}})


def component_registries(objective, family):
    result = {}
    for role, builder in (("model", model_factory), ("trainer", trainer_factory), ("predictor", predictor_factory)):
        registry = ComponentRegistry()
        registry.register_version(entry(role, objective, family), builder)
        result[role] = registry
    return result


def component_bindings(document, inputs, *, matrix_cells, registries):
    return {role: execution_binding(entry(role, document["objective"], document["family"]), document, inputs,
                                   matrix_cells=matrix_cells) for role in registries}


def construct_components(adapter_entry, bindings, *, matrix_cells, seed, registries, initial_checkpoint):
    # Validate the complete combination before the first allocation/factory.
    plan = component_plan(adapter_entry, registries, bindings, matrix_cells=matrix_cells, seed=seed)
    sealed = json.loads(encode(bindings))
    checked = component_plan(adapter_entry, registries, sealed, matrix_cells=matrix_cells, seed=seed)
    if plan != checked:
        raise ResearchError("CONTRACT_MISMATCH", "component combination changed during preflight")
    preflight_model(initial_checkpoint, sealed["model"]["config"], sealed["model"]["inputs"])
    parts = {role: registries[role].create_bound(sealed[role], matrix_cells=matrix_cells,
            context={"initial_checkpoint": initial_checkpoint} if role == "model" else None)
             for role in ("model", "trainer", "predictor")}
    return parts, plan
