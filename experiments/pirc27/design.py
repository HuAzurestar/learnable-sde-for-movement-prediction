"""Bounded, content-frozen synthetic design preparation; never a research launch.

The complete expected matrix includes unsupported/ineligible rows. Compiling a
design creates no store, grant, reservation, worker or scientific qualification.
Budget identities are explicit existing family bindings, not study/config IDs.
"""

from dataclasses import asdict, dataclass
from itertools import product
import json
import math

from application.propagation_inputs import validate_oracle_input, validate_nonlinear_input
from application.research_execution import execution_binding
from domain.errors import DataValidationError
from domain.frozen_dynamics import FrozenDynamicsPackage, content_hash, _encode, _hash
from domain.nonlinear_dynamics import FrozenNonlinearPackage
from domain.propagation import PropagationRequest
from domain.mixture import MixtureSettings
from infrastructure.research_store import identifier

from .oracles import matrix_cardinality
from .plugin import propagation_plugin, execution_config, execution_inputs
from .input_cases import StudyInputCase, StudyInputPolicy, validate_cases, input_binding
from .calibration_bindings import StudyCalibration, MAX_DOCUMENT_BYTES


METHODS = frozenset({"exact", "gaussian", "euler", "heun", "mlmc", "importance",
                    "cubature", "mixture", "reversible-heun", "pde"})
MAX_MANIFEST_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class StudyModel:
    family_id: str
    package: FrozenDynamicsPackage | FrozenNonlinearPackage
    initial_mean: tuple[float, ...]
    initial_covariance: tuple[tuple[float, ...], ...]
    configuration_id: str | None = None


@dataclass(frozen=True)
class StudyMethod:
    method: str
    steps: int = 32
    samples: int = 1024
    chunk_size: int = 256
    level_samples: tuple[int, ...] = ()
    proposal: tuple[float, float] = (0.0, 0.0)
    recovery: bool = False
    mixture_settings: MixtureSettings | None = None
    configuration_id: str | None = None
    horizon_steps: tuple[tuple[float, int], ...] = ()


@dataclass(frozen=True)
class StudyFunctional:
    functional_id: str
    kind: str
    normal: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0)
    threshold: float = 0.0
    closed: bool = True
    tolerance: float = 0.01
    target_probability: float | None = None


@dataclass(frozen=True)
class StudyArm:
    arm_id: str
    model_family_id: str
    method_family_id: str
    objective_id: str


@dataclass(frozen=True)
class PropagationStudyDesign:
    study_id: str
    experiment_id: str
    comparison_family: str
    models: tuple[StudyModel, ...]
    methods: tuple[StudyMethod, ...]
    functionals: tuple[StudyFunctional, ...]
    horizons: tuple[float, ...]
    seeds: tuple[int, ...]
    arms: tuple[StudyArm, ...]
    protocol_hash: str
    data_hash: str
    feature_hash: str
    selection_hash: str
    origin: float = 0.0
    history_cutoff: float = 0.0
    stopping_rule: str = "fixed-design-no-auto-expansion"
    primary_metrics: tuple[str, ...] = ("functional-estimate", "sampling-uncertainty", "charged-slot-ms")
    input_cases: tuple[StudyInputCase, ...] = ()
    input_policy: StudyInputPolicy | None = None
    calibrations: tuple[StudyCalibration, ...] = ()


@dataclass(frozen=True)
class PropagationStudyManifest:
    """Canonical bytes only; returned dictionaries never share mutable state."""

    _document: bytes

    @property
    def manifest_hash(self):
        return content_hash(self.manifest())

    def manifest(self):
        return json.loads(self._document)

    def study_spec(self, *, expected_hash):
        """Return the complete unadmitted draft, including non-executable rows.

        Declared refusals use shared PREFLIGHT_FAILED metadata, not workers.
        This draft intentionally has neither runtime_binding nor admission.
        """
        from experiments.pirc25.affine import code_hash
        document = self.manifest()
        if expected_hash != self.manifest_hash or document["code_hash"] != code_hash():
            raise DataValidationError("frozen study content or current source differs")
        return {"schema_version": "pirc25-contract-v1",
                **{key: document[key] for key in ("study_id", "experiment_id", "comparison_family",
                    "code_hash", "protocol_hash", "data_hash", "feature_hash", "selection_hash", "arms")},
                "cells": [row["cell"] for row in document["matrix"]],
                "propagation_design_hash": self.manifest_hash,
                "propagation_design": {key: document[key] for key in
                    ("schema_version", "axis_manifest", "expected_cells", "stopping_rule", "primary_metrics")}}


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _axis(values, kind, name):
    if type(values) is not tuple or not 0 < len(values) <= 64 or any(type(v) is not kind for v in values):
        raise DataValidationError("invalid bounded immutable " + name + " axis")


def _state_vector(value):
    return type(value) is tuple and len(value) == 4 and all(_finite(x) for x in value)


def _method_manifest(method):
    value = asdict(method)
    if method.configuration_id is None:
        del value["configuration_id"]
    if not method.horizon_steps:
        del value["horizon_steps"]
    return value


def _steps(method, horizon):
    # Coverage/order/types have been checked before copying or expansion.
    return next(n for h, n in method.horizon_steps if h == horizon) if method.horizon_steps else method.steps


def _model_manifest(model):
    value = {"family_id": model.family_id, "package": model.package.manifest(),
             "initial_mean": model.initial_mean, "initial_covariance": model.initial_covariance}
    if model.configuration_id is not None:
        value["configuration_id"] = model.configuration_id
    return value


def _functional_manifest(functional):
    value = asdict(functional)
    if functional.target_probability is None:
        del value["target_probability"]
    return value


def _calibration_key(model, functional, horizon, case):
    return (model.family_id, model.configuration_id, model.package.package_hash,
            functional.functional_id, horizon, case.instance_id if case is not None else None)


def _request(design, model, method, functional, horizon, seed, arm_id, case=None, calibration=None):
    initial = case if case is not None else model
    origin, cutoff = (case.origin, case.history_cutoff) if case is not None else (design.origin, design.history_cutoff)
    actual_functional = _functional_manifest(functional)
    actual_functional.pop("target_probability", None)
    if calibration is not None and calibration["status"] == "CALIBRATED":
        actual_functional["threshold"] = calibration["geometry"]["threshold"]
    paired = {"model_package_hash": model.package.package_hash,
              "initial_mean": initial.initial_mean, "initial_covariance": initial.initial_covariance,
              "origin": origin, "history_cutoff": cutoff,
              "horizon": horizon, "functional": actual_functional, "seed": seed}
    if calibration is not None:
        paired["calibration_binding_hash"] = calibration["binding_hash"]
    if case is not None:
        paired["input_binding_hash"] = input_binding(case, design.input_policy)["binding_hash"]
    coupling_id = "paired-" + content_hash(paired)
    request_id = "cell-" + content_hash({"paired": paired, "method": _method_manifest(method), "arm": arm_id})
    return PropagationRequest(request_id, model.package.package_hash, initial.initial_mean, initial.initial_covariance,
        origin, cutoff, (horizon,), functional.kind, seed, coupling_id, arm_id,
        steps=_steps(method, horizon), samples=method.samples, chunk_size=method.chunk_size, normal=functional.normal,
        threshold=actual_functional["threshold"], closed=functional.closed, tolerance=functional.tolerance)


def _calibration_table(design, arms):
    from domain.probability_calibration import halfspace_geometry
    from experiments.pirc25.affine import code_hash
    if (type(design.calibrations) is not tuple or len(design.calibrations) > 10000
            or any(type(entry) is not StudyCalibration for entry in design.calibrations)):
        raise DataValidationError("bounded immutable calibration table required")
    calibrated = tuple(f for f in design.functionals if f.target_probability is not None)
    slots = product(design.models, calibrated, design.horizons, design.input_cases or (None,))
    expected = [(m, f, h, c) for m, f, h, c in slots]
    if len(design.calibrations) != len(expected):
        raise DataValidationError("calibration table must cover exactly all physical geometry slots")
    documents, lookup = [], {}
    current_code = code_hash() if expected else None
    for entry, (model, functional, horizon, case) in zip(design.calibrations, expected):
        # Manifest validates primitive keys before they can enter a hash map.
        value = entry.manifest()
        key = _calibration_key(model, functional, horizon, case)
        if entry.key != key or key in lookup:
            raise DataValidationError("calibration table duplicate, order or slot differs")
        if value["source_pointer"] is not None:
            policy = value["source_pointer"]["policy"]
            if (policy["model_package_hash"] != model.package.package_hash
                    or policy["target_probability"] != functional.target_probability
                    or policy["code_hash"] != current_code):
                raise DataValidationError("calibration source policy/model/probability/code differs")
        if entry.status == "CALIBRATED":
            if type(model.package) is not FrozenDynamicsPackage or entry.consumer_study_id != design.study_id:
                raise DataValidationError("calibration requires affine geometry and exact consumer study")
            request = _request(design, model, design.methods[0], functional, horizon, design.seeds[0],
                arms[(model.family_id, design.methods[0].method, functional.kind)], case, value)
            geometry = halfspace_geometry(model.package, request, causal_input_hash=value["geometry"]["causal_input_hash"])
            if content_hash(geometry) != value["geometry_hash"]:
                raise DataValidationError("calibration geometry differs from actual model/input/event")
        documents.append(value)
        lookup[key] = value
    return documents, lookup


def _disposition(model, method):
    if method.method == "mixture" and method.mixture_settings is None:
        return "INELIGIBLE", "no explicit frozen bounded mixture settings supplied"
    if method.method == "pde":
        return "NOT_IMPLEMENTED", "no versioned qualified implementation in this adapter"
    if isinstance(model.package, FrozenNonlinearPackage) and method.method in {"exact", "gaussian"}:
        return "INELIGIBLE", "affine-only analytic method cannot replace nonlinear dynamics"
    return "PLANNED", "engineering capability only; shared admission and scientific qualification still required"


def _mixture_configuration(model, method, request):
    from experiments.pirc25.affine import code_hash
    from .mixture_plugin import mixture_config
    policy = method.mixture_settings.bind(model.package, request, code_hash())
    return policy, mixture_config(request, policy, synthetic=isinstance(model.package, FrozenNonlinearPackage))


def freeze_design(design):
    """Validate bounds and input/configuration before allocating the full matrix."""
    if type(design) is not PropagationStudyDesign:
        raise DataValidationError("explicit propagation study design is required")
    for name, kind in (("models", StudyModel), ("methods", StudyMethod), ("functionals", StudyFunctional)):
        _axis(getattr(design, name), kind, name)
    for name in ("horizons", "seeds"):
        values = getattr(design, name)
        if type(values) is not tuple or not 0 < len(values) <= 64:
            raise DataValidationError("invalid bounded immutable " + name + " axis")
    if type(design.input_cases) is not tuple or len(design.input_cases) > 64:
        raise DataValidationError("bounded immutable input case axis required")
    count = matrix_cardinality(tuple(len(getattr(design, name)) for name in
                                    ("models", "methods", "functionals", "horizons", "seeds"))
                                + (len(design.input_cases) or 1,))
    # The byte quota is independent of cell count. Conservative per-cell bound
    # includes the repeated package and bounded request/execution/row metadata.
    if any(type(model.package) not in (FrozenDynamicsPackage, FrozenNonlinearPackage) for model in design.models):
        raise DataValidationError("only explicit frozen synthetic packages are supported")
    per_cell_bytes = 12288 if design.input_cases else 8192
    if any(f.target_probability is not None for f in design.functionals):
        # Reserve repeated geometry/pointer/binding bytes plus the axis copy
        # before allocating either the calibration table or the full matrix.
        per_cell_bytes += 2 * MAX_DOCUMENT_BYTES + 2048
    if count * (max(len(model.package._document) for model in design.models) + per_cell_bytes) > MAX_MANIFEST_BYTES:
        raise DataValidationError("design exceeds bounded manifest byte quota before expansion")
    for value in (design.study_id, design.experiment_id, design.comparison_family):
        identifier(value)
    if any(not _hash(getattr(design, key)) for key in
           ("protocol_hash", "data_hash", "feature_hash", "selection_hash")):
        raise DataValidationError("study requires explicit frozen input/protocol hashes")
    validate_cases(design.input_cases, design.input_policy, selection_hash=design.selection_hash)
    if (not _finite(design.origin) or not _finite(design.history_cutoff) or design.history_cutoff > design.origin
            or design.stopping_rule != "fixed-design-no-auto-expansion"
            or type(design.primary_metrics) is not tuple or not 0 < len(design.primary_metrics) <= 16):
        raise DataValidationError("invalid origin/history, fixed stopping rule or primary metrics")
    for metric in design.primary_metrics:
        identifier(metric)
    if (any(not _finite(h) or h <= 0 for h in design.horizons)
            or len(set(design.horizons)) != len(design.horizons)
            or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in design.seeds)
            or len(set(design.seeds)) != len(design.seeds)
            or len(set(design.primary_metrics)) != len(design.primary_metrics)):
        raise DataValidationError("duplicate or invalid horizon/seed/metric axis")
    functional_ids = tuple(f.functional_id for f in design.functionals)
    for value in functional_ids:
        identifier(value)
    if len(set(functional_ids)) != len(functional_ids):
        raise DataValidationError("duplicate family/functional axis")
    configured_models = any(model.configuration_id is not None for model in design.models)
    model_identities = set()
    for model in design.models:
        identifier(model.family_id)
        if configured_models:
            identifier(model.configuration_id)
        identity = (model.family_id, model.configuration_id)
        if identity in model_identities:
            raise DataValidationError("duplicate model family/configuration axis; declare configuration once")
        model_identities.add(identity)
    configured = any(method.configuration_id is not None for method in design.methods)
    identities = set()
    for method in design.methods:
        identifier(method.method)
        if configured:
            identifier(method.configuration_id)
        identity = (method.method, method.configuration_id)
        if identity in identities:
            raise DataValidationError("duplicate method/configuration axis; declare configuration once")
        identities.add(identity)
    # Bound primitive shape/type before any dataclass copying, package evaluation
    # or numeric allocation. PSD/symmetry and per-method work checks still run
    # below; these are the existing request constraints, not new numeric policy.
    for model in design.models:
        if (not _state_vector(model.initial_mean) or type(model.initial_covariance) is not tuple
                or len(model.initial_covariance) != 4
                or not all(_state_vector(row) for row in model.initial_covariance)):
            raise DataValidationError("invalid bounded immutable initial distribution")
    for functional in design.functionals:
        if (type(functional.kind) is not str or functional.kind not in {"endpoint-x", "endpoint-halfspace"}
                or not _state_vector(functional.normal) or not any(x != 0 for x in functional.normal)
                or not _finite(functional.threshold) or type(functional.closed) is not bool
                or not _finite(functional.tolerance) or functional.tolerance <= 0):
            raise DataValidationError("invalid bounded endpoint functional")
        if functional.target_probability is not None and (not _finite(functional.target_probability)
                or not 0 < functional.target_probability < 1 or functional.kind != "endpoint-halfspace"
                or functional.normal[2:] != (0., 0.) or functional.threshold != 0):
            raise DataValidationError("calibrated probability requires a zero-template spatial halfspace")
    for method in design.methods:
        if (method.method not in METHODS or type(method.recovery) is not bool
                or type(method.steps) is not int or not 1 <= method.steps <= 8192
                or type(method.samples) is not int or not 2 <= method.samples <= 1_000_000
                or type(method.chunk_size) is not int or not 1 <= method.chunk_size <= 256
                or type(method.level_samples) is not tuple or type(method.proposal) is not tuple
                or len(method.proposal) != 2 or not all(_finite(x) for x in method.proposal)):
            raise DataValidationError("unknown or mutable method configuration")
        if (type(method.horizon_steps) is not tuple or len(method.horizon_steps) > 64
                or any(type(pair) is not tuple or len(pair) != 2
                    or not _finite(pair[0]) or pair[0] <= 0
                    or type(pair[1]) is not int or not 1 <= pair[1] <= 8192
                    for pair in method.horizon_steps)):
            raise DataValidationError("invalid bounded immutable horizon step table")
        if method.horizon_steps and (tuple(h for h, _ in method.horizon_steps) != design.horizons
                or method.horizon_steps[0][1] != method.steps):
            raise DataValidationError("horizon step table must cover the ordered horizon axis and bind first steps")
        if (method.mixture_settings is not None
                and (type(method.mixture_settings) is not MixtureSettings or method.method != "mixture" or not method.recovery)):
            raise DataValidationError("mixture requires explicit immutable settings and chunk adapter")
        if method.mixture_settings is not None:
            method.mixture_settings.validate()
        if method.recovery and method.method not in {"euler", "heun", "reversible-heun", "mlmc", "importance", "mixture"}:
            raise DataValidationError("method has no declared chunk continuation")
        if method.method == "mlmc":
            if (not 1 <= len(method.level_samples) <= 9
                    or any(type(n) is not int or not 2 <= n <= 1_000_000 for n in method.level_samples)
                    or sum(method.level_samples) != method.samples):
                raise DataValidationError("invalid fixed MLMC allocation")
        elif method.level_samples:
            raise DataValidationError("MLMC allocation supplied to another method")
        if method.method != "importance" and method.proposal != (0.0, 0.0):
            raise DataValidationError("proposal supplied to another method")
    if type(design.arms) is not tuple or not 0 < len(design.arms) <= 4096 or any(type(a) is not StudyArm for a in design.arms):
        raise DataValidationError("explicit existing immutable arm family bindings are required")
    arms = {}
    for arm in design.arms:
        for field in ("arm_id", "model_family_id", "method_family_id", "objective_id"):
            identifier(getattr(arm, field))
        family = (arm.model_family_id, arm.method_family_id, arm.objective_id)
        if family in arms or arm.arm_id in arms.values():
            raise DataValidationError("duplicate budget identity or family")
        arms[family] = arm.arm_id
    expected_families = {(m.family_id, method.method, f.kind) for m in design.models
                         for method in design.methods for f in design.functionals}
    if set(arms) != expected_families:
        raise DataValidationError("arm bindings must cover exactly the model/method/objective families")
    calibration_documents, calibration_lookup = _calibration_table(design, arms)
    calibration_table_hash = content_hash(calibration_documents) if calibration_documents else None
    # Different labels cannot disguise duplicate numerical configurations as
    # independent cells. All primitive and mixture settings checks precede copy.
    numerical_configs = []
    for method in design.methods:
        numerical = _method_manifest(method)
        numerical.pop("configuration_id", None)
        numerical.pop("horizon_steps", None)
        # Compare actual per-horizon configurations, not table labels or an
        # implicit-versus-explicit representation of the same constant grid.
        numerical["steps"] = tuple(_steps(method, h) for h in design.horizons)
        # These kernels propagate moments or a bounded deterministic point/
        # component set, not request.samples paths. Retain the original field
        # in requests/resource bindings, but it cannot create another numerical
        # stratum for the same deterministic algorithm. No manifest is rewritten.
        if method.method in {"exact", "gaussian", "cubature", "mixture"}:
            numerical.pop("samples")
        # All fields are already bounded and typed. Numeric equality also
        # identifies int/float zero and signed-zero aliases, unlike JSON hashes.
        if numerical in numerical_configs:
            raise DataValidationError("duplicate numerical configuration under different labels")
        numerical_configs.append(numerical)
    # Validate each model and each small configuration combination first. No
    # estimator is evaluated, and no horizon/seed matrix is materialized here.
    from inference.affine_oracle import _covariance
    import numpy as np
    plugins = {}
    model_configs = []
    for case in design.input_cases:
        covariance = _covariance(case.initial_covariance, "frozen case covariance").numpy()
        if np.linalg.eigvalsh(covariance).min() < 0:
            raise DataValidationError("input covariance requires projection; design refuses it")
    for model in design.models:
        validator = validate_nonlinear_input if isinstance(model.package, FrozenNonlinearPackage) else validate_oracle_input
        validator(model.package, expected_package_hash=model.package.package_hash)
        covariance = _covariance(model.initial_covariance, "frozen initial covariance").numpy()
        if np.linalg.eigvalsh(covariance).min() < 0:
            raise DataValidationError("initial covariance requires projection; design refuses it")
        # Labels are comparison strata, not extra independent samples. Compare
        # bounded validated inputs numerically so initial scalar aliases cannot
        # disguise repeated configurations of the exact same frozen package.
        model_config = (model.family_id, model.package.package_hash,
                        None if design.input_cases else model.initial_mean,
                        None if design.input_cases else model.initial_covariance)
        if model_config in model_configs:
            raise DataValidationError("duplicate model configuration under different labels")
        model_configs.append(model_config)
        for method, functional, horizon, case in product(design.methods, design.functionals,
                                                       design.horizons, design.input_cases or (None,)):
            arm_id = arms[(model.family_id, method.method, functional.kind)]
            calibration = calibration_lookup.get(_calibration_key(model, functional, horizon, case))
            request = _request(design, model, method, functional, horizon, design.seeds[0], arm_id, case, calibration)
            request.validate()
            if method.method == "importance" and functional.kind != "endpoint-halfspace":
                raise DataValidationError("rare-event proposal requires an endpoint probability objective")
            if (_disposition(model, method)[0] != "PLANNED"
                    or calibration is not None and calibration["status"] != "CALIBRATED"):
                continue
            synthetic = isinstance(model.package, FrozenNonlinearPackage)
            key = (synthetic, method.recovery, method.method == "mixture", method.method == "cubature")
            if key not in plugins:
                if method.method == "mixture":
                    from .mixture_plugin import mixture_plugin
                    plugins[key] = mixture_plugin(synthetic=synthetic)
                elif method.method == "cubature" and not synthetic:
                    from .cubature_plugin import cubature_plugin
                    plugins[key] = cubature_plugin()
                else:
                    plugins[key] = propagation_plugin(synthetic=synthetic, recovery=method.recovery)
            # Even restart-only affine work is bounded in the preparation layer;
            # total work must not be hidden behind a temporal-grid-only count.
            if method.method == "mixture":
                _, config = _mixture_configuration(model, method, request)
            elif method.method == "cubature" and not synthetic:
                from .cubature_plugin import cubature_config
                config = cubature_config(request)
            else:
                config = execution_config(request, method.method, level_samples=method.level_samples,
                                          proposal=method.proposal, recovery=method.recovery, synthetic=synthetic)
            total = (sum(n * (request.steps * 2**level + (request.steps * 2**(level-1) if level else 0))
                         for level, n in enumerate(method.level_samples)) if method.method == "mlmc"
                     else config["work_steps"] if method.method == "mixture"
                     else 8*request.steps if method.method == "cubature"
                     else request.steps if method.method in {"exact", "gaussian"} else method.samples*request.steps)
            if config["steps"] > 8192 or total > 1_000_000:
                raise DataValidationError("frozen method grid or total work exceeds quota")
            execution_binding(plugins[key].registry_entry, config, execution_inputs(request), matrix_cells=count)
    from experiments.pirc25.affine import code_hash
    rows = []
    for model, method, functional, horizon, seed, case in product(design.models, design.methods, design.functionals,
                                                          design.horizons, design.seeds, design.input_cases or (None,)):
        arm_id = arms[(model.family_id, method.method, functional.kind)]
        calibration = calibration_lookup.get(_calibration_key(model, functional, horizon, case))
        request = _request(design, model, method, functional, horizon, seed, arm_id, case, calibration)
        disposition, reason = _disposition(model, method)
        if calibration is not None and calibration["status"] != "CALIBRATED" and disposition == "PLANNED":
            disposition, reason = "INELIGIBLE", "calibration " + calibration["status"] + ": " + calibration["reason"]
        if case is not None and case.source_kind == "private-past-prefix" and disposition == "PLANNED":
            disposition, reason = "INELIGIBLE", "no admitted past-only source initialization adapter; source declaration is not read authority"
        row = {"row_id": request.request_id, "model_family_id": model.family_id, "method": method.method,
               "functional_id": functional.functional_id, "horizon": horizon, "seed": seed,
               "arm_id": arm_id, "disposition": disposition, "reason": reason,
               "model_package_hash": model.package.package_hash, "request_hash": request.request_hash}
        dimensions = {"functional_id": functional.functional_id, "functional_kind": functional.kind,
                      "functional_version": request.functional_version,
                      "region": {"coordinate_system": request.region_coordinate_system,
                                 "normal": functional.normal, "threshold": request.threshold,
                                 "closed": functional.closed},
                      "initialization": content_hash({"mean": model.initial_mean, "covariance": model.initial_covariance}),
                      "prediction_origin": design.origin}
        if calibration is not None:
            # Keep different source blocks in the same frozen probability-rule
            # stratum. Literal per-block thresholds remain in requests/bindings.
            dimensions["region"] = {"coordinate_system": request.region_coordinate_system,
                "normal": functional.normal, "closed": functional.closed,
                "target_probability": functional.target_probability, "calibration_table_hash": calibration_table_hash}
        if configured:
            row["configuration_id"] = method.configuration_id
            dimensions["numerical_configuration"] = method.configuration_id
        if configured_models:
            row["model_configuration_id"] = model.configuration_id
            dimensions["model_configuration"] = model.configuration_id
        if case is not None:
            dimensions.update(input_policy=design.input_policy.policy_hash,
                initialization={"policy_hash": design.input_policy.initialization_policy_hash},
                prediction_origin={"policy_hash": design.input_policy.origin_policy_hash},
                input_population={"dataset_id": case.dataset_id, "release_id": case.release_id,
                    "source_kind": case.source_kind, "split_role": case.split_role})
        row["cell"] = {"arm_id": arm_id, "block_id": case.block_id if case is not None else model.family_id,
                       "seed": seed, "horizon": horizon,
                       "visibility": "restricted" if case is not None and case.source_kind == "private-past-prefix" else "synthetic",
                       "frozen_dynamics": model.package.manifest(),
                       "propagation_request": json.loads(_encode(asdict(request))), "resource_class": "cpu",
                       "functional_id": functional.functional_id, "comparison_dimensions": dimensions}
        if calibration is not None:
            row["cell"]["calibration_binding"] = calibration
        if case is not None:
            row["input_case_id"] = case.instance_id
            row["cell"].update(instance_id=case.instance_id, input_binding=input_binding(case, design.input_policy))
        if disposition == "PLANNED":
            synthetic = isinstance(model.package, FrozenNonlinearPackage)
            plugin = plugins[(synthetic, method.recovery, method.method == "mixture", method.method == "cubature")]
            if method.method == "mixture":
                policy, config = _mixture_configuration(model, method, request)
                row["cell"]["mixture_policy"] = policy.manifest()
            elif method.method == "cubature" and not synthetic:
                from .cubature_plugin import cubature_config
                config = cubature_config(request)
            else:
                config = execution_config(request, method.method, level_samples=method.level_samples,
                                          proposal=method.proposal, recovery=method.recovery, synthetic=synthetic)
            row["cell"].update({"plugin_id": plugin.plugin_id, "capability": {"exact": "exact-transition", "mlmc": "coupled-level",
                    "importance": "rare-event"}.get(method.method, "generic-rollout"),
                "execution": execution_binding(plugin.registry_entry, config, execution_inputs(request), matrix_cells=count)})
        else:
            row["cell"].update({"plugin_id": "propagation-declared-unavailable", "capability": "generic-rollout",
                "execution_disposition": {"schema_version": "pirc25-execution-disposition-v1",
                                          "status": disposition, "reason": reason}})
        rows.append(row)
    document = {"schema_version": "propagation-study-manifest-v1", "study_id": design.study_id,
        "experiment_id": design.experiment_id, "comparison_family": design.comparison_family,
        "code_hash": code_hash(), **{key: getattr(design, key) for key in
            ("protocol_hash", "data_hash", "feature_hash", "selection_hash", "stopping_rule", "primary_metrics")},
        "qualification": "preparation-only-not-scientific", "expected_cells": count,
        "axis_manifest": {"models": [_model_manifest(model) for model in design.models],
            "methods": [_method_manifest(method) for method in design.methods], "functionals": [_functional_manifest(f) for f in design.functionals],
            "horizons": design.horizons, "seeds": design.seeds, "origin": design.origin, "history_cutoff": design.history_cutoff,
            "state_names": ("x", "y", "vx", "vy"), "time_unit": "s", "coordinate_system": "local-cartesian"},
        "arms": [{**asdict(arm), "budget_seconds": 86400} for arm in design.arms], "matrix": rows}
    if calibration_documents:
        document["axis_manifest"]["calibrations"] = calibration_documents
    if design.input_cases:
        document["axis_manifest"].update(input_cases=[case.manifest() for case in design.input_cases],
            input_policy=design.input_policy.manifest())
    encoded = _encode(document)
    if len(encoded) > MAX_MANIFEST_BYTES:
        raise DataValidationError("compiled manifest exceeds byte quota")
    return PropagationStudyManifest(encoded)
