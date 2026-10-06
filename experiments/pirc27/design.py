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
from infrastructure.research_store import identifier

from .oracles import matrix_cardinality
from .plugin import propagation_plugin, execution_config, execution_inputs


METHODS = frozenset({"exact", "gaussian", "euler", "heun", "mlmc", "importance",
                    "cubature", "mixture", "reversible-heun", "pde"})
MAX_MANIFEST_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class StudyModel:
    family_id: str
    package: FrozenDynamicsPackage | FrozenNonlinearPackage
    initial_mean: tuple[float, ...]
    initial_covariance: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class StudyMethod:
    method: str
    steps: int = 32
    samples: int = 1024
    chunk_size: int = 256
    level_samples: tuple[int, ...] = ()
    proposal: tuple[float, float] = (0.0, 0.0)
    recovery: bool = False


@dataclass(frozen=True)
class StudyFunctional:
    functional_id: str
    kind: str
    normal: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0)
    threshold: float = 0.0
    closed: bool = True
    tolerance: float = 0.01


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


def _request(design, model, method, functional, horizon, seed, arm_id):
    paired = {"model_package_hash": model.package.package_hash,
              "initial_mean": model.initial_mean, "initial_covariance": model.initial_covariance,
              "origin": design.origin, "history_cutoff": design.history_cutoff,
              "horizon": horizon, "functional": asdict(functional), "seed": seed}
    coupling_id = "paired-" + content_hash(paired)
    request_id = "cell-" + content_hash({"paired": paired, "method": asdict(method), "arm": arm_id})
    return PropagationRequest(request_id, model.package.package_hash, model.initial_mean, model.initial_covariance,
        design.origin, design.history_cutoff, (horizon,), functional.kind, seed, coupling_id, arm_id,
        steps=method.steps, samples=method.samples, chunk_size=method.chunk_size, normal=functional.normal,
        threshold=functional.threshold, closed=functional.closed, tolerance=functional.tolerance)


def _disposition(model, method):
    if method.method in {"mixture", "reversible-heun", "pde"}:
        return "NOT_IMPLEMENTED", "no versioned qualified implementation in this adapter"
    if isinstance(model.package, FrozenNonlinearPackage) and method.method in {"exact", "gaussian"}:
        return "INELIGIBLE", "affine-only analytic method cannot replace nonlinear dynamics"
    if isinstance(model.package, FrozenDynamicsPackage) and method.method == "cubature":
        return "INELIGIBLE", "cubature adapter currently declares synthetic nonlinear input only"
    return "PLANNED", "engineering capability only; shared admission and scientific qualification still required"


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
    count = matrix_cardinality(tuple(len(getattr(design, name)) for name in
                                    ("models", "methods", "functionals", "horizons", "seeds")))
    # The byte quota is independent of cell count. Conservative per-cell bound
    # includes the repeated package and bounded request/execution/row metadata.
    if any(type(model.package) not in (FrozenDynamicsPackage, FrozenNonlinearPackage) for model in design.models):
        raise DataValidationError("only explicit frozen synthetic packages are supported")
    if count * (max(len(model.package._document) for model in design.models) + 8192) > MAX_MANIFEST_BYTES:
        raise DataValidationError("design exceeds bounded manifest byte quota before expansion")
    for value in (design.study_id, design.experiment_id, design.comparison_family):
        identifier(value)
    if any(not _hash(getattr(design, key)) for key in
           ("protocol_hash", "data_hash", "feature_hash", "selection_hash")):
        raise DataValidationError("study requires explicit frozen input/protocol hashes")
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
    for values in ((m.family_id for m in design.models), (m.method for m in design.methods),
                   (f.functional_id for f in design.functionals)):
        values = tuple(values)
        for value in values:
            identifier(value)
        if len(set(values)) != len(values):
            raise DataValidationError("duplicate family/method/functional axis; declare configuration once")
    for method in design.methods:
        if (method.method not in METHODS or type(method.recovery) is not bool
                or type(method.level_samples) is not tuple or type(method.proposal) is not tuple
                or len(method.proposal) != 2 or not all(_finite(x) for x in method.proposal)):
            raise DataValidationError("unknown or mutable method configuration")
        if method.recovery and method.method not in {"euler", "heun", "mlmc", "importance"}:
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
        for value in asdict(arm).values():
            identifier(value)
        family = (arm.model_family_id, arm.method_family_id, arm.objective_id)
        if family in arms or arm.arm_id in arms.values():
            raise DataValidationError("duplicate budget identity or family")
        arms[family] = arm.arm_id
    expected_families = {(m.family_id, method.method, f.kind) for m in design.models
                         for method in design.methods for f in design.functionals}
    if set(arms) != expected_families:
        raise DataValidationError("arm bindings must cover exactly the model/method/objective families")
    # Validate each model and each small configuration combination first. No
    # estimator is evaluated, and no horizon/seed matrix is materialized here.
    from inference.affine_oracle import _covariance
    import numpy as np
    plugins = {}
    for model in design.models:
        validator = validate_nonlinear_input if isinstance(model.package, FrozenNonlinearPackage) else validate_oracle_input
        validator(model.package, expected_package_hash=model.package.package_hash)
        covariance = _covariance(model.initial_covariance, "frozen initial covariance").numpy()
        if np.linalg.eigvalsh(covariance).min() < 0:
            raise DataValidationError("initial covariance requires projection; design refuses it")
        for method, functional in product(design.methods, design.functionals):
            arm_id = arms[(model.family_id, method.method, functional.kind)]
            request = _request(design, model, method, functional, design.horizons[0], design.seeds[0], arm_id)
            request.validate()
            if method.method == "importance" and functional.kind != "endpoint-halfspace":
                raise DataValidationError("rare-event proposal requires an endpoint probability objective")
            if _disposition(model, method)[0] != "PLANNED":
                continue
            synthetic = isinstance(model.package, FrozenNonlinearPackage)
            key = (synthetic, method.recovery)
            if key not in plugins:
                plugins[key] = propagation_plugin(synthetic=synthetic, recovery=method.recovery)
            # Even restart-only affine work is bounded in the preparation layer;
            # total work must not be hidden behind a temporal-grid-only count.
            config = execution_config(request, method.method, level_samples=method.level_samples,
                                      proposal=method.proposal, recovery=method.recovery, synthetic=synthetic)
            total = (sum(n * (method.steps * 2**level + (method.steps * 2**(level-1) if level else 0))
                         for level, n in enumerate(method.level_samples)) if method.method == "mlmc"
                     else 8*method.steps if method.method == "cubature"
                     else method.steps if method.method in {"exact", "gaussian"} else method.samples*method.steps)
            if config["steps"] > 8192 or total > 1_000_000:
                raise DataValidationError("frozen method grid or total work exceeds quota")
            execution_binding(plugins[key].registry_entry, config, execution_inputs(request), matrix_cells=count)
    from experiments.pirc25.affine import code_hash
    rows = []
    for model, method, functional, horizon, seed in product(design.models, design.methods, design.functionals,
                                                          design.horizons, design.seeds):
        arm_id = arms[(model.family_id, method.method, functional.kind)]
        request = _request(design, model, method, functional, horizon, seed, arm_id)
        disposition, reason = _disposition(model, method)
        row = {"row_id": request.request_id, "model_family_id": model.family_id, "method": method.method,
               "functional_id": functional.functional_id, "horizon": horizon, "seed": seed,
               "arm_id": arm_id, "disposition": disposition, "reason": reason,
               "model_package_hash": model.package.package_hash, "request_hash": request.request_hash}
        dimensions = {"functional_id": functional.functional_id, "functional_kind": functional.kind,
                      "functional_version": request.functional_version,
                      "region": {"coordinate_system": request.region_coordinate_system,
                                 "normal": functional.normal, "threshold": functional.threshold,
                                 "closed": functional.closed},
                      "initialization": content_hash({"mean": model.initial_mean, "covariance": model.initial_covariance}),
                      "prediction_origin": design.origin}
        row["cell"] = {"arm_id": arm_id, "block_id": model.family_id, "seed": seed, "horizon": horizon,
                       "visibility": "synthetic", "frozen_dynamics": model.package.manifest(),
                       "propagation_request": json.loads(_encode(asdict(request))), "resource_class": "cpu",
                       "functional_id": functional.functional_id, "comparison_dimensions": dimensions}
        if disposition == "PLANNED":
            synthetic = isinstance(model.package, FrozenNonlinearPackage)
            plugin = plugins[(synthetic, method.recovery)]
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
        "axis_manifest": {"models": [{"family_id": model.family_id, "package": model.package.manifest(),
            "initial_mean": model.initial_mean, "initial_covariance": model.initial_covariance} for model in design.models],
            "methods": [asdict(method) for method in design.methods], "functionals": [asdict(f) for f in design.functionals],
            "horizons": design.horizons, "seeds": design.seeds, "origin": design.origin, "history_cutoff": design.history_cutoff,
            "state_names": ("x", "y", "vx", "vy"), "time_unit": "s", "coordinate_system": "local-cartesian"},
        "arms": [{**asdict(arm), "budget_seconds": 86400} for arm in design.arms], "matrix": rows}
    encoded = _encode(document)
    if len(encoded) > MAX_MANIFEST_BYTES:
        raise DataValidationError("compiled manifest exceeds byte quota")
    return PropagationStudyManifest(encoded)
