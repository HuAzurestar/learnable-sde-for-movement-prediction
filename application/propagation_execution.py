"""Shared-result adaptation for content-bound affine endpoint methods."""

from __future__ import annotations

from domain.frozen_dynamics import FrozenDynamicsPackage
from domain.nonlinear_dynamics import FrozenNonlinearPackage
from domain.errors import DataValidationError
from domain.propagation import PropagationRequest
from infrastructure.research_store import ResearchError, digest

from .propagation_inputs import validate_oracle_input, validate_nonlinear_input
from .research_execution import execution_plan


def request_from_manifest(document):
    if type(document) is not dict:
        raise ResearchError("CONTRACT_MISMATCH", "registered propagation request is required")
    document = dict(document)
    try:
        for name in ("initial_mean", "horizons", "normal"):
            if type(document[name]) is not list:
                raise ValueError("request arrays must be JSON lists")
            document[name] = tuple(document[name])
        covariance = document["initial_covariance"]
        if type(covariance) is not list or any(type(row) is not list for row in covariance):
            raise ValueError("initial covariance must be a JSON matrix")
        document["initial_covariance"] = tuple(tuple(row) for row in covariance)
        request = PropagationRequest(**document)
        request.validate()
        return request
    except (KeyError, TypeError, ValueError) as exc:
        raise ResearchError("CONTRACT_MISMATCH", "invalid registered propagation request") from exc


def validate_propagation_cell(spec, cell):
    """Validate method/request accounting before launch and again in the worker."""
    from experiments.pirc25.affine import code_hash
    from experiments.pirc27.plugin import propagation_plugin, execution_config, execution_inputs
    if spec.get("code_hash") != code_hash() or cell.get("visibility") != "synthetic":
        raise ResearchError("CONTRACT_MISMATCH", "only the frozen synthetic source is supported by this adapter")
    qualification = cell["plugin_id"] == "affine-propagation-qualification"
    if cell["plugin_id"] not in {"affine-propagation", "affine-propagation-chunk", "synthetic-propagation", "synthetic-propagation-chunk", "affine-propagation-qualification"}:
        raise ResearchError("CONTRACT_MISMATCH", "unknown explicit propagation adapter")
    recovery = cell["plugin_id"].endswith("-chunk")
    synthetic = cell["plugin_id"].startswith("synthetic-")
    plugin = propagation_plugin(recovery=recovery, synthetic=synthetic)
    if qualification:
        from experiments.pirc27.qualification_plugin import qualification_plugin
        plugin = qualification_plugin()
    execution_plan(spec, cell, plugin)
    request = request_from_manifest(cell["propagation_request"])
    package_type = FrozenNonlinearPackage if synthetic else FrozenDynamicsPackage
    try:
        package = package_type.from_manifest(cell["frozen_dynamics"], expected_hash=request.model_package_hash)
        (validate_nonlinear_input if synthetic else validate_oracle_input)(package, expected_package_hash=request.model_package_hash)
    except DataValidationError as exc:
        raise ResearchError("CONTRACT_MISMATCH", "frozen propagation package schema/content/code differs") from exc
    config = cell["execution"]["config"]
    expected = execution_config(request, config["method"], level_samples=tuple(config["level_samples"]), proposal=tuple(config["proposal"]), recovery=recovery, synthetic=synthetic)
    if qualification:
        from experiments.pirc27.qualification_plugin import qualification_policy, qualification_config
        policy = qualification_policy(spec, cell, package, request, config)
        expected = qualification_config(request, config["method"], policy)
    if config != expected or cell["execution"]["inputs"] != execution_inputs(request) or request.arm_id != cell["arm_id"] or request.seed != cell["seed"]:
        raise ResearchError("CONTRACT_MISMATCH", "registered request, method/resource configuration or arm differs")
    required_capability = {"exact": "exact-transition", "gaussian": "generic-rollout", "euler": "generic-rollout",
                           "heun": "generic-rollout", "reversible-heun": "generic-rollout", "mlmc": "coupled-level", "mlmc-pilot": "coupled-level", "importance": "rare-event", "cubature": "generic-rollout"}[config["method"]]
    if cell["capability"] != required_capability:
        raise ResearchError("CONTRACT_MISMATCH", "method differs from the cell's registered capability")
    if config["method"] == "mlmc-pilot":
        pilot_policy(spec, cell, request, config)
    return package, request, config, plugin


def pilot_policy(spec, cell, request, config):
    from domain.mlmc_pilot import MLMCPilotPolicy
    try:
        policy = MLMCPilotPolicy(**cell["mlmc_pilot_policy"])
        policy.validate(request, tuple(config["level_samples"]))
        arms = [arm for arm in spec["arms"] if arm["arm_id"] == request.arm_id]
        if (len(arms) != 1 or arms[0]["method_family_id"] != "mlmc" or cell.get("execution_role") != "pilot"
                or (spec.get("admission") or {}).get("mode") != "pilot"):
            raise ValueError("pilot must retain the original MLMC arm and explicit pilot role/mode")
        return policy
    except (KeyError, TypeError, ValueError, DataValidationError) as exc:
        raise ResearchError("UNQUALIFIED", "frozen independent pilot policy/role/shared MLMC arm is required") from exc


def execute_propagation(spec, cell, *, resume_state=None, checkpoint=None):
    from inference.propagation_methods import (analytic_estimate, monte_carlo, mlmc_estimate, importance_sampling)
    from inference.nonlinear_propagation import cubature_estimate
    package, request, config, plugin = validate_propagation_cell(spec, cell)
    recovery = plugin.resume_level == "chunk"
    if not recovery and (resume_state is not None or checkpoint is not None):
        raise ResearchError("CONTRACT_MISMATCH", "restart-only methods cannot accept chunk state")
    continuation = {"resume_state": resume_state, "checkpoint": checkpoint}
    methods = {"exact": lambda: analytic_estimate(package, request),
        "gaussian": lambda: analytic_estimate(package, request, discrete=True),
        "euler": lambda: monte_carlo(package, request, **continuation),
        "heun": lambda: monte_carlo(package, request, solver="additive-heun", **continuation),
        "reversible-heun": lambda: monte_carlo(package, request, solver="reversible-heun", **continuation),
        "mlmc": lambda: mlmc_estimate(package, request, level_samples=tuple(config["level_samples"]), **continuation),
        "mlmc-pilot": lambda: mlmc_estimate(package, request, level_samples=tuple(config["level_samples"]), phase=2, pilot=True, **continuation),
        "importance": lambda: importance_sampling(package, request, proposal=tuple(config["proposal"]), **continuation),
        "cubature": lambda: cubature_estimate(package, request)}
    result = methods[config["method"]]()
    synthetic = isinstance(package, FrozenNonlinearPackage)
    metrics = ({"functional_estimate": result.estimate} if synthetic else
               {"absolute_error_vs_float64_reference": abs(result.estimate-analytic_estimate(package, request).estimate)})
    unit = "m" if request.functional == "endpoint-x" else "1"
    output = {"metrics": metrics,
        "forecast": {"kind": result.kind, "horizons": list(request.horizons), "functional": result.manifest(),
                     "model_package_hash": package.package_hash, "request_hash": request.request_hash},
        "fit": {"training": "none; frozen synthetic generator"},
        "source_schema": "synthetic-endpoint-propagation-v1" if synthetic else "endpoint-propagation-v1"}
    if config["method"] == "mlmc-pilot":
        from inference.mlmc_pilot import analyze_mlmc_pilot
        output["forecast"]["pilot_analysis"] = analyze_mlmc_pilot(request, result, pilot_policy(spec, cell, request, config))
    if plugin.plugin_id == "affine-propagation-qualification":
        from experiments.pirc27.qualification_plugin import qualification_policy
        from inference.affine_qualification import analyze_affine_qualification
        policy = qualification_policy(spec, cell, package, request, config)
        output["forecast"]["qualification_analysis"] = analyze_affine_qualification(package, request,
            config["method"], policy, result)
    # Computational completion keeps low-ESS/unresolved estimates as artifacts;
    # estimator status remains visible and cannot be scientific qualification.
    return {"schema_version": "pirc25-result-v1", "status": "SUCCEEDED", "qualification": "fixture",
            "spec_hash": digest(spec), "cell_hash": digest(cell), "protocol_hash": spec["protocol_hash"],
            "state_order": list(plugin.state_order), "units": list(plugin.units), "time_unit": "s",
            "resume_level": plugin.resume_level, "input_hash": spec["data_hash"], "output_hash": digest(output),
            "metric_units": {name: unit for name in metrics}, **output}
