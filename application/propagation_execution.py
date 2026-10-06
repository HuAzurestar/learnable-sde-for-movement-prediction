"""Shared-result adaptation for content-bound affine endpoint methods."""

from __future__ import annotations

from domain.frozen_dynamics import FrozenDynamicsPackage
from domain.propagation import PropagationRequest
from infrastructure.research_store import ResearchError, digest

from .propagation_inputs import validate_oracle_input
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


def execute_propagation(spec, cell):
    from experiments.pirc25.affine import code_hash
    from experiments.pirc27.plugin import propagation_plugin, execution_config, execution_inputs
    from inference.propagation_methods import (analytic_estimate, monte_carlo, mlmc_estimate, importance_sampling)
    if spec.get("code_hash") != code_hash() or cell.get("visibility") != "synthetic":
        raise ResearchError("CONTRACT_MISMATCH", "only the frozen synthetic source is supported by this adapter")
    plugin = propagation_plugin()
    execution_plan(spec, cell, plugin)
    request = request_from_manifest(cell["propagation_request"])
    package = FrozenDynamicsPackage.from_manifest(cell["frozen_dynamics"], expected_hash=request.model_package_hash)
    validate_oracle_input(package, expected_package_hash=request.model_package_hash)
    config = cell["execution"]["config"]
    expected = execution_config(request, config["method"], level_samples=tuple(config["level_samples"]), proposal=tuple(config["proposal"]))
    if config != expected or cell["execution"]["inputs"] != execution_inputs(request) or request.arm_id != cell["arm_id"] or request.seed != cell["seed"]:
        raise ResearchError("CONTRACT_MISMATCH", "registered request, method/resource configuration or arm differs")
    required_capability = {"exact": "exact-transition", "gaussian": "generic-rollout", "euler": "generic-rollout",
                           "heun": "generic-rollout", "mlmc": "coupled-level", "importance": "rare-event"}[config["method"]]
    if cell["capability"] != required_capability:
        raise ResearchError("CONTRACT_MISMATCH", "method differs from the cell's registered capability")
    methods = {"exact": lambda: analytic_estimate(package, request),
        "gaussian": lambda: analytic_estimate(package, request, discrete=True),
        "euler": lambda: monte_carlo(package, request),
        "heun": lambda: monte_carlo(package, request, solver="additive-heun"),
        "mlmc": lambda: mlmc_estimate(package, request, level_samples=tuple(config["level_samples"])),
        "importance": lambda: importance_sampling(package, request, proposal=tuple(config["proposal"]))}
    result = methods[config["method"]]()
    reference = analytic_estimate(package, request).estimate
    unit = "m" if request.functional == "endpoint-x" else "1"
    output = {"metrics": {"absolute_error_vs_float64_reference": abs(result.estimate-reference)},
        "forecast": {"kind": result.kind, "horizons": list(request.horizons), "functional": result.manifest(),
                     "model_package_hash": package.package_hash, "request_hash": request.request_hash},
        "fit": {"training": "none; frozen synthetic generator"}, "source_schema": "endpoint-propagation-v1"}
    # Computational completion keeps low-ESS/unresolved estimates as artifacts;
    # estimator status remains visible and cannot be scientific qualification.
    return {"schema_version": "pirc25-result-v1", "status": "SUCCEEDED", "qualification": "fixture",
            "spec_hash": digest(spec), "cell_hash": digest(cell), "protocol_hash": spec["protocol_hash"],
            "state_order": list(plugin.state_order), "units": list(plugin.units), "time_unit": "s",
            "resume_level": "restart-only", "input_hash": spec["data_hash"], "output_hash": digest(output),
            "metric_units": {"absolute_error_vs_float64_reference": unit}, **output}
