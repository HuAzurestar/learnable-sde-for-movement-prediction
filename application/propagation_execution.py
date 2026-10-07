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
    mlmc_reference = cell["plugin_id"] == "affine-mlmc-qualification-chunk"
    mlmc_production = cell["plugin_id"] == "affine-mlmc-production-chunk"
    mixture_qualification = cell["plugin_id"] == "affine-mixture-qualification"
    mixture_production = cell["plugin_id"] == "affine-mixture-production-chunk"
    path_qualification = cell["plugin_id"] == "affine-path-qualification"
    path_production = cell["plugin_id"] == "affine-path-production-chunk"
    from experiments.pirc27.mixture_plugin import PLUGIN_IDS, mixture_plugin, mixture_policy, mixture_config
    mixture = cell["plugin_id"] in PLUGIN_IDS
    if cell["plugin_id"] not in {"affine-propagation", "affine-propagation-chunk", "synthetic-propagation", "synthetic-propagation-chunk", "affine-propagation-qualification", "affine-mlmc-qualification-chunk", "affine-mlmc-production-chunk", "affine-mixture-qualification", "affine-mixture-production-chunk", "affine-path-qualification", "affine-path-production-chunk"}|PLUGIN_IDS:
        raise ResearchError("CONTRACT_MISMATCH", "unknown explicit propagation adapter")
    recovery = cell["plugin_id"].endswith("-chunk")
    synthetic = cell["plugin_id"].startswith("synthetic-")
    plugin = propagation_plugin(recovery=recovery, synthetic=synthetic)
    if qualification:
        from experiments.pirc27.qualification_plugin import qualification_plugin
        plugin = qualification_plugin()
    if mlmc_reference:
        from experiments.pirc27.mlmc_qualification_plugin import mlmc_qualification_plugin
        plugin = mlmc_qualification_plugin()
    if mlmc_production:
        from experiments.pirc27.mlmc_production_plugin import mlmc_production_plugin
        plugin = mlmc_production_plugin()
    if mixture:
        plugin = mixture_plugin(synthetic=synthetic)
    if mixture_qualification:
        from experiments.pirc27.mixture_qualification_plugin import mixture_qualification_plugin
        plugin = mixture_qualification_plugin()
    if mixture_production:
        from experiments.pirc27.mixture_production_plugin import mixture_production_plugin
        plugin = mixture_production_plugin()
    if path_qualification:
        from experiments.pirc27.path_qualification_plugin import path_qualification_plugin
        plugin = path_qualification_plugin()
    if path_production:
        from experiments.pirc27.path_production_plugin import path_production_plugin
        plugin = path_production_plugin()
    execution_plan(spec, cell, plugin)
    request = request_from_manifest(cell["propagation_request"])
    package_type = FrozenNonlinearPackage if synthetic else FrozenDynamicsPackage
    try:
        package = package_type.from_manifest(cell["frozen_dynamics"], expected_hash=request.model_package_hash)
        (validate_nonlinear_input if synthetic else validate_oracle_input)(package, expected_package_hash=request.model_package_hash)
    except DataValidationError as exc:
        raise ResearchError("CONTRACT_MISMATCH", "frozen propagation package schema/content/code differs") from exc
    config = cell["execution"]["config"]
    if path_qualification:
        from experiments.pirc27.path_qualification_plugin import path_qualification_policy, path_qualification_config
        policy = path_qualification_policy(spec, cell, package, request, config)
        expected = path_qualification_config(request, policy)
    elif path_production:
        from experiments.pirc27.path_production_plugin import production_policies, path_production_config
        qualification_bound, policy = production_policies(spec, cell, package, request, config)
        expected = path_production_config(request, qualification_bound, policy)
    elif mixture_production:
        from experiments.pirc27.mixture_production_plugin import production_policies, mixture_production_config
        mixture_bound, policy = production_policies(spec, cell, package, request, config)
        expected = mixture_production_config(request, mixture_bound, policy)
    elif mixture:
        policy = mixture_policy(spec, cell, package, request, config)
        expected = mixture_config(request, policy, synthetic=synthetic)
    elif mixture_qualification:
        from experiments.pirc27.mixture_qualification_plugin import mixture_qualification_policies, mixture_qualification_config
        mixture_bound, policy = mixture_qualification_policies(spec, cell, package, request, config)
        expected = mixture_qualification_config(request, mixture_bound, policy)
    else:
        expected = execution_config(request, config["method"], level_samples=tuple(config["level_samples"]), proposal=tuple(config["proposal"]), recovery=recovery, synthetic=synthetic)
    if qualification:
        from experiments.pirc27.qualification_plugin import qualification_policy, qualification_config
        policy = qualification_policy(spec, cell, package, request, config)
        expected = qualification_config(request, config["method"], policy)
    if mlmc_reference:
        from experiments.pirc27.mlmc_qualification_plugin import mlmc_reference_policy, mlmc_qualification_config
        reference = mlmc_reference_policy(spec, cell, package, request, config)
        expected = mlmc_qualification_config(request, reference)
    if mlmc_production:
        from experiments.pirc27.mlmc_production_plugin import production_policy, mlmc_production_config
        policy = production_policy(spec, cell, package, request, config)
        expected = mlmc_production_config(request, policy)
    if config != expected or cell["execution"]["inputs"] != execution_inputs(request) or request.arm_id != cell["arm_id"] or request.seed != cell["seed"]:
        raise ResearchError("CONTRACT_MISMATCH", "registered request, method/resource configuration or arm differs")
    required_capability = {"exact": "exact-transition", "gaussian": "generic-rollout", "euler": "generic-rollout",
                           "heun": "generic-rollout", "reversible-heun": "generic-rollout", "mlmc": "coupled-level", "mlmc-pilot": "coupled-level", "importance": "rare-event", "cubature": "generic-rollout", "mixture": "generic-rollout"}[config["method"]]
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


def execute_propagation(spec, cell, *, resume_state=None, checkpoint=None, admission=None):
    from inference.propagation_methods import (analytic_estimate, monte_carlo, mlmc_estimate, importance_sampling)
    from inference.nonlinear_propagation import cubature_estimate
    from inference.mixture_propagation import mixture_estimate
    from experiments.pirc27.mixture_plugin import mixture_policy
    package, request, config, plugin = validate_propagation_cell(spec, cell)
    from infrastructure.research_admission_selection import select_admission_package
    settings, _ = select_admission_package(spec, cell)
    formal = settings.get("mode") == "formal"
    if formal and admission is None or not formal and admission is not None:
        raise ResearchError("UNQUALIFIED", "formal propagation needs its owner admission, never a caller qualification flag")
    recovery = plugin.resume_level == "chunk"
    if not recovery and (resume_state is not None or checkpoint is not None):
        raise ResearchError("CONTRACT_MISMATCH", "restart-only methods cannot accept chunk state")
    continuation = {"resume_state": resume_state, "checkpoint": checkpoint}
    path_production = plugin.plugin_id == "affine-path-production-chunk"
    path_last = resume_state
    if path_production:
        def path_checkpoint(state, total):
            nonlocal path_last
            if total != request.samples*request.steps:
                raise DataValidationError("target path statistic work proxy differs")
            path_last = state
            if checkpoint is not None:
                checkpoint(state, total)
        continuation["checkpoint"] = path_checkpoint
    mixture_production = plugin.plugin_id == "affine-mixture-production-chunk"
    if mixture_production:
        from experiments.pirc27.mixture_production_plugin import production_policies
        production_mixture, _ = production_policies(spec, cell, package, request, config)
    methods = {"exact": lambda: analytic_estimate(package, request),
        "gaussian": lambda: analytic_estimate(package, request, discrete=True),
        "euler": lambda: monte_carlo(package, request, **continuation),
        "heun": lambda: monte_carlo(package, request, solver="additive-heun", **continuation),
        "reversible-heun": lambda: monte_carlo(package, request, solver="reversible-heun", **continuation),
        "mlmc": lambda: mlmc_estimate(package, request, level_samples=tuple(config["level_samples"]), **continuation),
        "mlmc-pilot": lambda: mlmc_estimate(package, request, level_samples=tuple(config["level_samples"]), phase=2, pilot=True, **continuation),
        "importance": lambda: importance_sampling(package, request, proposal=tuple(config["proposal"]), **continuation),
        "cubature": lambda: cubature_estimate(package, request),
        "mixture": lambda: mixture_estimate(package, request,
            production_mixture if mixture_production else mixture_policy(spec, cell, package, request, config), **continuation)}
    mixture_analysis = None
    path_analysis = None
    if plugin.plugin_id == "affine-path-qualification":
        from experiments.pirc27.path_qualification_plugin import path_qualification_policy
        from inference.path_qualification import qualify_affine_paths
        policy = path_qualification_policy(spec, cell, package, request, config)
        result, path_analysis = qualify_affine_paths(package, request, policy)
    elif plugin.plugin_id == "affine-mixture-qualification":
        from experiments.pirc27.mixture_qualification_plugin import mixture_qualification_policies
        from inference.mixture_qualification import qualify_affine_mixture
        mixture_bound, policy = mixture_qualification_policies(spec, cell, package, request, config)
        result, mixture_analysis = qualify_affine_mixture(package, request, mixture_bound, policy)
    else:
        result = methods[config["method"]]()
    synthetic = isinstance(package, FrozenNonlinearPackage)
    mlmc_production = plugin.plugin_id == "affine-mlmc-production-chunk"
    # Dedicated path targets derive their metric from owner-admitted saved
    # enclosures below. Do not compute and discard another analytic metric.
    # The original sampler's own float64 error-budget diagnostics are unchanged.
    metrics = {}
    if not path_production:
        metrics = ({"functional_estimate": result.estimate} if synthetic else
                   {"absolute_error_vs_float64_reference": abs(result.estimate-analytic_estimate(package, request).estimate)})
    qualified_components = None
    path_output = None
    if formal:
        if path_production:
            from .path_qualification_admission import target_output_analysis, METRIC
            path_output = target_output_analysis(spec, cell, package, request, result.manifest(), path_last, admission)
            error = path_output["total_observed_functional_error_upper"]
        elif mixture_production:
            from .mixture_qualification_admission import qualified_mixture_forecast, METRIC
            error, qualified_components = qualified_mixture_forecast(spec, cell, package, request, result.manifest(), admission)
        elif mlmc_production:
            from .mlmc_qualification_admission import qualified_mlmc_forecast, METRIC
            error, qualified_components = qualified_mlmc_forecast(spec, cell, package, request, result, admission)
        else:
            from .propagation_qualification_admission import qualified_analytic_forecast, METRIC
            error, qualified_components = qualified_analytic_forecast(spec, cell, package, request,
                config["method"], result, admission)
        metrics = {METRIC: error}
    unit = "m" if request.functional == "endpoint-x" else "1"
    output = {"metrics": metrics,
        "forecast": {"kind": result.kind, "horizons": list(request.horizons), "functional": result.manifest(),
                     "model_package_hash": package.package_hash, "request_hash": request.request_hash},
        "fit": {"training": "none; frozen synthetic generator"},
        "source_schema": "synthetic-endpoint-propagation-v1" if synthetic else "endpoint-propagation-v1"}
    if qualified_components is not None:
        output["forecast"]["qualified_error_components"] = qualified_components
        budget = output["forecast"]["functional"]["error_budget"]
        budget["reference"] = {"value": qualified_components["reference_width_upper"], "units": unit,
            "estimated_by": "outward max of continuous and Euler functional interval widths" if mixture_production
                else "outward declared-affine continuous functional interval width", "status": "BOUNDED"}
        budget["time_discretization"] = {"value": qualified_components["time_bias_absolute_upper"], "units": unit,
            "estimated_by": "outward absolute signed grid-minus-continuous expectation bound", "status": "BOUNDED"}
    if path_output is not None:
        output["forecast"]["path_output_analysis"] = path_output
        output["forecast"]["current_output_qualification"] = path_output["status"]
        budget = output["forecast"]["functional"]["error_budget"]
        budget["reference"] = {"value": path_output["reference_width_upper"], "units": unit,
            "estimated_by": "outward max of saved continuous and method-specific grid functional interval widths", "status": "BOUNDED"}
        budget["time_discretization"] = {"value": path_output["absolute_time_bias_upper"], "units": unit,
            "estimated_by": "outward absolute signed grid-minus-continuous expectation bound", "status": "BOUNDED"}
    if config["method"] == "mlmc-pilot":
        from inference.mlmc_pilot import analyze_mlmc_pilot
        output["forecast"]["pilot_analysis"] = analyze_mlmc_pilot(request, result, pilot_policy(spec, cell, request, config))
        if plugin.plugin_id == "affine-mlmc-qualification-chunk":
            from experiments.pirc27.mlmc_qualification_plugin import mlmc_reference_policy
            from inference.affine_mlmc_qualification import analyze_bounded_mlmc_pilot
            output["forecast"]["bounded_mlmc_pilot_analysis"] = analyze_bounded_mlmc_pilot(package, request, result,
                pilot_policy(spec, cell, request, config), mlmc_reference_policy(spec, cell, package, request, config))
    if plugin.plugin_id == "affine-propagation-qualification":
        from experiments.pirc27.qualification_plugin import qualification_policy
        from inference.affine_qualification import analyze_affine_qualification
        policy = qualification_policy(spec, cell, package, request, config)
        output["forecast"]["qualification_analysis"] = analyze_affine_qualification(package, request,
            config["method"], policy, result)
    if mixture_analysis is not None:
        output["forecast"]["mixture_qualification_analysis"] = mixture_analysis
    if path_analysis is not None:
        output["forecast"]["path_qualification_analysis"] = path_analysis
    # Computational completion keeps low-ESS/unresolved estimates as artifacts;
    # estimator status remains visible and cannot be scientific qualification.
    return {"schema_version": "pirc25-result-v1", "status": "SUCCEEDED", "qualification": "qualified" if formal else "fixture",
            **({"admission_hash": admission["admission_hash"]} if formal else {}),
            "spec_hash": digest(spec), "cell_hash": digest(cell), "protocol_hash": spec["protocol_hash"],
            "state_order": list(plugin.state_order), "units": list(plugin.units), "time_unit": "s",
            "resume_level": plugin.resume_level, "input_hash": spec["data_hash"], "output_hash": digest(output),
            "metric_units": {name: unit for name in metrics}, **output}
