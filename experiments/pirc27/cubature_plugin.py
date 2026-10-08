"""Explicit affine eight-point adapter and pilot, never a model relabeling."""

from dataclasses import replace

from application.research_contracts import ExecutionPlugin
from domain.cubature_qualification import CubatureQualificationPolicy
from domain.errors import DataValidationError
from infrastructure.research_store import ResearchError
from experiments.pirc27.plugin import propagation_plugin, propagation_command, execution_config


PLUGIN_IDS = frozenset({"affine-cubature", "affine-cubature-qualification"})


def cubature_plugin(*, qualification=False):
    # Reuse the bounded eight-point workspace schema, not the nonlinear model
    # package or its scientific claims. Actual admission validates affine input.
    base = propagation_plugin(synthetic=True)
    schema = base.registry_entry.manifest()["config_schema"]
    schema["properties"]["method"]["enum"] = ["cubature"]
    contract = base.registry_entry.manifest()["resource_contract"]
    if qualification:
        schema["properties"]["qualification_policy_hash"] = {"type": "string", "minLength": 64, "maxLength": 64}
        schema["required"].append("qualification_policy_hash")
        contract["counts"]["observations"] = {"constant": 256}
        for prefix in ("rational-reference", "certificate-serialization"):
            for index in range(32):
                contract["tensors"].append({"name": prefix+"-pool-"+str(index),
                    "axes": ["state_dim"]*8, "item_bytes": 16})
    identity = "affine-cubature-qualification" if qualification else "affine-cubature"
    capabilities = frozenset({"generic-rollout"})
    entry = replace(base.registry_entry, component_id=identity, version="1.0.0",
        config_schema=schema, resource_contract=contract, capabilities=capabilities)
    return ExecutionPlugin(identity, capabilities, base.state_order, base.units,
        "restart-only", propagation_command, entry)


def cubature_config(request, *, policy=None):
    config = execution_config(request, "cubature", synthetic=True)
    if policy is not None:
        from experiments.pirc25.affine import code_hash
        if (type(policy) is not CubatureQualificationPolicy or policy.request_hash != request.request_hash
                or policy.model_package_hash != request.model_package_hash or policy.code_hash != code_hash()
                or type(policy.maximum_operations) is not int or not 1 <= policy.maximum_operations <= 400001):
            raise ResearchError("CONTRACT_MISMATCH", "affine cubature qualification configuration differs")
        # Include actual output/replay and the frozen certifier allowance.
        # Qualification is not registered as just the ordinary eight-point job.
        config["work_steps"] = 16*request.steps+policy.maximum_operations
        config["qualification_policy_hash"] = policy.policy_hash
    return config


def cubature_policy(spec, cell, package, request, config):
    from experiments.pirc25.affine import code_hash
    from infrastructure.research_admission_selection import select_admission_package
    try:
        policy = CubatureQualificationPolicy.from_manifest(cell["cubature_qualification_policy"])
        policy.validate(package, request, code_hash())
        settings, _ = select_admission_package(spec, cell)
        if (settings.get("mode") != "pilot" or cell.get("execution_role") != "qualification"
                or config != cubature_config(request, policy=policy)):
            raise ValueError("cubature pilot role/configuration differs")
        return policy
    except (KeyError, TypeError, ValueError, DataValidationError) as exc:
        raise ResearchError("UNQUALIFIED", "explicit frozen affine cubature pilot policy required") from exc
