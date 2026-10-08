"""Restart-only path/IS evidence producer on the existing original-arm pilot."""

from dataclasses import replace

from application.research_contracts import ExecutionPlugin
from domain.errors import DataValidationError
from domain.path_qualification import PathQualificationPolicy, METHODS, REFERENCE_WORK_CAP
from infrastructure.research_store import ResearchError
from .plugin import propagation_plugin, propagation_command, execution_config


PLUGIN_ID = "affine-path-qualification"


def path_qualification_plugin():
    base = propagation_plugin(recovery=True)
    document = base.registry_entry.manifest()
    config, contract = document["config_schema"], document["resource_contract"]
    config["properties"]["method"]["enum"] = list(METHODS)
    config["properties"]["qualification_policy_hash"] = {"type": "string", "minLength": 64, "maxLength": 64}
    config["required"] += ["qualification_policy_hash"]
    for prefix in ("path-rational-reference", "path-certificate-serialization"):
        for index in range(32):
            contract["tensors"].append({"name": prefix+"-pool-"+str(index),
                "axes": ["state_dim"]*8, "item_bytes": 16})
    entry = replace(base.registry_entry, component_id=PLUGIN_ID, version="1.0.0",
        resume_level="restart-only", config_schema=config, resource_contract=contract)
    return ExecutionPlugin(PLUGIN_ID, base.capabilities, base.state_order, base.units,
        "restart-only", propagation_command, entry)


def path_qualification_config(request, policy):
    from experiments.pirc25.affine import code_hash
    if type(policy) is not PathQualificationPolicy:
        raise DataValidationError("explicit path qualification policy required")
    policy.validate_structure()
    request.validate()
    if (policy.request_hash != request.request_hash or policy.code_hash != code_hash()
            or policy.model_package_hash != request.model_package_hash
            or policy.maximum_total_observed_functional_error > request.tolerance
            or policy.minimum_effective_sample_size > request.samples
            or (policy.method == "importance" and request.functional != "endpoint-halfspace")):
        raise DataValidationError("path qualification configuration binding differs")
    config = execution_config(request, policy.method, proposal=policy.proposal, recovery=True)
    config["work_steps"] += REFERENCE_WORK_CAP
    if config["work_steps"] > 1_000_000:
        raise DataValidationError("path plus declared reference work exceeds quota")
    return {**config, "qualification_policy_hash": policy.policy_hash}


def path_qualification_policy(spec, cell, package, request, config):
    from experiments.pirc25.affine import code_hash
    from infrastructure.research_admission_selection import select_admission_package
    try:
        policy = PathQualificationPolicy.from_manifest(cell["path_qualification_policy"])
        policy.validate(package, request, code_hash())
        settings, _ = select_admission_package(spec, cell)
        arms = [a for a in spec["arms"] if a["arm_id"] == request.arm_id]
        if (settings.get("mode") != "pilot" or cell.get("execution_role") != "qualification"
                or len(arms) != 1 or arms[0]["method_family_id"] != policy.method
                or config != path_qualification_config(request, policy)):
            raise ValueError("path pilot role/mode/original arm/config differs")
        return policy
    except (KeyError, TypeError, ValueError) as exc:
        raise ResearchError("UNQUALIFIED", "frozen affine-path policy and original-arm pilot required") from exc
