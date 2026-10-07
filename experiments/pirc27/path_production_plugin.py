"""Dedicated formal path/IS target, owner receipt and original arm required."""

from dataclasses import replace

from application.research_contracts import ExecutionPlugin
from domain.errors import DataValidationError
from domain.path_production import PathProductionPolicy
from domain.path_qualification import PathQualificationPolicy, METHODS
from infrastructure.research_store import ResearchError
from .plugin import propagation_plugin, propagation_command, propagation_resume_command, execution_config


PLUGIN_ID = "affine-path-production-chunk"


def path_production_plugin():
    base = propagation_plugin(recovery=True)
    document = base.registry_entry.manifest()
    config = document["config_schema"]
    config["properties"]["method"]["enum"] = list(METHODS)
    for name in ("qualification_policy_hash", "production_policy_hash"):
        config["properties"][name] = {"type": "string", "minLength": 64, "maxLength": 64}
        config["required"] += [name]
    for index in range(64):
        document["resource_contract"]["tensors"].append({"name": "saved-path-evidence-pool-"+str(index),
            "axes": ["state_dim"]*8, "item_bytes": 16})
    entry = replace(base.registry_entry, component_id=PLUGIN_ID, version="1.0.0",
        config_schema=config, resource_contract=document["resource_contract"])
    return ExecutionPlugin(PLUGIN_ID, base.capabilities, base.state_order, base.units,
        "chunk", propagation_command, entry)


def path_production_recovery_plugin():
    from application.research_recovery import RecoveryPlugin
    return RecoveryPlugin(PLUGIN_ID, "chunk", propagation_resume_command, "1.0.0")


def path_production_config(request, qualification, policy):
    if type(qualification) is not PathQualificationPolicy or type(policy) is not PathProductionPolicy:
        raise DataValidationError("explicit frozen path source/production policies required")
    qualification.validate_structure()
    policy.validate_structure()
    request.validate()
    if (policy.request_hash != request.request_hash or policy.source_request_hash != qualification.request_hash
            or policy.qualification_policy_hash != qualification.policy_hash
            or policy.model_package_hash != request.model_package_hash
            or policy.code_hash != qualification.code_hash
            or qualification.maximum_total_observed_functional_error > request.tolerance
            or qualification.minimum_effective_sample_size > request.samples):
        raise DataValidationError("frozen independent path configuration binding differs")
    config = execution_config(request, qualification.method, proposal=qualification.proposal, recovery=True)
    return {**config, "qualification_policy_hash": qualification.policy_hash, "production_policy_hash": policy.policy_hash}


def production_policies(spec, cell, package, request, config):
    from experiments.pirc25.affine import code_hash
    from infrastructure.research_admission_selection import select_admission_package
    try:
        qualification = PathQualificationPolicy.from_manifest(cell["path_qualification_policy"])
        policy = PathProductionPolicy.from_manifest(cell["path_production_policy"])
        policy.validate(package, request, qualification, code_hash())
        settings, _ = select_admission_package(spec, cell)
        arms = [a for a in spec["arms"] if a["arm_id"] == request.arm_id]
        if (cell["plugin_id"] != PLUGIN_ID or settings.get("mode") != "formal"
                or cell.get("execution_role") != "production" or len(arms) != 1
                or arms[0]["method_family_id"] != qualification.method
                or config != path_production_config(request, qualification, policy)):
            raise ValueError
        return qualification, policy
    except (KeyError, TypeError, ValueError, DataValidationError) as exc:
        raise ResearchError("UNQUALIFIED", "explicit frozen independent formal path original-arm registration required") from exc
