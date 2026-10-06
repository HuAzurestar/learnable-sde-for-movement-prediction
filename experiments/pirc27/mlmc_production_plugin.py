"""Explicit formally admitted MLMC producer; no implicit execution or grants."""

from dataclasses import replace

from application.research_contracts import ExecutionPlugin
from domain.mlmc_production import MLMCProductionPolicy
from domain.errors import DataValidationError
from infrastructure.research_store import ResearchError
from experiments.pirc27.plugin import propagation_command, propagation_plugin, execution_config


PLUGIN_ID = "affine-mlmc-production-chunk"


def mlmc_production_plugin():
    base = propagation_plugin(recovery=True)
    entry = base.registry_entry.manifest()
    config = entry["config_schema"]
    config["properties"]["method"]["enum"] = ["mlmc"]
    config["properties"]["production_policy_hash"] = {"type": "string", "minLength": 64, "maxLength": 64}
    config["required"] += ["production_policy_hash"]
    # Saved bounded-rational evidence is checked inside the worker; reserve
    # conservative fixed pools too, without changing physical state_dim.
    for index in range(64):
        entry["resource_contract"]["tensors"].append({"name": "saved-evidence-pool-"+str(index),
            "axes": ["state_dim"]*8, "item_bytes": 16})
    registered = replace(base.registry_entry, component_id=PLUGIN_ID, version="1.0.0",
        config_schema=config, resource_contract=entry["resource_contract"])
    return ExecutionPlugin(PLUGIN_ID, base.capabilities, base.state_order, base.units,
        "chunk", propagation_command, registered)


def mlmc_production_recovery_plugin():
    from application.research_recovery import RecoveryPlugin
    from experiments.pirc27.plugin import propagation_resume_command
    return RecoveryPlugin(PLUGIN_ID, "chunk", propagation_resume_command, "1.0.0")


def mlmc_production_config(request, policy):
    return {**execution_config(request, "mlmc", level_samples=policy.level_samples, recovery=True),
        "production_policy_hash": policy.policy_hash}


def production_policy(spec, cell, package, request, config):
    from experiments.pirc25.affine import code_hash
    from infrastructure.research_admission_selection import select_admission_package
    try:
        policy = MLMCProductionPolicy.from_manifest(cell["mlmc_production_policy"])
        policy.validate(package, request, code_hash())
        settings, _ = select_admission_package(spec, cell)
        arms = [a for a in spec["arms"] if a["arm_id"] == request.arm_id]
        if (len(arms) != 1 or arms[0]["method_family_id"] != "mlmc"
                or settings.get("mode") != "formal" or cell.get("execution_role") != "production"
                or config != mlmc_production_config(request, policy)):
            raise ValueError
        return policy
    except (KeyError, TypeError, ValueError, DataValidationError) as exc:
        raise ResearchError("UNQUALIFIED", "independent frozen formal MLMC production registration required") from exc
