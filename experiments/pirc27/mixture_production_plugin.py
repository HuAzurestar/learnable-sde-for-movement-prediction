"""Explicit affine mixture target; original arm and owner admission required."""

from dataclasses import replace

from application.research_contracts import ExecutionPlugin
from domain.errors import DataValidationError
from domain.mixture import MixturePolicy
from domain.mixture_qualification import MixtureQualificationPolicy
from infrastructure.research_store import ResearchError
from .mixture_plugin import mixture_config, mixture_plugin
from .plugin import propagation_command, propagation_resume_command


PLUGIN_ID = "affine-mixture-production-chunk"


def mixture_production_plugin():
    base = mixture_plugin()
    document = base.registry_entry.manifest()
    config = document["config_schema"]
    config["properties"]["qualification_policy_hash"] = {"type": "string", "minLength": 64, "maxLength": 64}
    config["required"] += ["qualification_policy_hash"]
    # Saved proof/owner evidence is bounded too, without reference recomputation.
    for index in range(64):
        document["resource_contract"]["tensors"].append({"name": "saved-mixture-evidence-pool-"+str(index),
            "axes": ["state_dim"]*8, "item_bytes": 16})
    entry = replace(base.registry_entry, component_id=PLUGIN_ID, version="1.0.0",
        config_schema=config, resource_contract=document["resource_contract"])
    return ExecutionPlugin(PLUGIN_ID, base.capabilities, base.state_order, base.units,
        "chunk", propagation_command, entry)


def mixture_production_recovery_plugin():
    from application.research_recovery import RecoveryPlugin
    return RecoveryPlugin(PLUGIN_ID, "chunk", propagation_resume_command, "1.0.0")


def mixture_production_config(request, mixture, policy):
    policy.validate_structure()
    return {**mixture_config(request, mixture), "qualification_policy_hash": policy.policy_hash}


def production_policies(spec, cell, package, request, config):
    from experiments.pirc25.affine import code_hash
    from infrastructure.research_admission_selection import select_admission_package
    try:
        mixture = MixturePolicy.from_manifest(cell["mixture_policy"])
        policy = MixtureQualificationPolicy.from_manifest(cell["mixture_qualification_policy"])
        policy.validate(package, request, mixture, code_hash())
        settings, _ = select_admission_package(spec, cell)
        arms = [a for a in spec["arms"] if a["arm_id"] == request.arm_id]
        if (cell["plugin_id"] != PLUGIN_ID or settings.get("mode") != "formal"
                or cell.get("execution_role") != "production" or len(arms) != 1
                or arms[0]["method_family_id"] != "mixture"
                or config != mixture_production_config(request, mixture, policy)):
            raise ValueError
        return mixture, policy
    except (KeyError, TypeError, ValueError, DataValidationError) as exc:
        raise ResearchError("UNQUALIFIED", "explicit frozen formal mixture original-arm registration required") from exc
