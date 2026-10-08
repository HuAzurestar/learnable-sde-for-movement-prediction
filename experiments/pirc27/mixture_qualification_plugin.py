"""Pilot-only mixture functional evidence; shared original arm and supervisor."""

from dataclasses import replace

from application.research_contracts import ExecutionPlugin
from domain.errors import DataValidationError
from domain.mixture import MixturePolicy
from domain.mixture_qualification import MixtureQualificationPolicy, REFERENCE_WORK_CAP
from infrastructure.research_store import ResearchError
from .mixture_plugin import mixture_config, mixture_plugin
from .plugin import propagation_command


PLUGIN_ID = "affine-mixture-qualification"


def mixture_qualification_plugin():
    base = mixture_plugin()
    document = base.registry_entry.manifest()
    config, contract = document["config_schema"], document["resource_contract"]
    config["properties"]["qualification_policy_hash"] = {"type": "string", "minLength": 64, "maxLength": 64}
    config["required"] += ["qualification_policy_hash"]
    # Fixed64MiB allowance for bounded rational reference/certificate objects;
    # physical state remains4, mixture candidate/live workspaces retained.
    for prefix in ("mixture-rational-reference", "mixture-certificate-serialization"):
        for index in range(32):
            contract["tensors"].append({"name": prefix+"-pool-"+str(index),
                "axes": ["state_dim"]*8, "item_bytes": 16})
    entry = replace(base.registry_entry, component_id=PLUGIN_ID, version="1.0.0",
        resume_level="restart-only", config_schema=config, resource_contract=contract)
    return ExecutionPlugin(PLUGIN_ID, base.capabilities, base.state_order, base.units,
        "restart-only", propagation_command, entry)


def mixture_qualification_config(request, mixture, policy):
    from experiments.pirc25.affine import code_hash
    if type(policy) is not MixtureQualificationPolicy or type(mixture) is not MixturePolicy:
        raise DataValidationError("explicit mixture and qualification policies required")
    policy.validate_structure()
    source = code_hash()
    if (policy.request_hash != request.request_hash or policy.code_hash != source
            or policy.mixture_policy_hash != mixture.policy_hash):
        raise DataValidationError("mixture qualification configuration binding differs")
    config = mixture_config(request, mixture)
    config["work_steps"] += REFERENCE_WORK_CAP
    if config["work_steps"] > 1_000_000:
        raise DataValidationError("mixture plus declared reference work exceeds quota")
    return {**config, "qualification_policy_hash": policy.policy_hash}


def mixture_qualification_policies(spec, cell, package, request, config):
    from experiments.pirc25.affine import code_hash
    from infrastructure.research_admission_selection import select_admission_package
    try:
        mixture = MixturePolicy.from_manifest(cell["mixture_policy"])
        policy = MixtureQualificationPolicy.from_manifest(cell["mixture_qualification_policy"])
        policy.validate(package, request, mixture, code_hash())
        settings, _ = select_admission_package(spec, cell)
        arms = [a for a in spec["arms"] if a["arm_id"] == request.arm_id]
        if (settings.get("mode") != "pilot" or cell.get("execution_role") != "qualification"
                or len(arms) != 1 or arms[0]["method_family_id"] != "mixture"
                or config != mixture_qualification_config(request, mixture, policy)):
            raise ValueError("mixture pilot role/mode/original arm/config differs")
        return mixture, policy
    except (KeyError, TypeError, ValueError) as exc:
        raise ResearchError("UNQUALIFIED", "frozen affine-mixture qualification policies and original-arm pilot required") from exc
